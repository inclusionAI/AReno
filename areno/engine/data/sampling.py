"""Token sampling for rollout (greedy and stochastic) under vocab-parallel TP.

Two sampling paths exist because tensor parallelism splits the vocabulary
axis across ranks:
- Greedy sampling can be done shard-locally: each rank returns its best
  (value, local_id) and one all-gather of small tensors picks the global
  argmax without ever materializing the full vocabulary on any rank.
- Stochastic sampling (top-k/top-p/temperature) needs a normalized
  distribution over the full vocabulary, so we pay one all-gather over the
  last dimension and sample on rank-0-shape logits.

Helpers in this module also implement EOS suppression for `min_new_tokens`,
explicit suppress lists, and per-row finish-reason truncation.
"""

from __future__ import annotations

import torch

from areno.engine.data import SamplingParams
from areno.engine.parallel.collectives import all_gather_first_dim, all_gather_last_dim
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.logprobs import vocab_parallel_selected_logprobs


def _sample_greedy_sharded(
    logits_shard: torch.Tensor,
    vocab_size: int,
    tp_size: int,
    *,
    eos_token_id: int | tuple[int, ...] | None,
    sample_step: int,
    min_new_tokens: int,
    suppress_token_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    """Pick greedy tokens when each TP rank owns only one vocab shard.

    Each rank computes its local best token, then ranks exchange just the local
    maxima instead of gathering the full vocabulary logits.
    """
    ctx = get_tp_context()
    local_vocab = vocab_size // tp_size
    scores = logits_shard.float()
    eos_ids = _token_id_tuple(eos_token_id)
    # Translate global suppression ids into this rank's local vocab indices;
    # an id outside the rank's shard simply produces no local entry.
    suppress_ids = _local_token_ids(suppress_token_ids, ctx.rank * local_vocab, local_vocab)
    if suppress_ids:
        scores = scores.clone()
        scores[:, torch.tensor(suppress_ids, device=scores.device, dtype=torch.long)] = float("-inf")
    if eos_ids and sample_step < min_new_tokens:
        # Mask EOS tokens whose ids fall inside this rank's shard so the global
        # argmax cannot pick them before we have produced `min_new_tokens`.
        vocab_start = ctx.rank * local_vocab
        local_eos = [token_id - vocab_start for token_id in eos_ids if 0 <= token_id - vocab_start < local_vocab]
        if local_eos:
            if not suppress_ids:
                scores = scores.clone()
            scores[:, torch.tensor(local_eos, device=scores.device, dtype=torch.long)] = float("-inf")
    # local_values: per-row max, local_ids: per-row argmax inside the shard.
    local_values, local_ids = scores.max(dim=-1)
    # Each gathered tensor has shape (tp_size, batch); pick the rank with the
    # largest value, then index that rank's local-id row to get the chosen
    # token inside its shard.
    gathered_values = all_gather_first_dim(local_values)
    gathered_ids = all_gather_first_dim(local_ids)
    best_ranks = gathered_values.argmax(dim=0)
    best_local_ids = gathered_ids.gather(0, best_ranks.unsqueeze(0)).squeeze(0)
    # Reconstruct the global token id from (rank, local_id).
    return best_ranks.to(torch.long) * local_vocab + best_local_ids.to(torch.long)


def _sample_full_vocab(
    logits_shard: torch.Tensor,
    params: SamplingParams,
    vocab_size: int,
    tp_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None,
    eos_token_id: int | tuple[int, ...] | None,
    sample_step: int,
) -> torch.Tensor:
    """Gather vocab shards before stochastic sampling.

    Top-k/top-p sampling needs a normalized distribution over the full
    vocabulary, so this path intentionally pays the all-gather cost.
    """
    del vocab_size, tp_size, device
    logits = all_gather_last_dim(logits_shard)
    return _sample(
        logits,
        params,
        logits.device,
        generator=generator,
        eos_token_id=eos_token_id,
        sample_step=sample_step,
    )


def _sample(
    logits: torch.Tensor,
    params: SamplingParams,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    eos_token_id: int | tuple[int, ...] | None = None,
    sample_step: int = 0,
) -> torch.Tensor:
    """Sample next tokens from full-vocab logits with suppression and min length."""

    # Replace NaNs and positive infinities by clamp values so the downstream
    # softmax and multinomial calls always receive finite, monotone scores.
    scores = torch.nan_to_num(
        logits.float(),
        nan=float("-inf"),
        posinf=torch.finfo(torch.float32).max,
        neginf=float("-inf"),
    )
    suppress_ids = [token_id for token_id in params.suppress_token_ids if 0 <= token_id < scores.shape[-1]]
    eos_ids = [token_id for token_id in _token_id_tuple(eos_token_id) if 0 <= token_id < scores.shape[-1]]
    if suppress_ids or (eos_ids and sample_step < params.min_new_tokens):
        scores = scores.clone()
    if suppress_ids:
        scores[:, torch.tensor(suppress_ids, device=scores.device, dtype=torch.long)] = float("-inf")
    if eos_ids and sample_step < params.min_new_tokens:
        scores[:, torch.tensor(eos_ids, device=scores.device, dtype=torch.long)] = float("-inf")
    if params.temperature == 0.0:
        return torch.argmax(scores, dim=-1)
    # Compute a deterministic fallback in case stochastic sampling sees a row
    # whose probability mass is all zero after top-k/top-p filtering.
    fallback = torch.argmax(scores, dim=-1)
    probs = torch.softmax(scores / params.temperature, dim=-1)
    probs = _sanitize_probs(probs, fallback)
    if params.top_k > 0 or params.top_p < 1.0:
        # Sort then zero out tokens that fall outside top-k or top-p so the
        # remaining distribution is exactly the truncated one. The mask is
        # `prefix sum (excluding self) > top_p`, which keeps the smallest
        # prefix whose sum first exceeds `top_p`.
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
        if params.top_k > 0:
            vocab = probs.shape[-1]
            k = min(params.top_k, vocab)
            probs_sort[:, k:] = 0.0
        if params.top_p < 1.0:
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            probs_sort[(probs_sum - probs_sort) > params.top_p] = 0.0
        probs_sort = _sanitize_sorted_probs(probs_sort)
        sampled = torch.multinomial(probs_sort, 1, generator=generator)
        return probs_idx.gather(-1, sampled).squeeze(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def _policy_token_logprobs(
    logits_shard: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute raw policy logprobs with the same TP kernel used by training."""

    return vocab_parallel_selected_logprobs(logits_shard, tokens)


def _make_sample_generator(params: SamplingParams, device: torch.device) -> torch.Generator | None:
    """Create a deterministic CUDA/CPU generator when stochastic sampling uses a seed."""

    if params.temperature == 0.0 or params.seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(params.seed)
    return generator


def _sanitize_probs(probs: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Renormalize probabilities and fall back to the argmax row when degenerate."""

    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    row_sum = probs.sum(dim=-1, keepdim=True)
    normalized = probs / row_sum.clamp_min(torch.finfo(probs.dtype).tiny)
    fallback_probs = torch.zeros_like(probs)
    # If a row collapsed to all zeros, replace it with a one-hot on the
    # deterministic argmax fallback so multinomial still has a valid input.
    fallback_probs.scatter_(1, fallback.view(-1, 1), 1.0)
    return torch.where(row_sum > 0.0, normalized, fallback_probs)


def _sanitize_sorted_probs(probs: torch.Tensor) -> torch.Tensor:
    """Renormalize sorted top-k/top-p probabilities or fall back to the highest score."""

    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    row_sum = probs.sum(dim=-1, keepdim=True)
    normalized = probs / row_sum.clamp_min(torch.finfo(probs.dtype).tiny)
    fallback_probs = torch.zeros_like(probs)
    # Sorted view places the highest-probability index at column 0; selecting
    # it yields the deterministic argmax for the original vocab.
    fallback_probs[:, 0] = 1.0
    return torch.where(row_sum > 0.0, normalized, fallback_probs)


def _stop_token_ids(params: SamplingParams, eos_token_id: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """Merge EOS ids and user stop ids without duplicates."""

    ids = []
    ids.extend(_token_id_tuple(eos_token_id))
    ids.extend(int(token_id) for token_id in params.stop_token_ids)
    return tuple(dict.fromkeys(ids))


def _token_id_tuple(value: int | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    """Normalize an int/iterable/None argument into a tuple of ints."""

    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(token_id) for token_id in value)


def _local_token_ids(token_ids: tuple[int, ...], vocab_start: int, local_vocab: int) -> list[int]:
    """Translate global token ids into the local vocab range for this TP rank."""

    return [token_id - vocab_start for token_id in token_ids if 0 <= token_id - vocab_start < local_vocab]


def _tokens_match_any(tokens: torch.Tensor, token_ids: tuple[int, ...]) -> torch.Tensor:
    """Return a per-token boolean mask of `tokens` that equal any stop id."""

    if not token_ids:
        return torch.zeros_like(tokens, dtype=torch.bool)
    stop = torch.tensor(token_ids, device=tokens.device, dtype=tokens.dtype)
    return tokens.unsqueeze(-1).eq(stop).any(dim=-1)


def _truncate_generated(rows: list[list[int]], stop_token_ids: tuple[int, ...]) -> tuple[list[list[int]], list[str]]:
    """Cut generated rows at the first stop token and record finish reasons."""
    if not stop_token_ids:
        return rows, ["length" for _ in rows]
    out = []
    finish_reason = []
    stop_set = set(stop_token_ids)
    for row in rows:
        stop_idx = next((idx for idx, token_id in enumerate(row) if token_id in stop_set), None)
        if stop_idx is None:
            out.append(row)
            finish_reason.append("length")
        else:
            # Keep the stop token itself so downstream consumers can see why
            # generation halted; reason becomes "stop" instead of "length".
            out.append(row[: stop_idx + 1])
            finish_reason.append("stop")
    return out, finish_reason
