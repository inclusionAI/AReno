"""Selected next-token logprob computation under vocab-parallel TP.

Both training and rollout need to score targets against a vocabulary that is
sharded across TP ranks. A naive implementation would all-gather the full
logits and then call `log_softmax` and `gather`, which is bandwidth-heavy and
recomputes work each rank already did locally. The kernel here instead
reduces only three scalars per position (per-row max, exp-sum, target logit),
so the all-reduce volume scales with the number of positions, not the vocab.

The autograd version reduces only row maxima, vocab-chunked exp-sums, and
target logits in forward, and recomputes the per-vocab-chunk softmax in
backward to form the local gradient slice -- again avoiding any full-vocab
tensor on the training hot path.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from areno.engine.parallel.context import get_tp_context


def next_token_logprobs(
    logits_shard: torch.Tensor,
    tokens: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute selected next-token logprobs for padded train rows."""

    # For each row, the prediction at position `t` targets `tokens[:, t+1]`.
    # We process in chunks along the time axis to bound peak memory used by
    # the per-chunk all-reduce in `vocab_parallel_selected_logprobs`.
    steps = max(logits_shard.shape[1] - 1, 0)
    selected = torch.empty(logits_shard.shape[0], steps, device=logits_shard.device, dtype=torch.float32)
    for start in range(0, steps, chunk_size):
        end = min(start + chunk_size, steps)
        targets = tokens[:, start + 1 : end + 1]
        local_logits = logits_shard[:, start:end].reshape(-1, logits_shard.shape[-1])
        selected[:, start:end] = vocab_parallel_selected_logprobs(local_logits, targets.reshape(-1)).view_as(targets)
    return selected


def packed_next_token_logprobs(
    logits_shard: torch.Tensor,
    tokens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute selected next-token logprobs for packed varlen train rows."""

    # Packed layout: `tokens` is a flat sequence of all concatenated rows and
    # `cu_seqlens` marks per-row boundaries. We materialize a flat `positions`
    # tensor pointing at each prediction site and a matching `labels` tensor
    # of the next token, then run the same TP kernel over those.
    flat_tokens = tokens.reshape(-1)
    cu_seqlens = cu_seqlens.to(device=tokens.device, dtype=torch.long)
    # Training packs every row with at least one token, so the number of
    # next-token action sites is total_tokens minus one tail token per row.
    # Keep this as shape arithmetic to avoid a GPU sync from `.item()`.
    action_count = max(flat_tokens.numel() - (cu_seqlens.numel() - 1), 0)
    selected = torch.empty(action_count, device=logits_shard.device, dtype=torch.float32)
    if action_count == 0:
        return selected

    # `positions[k]` is the packed index whose logits predict `labels[k]`.
    # Drop each sequence tail because it has no next-token target.
    positions = torch.arange(flat_tokens.numel(), device=tokens.device)
    keep = torch.ones(flat_tokens.numel(), device=tokens.device, dtype=torch.bool)
    keep[cu_seqlens[1:] - 1] = False
    positions = positions[keep]
    labels = flat_tokens[positions + 1]

    for start in range(0, action_count, chunk_size):
        end = min(start + chunk_size, action_count)
        local_logits = logits_shard[:, positions[start:end]].squeeze(0)
        selected[start:end] = vocab_parallel_selected_logprobs(local_logits, labels[start:end])
    return selected


def vocab_parallel_selected_logprobs(logits_shard: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Select label logprobs without gathering full vocabulary logits.

    Tensor parallel ranks each hold a vocab shard. This computes the same
    selected log-softmax values as a full-vocab `log_softmax(...).gather(...)`
    by reducing only row maxima, denominator sums, and target logits.
    """

    if logits_shard.numel() == 0:
        return torch.empty(labels.shape, device=logits_shard.device, dtype=torch.float32)
    ctx = get_tp_context()
    vocab_start = ctx.rank * logits_shard.shape[-1]
    # Use the autograd-aware path only when grad is required. Inference and
    # forward-only scoring paths get the cheaper functional implementation.
    if not torch.is_grad_enabled() or not logits_shard.requires_grad:
        return _vocab_parallel_selected_logprobs_forward(logits_shard, labels, vocab_start, ctx.group, ctx.world_size)
    return _VocabParallelSelectedLogprobs.apply(logits_shard, labels, vocab_start, ctx.group, ctx.world_size)


def _vocab_parallel_selected_logprobs_forward(
    logits_shard: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    group,
    world_size: int,
) -> torch.Tensor:
    return _selected_logprobs_components_forward(logits_shard, labels, vocab_start, group, world_size)


def _selected_logprobs_components_forward(
    logits_shard: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    group,
    world_size: int,
    *,
    vocab_chunk_size: int = 8192,
) -> torch.Tensor:
    """Forward-only selected logprobs without materializing full-vocab probs."""

    out, _safe_labels, _local_mask, _global_max, _exp_sum = _selected_logprobs_forward(
        logits_shard, labels, vocab_start, group, world_size, vocab_chunk_size=vocab_chunk_size
    )
    return out


def _selected_logprobs_forward(
    logits_shard: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    group,
    world_size: int,
    *,
    vocab_chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distributed selected-logprob forward that never materializes full-vocab
    float logits or softmax probabilities.

    Computes the per-row selected log-softmax (`target_logit - logsumexp`) by
    reducing row maxima, vocab-chunked exp-sums, and target logits (each
    cross-rank step is one TP all-reduce). Returns the output plus the lean
    reductions needed to recompute the softmax gradient in backward without
    saving the full `[positions, vocab]` probs tensor.
    """
    labels = labels.to(device=logits_shard.device, dtype=torch.long)
    local_vocab = logits_shard.shape[-1]
    local_labels = labels - int(vocab_start)
    local_mask = (local_labels >= 0) & (local_labels < local_vocab)

    local_max = logits_shard.max(dim=-1).values.float()
    global_max = local_max.clone()
    if world_size > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

    exp_sum = torch.zeros_like(global_max, dtype=torch.float32)
    for start in range(0, local_vocab, vocab_chunk_size):
        end = min(start + vocab_chunk_size, local_vocab)
        exp_sum += torch.exp(logits_shard[..., start:end].float() - global_max.unsqueeze(-1)).sum(dim=-1)
    if world_size > 1:
        dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM, group=group)
    logsumexp = global_max + exp_sum.log()

    # Off-shard label indices are clamped to a valid local position; the
    # resulting target value is zeroed via `local_mask` so the SUM-reduce
    # picks the correct rank's contribution.
    safe_labels = local_labels.clamp(min=0, max=max(local_vocab - 1, 0))
    target = logits_shard.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).float()
    target = target.masked_fill(~local_mask, 0.0)
    if world_size > 1:
        dist.all_reduce(target, op=dist.ReduceOp.SUM, group=group)
    return target - logsumexp, safe_labels, local_mask, global_max, exp_sum


def _selected_logprobs_recompute_backward(
    grad_output: torch.Tensor,
    logits_shard: torch.Tensor,
    safe_labels: torch.Tensor,
    local_mask: torch.Tensor,
    global_max: torch.Tensor,
    exp_sum: torch.Tensor,
    *,
    vocab_chunk_size: int = 8192,
) -> torch.Tensor:
    """Backward recomputing softmax probs per vocab chunk.

    The gradient of `target - logsumexp` w.r.t. the local logits shard is
    `grad_output * (onehot(label) - softmax(logits))` restricted to the local
    shard, where `softmax = exp(logits - global_max) / exp_sum`. Recomputing it
    in vocab chunks keeps peak memory at `positions * vocab_chunk_size` float
    instead of the full `positions * vocab` probs tensor that the save-probs
    form would retain across every position chunk for backward.
    """
    input_dtype = logits_shard.dtype
    local_vocab = logits_shard.shape[-1]
    grad_output_row = grad_output.float().unsqueeze(-1)
    grad_logits = torch.empty_like(logits_shard)
    for start in range(0, local_vocab, vocab_chunk_size):
        end = min(start + vocab_chunk_size, local_vocab)
        probs_chunk = torch.exp(
            logits_shard[..., start:end].float() - global_max.unsqueeze(-1)
        ) / exp_sum.unsqueeze(-1)
        grad_chunk = -probs_chunk
        # Add the one-hot for labels whose target falls inside this vocab chunk;
        # out-of-shard or out-of-chunk labels contribute zero (local_mask False).
        in_range = local_mask & (safe_labels >= start) & (safe_labels < end)
        rel_labels = (safe_labels - start).clamp(min=0, max=max(end - start - 1, 0))
        grad_chunk.scatter_add_(
            -1, rel_labels.unsqueeze(-1), in_range.to(grad_chunk.dtype).unsqueeze(-1)
        )
        grad_chunk.mul_(grad_output_row)
        grad_logits[..., start:end] = grad_chunk.to(input_dtype)
    return grad_logits


class _VocabParallelSelectedLogprobs(torch.autograd.Function):
    """Autograd function for vocab-parallel selected logprobs.

    Forward computes distributed log-softmax only for target labels via
    vocab-chunked reductions, never materializing full-vocab float logits or
    probabilities. Backward recomputes the per-vocab-chunk softmax to form the
    local shard gradient equivalent to full-vocab cross entropy, avoiding both
    a large all-gather and saving a full `[positions, vocab]` probs tensor on
    the training hot path (which OOMs on long packed sequences with large
    vocabularies).
    """

    @staticmethod
    def forward(
        ctx, logits_shard: torch.Tensor, labels: torch.Tensor, vocab_start: int, group, world_size: int
    ) -> torch.Tensor:
        out, safe_labels, local_mask, global_max, exp_sum = _selected_logprobs_forward(
            logits_shard, labels, vocab_start, group, world_size
        )
        ctx.save_for_backward(logits_shard, safe_labels, local_mask, global_max, exp_sum)
        ctx.input_dtype = logits_shard.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Standard log-softmax gradient: `onehot(label) - softmax(logits)`,
        # multiplied by upstream `grad_output`, recomputed per vocab chunk. The
        # onehot piece is applied only on the rank that owns the label
        # (`local_mask`) so other ranks contribute only the negative softmax
        # term; the sum across ranks reproduces the full-vocab gradient.
        logits_shard, safe_labels, local_mask, global_max, exp_sum = ctx.saved_tensors
        grad_logits = _selected_logprobs_recompute_backward(
            grad_output, logits_shard, safe_labels, local_mask, global_max, exp_sum
        )
        return grad_logits, None, None, None, None
