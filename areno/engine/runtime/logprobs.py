"""Selected next-token logprob computation under vocab-parallel TP.

Both training and rollout need to score targets against a vocabulary that is
sharded across TP ranks. A naive implementation would all-gather the full
logits and then call `log_softmax` and `gather`, which is bandwidth-heavy and
recomputes work each rank already did locally. The kernel here instead
reduces only three scalars per position (per-row max, exp-sum, target logit),
so the all-reduce volume scales with the number of positions, not the vocab.

The autograd version saves the unnormalized exp-shard and produces the local
gradient slice in backward, again avoiding any full-vocab tensor on the
training hot path.
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
    if cu_seqlens.numel() > 1 and bool(torch.any(cu_seqlens[1:] <= cu_seqlens[:-1])):
        raise ValueError("packed_next_token_logprobs requires strictly increasing cu_seqlens")
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
    out, _, _, _ = _selected_logprobs_components(logits_shard, labels, vocab_start, group, world_size, save_probs=False)
    return out


def _selected_logprobs_components(
    logits_shard: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    group,
    world_size: int,
    *,
    save_probs: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Distributed log-softmax over a vocab shard, selecting label probabilities.

    Steps (each cross-rank step is one TP all-reduce):
    1. Row-wise max within this rank, then MAX-reduce to a global per-row max
       used to shift logits for numerical stability.
    2. Compute `exp(logits - global_max)` locally and SUM-reduce row sums to
       get the partition function (its log is `logsumexp`).
    3. Each rank fills in target logits only for labels inside its shard,
       SUM-reduce to combine into the full per-row target logit.
    Output is `target - logsumexp`, the selected log-softmax value.
    """
    logits = logits_shard.float()
    labels = labels.to(device=logits_shard.device, dtype=torch.long)
    local_vocab = logits.shape[-1]
    local_labels = labels - int(vocab_start)
    local_mask = (local_labels >= 0) & (local_labels < local_vocab)

    local_max = logits.max(dim=-1).values
    global_max = local_max.clone()
    if world_size > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

    exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
    exp_sum = exp_logits.sum(dim=-1)
    if world_size > 1:
        dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM, group=group)
    logsumexp = global_max + exp_sum.log()

    # Off-shard label indices are clamped to a valid local position; the
    # resulting target value is zeroed via `local_mask` so the SUM-reduce
    # picks the correct rank's contribution.
    safe_labels = local_labels.clamp(min=0, max=max(local_vocab - 1, 0))
    target = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    target = target.masked_fill(~local_mask, 0.0)
    if world_size > 1:
        dist.all_reduce(target, op=dist.ReduceOp.SUM, group=group)
    probs = exp_logits / exp_sum.unsqueeze(-1) if save_probs else None
    return target - logsumexp, probs, safe_labels, local_mask


class _VocabParallelSelectedLogprobs(torch.autograd.Function):
    """Autograd function for vocab-parallel selected logprobs.

    Forward computes distributed log-softmax only for target labels. Backward
    returns the local shard gradient equivalent to full-vocab cross entropy,
    avoiding a large all-gather on the training path.
    """

    @staticmethod
    def forward(
        ctx, logits_shard: torch.Tensor, labels: torch.Tensor, vocab_start: int, group, world_size: int
    ) -> torch.Tensor:
        out, probs, safe_labels, local_mask = _selected_logprobs_components(
            logits_shard,
            labels,
            vocab_start,
            group,
            world_size,
            save_probs=True,
        )
        ctx.save_for_backward(probs, safe_labels, local_mask)
        ctx.input_dtype = logits_shard.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Standard log-softmax gradient: `softmax(logits) - onehot(label)`,
        # multiplied by upstream `grad_output`. We do the onehot piece only on
        # the rank that owns the label (`local_mask`) so other ranks contribute
        # only the negative softmax term; the sum across ranks reproduces the
        # full-vocab gradient.
        probs, safe_labels, local_mask = ctx.saved_tensors
        grad = probs
        grad.neg_()
        grad.scatter_add_(
            -1,
            safe_labels.unsqueeze(-1),
            local_mask.to(dtype=grad.dtype).unsqueeze(-1),
        )
        grad.mul_(grad_output.float().unsqueeze(-1))
        return grad.to(dtype=ctx.input_dtype), None, None, None, None
