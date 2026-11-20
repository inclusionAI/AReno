"""Cut Cross-Entropy: chunked log-sum-exp without materializing logits.

``F.cross_entropy(hidden @ weight.T, targets)`` materializes a
``[rows, vocab]`` logit tensor, which is several gigabytes for long sequences
and large vocabularies. This module computes the same loss (and the same
selected log-probs) by streaming the vocabulary in chunks:

1. project one vocabulary chunk into ``[rows, chunk]`` logits,
2. reduce it immediately into a per-row ``(max, sum-of-exp)`` pair,
3. fold the per-chunk pairs into one log-sum-exp, and
4. score the target rows with a single ``rows x hidden`` dot product.

Peak activation memory therefore scales with ``rows * vocab_chunk`` instead of
``rows * vocab``. Each chunk reduction runs under activation checkpointing, so
backward recomputes the chunk instead of retaining every chunk's logits.

The dtype split mirrors what a normal ``lm_head`` does: the projection runs in
the head weight dtype, while every reduction (and the log-sum-exp itself)
accumulates in FP32, so bf16/fp16 training keeps the log-prob precision of the
materialized-logits path.

Only the arithmetic is fused here. This is the reference implementation used
for CPU tests and for devices without a fused kernel; a CUDA/Triton variant can
later replace :func:`_chunk_row_stats` without changing the public API.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

DEFAULT_VOCAB_CHUNK_SIZE = 4096

_REDUCTIONS = ("none", "sum", "mean")


def cut_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    *,
    label_smoothing: float = 0.0,
    logit_softcap: float | None = None,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Cross-entropy from hidden states, the head weight, and target ids.

    Mathematically equivalent to ``F.cross_entropy(hidden_states @ weight.T,
    target_ids, ignore_index=ignore_index, reduction=reduction)`` but without
    ever materializing the ``[rows, vocab]`` logits.

    Args:
        hidden_states: ``[..., hidden]`` activations; all leading dims are
            flattened into rows.
        weight: ``[vocab, hidden]`` projection weight (an LM head weight).
        target_ids: ``[...]`` target ids with one entry per row of
            ``hidden_states``.
        ignore_index: rows with this target are excluded from ``sum``/``mean``
            and contribute ``0`` under ``reduction="none"``.
        reduction: ``"none"``, ``"sum"``, or ``"mean"``.
        label_smoothing: only ``0.0`` is implemented; anything else raises.
        logit_softcap: optional ``cap * tanh(logits / cap)`` cap applied to the
            logits, as used by Gemma-style final logit soft-capping.
        vocab_chunk_size: vocabulary rows projected per chunk.

    When every target is ``ignore_index`` the result is ``0`` for ``sum`` and
    ``mean`` (``F.cross_entropy`` yields ``nan`` for ``mean`` there), so an
    all-masked microbatch keeps the optimizer step finite.
    """

    if label_smoothing != 0.0:
        raise NotImplementedError("cut_cross_entropy implements label_smoothing=0 only")
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")

    hidden, targets, weight = _prepare(hidden_states, weight, target_ids, vocab_chunk_size)
    if hidden.shape[0] == 0:
        if reduction == "none":
            return torch.empty(0, device=hidden.device, dtype=torch.float32)
        return (hidden_states.sum() * 0.0).float()

    logprobs = cut_cross_entropy_logprobs(
        hidden,
        weight,
        targets,
        logit_softcap=logit_softcap,
        vocab_chunk_size=vocab_chunk_size,
    )
    nll = -logprobs
    valid = targets != ignore_index
    if reduction == "none":
        return torch.where(valid, nll, torch.zeros_like(nll))
    total = nll.masked_fill(~valid, 0.0).sum()
    if reduction == "sum":
        return total
    return total / valid.sum().clamp(min=1).to(total.dtype)


def cut_cross_entropy_logprobs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    logit_softcap: float | None = None,
    vocab_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Selected log-softmax values without materializing the logits.

    Returns an FP32 ``[rows]`` tensor of ``logits[target] - logsumexp(logits)``,
    the quantity next-token scoring feeds into every AReno loss function. Rows
    whose target id is unused (``ignore_index``) still produce a finite value;
    masking is the caller's job via :func:`cut_cross_entropy`.
    """

    hidden, targets, weight = _prepare(hidden_states, weight, target_ids, vocab_chunk_size)
    if hidden.shape[0] == 0:
        return torch.empty(0, device=hidden.device, dtype=torch.float32)

    softcap = _validated_softcap(logit_softcap)
    logsumexp = _chunked_logsumexp(hidden, weight, softcap, vocab_chunk_size)
    return _target_logits(hidden, weight, targets, softcap) - logsumexp


def _validated_softcap(logit_softcap: float | None) -> float | None:
    if logit_softcap is None:
        return None
    cap = float(logit_softcap)
    if cap <= 0.0:
        raise ValueError(f"logit_softcap must be positive or None, got {logit_softcap!r}")
    return cap


def _prepare(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate inputs and flatten the row axis."""

    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [vocab, hidden], got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise TypeError(f"weight must be a floating-point tensor, got {weight.dtype}")
    if weight.shape[0] == 0:
        raise ValueError("weight must have at least one vocabulary row")
    if hidden_states.ndim == 0 or hidden_states.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"hidden_states last dim must match weight hidden dim {weight.shape[1]}, "
            f"got shape {tuple(hidden_states.shape)}"
        )
    if isinstance(vocab_chunk_size, bool) or not isinstance(vocab_chunk_size, int) or vocab_chunk_size < 1:
        raise ValueError(f"vocab_chunk_size must be a positive integer, got {vocab_chunk_size!r}")

    hidden = hidden_states.reshape(-1, weight.shape[1])
    targets = target_ids.reshape(-1).to(dtype=torch.long)
    if targets.numel() != hidden.shape[0]:
        raise ValueError(
            f"target_ids has {targets.numel()} entries but hidden_states has {hidden.shape[0]} rows"
        )
    return hidden, targets, weight


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """FP32 accumulation for bf16/fp16 inputs, unchanged for fp32/fp64."""

    return torch.promote_types(dtype, torch.float32)


def _chunk_row_stats(
    hidden: torch.Tensor,
    weight_chunk: torch.Tensor,
    softcap: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row max and ``sum(exp(x - max))`` for one vocabulary chunk."""

    acc_dtype = _accumulation_dtype(weight_chunk.dtype)
    logits = (hidden @ weight_chunk.transpose(0, 1)).to(acc_dtype)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    chunk_max = logits.amax(dim=-1)
    chunk_sum = torch.exp(logits - chunk_max.unsqueeze(-1)).sum(dim=-1)
    return chunk_max, chunk_sum


def _chunked_logsumexp(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    softcap: float | None,
    vocab_chunk_size: int,
) -> torch.Tensor:
    """Combine per-chunk log-sum-exp pieces into one FP32 ``[rows]`` value."""

    maxes: list[torch.Tensor] = []
    sums: list[torch.Tensor] = []
    # Backward recomputes each chunk instead of retaining its logits; the
    # forward-only path (inference, eval) skips checkpointing entirely.
    use_checkpoint = torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    for start in range(0, weight.shape[0], vocab_chunk_size):
        chunk = weight[start : start + vocab_chunk_size]
        if use_checkpoint:
            chunk_max, chunk_sum = checkpoint(_chunk_row_stats, hidden, chunk, softcap, use_reentrant=False)
        else:
            chunk_max, chunk_sum = _chunk_row_stats(hidden, chunk, softcap)
        maxes.append(chunk_max)
        sums.append(chunk_sum)

    chunk_maxes = torch.stack(maxes)
    chunk_sums = torch.stack(sums)
    row_max = chunk_maxes.amax(dim=0)
    total = (chunk_sums * torch.exp(chunk_maxes - row_max)).sum(dim=0)
    return row_max + total.log()


def _target_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None,
) -> torch.Tensor:
    """``rows x hidden`` dot product against the target rows only.

    The projection uses the same input and output dtypes as each vocabulary
    chunk. Only the subsequent softcap and log-sum-exp reductions are promoted,
    so the numerator and denominator see identical rounded logits.
    """

    safe_targets = targets.clamp(min=0, max=weight.shape[0] - 1)
    selected = weight.index_select(0, safe_targets)
    logits = torch.bmm(hidden.unsqueeze(1), selected.unsqueeze(2)).reshape(-1)
    logits = logits.to(_accumulation_dtype(weight.dtype))
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


__all__ = ["DEFAULT_VOCAB_CHUNK_SIZE", "cut_cross_entropy", "cut_cross_entropy_logprobs"]
