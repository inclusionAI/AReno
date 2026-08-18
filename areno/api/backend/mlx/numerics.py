"""Shared MLX log-probability numerics for rollout, scoring, and training."""

from __future__ import annotations

from typing import Any


def float32_logits_processor(_tokens: Any, logits: Any):
    """Promote generation logits before MLX-LM normalizes and samples them."""

    import mlx.core as mx

    return logits.astype(mx.float32)


def selected_token_logprobs(logits: Any, targets: Any):
    """Return FP32 log probabilities for selected target token ids."""

    import mlx.core as mx

    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(logprobs, targets[..., None], axis=-1).squeeze(-1)


def chunked_linear_selected_token_logprobs(hidden: Any, targets: Any, weight: Any, *, chunk_size: int, bias=None):
    """Compute exact selected log probabilities without materializing full-vocabulary logits."""

    import mlx.core as mx

    targets = mx.stop_gradient(targets)
    target_weight = mx.take(weight, targets, axis=0)
    target_logits = (hidden * target_weight).sum(axis=-1)
    if bias is not None:
        target_logits = target_logits + mx.take(bias, targets, axis=0)
    target_logits = target_logits.astype(mx.float32)

    def block_logsumexp(hidden_values, weight_values, bias_values=None):
        logits = hidden_values @ weight_values.T
        if bias_values is not None:
            logits = logits + bias_values
        return mx.logsumexp(logits.astype(mx.float32), axis=-1)

    normalizer = None
    step = max(int(chunk_size), 1)
    for start in range(0, int(weight.shape[0]), step):
        end = min(start + step, int(weight.shape[0]))
        weight_block = weight[start:end]
        if bias is None:
            block = mx.checkpoint(block_logsumexp)(hidden, weight_block)
        else:
            block = mx.checkpoint(block_logsumexp)(hidden, weight_block, bias[start:end])
        normalizer = block if normalizer is None else mx.logaddexp(normalizer, block)
    return target_logits - normalizer


__all__ = [
    "chunked_linear_selected_token_logprobs",
    "float32_logits_processor",
    "selected_token_logprobs",
]
