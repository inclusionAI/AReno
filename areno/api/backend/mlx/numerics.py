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


__all__ = ["float32_logits_processor", "selected_token_logprobs"]
