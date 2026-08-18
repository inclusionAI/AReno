"""MLX training-pack, learning-rate, and gradient helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from areno.api.models import TrainSequence


def make_train_batch(rows: list[TrainSequence]) -> dict[str, Any]:
    """Pack public training rows into right-padded MLX arrays."""

    import mlx.core as mx

    if not rows:
        raise ValueError("MLX microbatch is empty")
    lengths = [len(row.tokens) for row in rows]
    width = max(lengths)
    input_ids = np.full((len(rows), width), int(rows[0].eos_token_id), dtype=np.int32)
    prompt_mask = np.ones((len(rows), width), dtype=np.bool_)
    loss_mask = np.zeros((len(rows), width), dtype=np.bool_)
    old_logprobs = np.zeros((len(rows), width), dtype=np.float32)
    advantages = np.zeros((len(rows), width), dtype=np.float32)
    ref_logprobs = np.zeros((len(rows), width), dtype=np.float32)
    returns = np.zeros((len(rows), width), dtype=np.float32)
    values = np.zeros((len(rows), width), dtype=np.float32)
    has_ref = False
    has_returns = False
    has_values = False

    for index, row in enumerate(rows):
        length = lengths[index]
        input_ids[index, :length] = row.tokens
        row_prompt_mask = _prompt_mask(row, length)
        prompt_mask[index, :length] = row_prompt_mask
        active = ~row_prompt_mask
        if row.loss_mask:
            provided = np.asarray(row.loss_mask[:length], dtype=np.bool_)
            loss_mask[index, : len(provided)] = provided
            active = loss_mask[index, :length]
        else:
            loss_mask[index, :length] = active
        _copy_vector(old_logprobs[index], row.logprobs)
        if row.advantages:
            _copy_vector(advantages[index], row.advantages)
        elif row.scalar_advantage is not None:
            advantages[index, :length][active] = float(row.scalar_advantage)
        if row.ref_logprobs:
            _copy_vector(ref_logprobs[index], row.ref_logprobs)
            has_ref = True
        if row.returns:
            _copy_vector(returns[index], row.returns)
            has_returns = True
        if row.values:
            _copy_vector(values[index], row.values)
            has_values = True

    batch = {
        "input_ids": mx.array(input_ids),
        "prompt_mask": mx.array(prompt_mask),
        "loss_mask": mx.array(loss_mask),
        "response_mask": mx.array(((~prompt_mask[:, 1:]) & loss_mask[:, 1:]).astype(np.float32)),
        "old_logprobs": mx.array(old_logprobs[:, 1:]),
        "advantages": mx.array(advantages[:, 1:]),
    }
    if has_ref:
        batch["ref_logprobs"] = mx.array(ref_logprobs[:, 1:])
    if has_returns:
        batch["returns"] = mx.array(returns[:, 1:])
    if has_values:
        batch["values"] = mx.array(values[:, 1:])
    return batch


def sft_target_token_count(rows: list[TrainSequence]) -> int:
    """Count shifted SFT targets using the same mask construction as CUDA."""

    count = 0
    for row in rows:
        length = len(row.tokens)
        if length < 2:
            continue
        active = ~_prompt_mask(row, length)
        if row.loss_mask:
            loss_mask = np.zeros(length, dtype=np.bool_)
            values = np.asarray(row.loss_mask[:length], dtype=np.bool_)
            loss_mask[: len(values)] = values
            active &= loss_mask
        count += int(active[1:].sum())
    return count


def clip_grad_norm(grads: Any, max_norm: float | None):
    """Return clipped MLX gradients and their pre-clip global norm."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    leaves = [value for _, value in tree_flatten(grads)]
    squared = sum((value.astype(mx.float32) ** 2).sum() for value in leaves)
    norm = mx.sqrt(squared)
    if max_norm is None:
        return grads, norm
    scale = mx.minimum(mx.array(1.0), float(max_norm) / (norm + 1e-6))
    return tree_map(lambda value: value * scale, grads), norm


def learning_rate_for_step(optimizer: dict[str, Any], step: int) -> float:
    """Evaluate the shared constant/linear/cosine schedule for MLX."""

    start = float(optimizer.get("lr", 1e-6))
    minimum = float(optimizer.get("min_lr", 0.0))
    decay_steps = max(int(optimizer.get("lr_decay_steps", 0)), 0)
    if decay_steps == 0:
        return start
    progress = min(max(step, 0) / decay_steps, 1.0)
    style = str(optimizer.get("lr_decay_style", "constant"))
    if style == "linear":
        factor = 1.0 - progress
    elif style == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif style == "constant":
        factor = 1.0
    else:
        raise ValueError(f"unsupported lr_decay_style: {style}")
    return minimum + (start - minimum) * factor


def _prompt_mask(row: TrainSequence, length: int) -> np.ndarray:
    if row.prompt_mask:
        result = np.ones(length, dtype=np.bool_)
        values = np.asarray(row.prompt_mask[:length], dtype=np.bool_)
        result[: len(values)] = values
        return result
    prompt_len = length if row.prompt_len is None else int(row.prompt_len)
    return np.arange(length) < prompt_len


def _copy_vector(destination: np.ndarray, values: list[float]) -> None:
    count = min(len(destination), len(values))
    if count:
        destination[:count] = values[:count]


__all__ = ["clip_grad_norm", "learning_rate_for_step", "make_train_batch", "sft_target_token_count"]
