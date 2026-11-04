"""Configurable reward clipping and per-batch standardization.

Raw reward scores produced by ``reward_fn`` can have extreme magnitudes or
unstable variance, which propagates into advantage estimates and destabilises
training. This module provides three opt-in transforms that sit between raw
reward calculation and advantage computation:

* **disabled** (default) -- passes rewards through unchanged.
* **clip** -- clamps rewards to ``[clip_min, clip_max]``.
* **standardize** -- subtracts the batch mean and divides by the batch std.

All transforms accept and return ``list[float]`` to stay compatible with the
existing reward pipeline, which uses plain Python lists throughout. The
dispatcher returns a stats dict so callers can log raw and transformed
distribution summaries separately.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class RewardTransformMode(str, Enum):
    """Supported reward transformation modes."""

    DISABLED = "disabled"
    CLIP = "clip"
    STANDARDIZE = "standardize"


def _compute_reward_stats(rewards: list[float]) -> dict[str, float | int | None]:
    """Compute summary statistics for a reward list.

    Returns a dict with ``mean``, ``std``, ``min``, ``max``, and ``count``.
    For an empty list every numeric field is ``None`` and ``count`` is 0.
    """

    if not rewards:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    arr = np.asarray(rewards, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def clip_rewards(rewards: list[float], clip_min: float, clip_max: float) -> list[float]:
    """Clip rewards to ``[clip_min, clip_max]``.

    Raises:
        ValueError: if ``clip_min > clip_max`` or any reward is NaN.
    """

    if clip_min > clip_max:
        raise ValueError(f"clip_min ({clip_min}) must not exceed clip_max ({clip_max})")
    if any(r != r for r in rewards):
        raise ValueError("clip_rewards received NaN values")
    return [max(clip_min, min(clip_max, r)) for r in rewards]


def standardize_rewards(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Standardize rewards to zero mean and unit variance.

    ``(r - mean) / (std + eps)`` using population std (correction=0), matching
    the convention already used by ``compute_group_advantages``.

    Constant rewards (std == 0) produce all-zero output because the numerator
    is zero for every element.

    Raises:
        ValueError: if ``rewards`` is empty or contains NaN.
    """

    if not rewards:
        raise ValueError("standardize_rewards received empty rewards")
    if any(r != r for r in rewards):
        raise ValueError("standardize_rewards received NaN values")
    arr = np.asarray(rewards, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()
    return ((arr - mean) / (std + eps)).tolist()


def transform_rewards(
    rewards: list[float],
    mode: str = "disabled",
    *,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
    standardize_eps: float = 1e-8,
) -> tuple[list[float], dict[str, float | int | None | str]]:
    """Apply a reward transformation and return ``(transformed, stats)``.

    This is the single entry point used by the training loop. ``stats``
    always contains ``raw_*`` fields; when ``mode`` is not ``disabled`` it
    also contains ``transformed_*`` fields so raw and transformed
    distributions can be logged separately.

    The ``disabled`` mode returns a shallow copy of the input list -- values
    are numerically identical (verified with ``==`` in tests), preserving
    full backward compatibility.
    """

    # Validate mode up front so an invalid config fails before any expensive
    # model or worker initialisation.
    try:
        parsed_mode = RewardTransformMode(mode)
    except ValueError as exc:
        raise ValueError(
            f"reward_transform_mode must be one of {[m.value for m in RewardTransformMode]}, got {mode!r}"
        ) from exc

    raw_stats = _compute_reward_stats(rewards)

    if parsed_mode is RewardTransformMode.DISABLED:
        # Return a copy so downstream mutations don't alias the caller's list.
        return list(rewards), {"transform_mode": "disabled", **{f"raw_{k}": v for k, v in raw_stats.items()}}

    if parsed_mode is RewardTransformMode.CLIP:
        transformed = clip_rewards(rewards, clip_min, clip_max)
    else:
        transformed = standardize_rewards(rewards, eps=standardize_eps)

    transformed_stats = _compute_reward_stats(transformed)
    return transformed, {
        "transform_mode": parsed_mode.value,
        **{f"raw_{k}": v for k, v in raw_stats.items()},
        **{f"transformed_{k}": v for k, v in transformed_stats.items()},
    }
