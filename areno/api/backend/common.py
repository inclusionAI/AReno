"""Framework-neutral helpers shared by execution backends."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum

from areno.api.models import RolloutResult, RolloutSequence


class TrainMetric(str, Enum):
    """Metric names whose reduction semantics are shared by all backends."""

    RATIO_MEAN = "ratio_mean"
    RATIO_STD = "ratio_std"
    ROLLOUT_LOGPROBS_MEAN = "rollout_logprobs_mean"
    TRAIN_LOGPROBS_MEAN = "train_logprobs_mean"
    LOGP_DIFF_MEAN = "logp_diff_mean"
    LOGP_ABS_DIFF_MEAN = "logp_abs_diff_mean"

    def __str__(self) -> str:
        return self.value


class MetricReduction(str, Enum):
    """Supported reductions for microbatch metrics."""

    FIRST = "first"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"

    def __str__(self) -> str:
        return self.value


LOGP_METRIC_WEIGHT = "_logp_metric_weight"

_FIRST_MICROBATCH_METRICS = frozenset({TrainMetric.RATIO_MEAN, TrainMetric.RATIO_STD})
_TOKEN_WEIGHTED_METRICS = frozenset(
    {
        TrainMetric.ROLLOUT_LOGPROBS_MEAN,
        TrainMetric.TRAIN_LOGPROBS_MEAN,
        TrainMetric.LOGP_DIFF_MEAN,
        TrainMetric.LOGP_ABS_DIFF_MEAN,
    }
)


def accumulation_steps(microbatch_count: int, requested_steps: int | None) -> int:
    """Resolve gradient accumulation exactly as the CUDA training engine does."""

    if microbatch_count < 1:
        raise ValueError("microbatch_count must be positive")
    return microbatch_count if requested_steps is None else max(int(requested_steps), 1)


def accumulation_group_size(index: int, microbatch_count: int, steps: int) -> int:
    """Return the size of the accumulation window containing ``index``."""

    group_start = (index // steps) * steps
    return min(steps, microbatch_count - group_start)


def metric_reduction(key: str) -> MetricReduction:
    """Return the backend-independent reduction for a microbatch metric."""

    if key in _FIRST_MICROBATCH_METRICS:
        return MetricReduction.FIRST
    if key in _TOKEN_WEIGHTED_METRICS:
        return MetricReduction.WEIGHTED_MEAN
    return MetricReduction.MEAN


def reduce_microbatch_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Reduce microbatch metrics, weighting logprob means by active tokens."""

    first: dict[str, float] = {}
    totals: dict[str, float] = {}
    denominators: dict[str, float] = {}
    for row in rows:
        weight = float(row.get(LOGP_METRIC_WEIGHT, 1.0))
        for raw_key, raw_value in row.items():
            key = str(raw_key)
            if key == LOGP_METRIC_WEIGHT:
                continue
            value = float(raw_value)
            reduction = metric_reduction(key)
            if reduction is MetricReduction.FIRST:
                first.setdefault(key, value)
                continue
            item_weight = weight if reduction is MetricReduction.WEIGHTED_MEAN else 1.0
            totals[key] = totals.get(key, 0.0) + value * item_weight
            denominators[key] = denominators.get(key, 0.0) + item_weight
    reduced = {key: total / max(denominators[key], 1.0) for key, total in totals.items()}
    reduced.update(first)
    return reduced


def expand_prompts(prompt_tokens: list[list[int]], n_samples: int) -> list[list[int]]:
    """Expand prompts in prompt-major, sample-minor order."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return [tokens for tokens in prompt_tokens for _ in range(n_samples)]


def expand_prompt_features(
    prompt_features: list[dict | None] | None,
    prompt_count: int,
    n_samples: int,
) -> list[dict | None] | None:
    """Validate and expand prompt-aligned side inputs like ``expand_prompts``."""

    if prompt_features is None:
        return None
    if len(prompt_features) != prompt_count:
        raise ValueError("prompt_features must have the same length as prompt_tokens")
    return [feature for feature in prompt_features for _ in range(n_samples)]


def group_rollout_sequences(
    sequences: list[RolloutSequence],
    prompt_count: int,
    n_samples: int,
    *,
    adapter_version: int | None = None,
) -> list[RolloutResult]:
    """Restore a flat prompt-major sequence list to the public result layout."""

    expected = prompt_count * n_samples
    if len(sequences) != expected:
        raise ValueError(f"backend returned {len(sequences)} sequences; expected {expected}")
    return [
        RolloutResult(
            sequences=sequences[start : start + n_samples],
            adapter_version=adapter_version,
        )
        for start in range(0, expected, n_samples)
    ]


__all__ = [
    "accumulation_group_size",
    "accumulation_steps",
    "expand_prompt_features",
    "expand_prompts",
    "group_rollout_sequences",
    "LOGP_METRIC_WEIGHT",
    "metric_reduction",
    "MetricReduction",
    "reduce_microbatch_metrics",
    "TrainMetric",
]
