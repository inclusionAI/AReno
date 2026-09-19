"""Lightweight dataclasses that flow through the rollout/training pipeline.

`PromptItem` is the unit produced by `Trainer.load_prompt_batches` after
tokenising a dataset row. `PromptBatch` groups a fixed-size set of items
together and carries diagnostic counters so the trainer can surface how many
records were skipped for exceeding the prompt-length budget.

`TokenLengthReport` and `compute_token_length_report` provide standalone
token-length distribution diagnostics for a list of `PromptItem` without
requiring the full training pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PromptItem:
    """A dataset record after prompt tokenization and length filtering.

    `prompt` keeps the raw text used for downstream decoding/rewards,
    `input_tokens` holds the tokenized prefix that will be prepended to every
    rollout response, and `record` preserves the original row so reward
    functions can read task-specific fields (gold answers, test cases, ...).
    """

    prompt: str
    solutions: list[str] | None
    input_tokens: list[int]
    record: dict[str, Any]


@dataclass(slots=True)
class PromptBatch:
    """A batch of prompts plus counters for skipped over-length examples.

    `scanned` is how many raw dataset rows were inspected to build this batch
    (including skips), `skipped_long` is how many were dropped this round, and
    `total_skipped_long` accumulates the drop count across the epoch so the
    metric logger can report it as a cumulative counter.
    """

    items: list[PromptItem]
    scanned: int
    skipped_long: int
    total_skipped_long: int

    @property
    def prompts(self) -> list[str]:
        """Return raw prompt strings in batch order for rollout."""

        return [item.prompt for item in self.items]


# ---------------------------------------------------------------------------
# Token-length distribution report
# ---------------------------------------------------------------------------

_OVERLENGTH_POLICIES = ("drop", "truncate")


@dataclass(slots=True)
class LengthStats:
    """Statistical summary of token lengths for one category."""

    count: int
    min: int
    p50: int
    p90: int
    p95: int
    p99: int
    max: int
    mean: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "min": self.min,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.max,
            "mean": round(self.mean, 2),
        }


@dataclass(slots=True)
class TokenLengthReport:
    """Full token-length distribution report for a dataset."""

    total_samples: int
    sampled: int
    sampling_seed: int | None
    max_context: int
    prompt_stats: LengthStats
    response_stats: LengthStats | None
    total_stats: LengthStats | None
    over_context_count: int
    over_context_pct: float
    retained_under_policy: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "sampled": self.sampled,
            "sampling_seed": self.sampling_seed,
            "max_context": self.max_context,
            "prompt_stats": self.prompt_stats.as_dict(),
            "response_stats": self.response_stats.as_dict() if self.response_stats else None,
            "total_stats": self.total_stats.as_dict() if self.total_stats else None,
            "over_context_count": self.over_context_count,
            "over_context_pct": round(self.over_context_pct, 4),
            "retained_under_policy": dict(self.retained_under_policy),
        }


def _percentile_lengths(lengths: list[int]) -> LengthStats:
    arr = np.asarray(lengths, dtype=np.int64)
    return LengthStats(
        count=int(arr.size),
        min=int(arr.min()),
        p50=int(np.percentile(arr, 50)),
        p90=int(np.percentile(arr, 90)),
        p95=int(np.percentile(arr, 95)),
        p99=int(np.percentile(arr, 99)),
        max=int(arr.max()),
        mean=float(arr.mean()),
    )


def compute_token_length_report(
    items: list[PromptItem],
    *,
    max_context: int = 4096,
    sample_ratio: float = 1.0,
    sample_seed: int | None = None,
    response_field: str | None = None,
    tokenizer=None,
    overlength_policies: tuple[str, ...] = _OVERLENGTH_POLICIES,
) -> TokenLengthReport:
    """Compute token-length distribution statistics for a list of PromptItems.

    Args:
        items: List of PromptItem with pre-tokenized input_tokens.
        max_context: Maximum context length for over-context calculation.
        sample_ratio: Fraction of items to sample (1.0 = full scan).
        sample_seed: Random seed for deterministic sampling.
        response_field: If provided, also compute response length stats
            by tokenizing record[response_field] with the given tokenizer.
        tokenizer: Required when response_field is set.
        overlength_policies: Policies to estimate retained examples.

    Returns:
        TokenLengthReport with prompt/response/total length statistics.

    Raises:
        ValueError: If items is empty, sample_ratio out of (0, 1],
                    or response_field set without tokenizer.
    """
    if not items:
        raise ValueError("items must not be empty")
    if not 0 < sample_ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1.0], got {sample_ratio}")
    if response_field is not None and tokenizer is None:
        raise ValueError("tokenizer is required when response_field is set")

    total_samples = len(items)

    # Deterministic sampling
    if sample_ratio >= 1.0:
        sampled_items = items
        sampled_count = total_samples
    else:
        rng = random.Random(sample_seed)
        sampled_count = max(1, int(total_samples * sample_ratio))
        indices = sorted(rng.sample(range(total_samples), sampled_count))
        sampled_items = [items[i] for i in indices]

    prompt_lengths = [len(item.input_tokens) for item in sampled_items]

    response_lengths: list[int] | None = None
    total_lengths: list[int] | None = None
    if response_field is not None:
        response_lengths = []
        for item in sampled_items:
            text = item.record.get(response_field)
            if isinstance(text, str):
                ids = tokenizer.encode(text, add_special_tokens=False)
                response_lengths.append(len(ids))
            else:
                response_lengths.append(0)
        total_lengths = [p + r for p, r in zip(prompt_lengths, response_lengths)]

    prompt_stats = _percentile_lengths(prompt_lengths)
    response_stats = _percentile_lengths(response_lengths) if response_lengths is not None else None
    total_stats = _percentile_lengths(total_lengths) if total_lengths is not None else None

    # Over-context calculation
    ref_lengths = total_lengths if total_lengths is not None else prompt_lengths
    over_context_count = sum(1 for length in ref_lengths if length > max_context)
    over_context_pct = over_context_count / len(ref_lengths) * 100 if ref_lengths else 0.0

    # Retained under each policy
    retained: dict[str, int] = {}
    for policy in overlength_policies:
        if policy == "drop":
            retained[policy] = len(ref_lengths) - over_context_count
        elif policy == "truncate":
            retained[policy] = len(ref_lengths)
        else:
            retained[policy] = len(ref_lengths) - over_context_count

    return TokenLengthReport(
        total_samples=total_samples,
        sampled=len(sampled_items),
        sampling_seed=sample_seed if sample_ratio < 1.0 else None,
        max_context=max_context,
        prompt_stats=prompt_stats,
        response_stats=response_stats,
        total_stats=total_stats,
        over_context_count=over_context_count,
        over_context_pct=over_context_pct,
        retained_under_policy=retained,
    )
