"""Lightweight dataclasses that flow through the rollout/training pipeline.

`PromptItem` is the unit produced by `Trainer.load_prompt_batches` after
tokenising a dataset row. `PromptBatch` groups a fixed-size set of items
together and carries diagnostic counters so the trainer can surface how many
records were skipped for exceeding the prompt-length budget.

Duplicate detection and bounded resampling helpers (`DedupResult`,
`normalize_completion`, `detect_duplicates`) support the optional
``dedup_enabled`` trainer feature that replaces normalized duplicate
completions within a rollout group until a uniqueness target or hard request
budget is reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
# Duplicate detection and bounded resampling
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DedupResult:
    """Summary of one dedup pass over a single rollout group.

    ``duplicate_indices`` holds the positions (within the group's
    ``sequences`` list) that are normalized duplicates of an earlier
    sequence.  The first occurrence of each normalized form is always kept;
    only later copies are flagged.

    ``resample_requested`` is how many replacement samples the caller should
    request to reach ``target_unique`` unique completions, capped by
    ``max_resample``.
    """

    duplicate_count: int
    unique_count: int
    total_count: int
    duplicate_ratio: float
    resample_requested: int
    duplicate_indices: list[int] = field(default_factory=list)


def normalize_completion(text: str) -> str:
    """Normalize completion text for duplicate comparison.

    Strips leading/trailing whitespace and lowercases.  This is intentionally
    conservative: only exact textual matches after normalization are treated
    as duplicates.  More aggressive fuzzy matching can be added later as a
    separate option.
    """

    return text.strip().lower()


def detect_duplicates(
    completions: list[str],
    *,
    target_unique: int | None = None,
    max_resample: int | None = None,
) -> DedupResult:
    """Detect normalized duplicate completions within one rollout group.

    Parameters
    ----------
    completions:
        Decoded completion strings for one prompt's ``n_samples`` rollouts,
        in the same order as ``RolloutResult.sequences``.
    target_unique:
        Desired number of unique completions.  Defaults to ``len(completions)``
        (i.e. all samples should be unique).  If the group already has at
        least ``target_unique`` unique values, no resampling is requested.
    max_resample:
        Hard cap on the number of extra rollout requests.  Defaults to
        ``len(completions)`` (one full re-roll budget).  When the budget is
        exhausted the caller keeps whatever duplicates remain.

    Returns
    -------
    DedupResult
        Summary with duplicate indices and the bounded resample request count.

    Raises
    ------
    ValueError
        If ``completions`` is empty, ``target_unique`` is < 1, or
        ``max_resample`` is < 0.
    """

    if not completions:
        raise ValueError("completions must be non-empty")
    total = len(completions)
    if target_unique is not None and target_unique < 1:
        raise ValueError("target_unique must be >= 1")
    if max_resample is not None and max_resample < 0:
        raise ValueError("max_resample must be >= 0")

    target = target_unique if target_unique is not None else total
    cap = max_resample if max_resample is not None else total

    seen: set[str] = set()
    duplicate_indices: list[int] = []
    for idx, text in enumerate(completions):
        key = normalize_completion(text)
        if key in seen:
            duplicate_indices.append(idx)
        else:
            seen.add(key)

    unique_count = total - len(duplicate_indices)
    # Only request enough resamples to close the gap between current unique
    # count and the target, then cap by the hard budget.
    needed = max(target - unique_count, 0)
    resample_requested = min(needed, cap)

    return DedupResult(
        duplicate_count=len(duplicate_indices),
        unique_count=unique_count,
        total_count=total,
        duplicate_ratio=len(duplicate_indices) / total if total else 0.0,
        resample_requested=resample_requested,
        duplicate_indices=duplicate_indices,
    )
