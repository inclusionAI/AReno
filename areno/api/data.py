"""Lightweight dataclasses that flow through the rollout/training pipeline.

`PromptItem` is the unit produced by `Trainer.load_prompt_batches` after
tokenising a dataset row. `PromptBatch` groups a fixed-size set of items
together and carries diagnostic counters so the trainer can surface how many
records were skipped for exceeding the prompt-length budget.

This module also provides degenerate-sample detection utilities (see
``DegenerateReason``, ``SampleQualityReport``, and the ``check_*`` functions)
that are shared by the rollout, SFT, and DPO data paths.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Degenerate sample detection
# ---------------------------------------------------------------------------


class DegenerateReason(enum.Enum):
    """Reasons a sample is considered degenerate."""

    EMPTY = "empty"
    WHITESPACE_ONLY = "whitespace_only"
    SPECIAL_TOKENS_ONLY = "special_tokens_only"
    NO_TRAINABLE_TOKENS = "no_trainable_tokens"
    IDENTICAL_PREFERENCE_BRANCHES = "identical_preference_branches"


class DegeneratePolicy(enum.Enum):
    """Policy for handling degenerate samples."""

    SKIP = "skip"
    ERROR = "error"


@dataclass(slots=True)
class DegenerateFilterConfig:
    """Configuration for degenerate sample filtering.

    The default (``enabled=True``, ``policy=SKIP``) preserves the prior
    behaviour of silently skipping empty/degenerate examples.
    """

    policy: DegeneratePolicy = DegeneratePolicy.SKIP
    enabled: bool = True


@dataclass(slots=True)
class SampleQualityReport:
    """Result of checking one sample for degeneracy.

    ``stage`` is ``"pre_tokenization"`` for text-level checks or
    ``"post_tokenization"`` for token-level checks.
    """

    is_degenerate: bool
    reason: DegenerateReason | None
    stage: str
    detail: str

    @classmethod
    def ok(cls) -> "SampleQualityReport":
        """Construct a non-degenerate report."""

        return cls(is_degenerate=False, reason=None, stage="", detail="")

    @classmethod
    def degenerate(cls, reason: DegenerateReason, stage: str, detail: str) -> "SampleQualityReport":
        """Construct a degenerate report."""

        return cls(is_degenerate=True, reason=reason, stage=stage, detail=detail)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def check_prompt_text(prompt: str) -> SampleQualityReport:
    """Check a raw prompt string before tokenization."""

    if not prompt:
        return SampleQualityReport.degenerate(
            DegenerateReason.EMPTY, "pre_tokenization", "prompt is an empty string"
        )
    if not prompt.strip():
        return SampleQualityReport.degenerate(
            DegenerateReason.WHITESPACE_ONLY, "pre_tokenization", "prompt contains only whitespace"
        )
    return SampleQualityReport.ok()


def check_response_text(response: str) -> SampleQualityReport:
    """Check a raw response string before tokenization."""

    if not response:
        return SampleQualityReport.degenerate(
            DegenerateReason.EMPTY, "pre_tokenization", "response is an empty string"
        )
    if not response.strip():
        return SampleQualityReport.degenerate(
            DegenerateReason.WHITESPACE_ONLY, "pre_tokenization", "response contains only whitespace"
        )
    return SampleQualityReport.ok()


def check_tokenized_prompt(token_ids: list[int], tokenizer: Any) -> SampleQualityReport:
    """Check tokenized prompt for zero-length or special-token-only degeneracy."""

    if not token_ids:
        return SampleQualityReport.degenerate(
            DegenerateReason.EMPTY, "post_tokenization", "prompt produced zero tokens"
        )
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if special_ids and all(tid in special_ids for tid in token_ids):
        return SampleQualityReport.degenerate(
            DegenerateReason.SPECIAL_TOKENS_ONLY,
            "post_tokenization",
            f"all {len(token_ids)} prompt tokens are special tokens",
        )
    return SampleQualityReport.ok()


def check_trainable_tokens(prompt_mask: list[bool]) -> SampleQualityReport:
    """Check that at least one position has a trainable (non-prompt) token.

    ``prompt_mask[1:]`` is used because the backend loss is next-token
    aligned: position *i* predicts *i+1*, so the trainable positions are
    those where ``prompt_mask[1:][j]`` is ``False``.
    """

    if not any(not is_prompt for is_prompt in prompt_mask[1:]):
        return SampleQualityReport.degenerate(
            DegenerateReason.NO_TRAINABLE_TOKENS,
            "post_tokenization",
            "no trainable tokens after prompt prefix",
        )
    return SampleQualityReport.ok()


def check_preference_pair(chosen: Any, rejected: Any) -> SampleQualityReport:
    """Check that DPO chosen and rejected branches are not identical."""

    if chosen == rejected:
        return SampleQualityReport.degenerate(
            DegenerateReason.IDENTICAL_PREFERENCE_BRANCHES,
            "pre_tokenization",
            "chosen and rejected branches are identical",
        )
    return SampleQualityReport.ok()


def apply_degenerate_policy(report: SampleQualityReport, config: DegenerateFilterConfig) -> bool:
    """Apply the configured policy to a quality report.

    Returns ``True`` if the sample should be skipped, ``False`` if it should
    be kept.  Raises ``ValueError`` when the policy is ``ERROR`` and the
    sample is degenerate.
    """

    if not report.is_degenerate:
        return False
    if not config.enabled:
        return False
    if config.policy is DegeneratePolicy.ERROR:
        raise ValueError(f"degenerate sample detected ({report.stage}): {report.detail}")
    return True


def record_degenerate_reason(counts: dict[str, int], report: SampleQualityReport) -> None:
    """Increment the reason counter in ``counts`` for a degenerate report."""

    if report.reason is not None:
        key = report.reason.value
        counts[key] = counts.get(key, 0) + 1


def format_degenerate_reasons(counts: dict[str, int]) -> str:
    """Format reason counts into a human-readable string for logging."""

    if not counts:
        return ""
    parts = [f"{reason}={n}" for reason, n in sorted(counts.items())]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Pipeline dataclasses
# ---------------------------------------------------------------------------


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
    """A batch of prompts plus counters for skipped examples.

    `scanned` is how many raw dataset rows were inspected to build this batch
    (including skips), `skipped_long` is how many were dropped this round, and
    `total_skipped_long` accumulates the drop count across the epoch so the
    metric logger can report it as a cumulative counter.

    `skipped_degenerate` / `total_skipped_degenerate` and
    `degenerate_reasons` track samples dropped because they were empty,
    whitespace-only, special-token-only, or had no trainable tokens.
    """

    items: list[PromptItem]
    scanned: int
    skipped_long: int
    total_skipped_long: int
    skipped_degenerate: int = 0
    total_skipped_degenerate: int = 0
    degenerate_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def prompts(self) -> list[str]:
        """Return raw prompt strings in batch order for rollout."""

        return [item.prompt for item in self.items]
