"""Lightweight dataclasses that flow through the rollout/training pipeline.

`PromptItem` is the unit produced by `Trainer.load_prompt_batches` after
tokenising a dataset row. `PromptBatch` groups a fixed-size set of items
together and carries diagnostic counters so the trainer can surface how many
records were skipped for exceeding the prompt-length budget.

The `Overlength*` types below are the unified contract for issue #216: a single
policy (`reject` / `warn` / `truncate`) and per-reason counters shared by SFT,
DPO, and agentic trajectories, replacing the ad-hoc reject-only checks that
used to live in each trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OverlengthPolicy(str, Enum):
    """How to handle a sample that exceeds the configured token budget."""

    REJECT = "reject"
    WARN = "warn"
    TRUNCATE = "truncate"


class OverlengthReason(str, Enum):
    """Why a sample was flagged by the overlength policy.

    `WITHIN_BUDGET` marks a sample that fit the budgets (no action needed); it
    exists so callers/tests can assert the non-overlength path deterministically.
    `EXACT_LIMIT` marks the boundary where a length equals (not exceeds) its cap.
    """

    WITHIN_BUDGET = "within_budget"
    EXACT_LIMIT = "exact_limit"
    PROMPT_TOO_LONG = "prompt_too_long"
    RESPONSE_TOO_LONG = "response_too_long"
    SINGLE_MESSAGE_OVERSIZED = "single_message_oversized"
    TRAJECTORY_TOO_LONG = "trajectory_too_long"


@dataclass(slots=True)
class OverlengthDecision:
    """The result of classifying one sample (or pair / trajectory) under a policy.

    `detail` carries bounded diagnostics (stage, over-by token count, truncation
    point) and never the full training-sample text, so logs/metrics stay safe to
    surface. `truncated` is True only when the policy actually cut the sample.
    """

    action: OverlengthPolicy
    reason: OverlengthReason
    truncated: bool = False
    detail: dict[str, Any] | None = None


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

    `overlength_counters` carries per-reason, per-action counts
    (``{f"{reason}/{action}": count}``) for issue #216's unified diagnostics.
    The legacy `skipped_long` / `total_skipped_long` fields stay so existing
    callers and tests keep working; under the default `reject` policy their
    values match the pre-#216 behavior exactly.
    """

    items: list[PromptItem]
    scanned: int
    skipped_long: int
    total_skipped_long: int
    overlength_counters: dict[str, int] = field(default_factory=dict)

    @property
    def prompts(self) -> list[str]:
        """Return raw prompt strings in batch order for rollout."""

        return [item.prompt for item in self.items]
