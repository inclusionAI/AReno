"""Reward function loading and group-relative advantage normalisation.

GRPO/GSPO compute advantages by standardising rewards within the group of
`n_samples` rollouts that share a prompt; that helper lives here. Reward
functions receive one :class:`RewardRecord` per prompt/sample row and return
one scalar score, which keeps prompt and agentic demos on the same contract.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)


class RewardEvent(BaseModel):
    """Normalized event in a rollout trajectory."""

    type: Literal["request", "assistant_text", "assistant_tool_call", "tool_result", "finish", "error"]
    text: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | str | None = None
    content: str | None = None
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardRecord(BaseModel):
    """Unified reward input for prompt and agentic rollouts."""

    prompt: str
    completion: str
    rendered_completion: str | None = None
    final_answer: str | None = None
    answer: Any | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[RewardEvent] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)
    loss_mask: list[bool] = Field(default_factory=list)
    source_record: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Normalize rewards within one prompt group for GRPO/GSPO training.

    For a group with rewards r_1..r_n the advantage is
    ``A_i = (r_i - mean(r)) / (std(r) + eps)``. The small `eps` avoids
    division-by-zero when all rollouts return the same reward.
    """

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    return ((rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + eps)).tolist()


def load_reward_fn(path: str) -> Callable[[RewardRecord], float]:
    """Load a user reward function from a Python file.

    The file must define `reward_fn(record)`, where `record` is a
    :class:`RewardRecord`. Keeping rewards as a loaded callable lets algorithm
    scripts swap verifiers without changing backend or training-loop code.
    """

    # spec_from_file_location lets us import a module whose path is supplied
    # at runtime without polluting `sys.modules` with a stable name.
    module_path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load reward function from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        reward_fn = module.reward_fn
    except AttributeError as exc:
        raise ValueError(f"{module_path} must define callable reward_fn(record)") from exc
    if not callable(reward_fn):
        raise ValueError(f"{module_path} must define callable reward_fn(record)")
    return reward_fn


def make_reward_record(
    *,
    prompt: str,
    completion: str,
    source_record: dict[str, Any],
    answer: Any | None = None,
    tokens: list[int] | None = None,
    logprobs: list[float] | None = None,
    loss_mask: list[bool] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RewardRecord:
    """Build the canonical reward input for a single prompt/completion pair."""

    return RewardRecord(
        prompt=prompt,
        completion=completion,
        rendered_completion=completion,
        final_answer=completion,
        answer=answer,
        tokens=list(tokens or []),
        logprobs=[float(value) for value in (logprobs or [])],
        loss_mask=list(loss_mask or []),
        source_record=dict(source_record),
        metadata=dict(metadata or {}),
    )


def compose_reward_fn(
    components: list[tuple[str, Callable[[RewardRecord], float], float]],
    *,
    on_error: Literal["raise", "mark_invalid"] = "raise",
) -> Callable[[RewardRecord], float]:
    """Combine multiple named reward functions into one weighted callable.

    Each component is a ``(name, fn, weight)`` tuple. The composed function
    calls every component with the same :class:`RewardRecord`, computes the
    weighted sum ``total = sum(w_i * r_i)``, and stores per-component values
    in ``record.metadata["reward_components"]`` for downstream metrics.

    Args:
        components: List of (name, fn, weight) tuples. Names must be unique.
            Weights can be any float; no implicit normalization is applied.
        on_error: Error handling strategy. ``"raise"`` (default) re-raises
            any exception or non-finite return value immediately. ``"mark_invalid"``
            sets the offending component's score to 0.0 and continues.

    Returns:
        A callable ``fn(record) -> float`` with the same contract as a
        single reward function.

    Raises:
        ValueError: If component names are duplicated, the list is empty,
            or any weight is non-finite.
    """

    if not components:
        raise ValueError("compose_reward_fn requires at least one component")
    names = [name for name, _, _ in components]
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        raise ValueError(f"duplicate reward component names: {sorted(set(duplicates))}")
    for name, _, weight in components:
        if not math.isfinite(weight):
            raise ValueError(f"weight for component '{name}' must be finite, got {weight}")

    def composed_fn(record: RewardRecord) -> float:
        component_scores: dict[str, float] = {}
        total = 0.0
        for name, fn, weight in components:
            try:
                score = float(fn(record))
                if not math.isfinite(score):
                    raise ValueError(f"reward component '{name}' returned non-finite value: {score}")
            except Exception as exc:
                if on_error == "raise":
                    raise
                # mark_invalid: log the failure and treat score as 0.0
                _logger.warning(
                    "reward component '%s' failed (%s); treating score as 0.0",
                    name,
                    exc,
                )
                score = 0.0
            component_scores[name] = score
            total += weight * score
        # Store component values for downstream metrics collection.
        record.metadata["reward_components"] = dict(component_scores)
        return total

    # Expose component names and weights for downstream metrics wiring.
    # Accessed by metrics code to detect whether a reward_fn is composed and
    # to enumerate component names. The leading underscore signals that the
    # attribute is set by compose_reward_fn and not part of Callable's type.
    composed_fn._reward_components = [(name, weight) for name, _, weight in components]  # type: ignore[attr-defined]
    return composed_fn
