"""Reward function loading and group-relative advantage normalisation.

GRPO/GSPO compute advantages by standardising rewards within the group of
`n_samples` rollouts that share a prompt; that helper lives here. Reward
functions receive one :class:`RewardRecord` per prompt/sample row and return
one scalar score, which keeps prompt and agentic demos on the same contract.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


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


@dataclass(slots=True)
class CompositeScore:
    """Weighted reward for one record, plus its per-component breakdown.

    `total` is what rollout trainers consume (via `CompositeReward.__call__`).
    `components` carries each component's value so metrics can report them
    independently; `invalid` lists components that raised or returned a
    non-finite value so diagnostics can attribute the failure by name without
    dumping the full training sample.
    """

    total: float
    components: dict[str, float] = field(default_factory=dict)
    invalid: list[str] = field(default_factory=list)


class CompositeReward:
    """Weighted sum of named reward callables sharing the `RewardRecord` contract.

    Components are registered as ``(name, reward_fn, weight)`` and combined into
    a single callable so existing rollout trainers need no change: ``__call__``
    returns the weighted ``total`` just like a plain ``reward_fn(record) -> float``.

    Two error modes control what happens when a component raises or returns a
    non-finite value:

    * ``"raise"`` (default) re-raises immediately with the component name, so a
      misconfigured verifier fails fast and loudly — preserving today's
      single-reward behavior.
    * ``"mark_invalid"`` records the failure in :class:`CompositeScore.invalid`,
      substitutes ``invalid_value`` for that component, and recomputes the total
      from the surviving components so the rest of the rollout still trains.
      Keep ``invalid_value`` at its default ``0.0`` for the re-normalisation to
      match the description above: a non-zero ``invalid_value`` still enters the
      numerator while its weight is dropped from the denominator, so the failed
      component ends up scaled by ``1 / surviving_weight`` rather than excluded.
      (The CLI never sets ``invalid_value``, so this only matters for direct
      SDK construction.)

    Weights are normalised (``sum(w_i * v_i) / sum(w_i)``) rather than summed
    absolutely: users pass ratios such as ``0.7 / 0.3`` to express relative
    importance, and normalisation also lets ``mark_invalid`` drop a failed
    component and re-normalise cleanly against the remaining weight.
    """

    def __init__(
        self,
        components: list[tuple[str, Callable[[RewardRecord], float], float]],
        *,
        on_error: Literal["raise", "mark_invalid"] = "raise",
        invalid_value: float = 0.0,
    ) -> None:
        # Validate everything up front so misconfiguration (duplicate names,
        # bad weights) surfaces at construction — inside the CLI preflight —
        # instead of on the Nth rollout inside the worker.
        if not components:
            raise ValueError("CompositeReward requires at least one component")
        if on_error not in ("raise", "mark_invalid"):
            raise ValueError(f"on_error must be 'raise' or 'mark_invalid', got {on_error!r}")
        seen: set[str] = set()
        weight_sum = 0.0
        for name, _fn, weight in components:
            if not name:
                raise ValueError("reward component name must be non-empty")
            if name in seen:
                raise ValueError(f"duplicate reward component name {name!r}")
            seen.add(name)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"reward component {name!r} weight must be a finite non-negative number, got {weight}")
            weight_sum += weight
        if weight_sum <= 0:
            # All-zero (or weights that cancel) would make the normalised total
            # undefined; reject rather than silently produce 0/0 -> nan.
            raise ValueError("CompositeReward weights must sum to a positive number")
        self._components = components
        self._on_error = on_error
        self._invalid_value = invalid_value

    def score(self, record: RewardRecord) -> CompositeScore:
        """Score one record through every component and aggregate the total.

        A component that raises or returns a non-finite value is either
        re-raised (``raise`` mode) or recorded as invalid and substituted with
        ``invalid_value`` (``mark_invalid`` mode). In the latter case the total
        is re-normalised over the weights of the surviving components so a single
        bad component does not collapse the whole reward to zero.
        """

        component_values: dict[str, float] = {}
        invalid: list[str] = []
        active_weight_sum = 0.0
        for name, fn, weight in self._components:
            try:
                value = float(fn(record))
            except Exception as exc:  # noqa: BLE001 — user components raise anything
                if self._on_error == "raise":
                    raise ValueError(f"reward component {name!r} raised: {type(exc).__name__}: {exc}") from exc
                invalid.append(name)
                component_values[name] = self._invalid_value
                continue
            if not math.isfinite(value):
                if self._on_error == "raise":
                    raise ValueError(f"reward component {name!r} returned a non-finite value: {value}")
                invalid.append(name)
                component_values[name] = self._invalid_value
                continue
            component_values[name] = value
            active_weight_sum += weight
        # Re-normalise against surviving components; if everything failed to the
        # invalid_value we fall back to returning that constant weighted total.
        numerator = sum(weight * component_values[name] for name, _fn, weight in self._components)
        denominator = active_weight_sum if active_weight_sum > 0 else float(sum(weight for _n, _f, weight in self._components))
        total = numerator / denominator if denominator != 0 else 0.0
        return CompositeScore(total=total, components=component_values, invalid=invalid)

    def __call__(self, record: RewardRecord) -> float:
        """Return the weighted total, keeping the rollout-trainer contract intact.

        Trainers call ``float(self.reward_fn(record))``; returning ``total`` here
        means a ``CompositeReward`` is a drop-in replacement for a single reward
        function without any trainer-side branching on the public path.
        """

        return self.score(record).total


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
