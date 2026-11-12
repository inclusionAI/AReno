"""Configuration for the experimental async policy trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from areno.api.trainer_config import PolicyTrainerConfig


@dataclass(slots=True)
class AsyncPolicyTrainerConfig(PolicyTrainerConfig):
    """Async rollout/train pipeline settings on top of policy RL config.

    ``queue_max_groups`` bounds how many completed prompt groups the rollout
    producer may buffer ahead of the train consumer; when the queue is full
    the producer blocks, which is the backpressure mechanism. ``max_staleness``
    limits how many policy versions a queued group may lag the current train
    policy, and ``stale_policy`` selects whether over-stale groups are dropped
    or abort the run. Async training requires a separate rollout engine
    (``rollout_devices``) so rollout and optimizer steps use disjoint GPUs.
    """

    queue_max_groups: int = 8
    queue_get_timeout_s: float = 1.0
    max_staleness: int | None = None
    stale_policy: Literal["drop", "error"] = "drop"

    def __post_init__(self) -> None:
        # Explicit base call: zero-arg super() breaks under dataclass(slots=True).
        PolicyTrainerConfig.__post_init__(self)
        if self.queue_max_groups < 1:
            raise ValueError("queue_max_groups must be >= 1")
        if self.queue_get_timeout_s <= 0:
            raise ValueError("queue_get_timeout_s must be positive")
        if self.max_staleness is not None and self.max_staleness < 0:
            raise ValueError("max_staleness must be >= 0 or None")
        if self.stale_policy not in {"drop", "error"}:
            raise ValueError("stale_policy must be 'drop' or 'error'")
        if self.rollout_devices is None:
            raise ValueError("async policy training requires rollout_devices (a separate rollout engine)")
