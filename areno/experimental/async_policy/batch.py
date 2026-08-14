"""Batch metadata for the experimental asynchronous policy trainer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from areno.api.models import TrainSequence


@dataclass(slots=True)
class AsyncTrainBatch:
    """A materialized prompt batch ready for a training step.

    This is the unit that flows through the bounded queue between the rollout
    worker and the train loop. It carries the policy version under which the
    rollout was generated so the train loop can enforce a staleness bound, plus
    the timing and size fields needed for metrics and debugging.
    """

    train_sequences: list[TrainSequence]
    rollout_policy_version: int
    epoch: int
    prompt_batch_id: int
    prompt_count: int
    sequence_count: int
    token_count: int
    reward_mean: float | None
    rollout_time_s: float
    reward_time_s: float
    materialize_time_s: float
    created_at_s: float = field(default_factory=time.time)
