"""Policy-version staleness helpers for the asynchronous policy trainer."""

from __future__ import annotations


def staleness_delta(train_policy_version: int, rollout_policy_version: int) -> int:
    """Return how far a batch's rollout policy lags the training policy."""

    return train_policy_version - rollout_policy_version


def is_stale(train_policy_version: int, rollout_policy_version: int, *, max_staleness: int) -> bool:
    """Return whether a batch is older than the configured staleness bound.

    A batch is stale when its rollout policy version lags the current training
    policy version by more than ``max_staleness``.
    """

    if max_staleness < 0:
        raise ValueError("max_staleness must be non-negative")
    return staleness_delta(train_policy_version, rollout_policy_version) > max_staleness
