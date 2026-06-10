"""Generalized advantage estimation used by the PPO trainer.

PPO needs per-token advantages `A_t = sum_{l>=0} (gamma*lam)^l delta_{t+l}`
where `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`. The fused `cugae` kernel
is preferred when available; the Python implementation below mirrors the same
math sequence-by-sequence and is used as a portable fallback for tests and
CPU-only debugging.
"""

from __future__ import annotations

from typing import Callable


def compute_gae(
    rewards,
    values,
    *,
    gamma: float,
    lam: float,
):
    """Compute PPO advantages with cugae when available.

    Production PPO should install cugae because GAE sits on the actor-critic
    critical path. The Python path is kept as a correctness fallback for local
    development and CPU-only tests.
    """

    fn = _resolve_cugae()
    if fn is not None:
        return fn(rewards, values, gamma=gamma, lam=lam)
    return _compute_gae_python(list(rewards), list(values), gamma=gamma, lam=lam)


def _resolve_cugae() -> Callable | None:
    # The cugae package historically exported the GAE entry point under a few
    # different names; try them in priority order so older installs keep
    # working without changes here.
    try:
        import cugae
    except ImportError:
        return None

    for name in ("compute_gae", "gae", "generalized_advantage_estimation"):
        fn = getattr(cugae, name, None)
        if callable(fn):
            return fn
    return None


def _compute_gae_python(
    rewards: list[float],
    values: list[float],
    *,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    if len(values) != len(rewards):
        raise ValueError(f"GAE requires one value per reward, got {len(values)} and {len(rewards)}")
    advantages = [0.0 for _ in rewards]
    # Iterate from the last timestep backwards so we can accumulate the
    # geometric series A_t = delta_t + (gamma*lam) * A_{t+1} in one pass.
    last_advantage = 0.0
    next_value = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t); the bootstrap V at the
        # tail is 0 because rollouts terminate at EOS.
        delta = rewards[idx] + gamma * next_value - values[idx]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[idx] = last_advantage
        next_value = values[idx]
    # The target returns the critic regresses against are A_t + V(s_t).
    returns = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return advantages, returns
