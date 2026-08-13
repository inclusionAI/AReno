"""Policy-only trainer specialization for Dr. GRPO."""

from __future__ import annotations

from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.experimental.drgrpo.loss import compute_drgrpo_advantages


class DrGRPOTrainer(PolicyOnlyTrainer):
    """Use centered rewards and the Dr. GRPO fixed-normalizer policy loss."""

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        if getattr(config, "agent_fn", None):
            raise ValueError("Dr. GRPO does not support agentic rollouts")
        super().__init__(
            config,
            instance=instance,
            dataset=dataset,
            reward_fn=reward_fn,
            loss_fn=loss_fn,
        )

    def _compute_group_advantages(self, rewards: list[float]) -> list[float]:
        return compute_drgrpo_advantages(rewards)


__all__ = ["DrGRPOTrainer"]
