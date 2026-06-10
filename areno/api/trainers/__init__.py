"""Concrete trainer loops dispatched by `trainer_factory.build_trainer`.

`PolicyOnlyTrainer` drives GSPO/GRPO (single trainable policy, group-relative
advantages, per-prompt rollouts). `SFTTrainer` runs supervised next-token
training directly from dataset rows. `DPOTrainer` trains on offline preference
pairs with a frozen reference policy. `PPOTrainer` inherits from the
policy-only loop and overrides batch assembly to include reference/critic/reward
roles and GAE.
"""

from areno.api.trainers.dpo import DPOTrainer
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.api.trainers.ppo import PPOTrainer
from areno.api.trainers.sft import SFTTrainer

__all__ = ["DPOTrainer", "PolicyOnlyTrainer", "PPOTrainer", "SFTTrainer"]
