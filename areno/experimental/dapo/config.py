"""Configuration for the experimental DAPO trainer."""

from __future__ import annotations

from dataclasses import dataclass

from areno.api.trainer_config import PolicyTrainerConfig


@dataclass(slots=True)
class DAPOTrainerConfig(PolicyTrainerConfig):
    """DAPO-specific rollout, clipping, and reward-shaping settings."""

    gradient_accumulation_steps: int | None = 1
    dapo_gen_batch_size: int | None = None
    dapo_max_num_gen_batches: int = 10
    dapo_clip_eps_low: float = 0.2
    dapo_clip_eps_high: float = 0.28
    dapo_overlong_buffer_len: int = 0
    dapo_overlong_penalty_factor: float = 1.0

    def __post_init__(self) -> None:
        PolicyTrainerConfig.__post_init__(self)
        if self.n_samples < 2:
            raise ValueError("DAPO requires n_samples >= 2")
        if self.greedy:
            raise ValueError("DAPO requires stochastic sampling")
        if self.agent_fn is not None:
            raise ValueError("DAPO does not support agent_fn")
        if self.dapo_gen_batch_size is not None and self.dapo_gen_batch_size <= 0:
            raise ValueError("dapo_gen_batch_size must be positive")
        if self.dapo_max_num_gen_batches <= 0:
            raise ValueError("dapo_max_num_gen_batches must be positive")
        if self.dapo_clip_eps_low <= 0 or self.dapo_clip_eps_high <= 0:
            raise ValueError("DAPO clip epsilons must be positive")
        if self.dapo_clip_eps_high < self.dapo_clip_eps_low:
            raise ValueError("dapo_clip_eps_high must be >= dapo_clip_eps_low")
        if self.dapo_overlong_buffer_len < 0:
            raise ValueError("dapo_overlong_buffer_len must be non-negative")
        if self.dapo_overlong_buffer_len > self.max_new_tokens:
            raise ValueError("dapo_overlong_buffer_len must not exceed max_new_tokens")
        if self.dapo_overlong_penalty_factor < 0:
            raise ValueError("dapo_overlong_penalty_factor must be non-negative")

    def resolved_gen_batch_size(self) -> int:
        """Return the candidate prompt count generated per dynamic-sampling attempt."""

        if self.dapo_gen_batch_size is not None:
            return self.dapo_gen_batch_size
        return self.batch_size

    def resolved_max_running_prompts(self) -> int:
        """Default rollout concurrency follows the larger candidate batch."""

        if self.max_running_prompts is not None:
            return self.max_running_prompts
        return max(self.resolved_gen_batch_size() * self.n_samples, 1)
