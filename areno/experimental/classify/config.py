"""Trainer config for grouped-softmax sequence classification."""

from __future__ import annotations

from dataclasses import dataclass

from areno.api.trainer_config import TrainerConfig


@dataclass(slots=True)
class ClassifyTrainerConfig(TrainerConfig):
    """Settings for `ClassifyTrainer`.

    `batch_size` counts questions per optimizer step. Each question's candidate
    paths are packed whole into per-DP-rank microbatches of at most
    `microbatch_tokens` real tokens; `mini_bs` and
    `gradient_accumulation_steps` are derived per step and ignored here.
    Questions whose longest candidate path exceeds `max_seq_len` are skipped.
    """

    brier_weight: float = 0.5
    microbatch_tokens: int = 24000
    max_seq_len: int = 512
    score_head_lr: float = 2.0e-4
    score_head_warmup_steps: int = 12
    seed: int = 17

    def __post_init__(self) -> None:
        TrainerConfig.__post_init__(self)
        if self.backend != "cuda":
            raise ValueError("classify training is only supported by the CUDA backend")
        if self.lora is not None:
            raise ValueError("classify training does not support native LoRA")
        if self.brier_weight < 0:
            raise ValueError("brier_weight must be non-negative")
        if self.microbatch_tokens < 1:
            raise ValueError("microbatch_tokens must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.score_head_lr <= 0:
            raise ValueError("score_head_lr must be positive")
        if self.score_head_warmup_steps < 0:
            raise ValueError("score_head_warmup_steps must be non-negative")

    def optimizer_config(self) -> dict:
        """Add the score-head LR group and head-only warmup."""

        config = TrainerConfig.optimizer_config(self)
        config["score_head_lr"] = self.score_head_lr
        config["score_head_warmup_steps"] = self.score_head_warmup_steps
        return config

    def cuda_config(self):
        """Enable the actor score head on the CUDA engine."""

        config = TrainerConfig.cuda_config(self)
        config.runtime["score_head"] = True
        return config


__all__ = ["ClassifyTrainerConfig"]
