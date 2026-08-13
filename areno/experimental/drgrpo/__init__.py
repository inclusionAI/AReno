"""Experimental Dr. GRPO algorithm registration."""

from __future__ import annotations

from functools import partial

from areno.api.algorithms import AlgorithmSpec, register_algorithm
from areno.api.trainer_config import TrainerConfig
from areno.experimental.drgrpo.loss import drgrpo_loss_fn


def _load_trainer() -> type:
    from areno.experimental.drgrpo.trainer import DrGRPOTrainer

    return DrGRPOTrainer


def _bind_loss(config: TrainerConfig, loss_fn):
    clip_eps = float(getattr(config, "grpo_clip_eps"))
    if clip_eps <= 0:
        raise ValueError("grpo_clip_eps must be positive")
    bound_loss_fn = partial(
        loss_fn,
        clip_eps=clip_eps,
        max_completion_length=config.max_new_tokens,
    )
    setattr(bound_loss_fn, "_areno_loss_reduction", "fixed_sequence_mean")
    return bound_loss_fn


register_algorithm(
    AlgorithmSpec(
        name="drgrpo",
        trainer_cls=_load_trainer,
        default_loss_fn=drgrpo_loss_fn,
        requires_rollout=True,
        loss_fn_factory=_bind_loss,
        experimental=True,
    )
)
