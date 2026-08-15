"""Experimental DAPO algorithm registration."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from areno.api.algorithms import AlgorithmSpec, register_algorithm
from areno.api.trainer_config import TrainerConfig
from areno.experimental.dapo.loss import dapo_loss_fn


def _load_trainer() -> type:
    from areno.experimental.dapo.trainer import DAPOTrainer

    return DAPOTrainer


def _bind_loss(config: TrainerConfig, loss_fn: Callable) -> Callable:
    bound_loss = partial(
        loss_fn,
        clip_eps_low=getattr(config, "dapo_clip_eps_low"),
        clip_eps_high=getattr(config, "dapo_clip_eps_high"),
    )
    setattr(bound_loss, "_areno_loss_reduction", "response_token_mean")
    return bound_loss


register_algorithm(
    AlgorithmSpec(
        name="dapo",
        trainer_cls=_load_trainer,
        default_loss_fn=dapo_loss_fn,
        requires_rollout=True,
        loss_fn_factory=_bind_loss,
        experimental=True,
    )
)


__all__ = ["dapo_loss_fn"]
