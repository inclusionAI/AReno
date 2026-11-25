"""Experimental asynchronous rollout/train policy pipeline.

Registers the ``async-grpo`` and ``async-gspo`` algorithms: a background
rollout producer feeds a bounded group queue while the train loop consumes
it, with per-sample staleness control over the policy-sync version. The
losses importance-weight against rollout logprobs so stale samples produce
correct off-policy gradients.
"""

from functools import partial

from areno.api.algorithms import AlgorithmSpec, register_algorithm
from areno.experimental.async_policy.losses import async_grpo_loss_fn, async_gspo_loss_fn


def _load_async_policy_trainer() -> type:
    from areno.experimental.async_policy.trainer import AsyncPolicyTrainer

    return AsyncPolicyTrainer


def _bind_async_grpo_loss(config, loss_fn):
    return partial(loss_fn, clip_eps=float(config.grpo_clip_eps))


def _bind_async_gspo_loss(config, loss_fn):
    return partial(loss_fn, clip_eps=float(config.gspo_clip_eps))


register_algorithm(
    AlgorithmSpec(
        name="async-grpo",
        trainer_cls=_load_async_policy_trainer,
        default_loss_fn=async_grpo_loss_fn,
        requires_rollout=True,
        loss_fn_factory=_bind_async_grpo_loss,
        experimental=True,
    )
)
register_algorithm(
    AlgorithmSpec(
        name="async-gspo",
        trainer_cls=_load_async_policy_trainer,
        default_loss_fn=async_gspo_loss_fn,
        requires_rollout=True,
        loss_fn_factory=_bind_async_gspo_loss,
        experimental=True,
    )
)

__all__ = ["async_grpo_loss_fn", "async_gspo_loss_fn"]
