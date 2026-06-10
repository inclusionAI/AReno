"""Trainer constructor dispatch keyed on the algorithm registry."""

from __future__ import annotations

from areno.api.algorithms import get_algorithm


def build_trainer(config, *, instance, dataset, reward_fn, loss_fn):
    """Create the trainer implementation selected by `config.algo`.

    The registry stores lazy trainer loaders, so PPO and other heavier trainers
    are imported only when their algorithm is selected.
    """

    trainer_cls = get_algorithm(config.algo).resolve_trainer_cls()
    return trainer_cls(config, instance=instance, dataset=dataset, reward_fn=reward_fn, loss_fn=loss_fn)
