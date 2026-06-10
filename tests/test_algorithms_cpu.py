from __future__ import annotations

import functools
import unittest

from areno.api.algorithms import AlgorithmSpec, get_algorithm, list_algorithms, register_algorithm
from areno.api.trainer_config import PolicyTrainerConfig, TrainerConfig
from areno.api.trainer_factory import build_trainer
from areno.api.loss_fns.gspo import gspo_loss_fn
from areno.api.loss_fns.ppo import ppo_loss_fn


class AlgorithmRegistryTest(unittest.TestCase):
    """Registry tests stay CPU-only and do not instantiate backend workers."""

    def test_builtin_algorithm_metadata_is_registered(self):
        """Built-ins should expose rollout and role requirements in one place."""
        algorithms = list_algorithms(include_experimental=False)

        self.assertEqual(set(algorithms), {"dpo", "grpo", "gspo", "ppo", "sft"})
        self.assertFalse(algorithms["sft"].requires_rollout)
        self.assertTrue(algorithms["gspo"].requires_rollout)
        self.assertIs(algorithms["ppo"].default_loss_fn, ppo_loss_fn)

    def test_unknown_algorithm_error_lists_registered_names(self):
        """Unknown names should fail with a useful registry-backed message."""
        with self.assertRaisesRegex(ValueError, "registered: .*gspo.*ppo"):
            get_algorithm("not_an_algorithm")

    def test_registered_gspo_loss_binds_config_clip_eps(self):
        """Algorithm specs can adapt the default loss with config parameters."""
        config = PolicyTrainerConfig(algo="gspo", ckpt="unused", dataset_path="unused", gspo_clip_eps=0.123)

        loss_fn = get_algorithm("gspo").make_loss_fn(config)

        self.assertIsInstance(loss_fn, functools.partial)
        self.assertIs(loss_fn.func, gspo_loss_fn)
        self.assertEqual(loss_fn.keywords, {"clip_eps": 0.123})

    def test_factory_uses_registered_trainer_class(self):
        """A contributed algorithm should be constructible without factory edits."""

        class DummyTrainer:
            def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
                self.config = config
                self.instance = instance
                self.dataset = dataset
                self.reward_fn = reward_fn
                self.loss_fn = loss_fn

        def dummy_loss(data_pack, logprobs):
            return data_pack, logprobs

        register_algorithm(
            AlgorithmSpec(
                name="unit_dummy_algo",
                trainer_cls=DummyTrainer,
                default_loss_fn=dummy_loss,
                requires_rollout=False,
                experimental=True,
            )
        )
        config = TrainerConfig(algo="unit_dummy_algo", ckpt="unused", dataset_path="unused")

        trainer = build_trainer(config, instance="api", dataset=["row"], reward_fn=None, loss_fn=dummy_loss)

        self.assertIsInstance(trainer, DummyTrainer)
        self.assertEqual(trainer.config, config)
        self.assertEqual(trainer.instance, "api")
        self.assertEqual(trainer.dataset, ["row"])
        self.assertIs(trainer.loss_fn, dummy_loss)


if __name__ == "__main__":
    unittest.main()
