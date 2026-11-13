from __future__ import annotations

import functools
import unittest
from unittest.mock import patch

from click import UsageError

from areno.api.algorithms import get_algorithm, opd_loss_fn
from areno.api.trainer_config import OPDTrainerConfig
from areno.cli.train import _trainer_config_from_options
from tests.test_train_cli_config_cpu import _options


def _opd_options(**overrides):
    base = dict(algo="opd", teacher_ckpt="teacher", reward_fn_path=None, reward_ckpt=None)
    base.update(overrides)
    return _options(**base)


class OpdConfigTest(unittest.TestCase):
    def test_opd_needs_no_reward_function(self):
        # The whole point of splitting requires_reward from requires_rollout:
        # OPD samples online but has no external reward_fn / reward_ckpt.
        config = _trainer_config_from_options(**_opd_options())

        self.assertIsInstance(config, OPDTrainerConfig)
        self.assertEqual(config.algo, "opd")
        self.assertEqual(config.teacher_ckpt, "teacher")
        self.assertEqual(config.opd_loss_type, "pg")
        self.assertIsNone(config.reward_fn_path)

    def test_existing_rollout_algorithms_still_require_reward(self):
        with self.assertRaisesRegex(UsageError, "reward-fn-path or --reward-ckpt is required"):
            _trainer_config_from_options(**_options(algo="gspo", reward_fn_path=None, reward_ckpt=None))

    def test_all_builtin_specs_keep_their_reward_behavior(self):
        spec = get_algorithm("opd")
        self.assertTrue(spec.requires_rollout)
        self.assertFalse(spec.requires_reward)
        self.assertFalse(spec.experimental)
        for name in ("sft", "dpo", "gspo", "grpo", "ppo", "jev"):
            self.assertTrue(get_algorithm(name).requires_reward, name)

    def test_loss_factory_binds_opd_hyperparameters(self):
        config = OPDTrainerConfig(
            algo="opd", ckpt="student", dataset_path="data",
            teacher_ckpt="teacher", opd_coef=2.0, opd_loss_type="k3", opd_clip=3.0,
        )

        loss_fn = get_algorithm("opd").make_loss_fn(config)

        self.assertIsInstance(loss_fn, functools.partial)
        self.assertIs(loss_fn.func, opd_loss_fn)
        self.assertEqual(
            loss_fn.keywords,
            {"opd_coef": 2.0, "opd_loss_type": "k3", "opd_clip": 3.0},
        )

    def test_config_validation(self):
        base = dict(algo="opd", ckpt="student", dataset_path="data", teacher_ckpt="teacher")
        cases = [
            ({"teacher_ckpt": None}, "teacher_ckpt"),
            ({"opd_coef": 0.0}, "opd_coef"),
            ({"opd_coef": float("nan")}, "opd_coef"),
            ({"opd_clip": -1.0}, "opd_clip"),
            ({"opd_loss_type": "mse"}, "opd_loss_type"),
            ({"opd_teacher_micro_bs": 0}, "opd_teacher_micro_bs"),
            ({"eval_interval": 0}, "eval_interval"),
            ({"eval_batches": 0}, "eval_batches"),
            ({"validation_dataset_loader_fn": "loader.py"}, "validation_dataset_loader_fn"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    OPDTrainerConfig(**{**base, **overrides})

    def test_cli_requires_teacher_ckpt(self):
        with self.assertRaisesRegex(UsageError, "--teacher-ckpt is required"):
            _trainer_config_from_options(**_opd_options(teacher_ckpt=None))

    def test_cli_validates_opd_hyperparameters(self):
        with self.assertRaisesRegex(UsageError, "--opd-coef"):
            _trainer_config_from_options(**_opd_options(opd_coef=0.0))
        with self.assertRaisesRegex(UsageError, "--opd-loss-type"):
            _trainer_config_from_options(**_opd_options(opd_loss_type="mse"))

    def test_reward_fn_path_is_optional_and_retained_for_evaluation(self):
        # --reward-fn-path is not required for OPD training, but when supplied
        # it reaches the trainer so the held-out generation benchmark can use
        # the standard reward_fn contract for task accuracy.
        with patch("areno.cli.train._validate_python_callable"):
            config = _trainer_config_from_options(
                **_opd_options(reward_fn_path="examples/math/math_verify_reward.py")
            )

        self.assertIsInstance(config, OPDTrainerConfig)
        self.assertEqual(config.reward_fn_path, "examples/math/math_verify_reward.py")

    def test_teacher_ckpt_is_resolved_like_other_model_refs(self):
        """Remote teacher ids must become local paths before role loading."""

        from areno.cli.model_refs import resolve_model_refs_for_config

        config = OPDTrainerConfig(
            algo="opd", ckpt="student", dataset_path="data", teacher_ckpt="teacher"
        )

        def fake_resolve(ref, cache=None, *, model_hub="modelscope"):
            del cache, model_hub
            return f"/resolved/{ref}"

        with patch("areno.cli.model_refs.resolve_model_ref", side_effect=fake_resolve):
            resolved = resolve_model_refs_for_config(config)

        self.assertEqual(resolved.ckpt, "/resolved/student")
        self.assertEqual(resolved.teacher_ckpt, "/resolved/teacher")

    def test_opd_may_use_held_out_evaluation(self):
        config = _trainer_config_from_options(
            **_opd_options(validation_dataset_path="held-out", eval_interval=5, eval_batches=2)
        )

        self.assertEqual(config.validation_dataset_path, "held-out")
        self.assertEqual(config.eval_interval, 5)
        self.assertEqual(config.eval_batches, 2)

    def test_other_algorithms_still_reject_held_out_evaluation(self):
        with self.assertRaisesRegex(UsageError, "validation datasets"):
            _trainer_config_from_options(
                **_options(algo="gspo", validation_dataset_path="held-out")
            )

    def test_shuffle_remains_sft_only(self):
        with self.assertRaisesRegex(UsageError, "--shuffle"):
            _trainer_config_from_options(**_opd_options(shuffle=True))


if __name__ == "__main__":
    unittest.main()
