from __future__ import annotations

import functools
import unittest

import torch

from areno.api.algorithms import get_algorithm, list_algorithms
from areno.api.rewards import compute_group_advantages
from areno.api.trainer_config import PolicyTrainerConfig
from areno.api.trainer_factory import build_trainer
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.experimental.drgrpo.loss import compute_drgrpo_advantages, drgrpo_loss_fn


class DrGRPOAdvantageTest(unittest.TestCase):
    """Dr. GRPO advantages center group rewards without variance scaling."""

    def test_advantages_match_hand_computed_group_centering(self):
        advantages = compute_drgrpo_advantages([1.0, 2.0, 4.0])

        expected = [-4.0 / 3.0, -1.0 / 3.0, 5.0 / 3.0]
        torch.testing.assert_close(torch.tensor(advantages), torch.tensor(expected))

    def test_advantages_are_zero_for_constant_rewards(self):
        advantages = compute_drgrpo_advantages([3.0, 3.0, 3.0])

        self.assertEqual(advantages, [0.0, 0.0, 0.0])

    def test_advantages_are_translation_invariant(self):
        baseline = compute_drgrpo_advantages([1.0, 2.0, 4.0])
        translated = compute_drgrpo_advantages([11.0, 12.0, 14.0])

        torch.testing.assert_close(torch.tensor(translated), torch.tensor(baseline))

    def test_advantages_scale_with_reward_magnitude(self):
        baseline = compute_drgrpo_advantages([1.0, 2.0, 4.0])
        scaled = compute_drgrpo_advantages([2.0, 4.0, 8.0])

        torch.testing.assert_close(torch.tensor(scaled), 2.0 * torch.tensor(baseline))
        standardized = compute_group_advantages([2.0, 4.0, 8.0])
        self.assertNotAlmostEqual(float(scaled[-1]), float(standardized[-1]), places=5)


class DrGRPOLossTest(unittest.TestCase):
    """Dr. GRPO sums token losses under a fixed completion-length normalizer."""

    def test_padded_loss_uses_fixed_max_completion_normalizer(self):
        data_pack = {
            "prompt_mask": torch.tensor(
                [
                    [True, True, False, False, False],
                    [True, True, True, False, False],
                ]
            ),
            "advantages": torch.tensor(
                [
                    [0.0, 0.0, 2.0, 2.0, 2.0],
                    [0.0, 0.0, 0.0, -1.0, -1.0],
                ]
            ),
            "logprobs": torch.zeros((2, 5)),
        }
        logprobs = torch.zeros((2, 4), requires_grad=True)

        loss, stats = drgrpo_loss_fn(data_pack, logprobs, max_completion_length=4)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), -0.5, places=6)
        torch.testing.assert_close(
            logprobs.grad,
            torch.tensor(
                [
                    [0.0, -0.25, -0.25, -0.25],
                    [0.0, 0.0, 0.125, 0.125],
                ]
            ),
        )
        self.assertAlmostEqual(float(stats["response_len"]), 2.5, places=6)

    def test_packed_and_padded_layouts_have_equal_loss_and_gradients(self):
        padded_pack = {
            "prompt_mask": torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                ]
            ),
            "advantages": torch.tensor(
                [
                    [0.0, 2.0, 2.0, 2.0],
                    [0.0, 0.0, -1.0, -1.0],
                ]
            ),
            "logprobs": torch.zeros((2, 4)),
        }
        padded_logprobs = torch.zeros((2, 3), requires_grad=True)
        packed_pack = {
            "packed_response_mask": torch.tensor([True, True, True, True, True]),
            "packed_seq_ids": torch.tensor([0, 0, 0, 1, 1]),
            "packed_num_sequences": 2,
            "packed_advantages": torch.tensor([2.0, 2.0, 2.0, -1.0, -1.0]),
            "packed_logprobs": torch.zeros(5),
        }
        packed_logprobs = torch.zeros(5, requires_grad=True)

        padded_loss, padded_stats = drgrpo_loss_fn(
            padded_pack,
            padded_logprobs,
            max_completion_length=4,
        )
        packed_loss, packed_stats = drgrpo_loss_fn(
            packed_pack,
            packed_logprobs,
            max_completion_length=4,
        )
        padded_loss.backward()
        packed_loss.backward()

        torch.testing.assert_close(packed_loss, padded_loss)
        padded_response_grad = torch.cat((padded_logprobs.grad[0], padded_logprobs.grad[1, 1:]))
        torch.testing.assert_close(packed_logprobs.grad, padded_response_grad)
        for key in (
            "policy_loss",
            "total_loss",
            "ratio_mean",
            "ratio_std",
            "advantage_mean",
            "response_len",
            "rollout_logprobs_mean",
            "train_logprobs_mean",
            "logp_diff_mean",
            "logp_abs_diff_mean",
        ):
            torch.testing.assert_close(packed_stats[key], padded_stats[key])

    def test_accumulation_preserves_global_fixed_sequence_normalizer(self):
        full_pack = {
            "prompt_mask": torch.tensor(
                [
                    [True, False],
                    [True, False],
                    [True, False],
                ]
            ),
            "advantages": torch.tensor(
                [
                    [0.0, 2.0],
                    [0.0, -1.0],
                    [0.0, 3.0],
                ]
            ),
            "logprobs": torch.zeros((3, 2)),
        }
        full_logprobs = torch.zeros((3, 1), requires_grad=True)
        full_loss, _ = drgrpo_loss_fn(full_pack, full_logprobs, max_completion_length=4)
        full_loss.backward()

        first_pack = {
            "prompt_mask": full_pack["prompt_mask"][:2],
            "advantages": full_pack["advantages"][:2],
            "logprobs": full_pack["logprobs"][:2],
            "_fixed_sequence_total": 3,
            "_fixed_sequence_grad_scale": 2,
        }
        second_pack = {
            "prompt_mask": full_pack["prompt_mask"][2:],
            "advantages": full_pack["advantages"][2:],
            "logprobs": full_pack["logprobs"][2:],
            "_fixed_sequence_total": 3,
            "_fixed_sequence_grad_scale": 2,
        }
        first_logprobs = torch.zeros((2, 1), requires_grad=True)
        second_logprobs = torch.zeros((1, 1), requires_grad=True)
        first_loss, _ = drgrpo_loss_fn(first_pack, first_logprobs, max_completion_length=4)
        second_loss, _ = drgrpo_loss_fn(second_pack, second_logprobs, max_completion_length=4)
        accumulated_loss = (first_loss + second_loss) / 2
        accumulated_loss.backward()

        torch.testing.assert_close(accumulated_loss, full_loss)
        torch.testing.assert_close(
            torch.cat((first_logprobs.grad, second_logprobs.grad)),
            full_logprobs.grad,
        )

    def test_loss_mask_keeps_fixed_denominator_and_blocks_gradients(self):
        data_pack = {
            "prompt_mask": torch.tensor([[True, False, False]]),
            "loss_mask": torch.tensor([[False, True, False]]),
            "advantages": torch.tensor([[0.0, 2.0, 100.0]]),
            "logprobs": torch.zeros((1, 3)),
        }
        logprobs = torch.zeros((1, 2), requires_grad=True)

        loss, stats = drgrpo_loss_fn(data_pack, logprobs, max_completion_length=4)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), -0.5, places=6)
        torch.testing.assert_close(logprobs.grad, torch.tensor([[-0.5, 0.0]]))
        self.assertAlmostEqual(float(stats["response_len"]), 1.0, places=6)

    def test_fully_masked_batch_returns_finite_zero_loss(self):
        data_pack = {
            "prompt_mask": torch.tensor([[True, False, False]]),
            "loss_mask": torch.tensor([[False, False, False]]),
            "advantages": torch.tensor([[0.0, 2.0, -1.0]]),
            "logprobs": torch.zeros((1, 3)),
        }
        logprobs = torch.zeros((1, 2), requires_grad=True)

        loss, stats = drgrpo_loss_fn(data_pack, logprobs, max_completion_length=4)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        torch.testing.assert_close(logprobs.grad, torch.zeros_like(logprobs))
        self.assertTrue(all(torch.isfinite(value) for value in stats.values()))

    def test_stored_rollout_logprobs_do_not_change_surrogate_loss_or_gradient(self):
        common = {
            "prompt_mask": torch.tensor([[True, False, False]]),
            "advantages": torch.tensor([[0.0, 2.0, -1.0]]),
        }
        close_pack = {**common, "logprobs": torch.tensor([[0.0, -0.1, -0.2]])}
        far_pack = {**common, "logprobs": torch.tensor([[0.0, -9.0, -7.0]])}
        close_logprobs = torch.tensor([[0.0, -0.1]], requires_grad=True)
        far_logprobs = close_logprobs.detach().clone().requires_grad_(True)

        close_loss, close_stats = drgrpo_loss_fn(close_pack, close_logprobs, max_completion_length=4)
        far_loss, far_stats = drgrpo_loss_fn(far_pack, far_logprobs, max_completion_length=4)
        close_loss.backward()
        far_loss.backward()

        torch.testing.assert_close(far_loss, close_loss)
        torch.testing.assert_close(far_logprobs.grad, close_logprobs.grad)
        self.assertEqual(float(close_stats["ratio_mean"]), 1.0)
        self.assertEqual(float(close_stats["ratio_std"]), 0.0)
        self.assertNotEqual(float(far_stats["logp_abs_diff_mean"]), float(close_stats["logp_abs_diff_mean"]))

    def test_loss_rejects_non_positive_max_completion_length(self):
        data_pack = {
            "prompt_mask": torch.tensor([[True, False]]),
            "advantages": torch.tensor([[0.0, 1.0]]),
            "logprobs": torch.zeros((1, 2)),
        }
        logprobs = torch.zeros((1, 1))

        with self.assertRaisesRegex(ValueError, "max_completion_length must be positive"):
            drgrpo_loss_fn(data_pack, logprobs, max_completion_length=0)


class DrGRPORegistryTest(unittest.TestCase):
    """Experimental discovery should expose Dr. GRPO without built-in branches."""

    def test_registry_discovers_experimental_algorithm(self):
        algorithm = get_algorithm("drgrpo")

        self.assertEqual(algorithm.name, "drgrpo")
        self.assertTrue(algorithm.requires_rollout)
        self.assertTrue(algorithm.experimental)
        self.assertIs(algorithm.default_loss_fn, drgrpo_loss_fn)

    def test_builtin_listing_excludes_loaded_experimental_algorithm(self):
        get_algorithm("drgrpo")

        self.assertNotIn("drgrpo", list_algorithms(include_experimental=False))

    def test_binding_reuses_grpo_clip_and_max_new_tokens(self):
        config = PolicyTrainerConfig(
            algo="drgrpo",
            ckpt="unused",
            dataset_path="unused",
            grpo_clip_eps=0.17,
            max_new_tokens=321,
        )

        loss_fn = get_algorithm("drgrpo").make_loss_fn(config)

        self.assertIsInstance(loss_fn, functools.partial)
        self.assertIs(loss_fn.func, drgrpo_loss_fn)
        self.assertEqual(
            loss_fn.keywords,
            {"clip_eps": 0.17, "max_completion_length": 321},
        )
        self.assertEqual(getattr(loss_fn, "_areno_loss_reduction"), "fixed_sequence_mean")

    def test_binding_rejects_non_positive_clip_epsilon(self):
        config = PolicyTrainerConfig(
            algo="drgrpo",
            ckpt="unused",
            dataset_path="unused",
            grpo_clip_eps=0.0,
        )

        with self.assertRaisesRegex(ValueError, "grpo_clip_eps must be positive"):
            get_algorithm("drgrpo").make_loss_fn(config)


class DrGRPOBackendTest(unittest.TestCase):
    """Backend annotations should preserve fixed-normalizer gradients."""

    def test_backend_annotations_account_for_accumulation_and_dp_replication(self):
        from areno.api.backend.areno.backend import _annotate_fixed_sequence_mean_packs

        packs = [{}, {}]
        _annotate_fixed_sequence_mean_packs(
            packs,
            [2, 1],
            gradient_accumulation_steps=None,
            dp_size=2,
        )

        self.assertEqual(packs[0]["_fixed_sequence_total"], 3)
        self.assertEqual(packs[1]["_fixed_sequence_total"], 3)
        self.assertEqual(packs[0]["_fixed_sequence_grad_scale"], 4)
        self.assertEqual(packs[1]["_fixed_sequence_grad_scale"], 2)

        replicated_packs = [{}, {}]
        _annotate_fixed_sequence_mean_packs(
            replicated_packs,
            [2, 1],
            gradient_accumulation_steps=None,
            dp_size=4,
        )

        self.assertEqual(replicated_packs[0]["_fixed_sequence_grad_scale"], 2)
        self.assertEqual(replicated_packs[1]["_fixed_sequence_grad_scale"], 2)

    def test_uneven_dp_shards_match_unsplit_loss_and_gradients(self):
        from areno.engine.runtime.common import split_data_pack_by_dp

        data_pack = {
            "input_ids": torch.ones((3, 2), dtype=torch.long),
            "prompt_mask": torch.tensor(
                [
                    [True, False],
                    [True, False],
                    [True, False],
                ]
            ),
            "advantages": torch.tensor(
                [
                    [0.0, 2.0],
                    [0.0, -1.0],
                    [0.0, 3.0],
                ]
            ),
            "logprobs": torch.zeros((3, 2)),
        }
        full_logprobs = torch.zeros((3, 1), requires_grad=True)
        full_loss, _ = drgrpo_loss_fn(data_pack, full_logprobs, max_completion_length=4)
        full_loss.backward()

        sharded_pack = {
            **data_pack,
            "_fixed_sequence_total": 3,
            "_fixed_sequence_grad_scale": 2,
        }
        shards = split_data_pack_by_dp(sharded_pack, 2)
        sharded_logprobs = torch.zeros((3, 1), requires_grad=True)
        rank_losses = [
            drgrpo_loss_fn(
                shard,
                sharded_logprobs[rank::2],
                max_completion_length=4,
            )[0]
            for rank, shard in enumerate(shards)
        ]
        dp_averaged_loss = torch.stack(rank_losses).mean()
        dp_averaged_loss.backward()

        torch.testing.assert_close(dp_averaged_loss, full_loss)
        torch.testing.assert_close(sharded_logprobs.grad, full_logprobs.grad)

    def test_uneven_microbatches_and_dp_shards_match_unsplit_gradients(self):
        from areno.api.backend.areno.backend import _annotate_fixed_sequence_mean_packs
        from areno.engine.runtime.common import split_data_pack_by_dp

        full_pack = {
            "input_ids": torch.ones((3, 2), dtype=torch.long),
            "prompt_mask": torch.tensor(
                [
                    [True, False],
                    [True, False],
                    [True, False],
                ]
            ),
            "advantages": torch.tensor(
                [
                    [0.0, 2.0],
                    [0.0, -1.0],
                    [0.0, 3.0],
                ]
            ),
            "logprobs": torch.zeros((3, 2)),
        }
        full_logprobs = torch.zeros((3, 1), requires_grad=True)
        full_loss, _ = drgrpo_loss_fn(full_pack, full_logprobs, max_completion_length=4)
        full_loss.backward()

        packs = [
            {key: value[:2] for key, value in full_pack.items()},
            {key: value[2:] for key, value in full_pack.items()},
        ]
        _annotate_fixed_sequence_mean_packs(
            packs,
            [2, 1],
            gradient_accumulation_steps=None,
            dp_size=2,
        )
        shards_by_pack = [split_data_pack_by_dp(pack, 2) for pack in packs]
        sharded_logprobs = torch.zeros((3, 1), requires_grad=True)
        rank_objectives = []
        for rank in range(2):
            first_loss, _ = drgrpo_loss_fn(
                shards_by_pack[0][rank],
                sharded_logprobs[rank:2:2],
                max_completion_length=4,
            )
            second_loss, _ = drgrpo_loss_fn(
                shards_by_pack[1][rank],
                sharded_logprobs[2:],
                max_completion_length=4,
            )
            rank_objectives.append((first_loss + second_loss) / 2)
        dp_averaged_objective = torch.stack(rank_objectives).mean()
        dp_averaged_objective.backward()

        torch.testing.assert_close(dp_averaged_objective, full_loss)
        torch.testing.assert_close(sharded_logprobs.grad, full_logprobs.grad)


class DrGRPOTrainerTest(unittest.TestCase):
    """The experimental trainer should specialize only advantage handling."""

    @staticmethod
    def _config(*, agent_fn: str | None = None) -> PolicyTrainerConfig:
        return PolicyTrainerConfig(
            algo="drgrpo",
            ckpt="unused",
            dataset_path="unused",
            agent_fn=agent_fn,
        )

    def test_trainer_dispatch_resolves_drgrpo_specialization(self):
        from areno.experimental.drgrpo.trainer import DrGRPOTrainer

        trainer = build_trainer(
            self._config(),
            instance="api",
            dataset=["row"],
            reward_fn=lambda _record: 0.0,
            loss_fn=drgrpo_loss_fn,
        )

        self.assertIsInstance(trainer, DrGRPOTrainer)
        self.assertEqual(
            trainer._compute_group_advantages([1.0, 2.0, 4.0]),
            compute_drgrpo_advantages([1.0, 2.0, 4.0]),
        )

    def test_policy_trainer_default_advantage_remains_standardized(self):
        trainer = object.__new__(PolicyOnlyTrainer)

        advantages = trainer._compute_group_advantages([1.0, 2.0, 4.0])

        torch.testing.assert_close(
            torch.tensor(advantages),
            torch.tensor(compute_group_advantages([1.0, 2.0, 4.0])),
        )

    def test_trainer_rejects_agentic_rollouts(self):
        with self.assertRaisesRegex(ValueError, "Dr. GRPO does not support agentic rollouts"):
            build_trainer(
                self._config(agent_fn="examples/agent.py"),
                instance="api",
                dataset=["row"],
                reward_fn=lambda _record: 0.0,
                loss_fn=drgrpo_loss_fn,
            )


if __name__ == "__main__":
    unittest.main()
