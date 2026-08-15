from __future__ import annotations

import math
import unittest
from functools import partial
from types import SimpleNamespace

import torch

from areno.api.algorithms import get_algorithm, list_algorithms
from areno.api.data import PromptBatch, PromptItem
from areno.api.models import RolloutResult, RolloutSequence
from areno.experimental.dapo.config import DAPOTrainerConfig
from areno.experimental.dapo.loss import dapo_loss_fn
from areno.experimental.dapo.trainer import (
    DAPOTrainer,
    _DAPOPromptGroup,
    _has_reward_variance,
    _overlong_penalty,
)


class DAPOLossTest(unittest.TestCase):
    """DAPO loss tests cover real ratios, Clip-Higher, and packed batches."""

    @staticmethod
    def _packed_pack(*, old_logprob: float, advantage: float) -> dict:
        return {
            "packed_response_mask": torch.tensor([True]),
            "packed_logprobs": torch.tensor([old_logprob]),
            "packed_advantages": torch.tensor([advantage]),
            "packed_seq_ids": torch.tensor([0]),
            "packed_num_sequences": 1,
        }

    def test_positive_advantage_uses_upper_clip(self):
        """Clip-Higher caps an improving positive-advantage token at 1 + eps_high."""
        logprobs = torch.tensor([math.log(1.5)], requires_grad=True)

        loss, stats = dapo_loss_fn(
            self._packed_pack(old_logprob=0.0, advantage=1.0),
            logprobs,
            clip_eps_low=0.2,
            clip_eps_high=0.28,
        )
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), -1.28, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac_upper"]), 1.0, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac_lower"]), 0.0, places=6)
        self.assertAlmostEqual(float(logprobs.grad.item()), 0.0, places=6)

    def test_negative_advantage_uses_lower_clip(self):
        """The lower clip protects a negative-advantage token when its ratio falls."""
        logprobs = torch.tensor([math.log(0.5)], requires_grad=True)

        loss, stats = dapo_loss_fn(
            self._packed_pack(old_logprob=0.0, advantage=-1.0),
            logprobs,
            clip_eps_low=0.2,
            clip_eps_high=0.28,
        )
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.8, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac_lower"]), 1.0, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac_upper"]), 0.0, places=6)
        self.assertAlmostEqual(float(logprobs.grad.item()), 0.0, places=6)

    def test_rollout_logprobs_change_ratio_loss_and_gradient(self):
        """DAPO must use stored rollout logprobs rather than a detached unit ratio."""
        current_a = torch.tensor([-0.05], requires_grad=True)
        current_b = torch.tensor([-0.05], requires_grad=True)

        loss_a, stats_a = dapo_loss_fn(self._packed_pack(old_logprob=-0.10, advantage=1.0), current_a)
        loss_b, stats_b = dapo_loss_fn(self._packed_pack(old_logprob=-0.20, advantage=1.0), current_b)
        loss_a.backward()
        loss_b.backward()

        self.assertNotAlmostEqual(float(loss_a.detach()), float(loss_b.detach()), places=6)
        self.assertNotAlmostEqual(float(current_a.grad.item()), float(current_b.grad.item()), places=6)
        self.assertAlmostEqual(float(stats_a["ratio_mean"]), math.exp(0.05), places=6)
        self.assertAlmostEqual(float(stats_b["ratio_mean"]), math.exp(0.15), places=6)

    def test_unclipped_gradient_matches_finite_difference(self):
        """A central finite difference should agree with the analytical DAPO gradient."""
        data_pack = self._packed_pack(old_logprob=-0.2, advantage=0.7)
        current = torch.tensor([-0.15], dtype=torch.float64, requires_grad=True)

        loss, _ = dapo_loss_fn(data_pack, current)
        loss.backward()

        step = 1.0e-5
        plus, _ = dapo_loss_fn(data_pack, torch.tensor([-0.15 + step], dtype=torch.float64))
        minus, _ = dapo_loss_fn(data_pack, torch.tensor([-0.15 - step], dtype=torch.float64))
        finite_difference = float((plus - minus) / (2 * step))
        self.assertAlmostEqual(float(current.grad.item()), finite_difference, places=5)

    def test_clip_higher_matches_hand_computed_mixed_token_reference(self):
        """Every Clip-Higher branch should match an independent literal reference."""
        data_pack = {
            "packed_response_mask": torch.tensor([True, True, True, True]),
            "packed_logprobs": torch.zeros(4),
            "packed_advantages": torch.tensor([1.0, -1.0, 1.0, -1.0]),
            "packed_seq_ids": torch.tensor([0, 0, 1, 1]),
            "packed_num_sequences": 2,
        }
        current = torch.tensor(
            [math.log(1.5), math.log(0.5), math.log(0.5), math.log(1.5)],
            requires_grad=True,
        )

        loss, stats = dapo_loss_fn(data_pack, current, clip_eps_low=0.2, clip_eps_high=0.28)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.13, places=6)
        self.assertTrue(torch.allclose(current.grad, torch.tensor([0.0, 0.0, -0.125, 0.375]), atol=1e-6))
        self.assertAlmostEqual(float(stats["ratio_mean"]), 1.0, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac_lower"]), 0.25, places=6)
        self.assertAlmostEqual(float(stats["pg_clipfrac_upper"]), 0.25, places=6)
        self.assertAlmostEqual(float(stats["advantage_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["response_len"]), 2.0, places=6)

    def test_packed_response_mask_excludes_inactive_tokens(self):
        """Packed padding slots must not affect the DAPO objective or metrics."""
        packed_pack = {
            "packed_response_mask": torch.tensor([True, True, False, True]),
            "packed_logprobs": torch.tensor([-0.2, -0.4, 9.0, -0.3]),
            "packed_advantages": torch.tensor([1.0, 1.0, 9.0, -0.5]),
            "packed_seq_ids": torch.tensor([0, 0, 0, 1]),
            "packed_num_sequences": 2,
        }
        compact_pack = {
            "packed_response_mask": torch.tensor([True, True, True]),
            "packed_logprobs": torch.tensor([-0.2, -0.4, -0.3]),
            "packed_advantages": torch.tensor([1.0, 1.0, -0.5]),
            "packed_seq_ids": torch.tensor([0, 0, 1]),
            "packed_num_sequences": 2,
        }

        masked_loss, masked_stats = dapo_loss_fn(packed_pack, torch.tensor([-0.1, -0.5, -9.0, -0.35]))
        compact_loss, compact_stats = dapo_loss_fn(compact_pack, torch.tensor([-0.1, -0.5, -0.35]))

        self.assertAlmostEqual(float(masked_loss), float(compact_loss), places=6)
        for name in ("ratio_mean", "pg_clipfrac", "advantage_mean", "response_len"):
            self.assertAlmostEqual(float(masked_stats[name]), float(compact_stats[name]), places=6)

    def test_annotated_microbatches_use_global_response_token_mean(self):
        """Unequal microbatches should contribute one global token-mean gradient."""
        short_logprobs = torch.tensor([0.0], requires_grad=True)
        long_logprobs = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        short_pack = {
            "packed_response_mask": torch.tensor([True]),
            "packed_logprobs": torch.tensor([0.0]),
            "packed_advantages": torch.tensor([1.0]),
            "packed_seq_ids": torch.tensor([0]),
            "packed_num_sequences": 1,
            "_response_token_total": 4,
            "_response_token_grad_scale": 2,
        }
        long_pack = {
            "packed_response_mask": torch.tensor([True, True, True]),
            "packed_logprobs": torch.tensor([0.0, 0.0, 0.0]),
            "packed_advantages": torch.tensor([3.0, 3.0, 3.0]),
            "packed_seq_ids": torch.tensor([0, 0, 0]),
            "packed_num_sequences": 1,
            "_response_token_total": 4,
            "_response_token_grad_scale": 2,
        }

        short_loss, _ = dapo_loss_fn(short_pack, short_logprobs)
        long_loss, _ = dapo_loss_fn(long_pack, long_logprobs)
        accumulated_loss = (short_loss + long_loss) / 2
        accumulated_loss.backward()

        self.assertAlmostEqual(float(accumulated_loss.detach()), -2.5, places=6)
        self.assertAlmostEqual(float(short_logprobs.grad.item()), -0.25, places=6)
        self.assertTrue(torch.allclose(long_logprobs.grad, torch.full((3,), -0.75)))


class DAPOBackendNormalizationTest(unittest.TestCase):
    """Backend metadata compensates for accumulation and DP gradient averaging."""

    @staticmethod
    def _train_sequence(advantage: float, *, response_tokens: int = 1):
        from areno.api.models import TrainSequence

        return TrainSequence(
            prompt_mask=[True] + [False] * response_tokens,
            tokens=[1] + list(range(2, response_tokens + 2)),
            logprobs=[0.0] * (response_tokens + 1),
            advantages=[0.0] + [advantage] * response_tokens,
            eos_token_id=99,
        )

    def test_annotations_cover_sharded_and_replicated_microbatches(self):
        from areno.api.backend.cuda.training import annotate_response_token_mean_packs

        packs = [
            {"input_ids": torch.zeros((1, 2), dtype=torch.long)},
            {"input_ids": torch.zeros((2, 3), dtype=torch.long)},
        ]

        annotate_response_token_mean_packs(
            packs,
            [1, 3],
            gradient_accumulation_steps=None,
            data_parallel_size=2,
        )

        self.assertEqual(packs[0]["_response_token_total"], 4)
        self.assertEqual(packs[1]["_response_token_total"], 4)
        self.assertEqual(packs[0]["_response_token_grad_scale"], 2)
        self.assertEqual(packs[1]["_response_token_grad_scale"], 4)

    def test_annotations_restart_at_optimizer_step_boundaries(self):
        from areno.api.backend.cuda.training import annotate_response_token_mean_packs

        packs = [{"input_ids": torch.zeros((2, 2), dtype=torch.long)} for _ in range(3)]

        annotate_response_token_mean_packs(
            packs,
            [1, 3, 5],
            gradient_accumulation_steps=2,
            data_parallel_size=1,
        )

        self.assertEqual([pack["_response_token_total"] for pack in packs], [4, 4, 5])
        self.assertEqual([pack["_response_token_grad_scale"] for pack in packs], [2, 2, 1])

    def test_dp_averaging_reconstructs_global_response_token_gradient(self):
        """DP rank averaging should equal one global token-mean objective."""
        theta = torch.tensor(0.0, requires_grad=True)
        rank_zero_pack = {
            "packed_response_mask": torch.tensor([True, True]),
            "packed_logprobs": torch.tensor([0.0, 0.0]),
            "packed_advantages": torch.tensor([1.0, 1.0]),
            "packed_seq_ids": torch.tensor([0, 0]),
            "packed_num_sequences": 1,
            "_response_token_total": 3,
            "_response_token_grad_scale": 2,
        }
        rank_one_pack = {
            "packed_response_mask": torch.tensor([True]),
            "packed_logprobs": torch.tensor([0.0]),
            "packed_advantages": torch.tensor([3.0]),
            "packed_seq_ids": torch.tensor([0]),
            "packed_num_sequences": 1,
            "_response_token_total": 3,
            "_response_token_grad_scale": 2,
        }

        rank_zero_loss, _ = dapo_loss_fn(rank_zero_pack, theta.expand(2))
        rank_one_loss, _ = dapo_loss_fn(rank_one_pack, theta.expand(1))
        dp_averaged_loss = (rank_zero_loss + rank_one_loss) / 2
        dp_averaged_loss.backward()

        self.assertAlmostEqual(float(dp_averaged_loss.detach()), -5.0 / 3.0, places=6)
        self.assertAlmostEqual(float(theta.grad), -5.0 / 3.0, places=6)

    def test_real_uneven_dp_split_matches_unsplit_response_token_gradient(self):
        """A real 3-to-2 DP split must preserve the global token mean."""
        from areno.api.backend.cuda.training import annotate_response_token_mean_packs, make_train_pack
        from areno.engine.runtime.common import split_data_pack_by_dp
        from areno.engine.runtime.train_step import _pack_train_data

        sequences = [self._train_sequence(value) for value in (1.0, 2.0, 3.0)]
        pack = make_train_pack(sequences)
        annotate_response_token_mean_packs(
            [pack],
            [3],
            gradient_accumulation_steps=1,
            data_parallel_size=2,
        )
        shards = split_data_pack_by_dp(pack, 2)
        theta = torch.tensor(0.0, requires_grad=True)
        shard_losses = []
        for shard in shards:
            packed = _pack_train_data(shard)
            loss, _ = dapo_loss_fn(
                packed,
                theta.expand(int(packed["packed_response_mask"].numel())),
            )
            shard_losses.append(loss)
        dp_averaged_loss = sum(shard_losses) / 2
        dp_averaged_loss.backward()

        self.assertEqual([int(shard["input_ids"].shape[0]) for shard in shards], [2, 1])
        self.assertAlmostEqual(float(dp_averaged_loss.detach()), -2.0, places=6)
        self.assertAlmostEqual(float(theta.grad), -2.0, places=6)

    def test_real_dp_split_and_accumulation_match_unsplit_response_token_gradient(self):
        """Sharded and replicated packs in one accumulation group must normalize together."""
        from areno.api.backend.cuda.training import annotate_response_token_mean_packs, make_train_pack
        from areno.engine.runtime.common import split_data_pack_by_dp
        from areno.engine.runtime.train_step import _pack_train_data

        sequences = [
            self._train_sequence(1.0),
            self._train_sequence(3.0),
            self._train_sequence(5.0, response_tokens=2),
        ]
        packs = [make_train_pack(sequences[:2]), make_train_pack(sequences[2:])]
        annotate_response_token_mean_packs(
            packs,
            [2, 2],
            gradient_accumulation_steps=2,
            data_parallel_size=2,
        )
        shards_by_pack = [split_data_pack_by_dp(pack, 2) for pack in packs]

        theta = torch.tensor(0.0, requires_grad=True)
        rank_losses = []
        for rank in range(2):
            accumulated_loss = torch.zeros(())
            for shards in shards_by_pack:
                packed = _pack_train_data(shards[rank])
                loss, _ = dapo_loss_fn(
                    packed,
                    theta.expand(int(packed["packed_response_mask"].numel())),
                )
                accumulated_loss = accumulated_loss + loss / 2
            rank_losses.append(accumulated_loss)
        dp_averaged_loss = sum(rank_losses) / 2
        dp_averaged_loss.backward()

        reference_theta = torch.tensor(0.0, requires_grad=True)
        reference_pack = _pack_train_data(make_train_pack(sequences))
        reference_loss, _ = dapo_loss_fn(
            reference_pack,
            reference_theta.expand(int(reference_pack["packed_response_mask"].numel())),
        )
        reference_loss.backward()

        self.assertEqual(
            [[int(shard["input_ids"].shape[0]) for shard in shards] for shards in shards_by_pack],
            [[1, 1], [1, 1]],
        )
        self.assertAlmostEqual(float(dp_averaged_loss.detach()), -3.5, places=6)
        self.assertAlmostEqual(float(theta.grad), -3.5, places=6)
        self.assertAlmostEqual(float(dp_averaged_loss.detach()), float(reference_loss.detach()), places=6)
        self.assertAlmostEqual(float(theta.grad), float(reference_theta.grad), places=6)


class DAPORegistrationTest(unittest.TestCase):
    """DAPO is discoverable through the opt-in experimental registry path."""

    def test_dapo_is_experimental_and_hidden_from_builtin_listing(self):
        self.assertNotIn("dapo", list_algorithms(include_experimental=False))

        spec = get_algorithm("dapo")

        self.assertTrue(spec.experimental)
        self.assertTrue(spec.requires_rollout)
        self.assertIn("dapo", list_algorithms(include_experimental=True))

    def test_dapo_loss_binds_asymmetric_clip_parameters(self):
        config = DAPOTrainerConfig(
            algo="dapo",
            ckpt="unused",
            dataset_path="unused",
            dapo_clip_eps_low=0.11,
            dapo_clip_eps_high=0.37,
        )

        loss_fn = get_algorithm("dapo").make_loss_fn(config)

        self.assertIsInstance(loss_fn, partial)
        self.assertIs(loss_fn.func, dapo_loss_fn)
        self.assertEqual(loss_fn.keywords, {"clip_eps_low": 0.11, "clip_eps_high": 0.37})
        self.assertEqual(getattr(loss_fn, "_areno_loss_reduction"), "response_token_mean")

    def test_dapo_config_uses_generation_batch_for_default_concurrency(self):
        config = DAPOTrainerConfig(
            algo="dapo",
            ckpt="unused",
            dataset_path="unused",
            batch_size=4,
            n_samples=8,
            dapo_gen_batch_size=12,
        )

        self.assertEqual(config.resolved_gen_batch_size(), 12)
        self.assertEqual(config.resolved_max_running_prompts(), 96)


class _TokenIdTokenizer:
    eos_token_id = 99

    def decode(self, tokens):
        return str(tokens[0]) if tokens else ""


def _prompt_batch(index: int) -> PromptBatch:
    return PromptBatch(
        items=[
            PromptItem(
                prompt=f"prompt-{index}",
                solutions=None,
                input_tokens=[index + 1],
                record={},
            )
        ],
        scanned=1,
        skipped_long=0,
        total_skipped_long=0,
    )


def _group(index: int, rewards: list[float]) -> _DAPOPromptGroup:
    result = RolloutResult(
        sequences=[
            RolloutSequence(resp_tokens=[index * 10 + sample + 1], resp_logprobs=[-0.1])
            for sample in range(len(rewards))
        ]
    )
    return _DAPOPromptGroup(
        item=_prompt_batch(index).items[0],
        rollout_result=result,
        raw_rewards=rewards,
        shaped_rewards=list(rewards),
    )


class _DAPOFakeAreno:
    def __init__(self, candidate_count: int):
        self.candidate_count = candidate_count
        self.train_calls = []
        self.train_result = {}
        self.finish_calls = 0

    def get_tokenizer(self):
        return _TokenIdTokenizer()

    def get_processor(self):
        return None

    def load_prompt_batches(self, _dataset, *, batch_size, max_prompt_tokens):
        del max_prompt_tokens
        assert batch_size == 1
        for index in range(self.candidate_count):
            yield _prompt_batch(index)

    def train(self, batch, loss_fn, *, mini_bs, gradient_accumulation_steps):
        self.train_calls.append((batch, loss_fn, mini_bs, gradient_accumulation_steps))
        return self.train_result

    def finish_step(self):
        self.finish_calls += 1


class _ScriptedDAPOTrainer(DAPOTrainer):
    def __init__(self, *args, groups, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.generation_calls = 0

    def _generate_candidate_groups(self, tokenizer, sampling_params, prompt_batch, *, epoch, step):
        del tokenizer, sampling_params, epoch, step
        index = int(prompt_batch.items[0].prompt.rsplit("-", 1)[1])
        self.generation_calls += 1
        groups = self.groups[index]
        return groups if isinstance(groups, list) else [groups]


class _ScriptedRolloutDAPOTrainer(DAPOTrainer):
    def __init__(self, *args, rollout_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_results = rollout_results

    async def _run_prompt_rollout(self, sampling_params, prompt_batch):
        del sampling_params, prompt_batch
        return self.rollout_results


def _dapo_config(**overrides) -> DAPOTrainerConfig:
    values = dict(
        algo="dapo",
        ckpt="unused",
        dataset_path="unused",
        epochs=1,
        max_steps=1,
        batch_size=2,
        mini_bs=2,
        n_samples=2,
        dapo_gen_batch_size=1,
        dapo_max_num_gen_batches=3,
        save_path=None,
    )
    values.update(overrides)
    return DAPOTrainerConfig(**values)


class DAPODynamicSamplingTest(unittest.TestCase):
    """Dynamic sampling fills complete informative batches before training."""

    def test_trainer_requires_callable_reward_function(self):
        trainer = DAPOTrainer(
            _dapo_config(),
            instance=SimpleNamespace(),
            dataset=[],
            reward_fn=None,
            loss_fn=dapo_loss_fn,
        )

        with self.assertRaisesRegex(ValueError, "callable reward_fn"):
            trainer._fit_initialized()

    def test_reward_variance_uses_raw_rewards_before_overlong_shaping(self):
        config = _dapo_config(max_new_tokens=4, dapo_overlong_buffer_len=2)
        trainer = DAPOTrainer(
            config,
            instance=SimpleNamespace(),
            dataset=[],
            reward_fn=lambda _record: 1.0,
            loss_fn=dapo_loss_fn,
        )
        item = _prompt_batch(0).items[0]
        result = RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=[1], resp_logprobs=[-0.1]),
                RolloutSequence(resp_tokens=[2, 3, 4, 5], resp_logprobs=[-0.1] * 4),
            ]
        )

        group = trainer._score_prompt_group(_TokenIdTokenizer(), item, result, prompt_index=0)

        self.assertFalse(group.informative)
        self.assertEqual(group.raw_rewards, [1.0, 1.0])
        self.assertEqual(group.shaped_rewards, [1.0, 0.0])

    def test_scoring_rejects_non_finite_rollout_logprobs(self):
        trainer = DAPOTrainer(
            _dapo_config(),
            instance=SimpleNamespace(),
            dataset=[],
            reward_fn=lambda record: float(record.metadata["sample_index"]),
            loss_fn=dapo_loss_fn,
        )
        result = RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=[1], resp_logprobs=[float("nan")]),
                RolloutSequence(resp_tokens=[2], resp_logprobs=[-0.1]),
            ]
        )

        with self.assertRaisesRegex(ValueError, "finite rollout logprobs"):
            trainer._score_prompt_group(
                _TokenIdTokenizer(),
                _prompt_batch(0).items[0],
                result,
                prompt_index=0,
            )

    def test_scoring_rejects_incomplete_prompt_group(self):
        """Dynamic sampling must never train a group smaller than n_samples."""
        trainer = DAPOTrainer(
            _dapo_config(n_samples=2),
            instance=SimpleNamespace(),
            dataset=[],
            reward_fn=lambda _record: 0.0,
            loss_fn=dapo_loss_fn,
        )
        result = RolloutResult(sequences=[RolloutSequence(resp_tokens=[1], resp_logprobs=[-0.1])])

        with self.assertRaisesRegex(ValueError, "expected n_samples=2, got 1"):
            trainer._score_prompt_group(
                _TokenIdTokenizer(),
                _prompt_batch(0).items[0],
                result,
                prompt_index=0,
            )

    def test_materialization_rejects_misaligned_raw_reward_count(self):
        trainer = DAPOTrainer(
            _dapo_config(),
            instance=SimpleNamespace(),
            dataset=[],
            reward_fn=lambda _record: 0.0,
            loss_fn=dapo_loss_fn,
        )
        group = _group(0, [0.0, 1.0])
        group.raw_rewards = [0.0]

        with self.assertRaisesRegex(ValueError, "sequence and raw reward counts"):
            trainer._materialize_scored_groups(_TokenIdTokenizer(), [group])

    def test_dynamic_sampling_retries_until_training_batch_is_full(self):
        areno = _DAPOFakeAreno(candidate_count=3)
        trainer = _ScriptedDAPOTrainer(
            _dapo_config(),
            instance=areno,
            dataset=[{}, {}, {}],
            reward_fn=lambda _record: 0.0,
            loss_fn=dapo_loss_fn,
            groups=[_group(0, [1.0, 1.0]), _group(1, [0.0, 1.0]), _group(2, [1.0, 2.0])],
        )

        trainer._fit_initialized()

        self.assertEqual(trainer.generation_calls, 3)
        self.assertEqual(len(areno.train_calls), 1)
        batch, _, mini_bs, gradient_accumulation_steps = areno.train_calls[0]
        self.assertEqual(len(batch), 4)
        self.assertEqual([sequence.reward for sequence in batch], [0.0, 1.0, 1.0, 2.0])
        self.assertEqual([sequence.tokens for sequence in batch], [[2, 11], [2, 12], [3, 21], [3, 22]])
        self.assertEqual([sequence.logprobs for sequence in batch], [[0.0, -0.1]] * 4)
        self.assertEqual(mini_bs, 2)
        self.assertEqual(gradient_accumulation_steps, 1)
        self.assertEqual(areno.train_result["dapo_gen_batches"], 3.0)
        self.assertEqual(areno.train_result["dapo_filtered_groups"], 1.0)
        self.assertEqual(areno.train_result["dapo_qualified_groups"], 2.0)

    def test_dynamic_sampling_cap_fails_without_training_partial_batch(self):
        areno = _DAPOFakeAreno(candidate_count=2)
        trainer = _ScriptedDAPOTrainer(
            _dapo_config(batch_size=1, dapo_max_num_gen_batches=2),
            instance=areno,
            dataset=[{}, {}],
            reward_fn=lambda _record: 0.0,
            loss_fn=dapo_loss_fn,
            groups=[_group(0, [1.0, 1.0]), _group(1, [2.0, 2.0])],
        )

        with self.assertRaisesRegex(RuntimeError, "generated_groups=2 qualified_groups=0 filtered_groups=2"):
            trainer._fit_initialized()

        self.assertEqual(areno.train_calls, [])

    def test_qualified_buffer_truncates_only_at_group_boundaries(self):
        areno = _DAPOFakeAreno(candidate_count=1)
        trainer = _ScriptedDAPOTrainer(
            _dapo_config(batch_size=2, dapo_max_num_gen_batches=1),
            instance=areno,
            dataset=[{}],
            reward_fn=lambda _record: 0.0,
            loss_fn=dapo_loss_fn,
            groups=[
                [
                    _group(0, [0.0, 1.0]),
                    _group(1, [1.0, 2.0]),
                    _group(2, [2.0, 3.0]),
                ]
            ],
        )

        trainer._fit_initialized()

        batch = areno.train_calls[0][0]
        self.assertEqual([sequence.reward for sequence in batch], [0.0, 1.0, 1.0, 2.0])
        self.assertEqual(areno.train_result["dapo_discarded_qualified_groups"], 1.0)

    def test_real_multi_prompt_generation_filters_and_truncates_whole_groups(self):
        """Real scoring preserves prompt order across filtering and overflow."""
        prompt_batch = PromptBatch(
            items=[_prompt_batch(index).items[0] for index in range(3)],
            scanned=3,
            skipped_long=0,
            total_skipped_long=0,
        )
        rollout_results = [
            RolloutResult(
                sequences=[
                    RolloutSequence(resp_tokens=[index * 10 + sample + 1], resp_logprobs=[-0.1]) for sample in range(2)
                ]
            )
            for index in range(3)
        ]

        def reward_fn(record):
            prompt_index = int(record.metadata["prompt_index"])
            sample_index = int(record.metadata["sample_index"])
            if prompt_index == 0:
                return 1.0
            return float(prompt_index + sample_index)

        trainer = _ScriptedRolloutDAPOTrainer(
            _dapo_config(batch_size=1, dapo_gen_batch_size=3, dapo_max_num_gen_batches=1),
            instance=SimpleNamespace(record_rollout_sample=lambda _sample: None),
            dataset=[],
            reward_fn=reward_fn,
            loss_fn=dapo_loss_fn,
            rollout_results=rollout_results,
        )

        collection = trainer._collect_qualified_groups(
            _TokenIdTokenizer(),
            SimpleNamespace(),
            iter([prompt_batch]),
            epoch=0,
            step=0,
        )

        self.assertEqual(collection.generated_groups, 3)
        self.assertEqual(collection.filtered_groups, 1)
        self.assertEqual(collection.discarded_qualified_groups, 1)
        self.assertEqual([group.item.prompt for group in collection.groups], ["prompt-1"])
        self.assertEqual(collection.groups[0].raw_rewards, [1.0, 2.0])
        self.assertEqual(
            [sequence.resp_tokens for sequence in collection.groups[0].rollout_result.sequences],
            [[11], [12]],
        )

    def test_dataset_exhaustion_drops_incomplete_batch(self):
        areno = _DAPOFakeAreno(candidate_count=1)
        trainer = _ScriptedDAPOTrainer(
            _dapo_config(batch_size=2, dapo_max_num_gen_batches=3),
            instance=areno,
            dataset=[{}],
            reward_fn=lambda _record: 0.0,
            loss_fn=dapo_loss_fn,
            groups=[_group(0, [0.0, 1.0])],
        )

        with self.assertRaisesRegex(RuntimeError, "no complete training batch"):
            trainer._fit_initialized()

        self.assertEqual(areno.train_calls, [])
        self.assertEqual(areno.finish_calls, 1)

    def test_overlong_penalty_is_linear_and_bounded(self):
        self.assertEqual(_overlong_penalty(5, max_response_length=10, buffer_length=4, factor=1.0), 0.0)
        self.assertAlmostEqual(_overlong_penalty(7, max_response_length=10, buffer_length=4, factor=1.0), -0.25)
        self.assertEqual(_overlong_penalty(10, max_response_length=10, buffer_length=4, factor=1.0), -1.0)
        self.assertEqual(_overlong_penalty(20, max_response_length=10, buffer_length=4, factor=1.0), -1.0)
        self.assertTrue(_has_reward_variance([0.0, 1.0]))
        self.assertFalse(_has_reward_variance([1.0, 1.0]))
        self.assertFalse(_has_reward_variance([1.0, 1.0 + 1.0e-9]))


class DAPODataPipelineAccuracyTest(unittest.TestCase):
    """Hand-derived fixtures verify each field from rollout through packed loss."""

    def test_rollout_reward_advantage_and_packed_actions_remain_aligned(self):
        from areno.api.backend.cuda.training import make_train_pack
        from areno.engine.runtime.train_step import _pack_train_data

        reward_records = []

        def reward_fn(record):
            reward_records.append(record)
            return [2.0, 4.0][record.metadata["sample_index"]]

        trainer = DAPOTrainer(
            _dapo_config(
                max_new_tokens=3,
                dapo_overlong_buffer_len=2,
                dapo_overlong_penalty_factor=0.5,
            ),
            instance=SimpleNamespace(),
            dataset=[],
            reward_fn=reward_fn,
            loss_fn=dapo_loss_fn,
        )
        item = PromptItem(
            prompt="check alignment",
            solutions="expected answer",
            input_tokens=[7, 8],
            record={"features": {"source": 3}, "row_id": "row-7"},
        )
        rollout = RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=[21, 22], resp_logprobs=[-0.2, -0.4]),
                RolloutSequence(resp_tokens=[31], resp_logprobs=[-0.6]),
            ]
        )

        group = trainer._score_prompt_group(_TokenIdTokenizer(), item, rollout, prompt_index=5)
        train_batch, raw_rewards, shaped_rewards, rollout_logprobs = trainer._materialize_scored_groups(
            _TokenIdTokenizer(), [group]
        )
        packed = _pack_train_data(make_train_pack(train_batch))

        self.assertEqual([record.tokens for record in reward_records], [[7, 8, 21, 22], [7, 8, 31]])
        self.assertEqual(
            [record.logprobs for record in reward_records],
            [[0.0, 0.0, -0.2, -0.4], [0.0, 0.0, -0.6]],
        )
        self.assertEqual(
            [record.loss_mask for record in reward_records],
            [[False, False, True, True], [False, False, True]],
        )
        self.assertEqual(
            [record.metadata for record in reward_records],
            [{"prompt_index": 5, "sample_index": 0}, {"prompt_index": 5, "sample_index": 1}],
        )
        self.assertEqual([record.prompt for record in reward_records], ["check alignment", "check alignment"])
        self.assertEqual([record.completion for record in reward_records], ["21", "31"])
        self.assertEqual([record.answer for record in reward_records], ["expected answer", "expected answer"])
        self.assertEqual([record.source_record for record in reward_records], [item.record, item.record])
        self.assertEqual(raw_rewards, [2.0, 4.0])
        self.assertEqual(shaped_rewards, [1.75, 4.0])
        self.assertEqual(rollout_logprobs, [-0.2, -0.4, -0.6])
        self.assertEqual([sequence.tokens for sequence in train_batch], [[7, 8, 21, 22], [7, 8, 31]])
        self.assertEqual(
            [sequence.prompt_mask for sequence in train_batch],
            [[True, True, False, False], [True, True, False]],
        )
        self.assertEqual([sequence.reward for sequence in train_batch], [1.75, 4.0])
        self.assertEqual([sequence.features for sequence in train_batch], [{"source": 3}, {"source": 3}])
        for actual, expected in zip(
            [sequence.advantages for sequence in train_batch],
            [[0.0, 0.0, -1.0, -1.0], [0.0, 0.0, 1.0]],
            strict=True,
        ):
            self.assertTrue(torch.allclose(torch.tensor(actual), torch.tensor(expected), atol=1e-6))
        self.assertEqual(packed["packed_response_mask"].tolist(), [False, True, True, False, True])
        self.assertTrue(
            torch.allclose(
                packed["packed_logprobs"],
                torch.tensor([0.0, -0.2, -0.4, 0.0, -0.6]),
            )
        )
        self.assertTrue(
            torch.allclose(
                packed["packed_advantages"],
                torch.tensor([0.0, -1.0, -1.0, 0.0, 1.0]),
                atol=1e-6,
            )
        )
        self.assertEqual(packed["packed_seq_ids"].tolist(), [0, 0, 0, 1, 1])

        current_logprobs = packed["packed_logprobs"].clone()
        current_logprobs += torch.tensor([9.0, math.log(0.5), math.log(1.5), -9.0, math.log(1.5)])
        current_logprobs.requires_grad_()
        loss, stats = dapo_loss_fn(packed, current_logprobs, clip_eps_low=0.2, clip_eps_high=0.28)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.34, places=6)
        self.assertTrue(
            torch.allclose(
                current_logprobs.grad,
                torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0]),
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(float(stats["ratio_mean"]), (0.5 + 1.5 + 1.5) / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
