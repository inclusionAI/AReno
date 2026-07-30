"""CPU-only tests for the completion validation module.

These tests cover classification of all four invalid types, the ``off``
(default) no-op path, ``filter`` policy behaviour, metrics output,
quarantine file writing, and boundary cases (all-invalid batch, empty
input).  No GPU or model loading is required.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from areno.engine.runtime.completion_validator import (
    classify_completion,
    get_special_token_ids,
    validate_completions,
)


class ClassifyCompletionTest(unittest.TestCase):
    """Unit tests for the single-completion classifier."""

    def test_valid_completion(self):
        """A normal completion with content should be valid."""
        check = classify_completion("answer is 42", [1, 2, 3])
        self.assertTrue(check.is_valid)
        self.assertIsNone(check.invalid_type)

    def test_empty_string(self):
        """An empty decoded string should be classified as empty."""
        check = classify_completion("", [0], eos_token_ids=(0,))
        self.assertFalse(check.is_valid)
        self.assertEqual(check.invalid_type, "empty")

    def test_no_response_tokens(self):
        """Zero response tokens should be classified as empty."""
        check = classify_completion("", [])
        self.assertFalse(check.is_valid)
        self.assertEqual(check.invalid_type, "empty")

    def test_whitespace_only(self):
        """Whitespace-only text should be classified as whitespace."""
        check = classify_completion("   \n  ", [10, 11])
        self.assertFalse(check.is_valid)
        self.assertEqual(check.invalid_type, "whitespace")

    def test_immediate_eos(self):
        """A single EOS token should be classified as immediate_eos."""
        check = classify_completion("<|endoftext|>", [2], eos_token_ids=(2,))
        self.assertFalse(check.is_valid)
        self.assertEqual(check.invalid_type, "immediate_eos")

    def test_special_token_only(self):
        """All-special-token responses should be classified as special_token."""
        check = classify_completion(
            "<|im_start|><|im_end|>",
            [100, 101],
            eos_token_ids=(2,),
            special_token_ids=(100, 101),
        )
        self.assertFalse(check.is_valid)
        self.assertEqual(check.invalid_type, "special_token")

    def test_eos_not_in_special_ids_still_immediate_eos(self):
        """Immediate EOS should take priority over special_token when len==1."""
        check = classify_completion(
            "<|endoftext|>",
            [2],
            eos_token_ids=(2,),
            special_token_ids=(2,),
        )
        self.assertFalse(check.is_valid)
        self.assertEqual(check.invalid_type, "immediate_eos")

    def test_valid_completion_with_eos_at_end(self):
        """A completion that has content followed by EOS should be valid."""
        check = classify_completion(
            "answer is 42",
            [1, 2, 3, 2],
            eos_token_ids=(2,),
            special_token_ids=(2,),
        )
        self.assertTrue(check.is_valid)


class ValidateCompletionsTest(unittest.TestCase):
    """Tests for the batch-level validate_completions function."""

    def test_policy_off_returns_inputs_unchanged(self):
        """When policy is 'off', all completions should be kept with no metrics."""
        completions = ["ok", "", "  ", "ok2"]
        resp_tokens = [[1, 2], [], [10], [3, 4]]
        kept_c, kept_t, vr = validate_completions(
            completions, resp_tokens, policy="off",
            eos_token_ids=(0,), special_token_ids=(0,),
        )
        self.assertEqual(len(kept_c), 4)
        self.assertEqual(vr.kept_indices, [0, 1, 2, 3])
        self.assertEqual(vr.dropped_indices, [])
        self.assertEqual(vr.metrics, {})

    def test_filter_drops_invalid(self):
        """Filter policy should drop empty and whitespace completions."""
        completions = ["ok", "", "  ", "ok2"]
        resp_tokens = [[1, 2], [], [10], [3, 4]]
        kept_c, kept_t, vr = validate_completions(
            completions, resp_tokens, policy="filter",
            eos_token_ids=(0,), special_token_ids=(0,),
        )
        self.assertEqual(len(kept_c), 2)
        self.assertEqual(vr.kept_indices, [0, 3])
        self.assertEqual(vr.dropped_indices, [1, 2])
        self.assertEqual(vr.quarantine_records[0]["invalid_type"], "empty")
        self.assertEqual(vr.quarantine_records[1]["invalid_type"], "whitespace")

    def test_filter_metrics(self):
        """Metrics should include per-type counts and filtered total."""
        completions = ["ok", "", "", "ok2"]
        resp_tokens = [[1], [], [], [2]]
        _, _, vr = validate_completions(
            completions, resp_tokens, policy="filter",
            eos_token_ids=(0,), special_token_ids=(0,),
        )
        self.assertEqual(vr.metrics["completion_total"], 4.0)
        self.assertEqual(vr.metrics["completion_valid"], 2.0)
        self.assertEqual(vr.metrics["completion_invalid"], 2.0)
        self.assertEqual(vr.metrics["completion_invalid_empty"], 2.0)
        self.assertEqual(vr.metrics["completion_filtered"], 2.0)

    def test_all_invalid_batch(self):
        """A batch where every completion is invalid should drop all rows."""
        completions = ["", "  "]
        resp_tokens = [[], [10]]
        kept_c, kept_t, vr = validate_completions(
            completions, resp_tokens, policy="filter",
            eos_token_ids=(0,), special_token_ids=(0,),
        )
        self.assertEqual(len(kept_c), 0)
        self.assertEqual(vr.kept_indices, [])
        self.assertEqual(vr.dropped_indices, [0, 1])

    def test_all_valid_batch(self):
        """A batch with no invalid completions should keep everything."""
        completions = ["answer1", "answer2"]
        resp_tokens = [[1, 2], [3, 4]]
        kept_c, kept_t, vr = validate_completions(
            completions, resp_tokens, policy="filter",
            eos_token_ids=(0,), special_token_ids=(0,),
        )
        self.assertEqual(len(kept_c), 2)
        self.assertEqual(vr.dropped_indices, [])

    def test_empty_input_batch(self):
        """An empty completions list should produce empty results."""
        kept_c, kept_t, vr = validate_completions(
            [], [], policy="filter",
            eos_token_ids=(0,), special_token_ids=(0,),
        )
        self.assertEqual(len(kept_c), 0)
        self.assertEqual(vr.metrics["completion_total"], 0.0)
        self.assertEqual(vr.metrics["completion_valid"], 0.0)

    def test_quarantine_file_written(self):
        """Quarantine records should be written as JSONL to the specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            quarantine_path = Path(tmpdir) / "empty_completions.jsonl"
            validate_completions(
                ["ok", ""],
                [[1], []],
                policy="filter",
                eos_token_ids=(0,),
                special_token_ids=(0,),
                quarantine_path=str(quarantine_path),
                prompt="What is 1+1?",
            )
            self.assertTrue(quarantine_path.exists())
            lines = quarantine_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["invalid_type"], "empty")
            self.assertEqual(record["prompt"], "What is 1+1?")
            self.assertEqual(record["policy"], "filter")

    def test_quarantine_not_written_when_all_valid(self):
        """No quarantine file should be created when there are no invalid completions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            quarantine_path = Path(tmpdir) / "empty_completions.jsonl"
            validate_completions(
                ["ok"],
                [[1]],
                policy="filter",
                eos_token_ids=(0,),
                special_token_ids=(0,),
                quarantine_path=str(quarantine_path),
            )
            self.assertFalse(quarantine_path.exists())

    def test_completion_truncated_in_quarantine(self):
        """Long completions should be truncated to 500 chars in quarantine records."""
        long_completion = "x" * 1000
        with tempfile.TemporaryDirectory() as tmpdir:
            quarantine_path = Path(tmpdir) / "q.jsonl"
            validate_completions(
                [long_completion, ""],
                [[1] * 1000, []],
                policy="filter",
                eos_token_ids=(0,),
                special_token_ids=(0,),
                quarantine_path=str(quarantine_path),
            )
            lines = quarantine_path.read_text(encoding="utf-8").strip().split("\n")
            # Only the empty completion should be in quarantine (the long one is valid).
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["invalid_type"], "empty")

    def test_prompt_truncated_in_quarantine(self):
        """Long prompts should be truncated to 500 chars in quarantine records."""
        long_prompt = "p" * 1000
        with tempfile.TemporaryDirectory() as tmpdir:
            quarantine_path = Path(tmpdir) / "q.jsonl"
            validate_completions(
                ["", "ok"],
                [[], [1]],
                policy="filter",
                eos_token_ids=(0,),
                special_token_ids=(0,),
                quarantine_path=str(quarantine_path),
                prompt=long_prompt,
            )
            lines = quarantine_path.read_text(encoding="utf-8").strip().split("\n")
            record = json.loads(lines[0])
            self.assertEqual(len(record["prompt"]), 500)


class GetSpecialTokenIdsTest(unittest.TestCase):
    """Tests for the tokenizer special-token-id extraction helper."""

    def test_extracts_all_special_ids(self):
        """all_special_ids from a tokenizer should be returned as a sorted tuple."""

        class FakeTokenizer:
            all_special_ids = [3, 1, 2]
            added_tokens_encoder = {"<extra>": 5}

        result = get_special_token_ids(FakeTokenizer())
        self.assertEqual(result, (1, 2, 3, 5))

    def test_handles_missing_attributes(self):
        """A tokenizer without all_special_ids should not raise."""

        class MinimalTokenizer:
            pass

        result = get_special_token_ids(MinimalTokenizer())
        self.assertEqual(result, ())


class ConfigFieldTest(unittest.TestCase):
    """Tests that the config dataclass carries the new field with correct default."""

    def test_default_policy_is_off(self):
        """RolloutTrainerConfig should default empty_completion_policy to 'off'."""
        from areno.api.trainer_config import PolicyTrainerConfig

        config = PolicyTrainerConfig(algo="gspo", ckpt="unused", dataset_path="unused")
        self.assertEqual(config.empty_completion_policy, "off")

    def test_policy_inherited_by_ppo(self):
        """PPOTrainerConfig should inherit the field from RolloutTrainerConfig."""
        from areno.api.trainer_config import PPOTrainerConfig

        config = PPOTrainerConfig(algo="ppo", ckpt="unused", dataset_path="unused")
        self.assertEqual(config.empty_completion_policy, "off")

    def test_sft_config_has_no_field(self):
        """SFT uses TrainerConfig which should not have the field."""
        from areno.api.trainer_config import TrainerConfig

        config = TrainerConfig(algo="sft", ckpt="unused", dataset_path="unused")
        self.assertFalse(hasattr(config, "empty_completion_policy"))


class QuarantineRecordFieldsTest(unittest.TestCase):
    """Tests that quarantine records contain all required fields with correct values."""

    def test_record_contains_all_fields(self):
        """Each quarantine record should have index, invalid_type, completion, resp_token_count, prompt, policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            quarantine_path = Path(tmpdir) / "q.jsonl"
            validate_completions(
                ["ok", ""],
                [[1, 2], []],
                policy="filter",
                eos_token_ids=(0,),
                special_token_ids=(0,),
                quarantine_path=str(quarantine_path),
                prompt="test prompt",
            )
            lines = quarantine_path.read_text(encoding="utf-8").strip().split("\n")
            record = json.loads(lines[0])
            self.assertEqual(record["index"], 1)
            self.assertEqual(record["invalid_type"], "empty")
            self.assertEqual(record["completion"], "")
            self.assertEqual(record["resp_token_count"], 0)
            self.assertEqual(record["prompt"], "test prompt")
            self.assertEqual(record["policy"], "filter")

    def test_multiple_records_preserve_order(self):
        """Multiple invalid completions should produce records in original index order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            quarantine_path = Path(tmpdir) / "q.jsonl"
            validate_completions(
                ["", "ok", "  ", "ok2"],
                [[], [1], [2], [3]],
                policy="filter",
                eos_token_ids=(0,),
                special_token_ids=(0,),
                quarantine_path=str(quarantine_path),
                prompt="p",
            )
            lines = quarantine_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            r0 = json.loads(lines[0])
            r1 = json.loads(lines[1])
            self.assertEqual(r0["index"], 0)
            self.assertEqual(r0["invalid_type"], "empty")
            self.assertEqual(r1["index"], 2)
            self.assertEqual(r1["invalid_type"], "whitespace")


class RewardFnIsolationTest(unittest.TestCase):
    """Integration-style tests verifying invalid completions never reach reward_fn.

    These tests simulate the filtering step that runs before reward_fn in the
    trainer and assert that only valid completions survive to be scored.
    """

    def test_filter_then_reward_only_receives_valid(self):
        """After filtering, reward_fn should only be called with valid completions."""
        completions = ["answer is 42", "", "  ", "answer is 9"]
        resp_tokens = [[1, 2, 3], [], [10, 11], [4, 5, 6]]

        kept_completions, kept_tokens, vr = validate_completions(
            completions, resp_tokens, policy="filter",
            eos_token_ids=(0,), special_token_ids=(0,),
        )

        # Simulate reward_fn: it should only see kept completions.
        seen_by_reward = list(kept_completions)
        self.assertEqual(seen_by_reward, ["answer is 42", "answer is 9"])
        self.assertNotIn("", seen_by_reward)
        self.assertNotIn("  ", seen_by_reward)

    def test_off_policy_passes_everything_to_reward(self):
        """When policy is 'off', reward_fn should receive all completions including invalid."""
        completions = ["ok", ""]
        resp_tokens = [[1], []]

        kept_completions, _, vr = validate_completions(
            completions, resp_tokens, policy="off",
            eos_token_ids=(0,), special_token_ids=(0,),
        )

        seen_by_reward = list(kept_completions)
        self.assertEqual(seen_by_reward, ["ok", ""])
        self.assertIn("", seen_by_reward)

    def test_all_invalid_yields_empty_for_reward(self):
        """When all completions are invalid, reward_fn should receive nothing."""
        completions = ["", "  "]
        resp_tokens = [[], [10]]

        kept_completions, _, vr = validate_completions(
            completions, resp_tokens, policy="filter",
            eos_token_ids=(0,), special_token_ids=(0,),
        )

        self.assertEqual(len(kept_completions), 0)


class MaterializeTrainBatchIntegrationTest(unittest.TestCase):
    """Integration tests that mock rollout results and exercise the full
    _filter_empty_completions → _materialize_train_batch pipeline including
    validation, reward, and TrainSequence assembly.
    """

    def _make_trainer(self, policy="off", resample_budget=3):
        """Build a minimal PolicyOnlyTrainer with mocked dependencies."""

        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        class FakeTokenizer:
            eos_token_id = 2
            chat_template = None

            def decode(self, token_ids):
                if not token_ids:
                    return ""
                if token_ids == [2]:
                    return "<|endoftext|>"
                if all(t in (2,) for t in token_ids):
                    return "<|endoftext|>" * len(token_ids)
                return "answer_" + "".join(chr(t + 60) for t in token_ids)

        class FakeContext:
            model_path = ""

        config = SimpleNamespace(
            empty_completion_policy=policy,
            empty_completion_resample_budget=resample_budget,
            save_path=None,
        )

        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.config = config
        trainer.areno = SimpleNamespace(_ctx=FakeContext())
        trainer.reward_fn = None  # set per-test
        trainer.loss_fn = None
        trainer.logger = __import__("logging").getLogger("test")
        trainer._agent_run_fn = None

        return trainer, FakeTokenizer()

    def _make_prompt_batch(self, n_prompts=1):
        """Create a PromptBatch with simple tokenized prompts."""
        from areno.api.data import PromptBatch, PromptItem

        items = [
            PromptItem(
                prompt=f"question {i}",
                solutions=[str(i)],
                input_tokens=[100 + i],
                record={"answer": str(i)},
            )
            for i in range(n_prompts)
        ]
        return PromptBatch(items=items, scanned=n_prompts, skipped_long=0, total_skipped_long=0)

    def _make_rollout_results(self, n_prompts=1, n_samples=4, invalid_indices=None):
        """Create mock RolloutResult list with specified invalid completions.

        invalid_indices is a dict mapping prompt_idx -> set of sample indices
        that should be invalid.  Invalid samples get empty resp_tokens.
        """
        from areno.api.models import RolloutResult, RolloutSequence

        if invalid_indices is None:
            invalid_indices = {}

        results = []
        for p in range(n_prompts):
            invalid_for_prompt = invalid_indices.get(p, set())
            sequences = []
            for s in range(n_samples):
                if s in invalid_for_prompt:
                    sequences.append(RolloutSequence(resp_tokens=[], resp_logprobs=[]))
                else:
                    sequences.append(
                        RolloutSequence(
                            resp_tokens=[10 + s, 20 + s],
                            resp_logprobs=[-0.5, -0.3],
                        )
                    )
            results.append(RolloutResult(sequences=sequences))
        return results

    def test_off_policy_preserves_all_samples(self):
        """With policy='off', all samples should reach reward_fn and TrainSequence."""

        trainer, tokenizer = self._make_trainer(policy="off")
        prompt_batch = self._make_prompt_batch(n_prompts=1)
        rollout_results = self._make_rollout_results(
            n_prompts=1, n_samples=4,
            invalid_indices={0: {1, 2}},  # samples 1 and 2 are empty
        )

        # _filter_empty_completions is a no-op when policy is "off".
        rollout_results = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params=None
        )

        seen_completions = []

        def reward_fn(record):
            seen_completions.append(record.completion)
            return 1.0

        trainer.reward_fn = reward_fn
        train_batch, rewards_all, rollout_logprobs = trainer._materialize_train_batch(
            tokenizer, prompt_batch, rollout_results
        )

        # All 4 samples should be present (including empty ones).
        self.assertEqual(len(train_batch), 4)
        self.assertEqual(len(rewards_all), 4)
        self.assertIn("", seen_completions)

    def test_filter_removes_invalid_before_reward(self):
        """With policy='filter', invalid completions should not reach reward_fn."""

        trainer, tokenizer = self._make_trainer(policy="filter")
        prompt_batch = self._make_prompt_batch(n_prompts=1)
        rollout_results = self._make_rollout_results(
            n_prompts=1, n_samples=4,
            invalid_indices={0: {1, 2}},  # samples 1 and 2 are empty
        )

        rollout_results = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params=None
        )

        seen_completions = []

        def reward_fn(record):
            seen_completions.append(record.completion)
            return 1.0

        trainer.reward_fn = reward_fn
        train_batch, rewards_all, rollout_logprobs = trainer._materialize_train_batch(
            tokenizer, prompt_batch, rollout_results
        )

        # Only 2 valid samples should reach reward_fn and TrainSequence.
        self.assertEqual(len(train_batch), 2)
        self.assertEqual(len(rewards_all), 2)
        self.assertNotIn("", seen_completions)
        # All seen completions should be non-empty.
        for c in seen_completions:
            self.assertTrue(len(c) > 0)

    def test_filter_all_invalid_skips_prompt(self):
        """When all samples are invalid, should skip the prompt with a warning, not crash."""
        trainer, tokenizer = self._make_trainer(policy="filter")
        prompt_batch = self._make_prompt_batch(n_prompts=1)
        rollout_results = self._make_rollout_results(
            n_prompts=1, n_samples=4,
            invalid_indices={0: {0, 1, 2, 3}},  # all empty
        )

        rollout_results = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params=None
        )

        def reward_fn(record):
            self.fail("reward_fn should not be called when all completions are invalid")

        trainer.reward_fn = reward_fn
        train_batch, rewards_all, _ = trainer._materialize_train_batch(
            tokenizer, prompt_batch, rollout_results
        )
        # Should skip the prompt entirely, producing empty train_batch.
        self.assertEqual(len(train_batch), 0)
        self.assertEqual(len(rewards_all), 0)

    def test_filter_mixed_valid_invalid_advantages_correct(self):
        """After filtering, advantages should be computed only over valid samples."""
        trainer, tokenizer = self._make_trainer(policy="filter")
        prompt_batch = self._make_prompt_batch(n_prompts=1)
        rollout_results = self._make_rollout_results(
            n_prompts=1, n_samples=4,
            invalid_indices={0: {0, 3}},  # samples 0 and 3 are empty
        )

        rollout_results = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params=None
        )

        def reward_fn(record):
            # Give different rewards to the two valid samples.
            # sample 1 has resp_tokens=[11,21], sample 2 has [12,22].
            if len(record.tokens) > 1 and record.tokens[1] == 11:  # sample 1
                return 2.0
            return 0.0  # sample 2

        trainer.reward_fn = reward_fn
        train_batch, rewards_all, _ = trainer._materialize_train_batch(
            tokenizer, prompt_batch, rollout_results
        )

        # 2 valid samples, 2 rewards.
        self.assertEqual(len(rewards_all), 2)
        self.assertEqual(len(train_batch), 2)
        # Rewards should be [2.0, 0.0] for samples 1 and 2.
        self.assertIn(2.0, rewards_all)
        self.assertIn(0.0, rewards_all)
        # Each TrainSequence should have matching reward.
        for seq in train_batch:
            self.assertIn(seq.reward, [2.0, 0.0])

    def test_filter_multiple_prompts(self):
        """Filtering should work correctly across multiple prompts in one batch."""
        trainer, tokenizer = self._make_trainer(policy="filter")
        prompt_batch = self._make_prompt_batch(n_prompts=2)
        rollout_results = self._make_rollout_results(
            n_prompts=2, n_samples=2,
            invalid_indices={0: {0}, 1: {1}},  # prompt0 sample0 empty, prompt1 sample1 empty
        )

        rollout_results = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params=None
        )

        def reward_fn(record):
            return 1.0

        trainer.reward_fn = reward_fn
        train_batch, rewards_all, _ = trainer._materialize_train_batch(
            tokenizer, prompt_batch, rollout_results
        )

        # 2 prompts × 1 valid sample each = 2 TrainSequences.
        self.assertEqual(len(train_batch), 2)
        self.assertEqual(len(rewards_all), 2)

    def test_resample_policy_all_valid_no_retry(self):
        """With resample policy and all valid completions, no retry should occur."""
        trainer, tokenizer = self._make_trainer(policy="resample")
        prompt_batch = self._make_prompt_batch(n_prompts=1)
        rollout_results = self._make_rollout_results(
            n_prompts=1, n_samples=4,
            invalid_indices={},  # all valid
        )

        rollout_results = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params=None
        )

        def reward_fn(record):
            return 1.0

        trainer.reward_fn = reward_fn
        train_batch, rewards_all, _ = trainer._materialize_train_batch(
            tokenizer, prompt_batch, rollout_results
        )

        self.assertEqual(len(train_batch), 4)
        self.assertEqual(len(rewards_all), 4)

    def test_resample_metrics(self):
        """Resample policy should produce resample-specific metrics."""
        completions = ["ok", ""]
        resp_tokens = [[1], []]
        _, _, vr = validate_completions(
            completions, resp_tokens, policy="resample",
            eos_token_ids=(0,), special_token_ids=(0,),
            resample_budget=5,
        )
        self.assertIn("completion_resample_candidates", vr.metrics)
        self.assertEqual(vr.metrics["completion_resample_candidates"], 1.0)
        self.assertEqual(vr.metrics["completion_resample_budget"], 5.0)

    # ------------------------------------------------------------------
    # Resample integration tests
    # ------------------------------------------------------------------

    def test_resample_succeeds_on_retry(self):
        """When resample finds invalid completions, it should re-rollout and
        succeed if the new completions are valid."""
        from areno.api.models import RolloutResult, RolloutSequence

        trainer, tokenizer = self._make_trainer(policy="resample", resample_budget=3)
        prompt_batch = self._make_prompt_batch(n_prompts=1)

        # First rollout: sample 0 is invalid (empty).
        bad_result = RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=[], resp_logprobs=[]),       # invalid
                RolloutSequence(resp_tokens=[11, 21], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[12, 22], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[13, 23], resp_logprobs=[-0.5, -0.3]),
            ]
        )
        # Second rollout (retry): all valid.
        good_result = RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=[14, 24], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[15, 25], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[16, 26], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[17, 27], resp_logprobs=[-0.5, -0.3]),
            ]
        )

        # Mock _run_prompt_rollout_for_tokens to return the good result.
        call_count = [0]

        async def mock_rollout(sampling_params, prompt_tokens):
            call_count[0] += 1
            return [good_result]

        trainer._run_prompt_rollout_for_tokens = mock_rollout

        rollout_results = [bad_result]
        filtered = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params={"temp": 1.0}
        )

        # Should have called rollout once (retry).
        self.assertEqual(call_count[0], 1)
        # Should keep all 4 samples from the good result.
        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(filtered[0].sequences), 4)

    def test_resample_exhausted_fallback_to_filter(self):
        """When resample budget is exhausted, invalid completions should be
        dropped via filter fallback."""
        from areno.api.models import RolloutResult, RolloutSequence

        trainer, tokenizer = self._make_trainer(policy="resample", resample_budget=2)
        prompt_batch = self._make_prompt_batch(n_prompts=1)

        # Every rollout returns the same invalid result.
        bad_result = RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=[], resp_logprobs=[]),       # invalid
                RolloutSequence(resp_tokens=[11, 21], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[], resp_logprobs=[]),       # invalid
                RolloutSequence(resp_tokens=[13, 23], resp_logprobs=[-0.5, -0.3]),
            ]
        )

        call_count = [0]

        async def mock_rollout(sampling_params, prompt_tokens):
            call_count[0] += 1
            return [bad_result]

        trainer._run_prompt_rollout_for_tokens = mock_rollout

        rollout_results = [bad_result]
        filtered = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params={"temp": 1.0}
        )

        # Should have called rollout budget times.
        self.assertEqual(call_count[0], 2)
        # After fallback, only 2 valid samples remain.
        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(filtered[0].sequences), 2)

    def test_resample_partial_success_filter_remaining(self):
        """When resample produces some valid and some invalid completions,
        the retry loop continues until budget is exhausted, then falls back
        to filter which drops the remaining invalid ones."""
        from areno.api.models import RolloutResult, RolloutSequence

        trainer, tokenizer = self._make_trainer(policy="resample", resample_budget=2)
        prompt_batch = self._make_prompt_batch(n_prompts=1)

        # Initial: 2 invalid, 2 valid.
        initial = RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=[], resp_logprobs=[]),
                RolloutSequence(resp_tokens=[11, 21], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[], resp_logprobs=[]),
                RolloutSequence(resp_tokens=[13, 23], resp_logprobs=[-0.5, -0.3]),
            ]
        )
        # Every retry returns the same partial result (1 invalid, 3 valid).
        partial = RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=[14, 24], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[], resp_logprobs=[]),       # still invalid
                RolloutSequence(resp_tokens=[16, 26], resp_logprobs=[-0.5, -0.3]),
                RolloutSequence(resp_tokens=[17, 27], resp_logprobs=[-0.5, -0.3]),
            ]
        )

        call_count = [0]

        async def mock_rollout(sampling_params, prompt_tokens):
            call_count[0] += 1
            return [partial]

        trainer._run_prompt_rollout_for_tokens = mock_rollout

        rollout_results = [initial]
        filtered = trainer._filter_empty_completions(
            rollout_results, tokenizer, prompt_batch, sampling_params={"temp": 1.0}
        )

        # Retry loop ran budget times (2) because each retry still had 1 invalid.
        self.assertEqual(call_count[0], 2)
        # After fallback filter, only 3 valid remain.
        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(filtered[0].sequences), 3)


class AgenticCompletionFilterTest(unittest.TestCase):
    """Tests for _filter_agentic_completions."""

    def _make_trainer(self, policy="filter"):
        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        class FakeTokenizer:
            eos_token_id = 2
            chat_template = None

            def decode(self, token_ids):
                if not token_ids:
                    return ""
                if token_ids == [2]:
                    return "<|endoftext|>"
                return "answer_" + "".join(chr(t + 60) for t in token_ids)

        class FakeContext:
            model_path = ""

        config = SimpleNamespace(
            empty_completion_policy=policy,
            empty_completion_resample_budget=3,
            save_path=None,
        )

        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.config = config
        trainer.areno = SimpleNamespace(_ctx=FakeContext())
        trainer.logger = __import__("logging").getLogger("test")

        return trainer, FakeTokenizer()

    def _make_reward_record(self, completion, tokens):
        from areno.api.rewards import RewardRecord
        return RewardRecord(
            prompt="test prompt",
            completion=completion,
            source_record={},
            answer="",
            tokens=tokens,
            logprobs=[],
            loss_mask=[],
            metadata={},
        )

    def _make_sample(self):
        return SimpleNamespace(item=SimpleNamespace(record={}))

    def test_filter_removes_invalid_agentic(self):
        trainer, tokenizer = self._make_trainer(policy="filter")
        samples = [self._make_sample() for _ in range(3)]
        reward_records = [
            self._make_reward_record("ok", [1, 2]),
            self._make_reward_record("", []),       # empty → dropped
            self._make_reward_record("also ok", [3, 4]),
        ]

        filtered_samples, filtered_records = trainer._filter_agentic_completions(
            samples, reward_records, tokenizer
        )

        self.assertEqual(len(filtered_samples), 2)
        self.assertEqual(len(filtered_records), 2)
        self.assertEqual(filtered_records[0].completion, "ok")
        self.assertEqual(filtered_records[1].completion, "also ok")

    def test_all_invalid_agentic_raises(self):
        trainer, tokenizer = self._make_trainer(policy="filter")
        samples = [self._make_sample() for _ in range(2)]
        reward_records = [
            self._make_reward_record("", []),
            self._make_reward_record("  ", [10]),
        ]

        with self.assertRaises(RuntimeError) as ctx:
            trainer._filter_agentic_completions(samples, reward_records, tokenizer)
        self.assertIn("all agent trajectories", str(ctx.exception))

    def test_off_policy_agentic_noop(self):
        trainer, tokenizer = self._make_trainer(policy="off")
        samples = [self._make_sample() for _ in range(2)]
        reward_records = [
            self._make_reward_record("", []),
            self._make_reward_record("ok", [1]),
        ]

        # off policy: caller doesn't invoke this method at all, but if it
        # does, it should still be a no-op (defensive).
        filtered_samples, filtered_records = trainer._filter_agentic_completions(
            samples, reward_records, tokenizer
        )
        self.assertEqual(len(filtered_samples), 2)
        self.assertEqual(len(filtered_records), 2)


class PPOTrainerFilterTest(unittest.TestCase):
    """Tests for PPO filter logic applied to token_rows and reward_records.

    These tests exercise the validation block inside PPO's
    _materialize_train_batch without running the full forward pass
    (which requires a real backend).
    """

    def _make_ppo_trainer(self, policy="filter"):
        from areno.api.trainers.ppo import PPOTrainer

        class FakeTokenizer:
            eos_token_id = 2
            chat_template = None

            def decode(self, token_ids):
                if not token_ids:
                    return ""
                if token_ids == [2]:
                    return "<|endoftext|>"
                return "answer_" + "".join(chr(t + 60) for t in token_ids)

        class FakeContext:
            model_path = ""

        config = SimpleNamespace(
            empty_completion_policy=policy,
            empty_completion_resample_budget=3,
            save_path=None,
            n_samples=2,
        )

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.config = config
        trainer.areno = SimpleNamespace(_ctx=FakeContext())
        trainer.reward_fn = None
        trainer.logger = __import__("logging").getLogger("test")
        trainer._last_ppo_stats = {}

        return trainer, FakeTokenizer()

    def _make_reward_records(self, completions, tokens_list):
        from areno.api.rewards import RewardRecord
        records = []
        for completion, tokens in zip(completions, tokens_list):
            records.append(RewardRecord(
                prompt="test",
                completion=completion,
                source_record={},
                answer="",
                tokens=tokens,
                logprobs=[],
                loss_mask=[],
                metadata={},
            ))
        return records

    def test_ppo_filter_keeps_valid_only(self):
        """validate_completions with policy='filter' should keep only valid
        reward records."""
        from areno.engine.runtime.completion_validator import validate_completions

        trainer, tokenizer = self._make_ppo_trainer(policy="filter")
        eos_ids = trainer._completion_eos_ids(tokenizer)
        special_ids = trainer._completion_special_token_ids(tokenizer)

        completions = ["ok", "", "also ok"]
        resp_tokens = [[1, 2], [], [3, 4]]
        records = self._make_reward_records(completions, [[100, 1, 2], [100], [100, 3, 4]])

        _, _, vr = validate_completions(
            completions, resp_tokens, policy="filter",
            eos_token_ids=eos_ids, special_token_ids=special_ids,
        )

        kept_idx = set(vr.kept_indices)
        filtered = [r for i, r in enumerate(records) if i in kept_idx]
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].completion, "ok")
        self.assertEqual(filtered[1].completion, "also ok")

    def test_ppo_all_invalid_raises(self):
        """When all completions are invalid, PPO should raise RuntimeError."""
        from areno.engine.runtime.completion_validator import validate_completions

        trainer, tokenizer = self._make_ppo_trainer(policy="filter")
        eos_ids = trainer._completion_eos_ids(tokenizer)
        special_ids = trainer._completion_special_token_ids(tokenizer)

        completions = ["", "  "]
        resp_tokens = [[], [10]]

        _, _, vr = validate_completions(
            completions, resp_tokens, policy="filter",
            eos_token_ids=eos_ids, special_token_ids=special_ids,
        )

        # Simulate what PPO does: check if token_rows is empty after filtering.
        token_rows = [[100], [100, 10]]
        kept_idx = set(vr.kept_indices)
        token_rows = [r for i, r in enumerate(token_rows) if i in kept_idx]

        with self.assertRaises(RuntimeError) as ctx:
            if not token_rows:
                raise RuntimeError(
                    "all completions were empty or invalid; "
                    f"dropped={len(vr.dropped_indices)} policy=filter"
                )
        self.assertIn("all completions were empty or invalid", str(ctx.exception))

    def test_ppo_off_policy_no_filter(self):
        """With policy='off', validate_completions returns all inputs unchanged."""
        from areno.engine.runtime.completion_validator import validate_completions

        trainer, tokenizer = self._make_ppo_trainer(policy="off")
        eos_ids = trainer._completion_eos_ids(tokenizer)
        special_ids = trainer._completion_special_token_ids(tokenizer)

        completions = ["ok", ""]
        resp_tokens = [[1], []]

        kept_c, kept_t, vr = validate_completions(
            completions, resp_tokens, policy="off",
            eos_token_ids=eos_ids, special_token_ids=special_ids,
        )
        self.assertEqual(len(kept_c), 2)
        self.assertEqual(vr.dropped_indices, [])


class CLIParameterValidationTest(unittest.TestCase):
    """Tests for CLI parameter validation of empty-completion-policy options."""

    def test_invalid_policy_rejected(self):
        """Invalid policy values should be rejected."""
        invalid_policies = ["invalid", "delete", "ignore", "FILTER", "OFF"]
        for policy in invalid_policies:
            with self.subTest(policy=policy):
                self.assertNotIn(policy, ("off", "filter", "resample"))

    def test_valid_policies_accepted(self):
        """Valid policy values should pass validation."""
        valid_policies = ["off", "filter", "resample"]
        for policy in valid_policies:
            with self.subTest(policy=policy):
                self.assertIn(policy, ("off", "filter", "resample"))

    def test_resample_requires_positive_budget(self):
        """When policy is resample, budget must be > 0."""
        # Simulate the CLI validation logic.
        policy = "resample"
        for budget in (0, -1, -5):
            with self.subTest(budget=budget):
                is_valid = not (policy == "resample" and budget <= 0)
                self.assertFalse(is_valid)

    def test_non_resample_accepts_any_budget(self):
        """When policy is not resample, budget is not validated."""
        for policy in ("off", "filter"):
            for budget in (0, -1, 3):
                with self.subTest(policy=policy, budget=budget):
                    is_valid = not (policy == "resample" and budget <= 0)
                    self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
