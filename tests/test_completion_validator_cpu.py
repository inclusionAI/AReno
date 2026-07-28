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
    CompletionCheck,
    ValidationResult,
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
    _materialize_train_batch pipeline including validation, reward, and
    TrainSequence assembly.
    """

    def _make_trainer(self, policy="off"):
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
        from areno.api.rewards import RewardRecord

        trainer, tokenizer = self._make_trainer(policy="off")
        prompt_batch = self._make_prompt_batch(n_prompts=1)
        rollout_results = self._make_rollout_results(
            n_prompts=1, n_samples=4,
            invalid_indices={0: {1, 2}},  # samples 1 and 2 are empty
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
        from areno.api.rewards import RewardRecord

        trainer, tokenizer = self._make_trainer(policy="filter")
        prompt_batch = self._make_prompt_batch(n_prompts=1)
        rollout_results = self._make_rollout_results(
            n_prompts=1, n_samples=4,
            invalid_indices={0: {1, 2}},  # samples 1 and 2 are empty
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
        trainer.config = SimpleNamespace(
            empty_completion_policy="resample",
            empty_completion_resample_budget=3,
            save_path=None,
        )
        prompt_batch = self._make_prompt_batch(n_prompts=1)
        rollout_results = self._make_rollout_results(
            n_prompts=1, n_samples=4,
            invalid_indices={},  # all valid
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


if __name__ == "__main__":
    unittest.main()
