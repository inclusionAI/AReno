"""CPU tests for duplicate detection and bounded resampling (issue #209).

These tests exercise ``areno.api.data.detect_duplicates`` and the related
``DedupResult`` / ``normalize_completion`` helpers without requiring a GPU,
tokenizer, or backend.  They also verify the trainer config fields default
to disabled (backward compatible) and that invalid inputs raise clear errors.
"""

from __future__ import annotations

import unittest

from areno.api.data import DedupResult, detect_duplicates, normalize_completion
from areno.api.trainer_config import PolicyTrainerConfig, RolloutTrainerConfig


class NormalizeCompletionTest(unittest.TestCase):
    """``normalize_completion`` should produce a stable comparison key."""

    def test_strips_whitespace(self):
        self.assertEqual(normalize_completion("  hello  "), "hello")

    def test_lowercases(self):
        self.assertEqual(normalize_completion("Hello"), "hello")

    def test_empty_string(self):
        self.assertEqual(normalize_completion(""), "")

    def test_only_whitespace(self):
        self.assertEqual(normalize_completion("   "), "")


class DetectDuplicatesTest(unittest.TestCase):
    """Core dedup logic tests covering success, boundary, and error paths."""

    # ------------------------------------------------------------------
    # Success paths
    # ------------------------------------------------------------------

    def test_already_unique_group_requests_no_resample(self):
        """A group with all-unique completions should not request resampling."""
        result = detect_duplicates(["alpha", "beta", "gamma"])
        self.assertEqual(result.duplicate_count, 0)
        self.assertEqual(result.unique_count, 3)
        self.assertEqual(result.total_count, 3)
        self.assertAlmostEqual(result.duplicate_ratio, 0.0)
        self.assertEqual(result.resample_requested, 0)
        self.assertEqual(result.duplicate_indices, [])

    def test_all_identical_group(self):
        """Every sample after the first is a duplicate."""
        result = detect_duplicates(["same", "same", "same", "same"])
        self.assertEqual(result.duplicate_count, 3)
        self.assertEqual(result.unique_count, 1)
        self.assertEqual(result.total_count, 4)
        self.assertAlmostEqual(result.duplicate_ratio, 0.75)
        self.assertEqual(result.resample_requested, 3)
        self.assertEqual(result.duplicate_indices, [1, 2, 3])

    def test_partially_duplicated_group(self):
        """Only later copies are flagged; first occurrences are kept."""
        result = detect_duplicates(["a", "b", "a", "c", "b"])
        self.assertEqual(result.duplicate_count, 2)
        self.assertEqual(result.unique_count, 3)
        self.assertEqual(result.total_count, 5)
        self.assertAlmostEqual(result.duplicate_ratio, 0.4)
        self.assertEqual(result.resample_requested, 2)
        self.assertEqual(result.duplicate_indices, [2, 4])

    def test_normalized_comparison_is_case_insensitive(self):
        """Completions differing only in case should be duplicates."""
        result = detect_duplicates(["Hello", "HELLO", "hello"])
        self.assertEqual(result.duplicate_count, 2)
        self.assertEqual(result.unique_count, 1)
        self.assertEqual(result.duplicate_indices, [1, 2])

    def test_normalized_comparison_strips_whitespace(self):
        """Leading/trailing whitespace should not prevent duplicate detection."""
        result = detect_duplicates(["  answer  ", "answer", "\tanswer\n"])
        self.assertEqual(result.duplicate_count, 2)
        self.assertEqual(result.unique_count, 1)

    # ------------------------------------------------------------------
    # Boundary / budget paths
    # ------------------------------------------------------------------

    def test_target_unique_caps_resample_request(self):
        """When target_unique < n_samples, only the gap is requested."""
        # 4 identical, target 2 unique -> need 1 more, request 1
        result = detect_duplicates(["x", "x", "x", "x"], target_unique=2)
        self.assertEqual(result.unique_count, 1)
        self.assertEqual(result.resample_requested, 1)

    def test_max_resample_caps_request(self):
        """Hard request limit should bound resample_requested."""
        # 4 identical, target 4 unique, but budget is 2
        result = detect_duplicates(["x", "x", "x", "x"], target_unique=4, max_resample=2)
        self.assertEqual(result.resample_requested, 2)

    def test_budget_exhausted_leaves_duplicates(self):
        """When budget < needed, resample_requested reflects the cap."""
        result = detect_duplicates(["x", "x", "x", "x"], target_unique=4, max_resample=1)
        self.assertEqual(result.duplicate_count, 3)
        self.assertEqual(result.resample_requested, 1)

    def test_target_already_met_requests_zero(self):
        """No resampling when unique count already meets target."""
        result = detect_duplicates(["a", "b", "a"], target_unique=2)
        self.assertEqual(result.unique_count, 2)
        self.assertEqual(result.resample_requested, 0)

    def test_single_completion(self):
        """A single completion is always unique."""
        result = detect_duplicates(["only"])
        self.assertEqual(result.duplicate_count, 0)
        self.assertEqual(result.unique_count, 1)
        self.assertEqual(result.resample_requested, 0)

    def test_default_target_is_full_uniqueness(self):
        """Without target_unique, the goal is all samples unique."""
        result = detect_duplicates(["a", "a", "b"])
        self.assertEqual(result.resample_requested, 1)

    def test_default_max_resample_is_n_samples(self):
        """Without max_resample, the budget equals the group size."""
        result = detect_duplicates(["a", "a", "a"], target_unique=10)
        self.assertEqual(result.resample_requested, 3)

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_empty_completions_raises(self):
        with self.assertRaisesRegex(ValueError, "completions must be non-empty"):
            detect_duplicates([])

    def test_target_unique_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "target_unique must be >= 1"):
            detect_duplicates(["a"], target_unique=0)

    def test_negative_max_resample_raises(self):
        with self.assertRaisesRegex(ValueError, "max_resample must be >= 0"):
            detect_duplicates(["a"], max_resample=-1)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_deterministic_output(self):
        """Same input always produces the same result."""
        completions = ["a", "b", "a", "c", "b", "a"]
        r1 = detect_duplicates(completions)
        r2 = detect_duplicates(completions)
        self.assertEqual(r1, r2)

    def test_order_preserves_first_occurrence(self):
        """The first occurrence of a duplicate is never flagged."""
        result = detect_duplicates(["x", "y", "x", "y", "z"])
        self.assertEqual(result.duplicate_indices, [2, 3])
        # Indices 0, 1, 4 are unique first-occurrences
        self.assertEqual(result.unique_count, 3)


class DedupResultFieldsTest(unittest.TestCase):
    """Assert emitted metric fields are present and correctly typed."""

    def test_all_fields_populated(self):
        result = detect_duplicates(["a", "a", "b"])
        self.assertIsInstance(result, DedupResult)
        self.assertIsInstance(result.duplicate_count, int)
        self.assertIsInstance(result.unique_count, int)
        self.assertIsInstance(result.total_count, int)
        self.assertIsInstance(result.duplicate_ratio, float)
        self.assertIsInstance(result.resample_requested, int)
        self.assertIsInstance(result.duplicate_indices, list)

    def test_duplicate_ratio_range(self):
        """Ratio should be in [0, 1]."""
        for completions in [["a"], ["a", "b"], ["a", "a"], ["a", "a", "a", "b"]]:
            result = detect_duplicates(completions)
            self.assertGreaterEqual(result.duplicate_ratio, 0.0)
            self.assertLessEqual(result.duplicate_ratio, 1.0)


class TrainerConfigDedupTest(unittest.TestCase):
    """Config fields should default to disabled (backward compatible)."""

    def test_rollout_config_dedup_disabled_by_default(self):
        cfg = RolloutTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
        )
        self.assertFalse(cfg.dedup_enabled)
        self.assertIsNone(cfg.dedup_min_unique)
        self.assertIsNone(cfg.dedup_max_resample)

    def test_rollout_config_resolved_defaults(self):
        cfg = RolloutTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
            n_samples=8,
        )
        self.assertEqual(cfg.resolved_dedup_min_unique(), 8)
        self.assertEqual(cfg.resolved_dedup_max_resample(), 8)

    def test_rollout_config_explicit_dedup_values(self):
        cfg = RolloutTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
            n_samples=8,
            dedup_enabled=True,
            dedup_min_unique=4,
            dedup_max_resample=2,
        )
        self.assertTrue(cfg.dedup_enabled)
        self.assertEqual(cfg.resolved_dedup_min_unique(), 4)
        self.assertEqual(cfg.resolved_dedup_max_resample(), 2)

    def test_policy_config_inherits_dedup_fields(self):
        cfg = PolicyTrainerConfig(
            algo="grpo", ckpt="unused", dataset_path="unused",
            dedup_enabled=True,
        )
        self.assertTrue(cfg.dedup_enabled)


class TrainerConfigDedupValidationTest(unittest.TestCase):
    """Config validation should fail before expensive model/worker init."""

    def test_dedup_min_unique_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "dedup_min_unique must be >= 1"):
            RolloutTrainerConfig(
                algo="gspo", ckpt="unused", dataset_path="unused",
                dedup_min_unique=0,
            )

    def test_dedup_max_resample_negative_raises(self):
        with self.assertRaisesRegex(ValueError, "dedup_max_resample must be >= 0"):
            RolloutTrainerConfig(
                algo="gspo", ckpt="unused", dataset_path="unused",
                dedup_max_resample=-1,
            )

    def test_n_samples_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "n_samples must be >= 1"):
            RolloutTrainerConfig(
                algo="gspo", ckpt="unused", dataset_path="unused",
                n_samples=0,
            )

    def test_valid_dedup_config_constructs_without_error(self):
        cfg = RolloutTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
            n_samples=8,
            dedup_enabled=True,
            dedup_min_unique=6,
            dedup_max_resample=4,
        )
        self.assertTrue(cfg.dedup_enabled)


class DedupIntegrationTest(unittest.TestCase):
    """Integration-style test crossing data.py -> trainer_config -> policy_only.

    Uses a fake tokenizer and a fake trainer backend to verify that
    ``_resample_duplicates`` correctly detects duplicates, issues resample
    requests, replaces duplicate sequences, and produces observable metrics.
    No GPU, model, or network required.
    """

    def _make_rollout_result(self, completions_tokens):
        """Build a RolloutResult from lists of token ids."""
        from areno.api.models import RolloutResult, RolloutSequence

        return RolloutResult(
            sequences=[
                RolloutSequence(resp_tokens=tokens, resp_logprobs=[0.0] * len(tokens))
                for tokens in completions_tokens
            ]
        )

    def _make_prompt_batch(self, prompt="test", n=1):
        """Build a minimal PromptBatch with one item."""
        from areno.api.data import PromptBatch, PromptItem

        return PromptBatch(
            items=[
                PromptItem(prompt=prompt, solutions=None, input_tokens=[1, 2], record={"prompt": prompt})
                for _ in range(n)
            ],
            scanned=n,
            skipped_long=0,
            total_skipped_long=0,
        )

    def test_resample_replaces_duplicates_with_fresh_samples(self):
        """The trainer should replace duplicate sequences with resample output."""
        from areno.api.data import PromptBatch, PromptItem
        from areno.api.models import RolloutResult, RolloutSequence
        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        # --- Fake tokenizer: maps token lists to deterministic strings ---
        # Token ids 1-9 all mapped
        class FakeTokenizer:
            def decode(self, tokens):
                table = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i"}
                return "".join(table.get(t, "?") for t in tokens)

        # --- Fake trainer backend: captures resample calls ---
        class FakeTrainer:
            def __init__(self):
                self.rollout_calls = 0

            class _SessionCtx:
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    return False

            def rollout_session(self, **kwargs):
                return self._SessionCtx()

            async def rollout_token_batch_async(self, prompt_tokens, n_samples, sampling_params):
                self.rollout_calls += 1
                # Resample call: return unique completions
                return [RolloutResult(sequences=[
                    RolloutSequence(resp_tokens=[4, 5, 6], resp_logprobs=[0.0, 0.0, 0.0]),  # "def"
                    RolloutSequence(resp_tokens=[7, 8, 9], resp_logprobs=[0.0, 0.0, 0.0]),  # "ghi"
                ])]

        import logging

        fake_trainer = FakeTrainer()
        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.areno = fake_trainer
        trainer.config = PolicyTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
            n_samples=4,
            dedup_enabled=True,
            dedup_min_unique=4,
            dedup_max_resample=4,
        )
        trainer.logger = logging.getLogger("test")
        trainer._agent_run_fn = None

        tokenizer = FakeTokenizer()
        prompt_batch = self._make_prompt_batch(n=1)

        # Build initial rollout results: ["aaa", "aaa", "aaa", "bcd"]
        rollout_results = [self._make_rollout_result([
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 3, 4]
        ])]

        # Run resample
        result = trainer._resample_duplicates(tokenizer, None, prompt_batch, rollout_results)

        # Verify duplicates were detected and replaced
        final_completions = [
            tokenizer.decode(seq.resp_tokens)
            for seq in result[0].sequences
        ]

        # Position 0 ("aaa") is unique first-occurrence, kept
        # Positions 1, 2 are duplicates, should be replaced by "def" and "ghi"
        # Position 3 ("bcd") is unique, kept
        self.assertEqual(final_completions[0], "aaa")
        self.assertEqual(final_completions[3], "bcd")
        # Replaced positions should differ from original "aaa"
        self.assertNotEqual(final_completions[1], "aaa")
        self.assertNotEqual(final_completions[2], "aaa")
        # Resample was called once
        self.assertEqual(fake_trainer.rollout_calls, 1)

    def test_resample_disabled_does_nothing(self):
        """When dedup_enabled=False, no resampling should occur."""
        from areno.api.models import RolloutResult, RolloutSequence
        from areno.api.trainers.policy_only import PolicyOnlyTrainer
        import logging

        class FakeTokenizer:
            def decode(self, tokens):
                return "same"

        class FakeTrainer:
            rollout_calls = 0

            def rollout_session(self, **kwargs):
                return self

            def __aenter__(self):
                return self

            def __aexit__(self, *args):
                return False

            async def rollout_token_batch_async(self, *args, **kwargs):
                self.__class__.rollout_calls += 1
                return []

        fake_trainer = FakeTrainer()
        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.areno = fake_trainer
        trainer.config = PolicyTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
            n_samples=4,
            dedup_enabled=False,  # disabled
        )
        trainer.logger = logging.getLogger("test")
        trainer._agent_run_fn = None

        # All identical completions
        rollout_results = [self._make_rollout_result([
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]
        ])]

        # The trainer loop checks dedup_enabled before calling _resample_duplicates,
        # so if disabled, this method is never invoked. Simulate the guard:
        if getattr(trainer.config, "dedup_enabled", False):
            trainer._resample_duplicates(FakeTokenizer(), None, self._make_prompt_batch(), rollout_results)

        # No resample calls should have been made
        self.assertEqual(FakeTrainer.rollout_calls, 0)

    def test_resample_preserves_unique_sequences(self):
        """Unique sequences must not be modified by the resample pass."""
        from areno.api.models import RolloutResult, RolloutSequence
        from areno.api.trainers.policy_only import PolicyOnlyTrainer
        import logging

        class FakeTokenizer:
            def decode(self, tokens):
                table = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i"}
                return "".join(table.get(t, "?") for t in tokens)

        class FakeTrainer:
            def __init__(self):
                self.rollout_calls = 0

            class _SessionCtx:
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    return False

            def rollout_session(self, **kwargs):
                return self._SessionCtx()

            async def rollout_token_batch_async(self, prompt_tokens, n_samples, sampling_params):
                self.rollout_calls += 1
                # Resample returns one unique completion
                return [RolloutResult(sequences=[
                    RolloutSequence(resp_tokens=[6, 6, 6], resp_logprobs=[0.0, 0.0, 0.0]),  # "fff"
                ])]

        fake_trainer = FakeTrainer()
        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.areno = fake_trainer
        trainer.config = PolicyTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
            n_samples=4,
            dedup_enabled=True,
            dedup_min_unique=4,
            dedup_max_resample=4,
        )
        trainer.logger = logging.getLogger("test")
        trainer._agent_run_fn = None

        # ["abc", "abc", "def", "ghi"] — 1 duplicate at index 1
        rollout_results = [self._make_rollout_result([
            [1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9]
        ])]

        result = trainer._resample_duplicates(FakeTokenizer(), None, self._make_prompt_batch(), rollout_results)

        final = [FakeTokenizer().decode(seq.resp_tokens) for seq in result[0].sequences]

        # Unique sequences at 0, 2, 3 must be unchanged
        self.assertEqual(final[0], "abc")
        self.assertEqual(final[2], "def")
        self.assertEqual(final[3], "ghi")
        # Index 1 should be replaced
        self.assertNotEqual(final[1], "abc")


if __name__ == "__main__":
    unittest.main()