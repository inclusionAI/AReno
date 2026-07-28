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


if __name__ == "__main__":
    unittest.main()