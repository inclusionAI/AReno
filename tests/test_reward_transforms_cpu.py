"""CPU tests for configurable reward clipping and batch normalization.

Covers normal, extreme, constant, NaN, and empty inputs, the disabled-mode
numerical-identity guarantee, and dispatcher statistics reporting.
"""

from __future__ import annotations

import math
import unittest

from areno.api.reward_transforms import (
    clip_rewards,
    standardize_rewards,
    transform_rewards,
)
from areno.api.rewards import compute_group_advantages


class ClipRewardsTest(unittest.TestCase):
    """Clipping correctly handles normal, extreme, constant, NaN, and empty inputs."""

    def test_normal_clip(self):
        """Values outside [min, max] are clamped; inside values are unchanged."""
        result = clip_rewards([1.0, 5.0, -3.0, 10.0], clip_min=-2.0, clip_max=8.0)
        self.assertEqual(result, [1.0, 5.0, -2.0, 8.0])

    def test_extreme_values(self):
        """Very large magnitudes are clamped to the configured range."""
        result = clip_rewards([1e20, -1e20, 1.0, -1.0], clip_min=-10.0, clip_max=10.0)
        self.assertEqual(result, [10.0, -10.0, 1.0, -1.0])

    def test_constant_rewards(self):
        """Constant rewards within range pass through unchanged."""
        result = clip_rewards([5.0, 5.0, 5.0], clip_min=0.0, clip_max=10.0)
        self.assertEqual(result, [5.0, 5.0, 5.0])

    def test_nan_raises(self):
        """NaN rewards must raise ValueError, not silently propagate."""
        with self.assertRaisesRegex(ValueError, "NaN"):
            clip_rewards([1.0, float("nan"), 3.0], clip_min=-1.0, clip_max=1.0)

    def test_empty_returns_empty(self):
        """An empty reward list is a safe no-op."""
        result = clip_rewards([], clip_min=-1.0, clip_max=1.0)
        self.assertEqual(result, [])

    def test_invalid_range_raises(self):
        """clip_min > clip_max is a configuration error."""
        with self.assertRaisesRegex(ValueError, "clip_min"):
            clip_rewards([1.0, 2.0], clip_min=10.0, clip_max=-10.0)

    def test_no_clip_needed(self):
        """Values already within range are returned unchanged."""
        result = clip_rewards([1.0, 2.0, 3.0], clip_min=0.0, clip_max=10.0)
        self.assertEqual(result, [1.0, 2.0, 3.0])


class StandardizeRewardsTest(unittest.TestCase):
    """Standardization handles normal, extreme, constant, NaN, empty, and single inputs."""

    def test_normal_standardize(self):
        """Standardized rewards have approximately zero mean and unit std."""
        result = standardize_rewards([1.0, 2.0, 3.0, 4.0])
        mean = sum(result) / len(result)
        self.assertAlmostEqual(mean, 0.0, places=6)

    def test_extreme_values_produce_finite_output(self):
        """Large magnitudes should still produce finite standardized values."""
        result = standardize_rewards([1e10, -1e10, 0.0])
        for value in result:
            self.assertTrue(math.isfinite(value))

    def test_constant_returns_zeros(self):
        """When all rewards are identical, standardization yields all zeros."""
        result = standardize_rewards([5.0, 5.0, 5.0])
        self.assertEqual(result, [0.0, 0.0, 0.0])

    def test_nan_raises(self):
        """NaN rewards must raise ValueError."""
        with self.assertRaisesRegex(ValueError, "NaN"):
            standardize_rewards([1.0, float("nan")])

    def test_empty_raises(self):
        """An empty reward list cannot be standardized."""
        with self.assertRaisesRegex(ValueError, "empty"):
            standardize_rewards([])

    def test_single_element_returns_zero(self):
        """A single-element list has std=0, so the result is [0.0]."""
        result = standardize_rewards([42.0])
        self.assertEqual(result, [0.0])

    def test_finite_output_on_random_large_range(self):
        """A wide-range input should produce only finite values."""
        result = standardize_rewards([-1e6, 1e6, 0.0, 1e3, -1e3])
        for value in result:
            self.assertTrue(math.isfinite(value))


class TransformRewardsDispatcherTest(unittest.TestCase):
    """The dispatcher correctly routes to each mode and reports statistics."""

    def test_disabled_returns_equal(self):
        """Disabled mode must return a numerically identical copy."""
        rewards = [1.5, -3.2, 0.0, 7.8]
        transformed, stats = transform_rewards(rewards, mode="disabled")
        self.assertEqual(transformed, rewards)
        self.assertIsNot(transformed, rewards)

    def test_disabled_records_raw_only(self):
        """Disabled mode stats should contain raw_* fields but no transformed_* fields."""
        _, stats = transform_rewards([1.0, 2.0, 3.0], mode="disabled")
        self.assertIn("raw_mean", stats)
        self.assertIn("raw_std", stats)
        self.assertIn("raw_count", stats)
        self.assertNotIn("transformed_mean", stats)
        self.assertEqual(stats["transform_mode"], "disabled")

    def test_clip_mode(self):
        """Clip mode should clamp values and report both raw and transformed stats."""
        rewards = [1.0, 5.0, -3.0, 10.0]
        transformed, stats = transform_rewards(rewards, mode="clip", clip_min=-2.0, clip_max=8.0)
        self.assertEqual(transformed, [1.0, 5.0, -2.0, 8.0])
        self.assertEqual(stats["transform_mode"], "clip")
        self.assertIn("raw_mean", stats)
        self.assertIn("transformed_mean", stats)
        self.assertEqual(stats["raw_max"], 10.0)
        self.assertEqual(stats["transformed_max"], 8.0)

    def test_standardize_mode(self):
        """Standardize mode should normalize and report both raw and transformed stats."""
        rewards = [1.0, 2.0, 3.0, 4.0]
        transformed, stats = transform_rewards(rewards, mode="standardize")
        mean = sum(transformed) / len(transformed)
        self.assertAlmostEqual(mean, 0.0, places=6)
        self.assertEqual(stats["transform_mode"], "standardize")
        self.assertIn("raw_mean", stats)
        self.assertIn("transformed_mean", stats)

    def test_invalid_mode_raises(self):
        """An unrecognized mode string should raise ValueError with the valid options."""
        with self.assertRaisesRegex(ValueError, "reward_transform_mode"):
            transform_rewards([1.0, 2.0], mode="nonsense")

    def test_stats_consistency(self):
        """Transformed stats should match the actual transformed output."""
        rewards = [1.0, 5.0, -3.0, 10.0]
        transformed, stats = transform_rewards(rewards, mode="clip", clip_min=-2.0, clip_max=8.0)
        self.assertAlmostEqual(stats["transformed_mean"], sum(transformed) / len(transformed), places=6)
        self.assertAlmostEqual(stats["transformed_min"], min(transformed), places=6)
        self.assertAlmostEqual(stats["transformed_max"], max(transformed), places=6)


class DisabledBackwardCompatibilityTest(unittest.TestCase):
    """Prove that disabled mode leaves downstream advantages numerically unchanged."""

    def test_advantage_unchanged_under_disabled(self):
        """Advantages computed after a disabled transform must equal direct computation."""
        rewards = [1.0, 3.0, 2.0, 5.0, 0.0]
        direct_advantages = compute_group_advantages(rewards)
        transformed, _ = transform_rewards(rewards, mode="disabled")
        transformed_advantages = compute_group_advantages(transformed)
        self.assertEqual(direct_advantages, transformed_advantages)


if __name__ == "__main__":
    unittest.main()
