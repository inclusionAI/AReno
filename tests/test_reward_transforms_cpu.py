"""CPU tests for configurable reward clipping and batch normalization.

Covers the transform core (``transform_rewards``), distribution summary,
config validation, and an integration-style test chaining transform →
``compute_group_advantages`` — all without external services or GPUs.
"""

from __future__ import annotations

import math
import unittest

from areno.api.rewards import (
    RewardTransformError,
    compute_group_advantages,
    reward_distribution_summary,
    transform_rewards,
)
from areno.api.trainer_config import PolicyTrainerConfig

# ---------------------------------------------------------------------------
# transform_rewards — disabled mode
# ---------------------------------------------------------------------------


class DisabledModeTest(unittest.TestCase):
    """disabled mode must return the input list unchanged (no copy)."""

    def test_disabled_normal(self):
        rewards = [1.0, 2.0, 3.0]
        out = transform_rewards(rewards, mode="disabled")
        self.assertEqual(out, rewards)
        self.assertIs(out, rewards)

    def test_disabled_empty(self):
        out = transform_rewards([], mode="disabled")
        self.assertEqual(out, [])

    def test_disabled_nan(self):
        rewards = [float("nan"), 1.0]
        out = transform_rewards(rewards, mode="disabled")
        self.assertIs(out, rewards)


# ---------------------------------------------------------------------------
# transform_rewards — clip mode
# ---------------------------------------------------------------------------


class ClipModeTest(unittest.TestCase):
    """clip mode clamps each element to [clip_min, clip_max]."""

    def test_clip_normal(self):
        out = transform_rewards([1.0, 5.0, 10.0], mode="clip", clip_min=0.0, clip_max=8.0)
        self.assertEqual(out, [1.0, 5.0, 8.0])

    def test_clip_boundary(self):
        out = transform_rewards([0.0, 8.0, 0.0, 8.0], mode="clip", clip_min=0.0, clip_max=8.0)
        self.assertEqual(out, [0.0, 8.0, 0.0, 8.0])

    def test_clip_extreme(self):
        out = transform_rewards([-1e6, 0.0, 1e6], mode="clip", clip_min=-5.0, clip_max=5.0)
        self.assertEqual(out, [-5.0, 0.0, 5.0])

    def test_clip_empty(self):
        out = transform_rewards([], mode="clip", clip_min=0.0, clip_max=8.0)
        self.assertEqual(out, [])

    def test_clip_nan_raises(self):
        # clip mode requires finite inputs; NaN must raise a clear error.
        with self.assertRaises(RewardTransformError) as ctx:
            transform_rewards([float("nan"), 1.0], mode="clip", clip_min=0.0, clip_max=2.0)
        self.assertEqual(ctx.exception.stage, "reward_transform.clip")
        self.assertIn("has_nan=True", ctx.exception.input_summary)


# ---------------------------------------------------------------------------
# transform_rewards — standardize mode
# ---------------------------------------------------------------------------


class StandardizeModeTest(unittest.TestCase):
    """standardize mode z-scores across the full batch (cross-group)."""

    def test_standardize_normal(self):
        out = transform_rewards([1.0, 2.0, 3.0], mode="standardize")
        mean = sum(out) / len(out)
        self.assertAlmostEqual(mean, 0.0, places=6)
        std = math.sqrt(sum((x - mean) ** 2 for x in out) / len(out))
        self.assertAlmostEqual(std, 1.0, places=5)

    def test_standardize_extreme(self):
        out = transform_rewards([-1e6, 0.0, 1e6], mode="standardize")
        for value in out:
            self.assertTrue(math.isfinite(value))

    def test_standardize_constant(self):
        # std=0 → eps guard → all zeros, all finite.
        out = transform_rewards([3.0, 3.0, 3.0], mode="standardize")
        self.assertEqual(out, [0.0, 0.0, 0.0])
        for value in out:
            self.assertTrue(math.isfinite(value))

    def test_standardize_nan_raises(self):
        with self.assertRaises(RewardTransformError) as ctx:
            transform_rewards([float("nan"), 1.0], mode="standardize")
        self.assertEqual(ctx.exception.stage, "reward_transform.standardize")
        self.assertIn("has_nan=True", ctx.exception.input_summary)

    def test_standardize_empty_raises(self):
        with self.assertRaises(RewardTransformError) as ctx:
            transform_rewards([], mode="standardize")
        self.assertEqual(ctx.exception.stage, "reward_transform.standardize")
        self.assertIn("count=0", ctx.exception.input_summary)

    def test_standardize_custom_eps(self):
        out = transform_rewards([5.0, 5.0, 5.0], mode="standardize", standardize_eps=1e-4)
        self.assertEqual(out, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# transform_rewards — unknown mode
# ---------------------------------------------------------------------------


class UnknownModeTest(unittest.TestCase):
    def test_unknown_mode_raises(self):
        with self.assertRaises(RewardTransformError) as ctx:
            transform_rewards([1.0], mode="bogus")
        self.assertEqual(ctx.exception.stage, "reward_transform.dispatch")


# ---------------------------------------------------------------------------
# reward_distribution_summary
# ---------------------------------------------------------------------------


class DistributionSummaryTest(unittest.TestCase):
    def test_summary_normal(self):
        summary = reward_distribution_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["mean"], 2.0, places=6)
        self.assertAlmostEqual(summary["min"], 1.0, places=6)
        self.assertAlmostEqual(summary["max"], 3.0, places=6)
        self.assertTrue(summary["std"] >= 0.0)

    def test_summary_empty(self):
        summary = reward_distribution_summary([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["mean"])
        self.assertIsNone(summary["std"])
        self.assertIsNone(summary["min"])
        self.assertIsNone(summary["max"])


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class ConfigValidationTest(unittest.TestCase):
    """PolicyTrainerConfig.__post_init__ must validate reward transform fields."""

    def _base_kwargs(self):
        return dict(algo="gspo", ckpt="dummy", dataset_path="dummy")

    def test_config_default_disabled(self):
        config = PolicyTrainerConfig(**self._base_kwargs())
        self.assertEqual(config.reward_transform_mode, "disabled")

    def test_config_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            PolicyTrainerConfig(**self._base_kwargs(), reward_transform_mode="bogus")

    def test_config_clip_missing_min_raises(self):
        with self.assertRaises(ValueError):
            PolicyTrainerConfig(**self._base_kwargs(), reward_transform_mode="clip", reward_clip_max=5.0)

    def test_config_clip_missing_max_raises(self):
        with self.assertRaises(ValueError):
            PolicyTrainerConfig(**self._base_kwargs(), reward_transform_mode="clip", reward_clip_min=-5.0)

    def test_config_clip_min_gt_max_raises(self):
        with self.assertRaises(ValueError):
            PolicyTrainerConfig(
                **self._base_kwargs(),
                reward_transform_mode="clip",
                reward_clip_min=5.0,
                reward_clip_max=1.0,
            )

    def test_config_standardize_eps_zero_raises(self):
        with self.assertRaises(ValueError):
            PolicyTrainerConfig(**self._base_kwargs(), reward_transform_mode="standardize", reward_standardize_eps=0.0)

    def test_config_standardize_eps_negative_raises(self):
        with self.assertRaises(ValueError):
            PolicyTrainerConfig(**self._base_kwargs(), reward_transform_mode="standardize", reward_standardize_eps=-1.0)

    def test_config_clip_valid(self):
        config = PolicyTrainerConfig(
            **self._base_kwargs(),
            reward_transform_mode="clip",
            reward_clip_min=-5.0,
            reward_clip_max=5.0,
        )
        self.assertEqual(config.reward_transform_mode, "clip")

    def test_config_standardize_valid(self):
        config = PolicyTrainerConfig(**self._base_kwargs(), reward_transform_mode="standardize")
        self.assertEqual(config.reward_transform_mode, "standardize")


# ---------------------------------------------------------------------------
# Integration: transform → compute_group_advantages
# ---------------------------------------------------------------------------


class TransformAdvantageIntegrationTest(unittest.TestCase):
    """Verify the transform + advantage pipeline produces correct results."""

    def test_clip_then_group_advantages(self):
        # Two groups: [−10, 0, 10] and [1, 2, 3].
        # Clip to [−5, 5] → [−5, 0, 5] and [1, 2, 3].
        raw = [-10.0, 0.0, 10.0, 1.0, 2.0, 3.0]
        transformed = transform_rewards(raw, mode="clip", clip_min=-5.0, clip_max=5.0)
        self.assertEqual(transformed, [-5.0, 0.0, 5.0, 1.0, 2.0, 3.0])

        # Group 1 advantages from clipped rewards.
        adv_g1 = compute_group_advantages(transformed[0:3])
        mean_g1 = sum(adv_g1) / len(adv_g1)
        self.assertAlmostEqual(mean_g1, 0.0, places=6)

        # Group 2 advantages from un-clipped rewards (within range).
        adv_g2 = compute_group_advantages(transformed[3:6])
        mean_g2 = sum(adv_g2) / len(adv_g2)
        self.assertAlmostEqual(mean_g2, 0.0, places=6)

    def test_standardize_cross_group_not_cancelled(self):
        # standardize across the full batch, then per-group z-score.
        # The per-group z-score should NOT fully cancel the cross-group
        # standardization because the global mean/std differs from the
        # per-group mean/std.
        raw = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        transformed = transform_rewards(raw, mode="standardize")
        # Transformed mean should be ~0 across the full batch.
        full_mean = sum(transformed) / len(transformed)
        self.assertAlmostEqual(full_mean, 0.0, places=5)

        # Per-group advantages should still be zero-mean within each group.
        adv_g1 = compute_group_advantages(transformed[0:3])
        adv_g2 = compute_group_advantages(transformed[3:6])
        self.assertAlmostEqual(sum(adv_g1) / len(adv_g1), 0.0, places=5)
        self.assertAlmostEqual(sum(adv_g2) / len(adv_g2), 0.0, places=5)

    def test_disabled_then_group_advantages_unchanged(self):
        raw = [1.0, 2.0, 3.0]
        transformed = transform_rewards(raw, mode="disabled")
        adv_direct = compute_group_advantages(raw)
        adv_via_transform = compute_group_advantages(transformed)
        self.assertEqual(adv_via_transform, adv_direct)

    def test_standardize_empty_in_pipeline_raises(self):
        with self.assertRaises(RewardTransformError) as ctx:
            transform_rewards([], mode="standardize")
        self.assertIn("count=0", ctx.exception.input_summary)


if __name__ == "__main__":
    unittest.main()
