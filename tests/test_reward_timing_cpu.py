"""CPU tests for the slow-reward-hook timing instrumentation (issue #242).

These tests cover the core logic, disabled/default behavior, outlier
flagging, timeout enforcement, boundary values, malformed input, and
deterministic output -- all without a GPU or external services.
"""

from __future__ import annotations

import math
import time
import unittest
from unittest.mock import MagicMock

from areno.api.reward_timing import (
    TIMEOUT_REWARD,
    RewardSampleTiming,
    RewardTimingConfig,
    RewardTimingReport,
    TimedRewardFn,
)
from areno.api.rewards import RewardRecord


def _record(prompt_index: int = 0, sample_index: int = 0) -> RewardRecord:
    """Build a minimal RewardRecord with metadata for sample identification."""

    return RewardRecord(
        prompt="test",
        completion="test",
        metadata={"prompt_index": prompt_index, "sample_index": sample_index},
    )


class DisabledTimingTest(unittest.TestCase):
    """When timing is disabled (the default), the wrapper is a transparent passthrough."""

    def test_disabled_returns_raw_result(self):
        """The wrapper should return the exact value from the underlying function."""

        def reward_fn(record):
            return 0.75

        timed = TimedRewardFn(reward_fn, RewardTimingConfig(enabled=False))
        self.assertAlmostEqual(timed(_record()), 0.75)

    def test_disabled_finalize_returns_none(self):
        """finalize_batch should return None when timing is disabled."""

        timed = TimedRewardFn(lambda r: 1.0, RewardTimingConfig(enabled=False))
        timed(_record())
        self.assertIsNone(timed.finalize_batch(0))

    def test_default_config_is_disabled(self):
        """A bare RewardTimingConfig should default to disabled."""

        config = RewardTimingConfig()
        self.assertFalse(config.enabled)


class EnabledTimingTest(unittest.TestCase):
    """Core timing, outlier flagging, and report generation."""

    def test_enabled_records_per_sample_timing(self):
        """Each call should produce one sample timing entry."""

        timed = TimedRewardFn(lambda r: 1.0, RewardTimingConfig(enabled=True))
        timed(_record(0, 0))
        timed(_record(1, 2))
        report = timed.finalize_batch(5)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.sample_timings), 2)
        self.assertEqual(report.step, 5)
        self.assertEqual(report.sample_timings[0].sample_id, "p0_s0")
        self.assertEqual(report.sample_timings[1].sample_id, "p1_s2")

    def test_slow_threshold_flags_outliers(self):
        """Samples above slow_threshold_s should appear in outliers."""

        def slow_fn(record):
            time.sleep(0.05)
            return 1.0

        config = RewardTimingConfig(enabled=True, slow_threshold_s=0.01)
        timed = TimedRewardFn(slow_fn, config)
        timed(_record(0, 0))
        report = timed.finalize_batch(0)
        self.assertEqual(len(report.outliers), 1)
        self.assertEqual(report.outliers[0].sample_id, "p0_s0")
        self.assertGreater(report.outliers[0].elapsed_s, 0.01)

    def test_fast_samples_not_flagged(self):
        """Samples below slow_threshold_s should not appear in outliers."""

        config = RewardTimingConfig(enabled=True, slow_threshold_s=10.0)
        timed = TimedRewardFn(lambda r: 1.0, config)
        timed(_record(0, 0))
        report = timed.finalize_batch(0)
        self.assertEqual(len(report.outliers), 0)

    def test_report_summary_statistics(self):
        """Report should carry total, mean, max, and p95 elapsed times."""

        def fn(record):
            # Deterministic-ish: fast call
            return 1.0

        config = RewardTimingConfig(enabled=True)
        timed = TimedRewardFn(fn, config)
        for i in range(5):
            timed(_record(i, 0))
        report = timed.finalize_batch(0)
        self.assertIsNotNone(report)
        self.assertEqual(report.hook_name, "reward_fn")
        self.assertGreater(report.total_elapsed_s, 0.0)
        self.assertGreater(report.mean_elapsed_s, 0.0)
        self.assertGreaterEqual(report.max_elapsed_s, report.mean_elapsed_s)
        self.assertGreaterEqual(report.p95_elapsed_s, 0.0)

    def test_finalize_clears_pending(self):
        """After finalize, a second call should produce a fresh report."""

        timed = TimedRewardFn(lambda r: 1.0, RewardTimingConfig(enabled=True))
        timed(_record())
        report1 = timed.finalize_batch(0)
        self.assertIsNotNone(report1)
        self.assertEqual(len(report1.sample_timings), 1)
        report2 = timed.finalize_batch(1)
        self.assertIsNone(report2)

    def test_no_samples_returns_none(self):
        """finalize_batch with no pending samples returns None."""

        timed = TimedRewardFn(lambda r: 1.0, RewardTimingConfig(enabled=True))
        self.assertIsNone(timed.finalize_batch(0))


class ReportFormatTest(unittest.TestCase):
    """Human-readable and structured output."""

    def test_to_dict_has_expected_fields(self):
        """to_dict should produce JSON-serializable fields without prompts."""

        timing = RewardSampleTiming(hook_name="reward_fn", sample_id="p0_s0", elapsed_s=0.123)
        report = RewardTimingReport(
            hook_name="reward_fn",
            step=3,
            sample_timings=[timing],
            total_elapsed_s=0.123,
            mean_elapsed_s=0.123,
            max_elapsed_s=0.123,
            p95_elapsed_s=0.123,
            outliers=[timing],
        )
        d = report.to_dict()
        self.assertEqual(d["hook_name"], "reward_fn")
        self.assertEqual(d["step"], 3)
        self.assertEqual(d["num_samples"], 1)
        self.assertEqual(d["outliers"][0]["sample_id"], "p0_s0")
        # No prompt or completion text should leak.
        for key in d:
            self.assertNotIn("prompt", key.lower())
            self.assertNotIn("completion", key.lower())

    def test_format_human_includes_outliers(self):
        """format_human should list slow sample IDs."""

        timing = RewardSampleTiming(hook_name="reward_fn", sample_id="p0_s0", elapsed_s=0.5)
        report = RewardTimingReport(
            hook_name="reward_fn",
            step=1,
            sample_timings=[timing],
            outliers=[timing],
        )
        text = report.format_human()
        self.assertIn("slow_samples=[p0_s0]", text)
        self.assertIn("hook=reward_fn", text)

    def test_format_human_includes_timeouts(self):
        """format_human should list timeout sample IDs."""

        timing = RewardSampleTiming(hook_name="reward_fn", sample_id="p1_s3", elapsed_s=2.0, timed_out=True)
        report = RewardTimingReport(
            hook_name="reward_fn",
            step=1,
            sample_timings=[timing],
            timeouts=[timing],
        )
        text = report.format_human()
        self.assertIn("timeouts=[p1_s3]", text)


class ConfigValidationTest(unittest.TestCase):
    """Configuration validation and error messages."""

    def test_negative_slow_threshold_raises(self):
        """A negative slow_threshold_s should raise ValueError."""

        with self.assertRaisesRegex(ValueError, "slow_threshold_s must be positive"):
            RewardTimingConfig(slow_threshold_s=-1.0).validate()

    def test_zero_slow_threshold_raises(self):
        """A zero slow_threshold_s should raise ValueError."""

        with self.assertRaisesRegex(ValueError, "slow_threshold_s must be positive"):
            RewardTimingConfig(slow_threshold_s=0.0).validate()

    def test_negative_timeout_raises(self):
        """A negative timeout_s should raise ValueError."""

        with self.assertRaisesRegex(ValueError, "timeout_s must be positive"):
            RewardTimingConfig(timeout_s=-0.5).validate()

    def test_timeout_less_than_threshold_raises(self):
        """timeout_s < slow_threshold_s should raise ValueError."""

        with self.assertRaisesRegex(ValueError, "timeout_s must be >= slow_threshold_s"):
            RewardTimingConfig(slow_threshold_s=1.0, timeout_s=0.5).validate()

    def test_none_threshold_and_timeout_are_valid(self):
        """None values for threshold and timeout should be valid."""

        RewardTimingConfig(slow_threshold_s=None, timeout_s=None).validate()
        RewardTimingConfig(slow_threshold_s=1.0, timeout_s=None).validate()
        RewardTimingConfig(slow_threshold_s=None, timeout_s=1.0).validate()

    def test_constructor_validates(self):
        """TimedRewardFn constructor should validate the config."""

        with self.assertRaises(ValueError):
            TimedRewardFn(lambda r: 1.0, RewardTimingConfig(slow_threshold_s=-1.0))


class TimeoutTest(unittest.TestCase):
    """Per-sample timeout enforcement (POSIX only)."""

    def test_timeout_returns_nan(self):
        """A timed-out sample should return NaN."""

        def slow_fn(record):
            time.sleep(0.3)
            return 1.0

        config = RewardTimingConfig(enabled=True, timeout_s=0.05)
        timed = TimedRewardFn(slow_fn, config)
        result = timed(_record(0, 0))
        self.assertTrue(math.isnan(result))

    def test_timeout_recorded_in_report(self):
        """Timed-out samples should appear in the report's timeouts list."""

        def slow_fn(record):
            time.sleep(0.2)
            return 1.0

        config = RewardTimingConfig(enabled=True, timeout_s=0.05)
        timed = TimedRewardFn(slow_fn, config)
        timed(_record(0, 0))
        report = timed.finalize_batch(0)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.timeouts), 1)
        self.assertTrue(report.timeouts[0].timed_out)

    def test_no_timeout_when_disabled(self):
        """Without timeout_s, even slow functions should complete normally."""

        def slow_fn(record):
            time.sleep(0.05)
            return 0.42

        config = RewardTimingConfig(enabled=True)
        timed = TimedRewardFn(slow_fn, config)
        result = timed(_record())
        self.assertAlmostEqual(result, 0.42)
        report = timed.finalize_batch(0)
        self.assertEqual(len(report.timeouts), 0)


class PercentileTest(unittest.TestCase):
    """The _percentile helper."""

    def test_single_value(self):
        """A single value should return itself for any percentile."""

        from areno.api.reward_timing import _percentile

        self.assertEqual(_percentile([5.0], 95), 5.0)

    def test_multiple_values(self):
        """The 95th percentile of [1,2,3,4,5] should be near 5."""

        from areno.api.reward_timing import _percentile

        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95)
        self.assertGreaterEqual(result, 4.0)
        self.assertLessEqual(result, 5.0)

    def test_empty_list(self):
        """An empty list should return 0.0."""

        from areno.api.reward_timing import _percentile

        self.assertEqual(_percentile([], 95), 0.0)


class MissingMetadataTest(unittest.TestCase):
    """Samples without prompt_index/sample_index metadata."""

    def test_missing_metadata_produces_default_id(self):
        """When metadata lacks indices, sample_id should use -1."""

        record = RewardRecord(prompt="test", completion="test")
        timed = TimedRewardFn(lambda r: 1.0, RewardTimingConfig(enabled=True))
        timed(record)
        report = timed.finalize_batch(0)
        self.assertEqual(report.sample_timings[0].sample_id, "p-1_s-1")


if __name__ == "__main__":
    unittest.main()
