"""CPU tests for reward distribution summarisation (issue #260).

Covers:
- ``compute_reward_statistics`` edge cases (constant, sparse, missing, non-finite, outliers, empty).
- ``compute_component_statistics`` with dynamically appearing components and
  missing-vs-zero distinction.
- ``normalize_reward_result`` for both float and dict return types.
- ``load_reward_samples`` JSONL reading and step filtering.
- CLI integration via ``CliRunner`` (table output, JSON output, step filter,
  empty dir, invalid threshold, non-existent dir).
- Formatting (table columns, JSON validity).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

from click.testing import CliRunner

from areno.api.metrics import load_reward_samples
from areno.api.reward_stats import (
    RewardSummaryReport,
    RewardStatistics,
    compute_component_statistics,
    compute_reward_statistics,
    format_reward_json,
    format_reward_table,
)
from areno.api.rewards import normalize_reward_result
from areno.cli.reward_summary import reward_summary_command


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


class RewardStatsTest(unittest.TestCase):
    """Core statistics edge cases."""

    def test_constant_rewards(self):
        """All-equal rewards: std=0, zero_fraction=0, outlier=0."""
        stats = compute_reward_statistics([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(stats.count, 4)
        self.assertAlmostEqual(stats.mean, 1.0)
        self.assertAlmostEqual(stats.std, 0.0)
        self.assertAlmostEqual(stats.zero_fraction, 0.0)
        self.assertAlmostEqual(stats.outlier_fraction, 0.0)

    def test_sparse_rewards(self):
        """Mostly-zero rewards: zero_fraction should be high."""
        values = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        stats = compute_reward_statistics(values)
        self.assertAlmostEqual(stats.zero_fraction, 4 / 6)
        self.assertAlmostEqual(stats.mean, 2 / 6)

    def test_missing_values(self):
        """None and NaN entries count as missing, not zero."""
        stats = compute_reward_statistics([0.0, 1.0, None, float("nan")])
        self.assertEqual(stats.count, 4)
        self.assertAlmostEqual(stats.missing_fraction, 0.5)
        # Zero fraction is relative to total, not just finite.
        self.assertAlmostEqual(stats.zero_fraction, 0.25)

    def test_non_finite_values(self):
        """inf and -inf are treated as missing (not usable for training)."""
        stats = compute_reward_statistics([1.0, float("inf"), float("-inf"), 2.0])
        self.assertEqual(stats.count, 4)
        self.assertAlmostEqual(stats.missing_fraction, 0.5)
        self.assertAlmostEqual(stats.mean, 1.5)  # only 1.0 and 2.0 are finite
        self.assertAlmostEqual(stats.min, 1.0)
        self.assertAlmostEqual(stats.max, 2.0)

    def test_outlier_detection(self):
        """Values beyond threshold * std from the mean are outliers."""
        # [0, 0, 0, 0, 100] — mean=20, std≈40, 100 is > 3 std away
        stats = compute_reward_statistics([0.0, 0.0, 0.0, 0.0, 100.0], outlier_threshold=3.0)
        self.assertGreater(stats.outlier_fraction, 0.0)
        # With a very high threshold, no outliers.
        stats_loose = compute_reward_statistics([0.0, 0.0, 0.0, 0.0, 100.0], outlier_threshold=100.0)
        self.assertAlmostEqual(stats_loose.outlier_fraction, 0.0)

    def test_empty_values(self):
        """An empty list should produce zeroed statistics without errors."""
        stats = compute_reward_statistics([])
        self.assertEqual(stats.count, 0)
        self.assertAlmostEqual(stats.mean, 0.0)

    def test_all_missing(self):
        """All-None input: missing_fraction=1, no crash on mean/std."""
        stats = compute_reward_statistics([None, None, float("nan")])
        self.assertAlmostEqual(stats.missing_fraction, 1.0)
        self.assertAlmostEqual(stats.mean, 0.0)

    def test_zero_std_no_outliers(self):
        """When std=0 (constant), outlier_fraction must be 0 (no div-by-zero)."""
        stats = compute_reward_statistics([5.0, 5.0, 5.0], outlier_threshold=1.0)
        self.assertAlmostEqual(stats.outlier_fraction, 0.0)


# ---------------------------------------------------------------------------
# Component statistics
# ---------------------------------------------------------------------------


class ComponentStatsTest(unittest.TestCase):
    """Named component aggregation with dynamically appearing keys."""

    def test_dynamically_appearing_components(self):
        """A component absent in some samples should be 'missing' for those."""
        samples = [
            {"reward": 1.5, "reward_components": {"correctness": 1.0, "format": 0.5}},
            {"reward": 1.0, "reward_components": {"correctness": 1.0, "format": 0.0}},
            {"reward": 0.0, "reward_components": {"correctness": 0.0}},
            {"reward": 0.5, "reward_components": {"correctness": 0.5}},
        ]
        report = compute_component_statistics(samples, outlier_threshold=3.0)
        self.assertEqual(report.sample_count, 4)
        self.assertIn("correctness", report.components)
        self.assertIn("format", report.components)
        # 'format' appears in samples 0,1 but not 2,3 → missing_fraction=0.5
        self.assertAlmostEqual(report.components["format"].missing_fraction, 0.5)

    def test_distinguishes_missing_from_zero(self):
        """A component value of 0.0 (zero) must not be counted as missing."""
        samples = [
            {"reward": 0.0, "reward_components": {"a": 0.0}},  # a=0 (zero, not missing)
            {"reward": 1.0, "reward_components": {"a": 1.0, "b": 1.0}},  # b present
            {"reward": 1.0, "reward_components": {"a": 1.0}},  # b missing
        ]
        report = compute_component_statistics(samples)
        # 'a' present in all 3, one of which is 0.0
        self.assertAlmostEqual(report.components["a"].missing_fraction, 0.0)
        self.assertAlmostEqual(report.components["a"].zero_fraction, 1 / 3)
        # 'b' present in 1 of 3
        self.assertAlmostEqual(report.components["b"].missing_fraction, 2 / 3)

    def test_no_components(self):
        """Samples without reward_components should only have a total row."""
        samples = [
            {"reward": 1.0, "reward_components": None},
            {"reward": 0.0, "reward_components": None},
        ]
        report = compute_component_statistics(samples)
        self.assertEqual(len(report.components), 0)
        self.assertEqual(report.total.count, 2)

    def test_empty_samples(self):
        """An empty sample list should produce an empty report."""
        report = compute_component_statistics([])
        self.assertEqual(report.sample_count, 0)
        self.assertEqual(report.total.count, 0)


# ---------------------------------------------------------------------------
# normalize_reward_result
# ---------------------------------------------------------------------------


class NormalizeRewardResultTest(unittest.TestCase):
    """Backward-compatible reward return-type normalisation."""

    def test_normalize_float_returns_total_and_none(self):
        total, components = normalize_reward_result(1.5)
        self.assertAlmostEqual(total, 1.5)
        self.assertIsNone(components)

    def test_normalize_int_returns_total_and_none(self):
        total, components = normalize_reward_result(0)
        self.assertAlmostEqual(total, 0.0)
        self.assertIsNone(components)

    def test_normalize_dict_returns_sum_and_components(self):
        total, components = normalize_reward_result({"a": 0.5, "b": 1.0})
        self.assertAlmostEqual(total, 1.5)
        self.assertIsNotNone(components)
        self.assertAlmostEqual(components["a"], 0.5)
        self.assertAlmostEqual(components["b"], 1.0)

    def test_normalize_dict_with_negative_values(self):
        total, components = normalize_reward_result({"a": -0.5, "b": 1.0})
        self.assertAlmostEqual(total, 0.5)
        self.assertAlmostEqual(components["a"], -0.5)


# ---------------------------------------------------------------------------
# load_reward_samples
# ---------------------------------------------------------------------------


class LoadRewardSamplesTest(unittest.TestCase):
    """JSONL file reading and step filtering."""

    def test_load_reward_samples_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reward_metrics.123.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"step": 0, "reward": 1.0, "reward_components": None}) + "\n")
                f.write(json.dumps({"step": 1, "reward": 0.5, "reward_components": {"a": 0.5}}) + "\n")
            samples = load_reward_samples(tmp)
            self.assertEqual(len(samples), 2)
            self.assertAlmostEqual(samples[0]["reward"], 1.0)
            self.assertAlmostEqual(samples[1]["reward"], 0.5)

    def test_load_reward_samples_step_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reward_metrics.123.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for s in range(5):
                    f.write(json.dumps({"step": s, "reward": float(s), "reward_components": None}) + "\n")
            samples = load_reward_samples(tmp, step=2)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["step"], 2)

    def test_load_reward_samples_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = load_reward_samples(tmp)
            self.assertEqual(len(samples), 0)

    def test_load_reward_samples_multiple_files(self):
        """Multiple reward_metrics.*.jsonl files should all be read."""
        with tempfile.TemporaryDirectory() as tmp:
            for pid in (111, 222):
                path = os.path.join(tmp, f"reward_metrics.{pid}.jsonl")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"step": 0, "reward": 1.0, "reward_components": None}) + "\n")
            samples = load_reward_samples(tmp)
            self.assertEqual(len(samples), 2)

    def test_load_reward_samples_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reward_metrics.123.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"step": 0, "reward": 1.0, "reward_components": None}) + "\n")
                f.write("\n")
                f.write(json.dumps({"step": 1, "reward": 0.5, "reward_components": None}) + "\n")
            samples = load_reward_samples(tmp)
            self.assertEqual(len(samples), 2)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def _write_jsonl(tmp: str, records: list[dict]) -> str:
    path = os.path.join(tmp, "reward_metrics.999.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


class RewardSummaryCliTest(unittest.TestCase):
    """CLI command integration via CliRunner."""

    def test_cli_table_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_jsonl(tmp, [
                {"step": 0, "epoch": 0, "prompt_idx": 0, "sample_idx": 0,
                 "reward": 1.0, "reward_components": {"a": 1.0}},
                {"step": 0, "epoch": 0, "prompt_idx": 0, "sample_idx": 1,
                 "reward": 0.0, "reward_components": {"a": 0.0}},
            ])
            result = CliRunner().invoke(reward_summary_command, ["--metrics-log-dir", tmp])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("total", result.output)
            self.assertIn("Mean", result.output)

    def test_cli_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_jsonl(tmp, [
                {"step": 0, "epoch": 0, "prompt_idx": 0, "sample_idx": 0,
                 "reward": 1.0, "reward_components": {"a": 1.0}},
            ])
            result = CliRunner().invoke(reward_summary_command, ["--metrics-log-dir", tmp, "--json"])
            self.assertEqual(result.exit_code, 0, result.output)
            parsed = json.loads(result.output)
            self.assertIn("total", parsed)
            self.assertIn("components", parsed)
            self.assertIn("sample_count", parsed)

    def test_cli_step_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_jsonl(tmp, [
                {"step": 0, "epoch": 0, "prompt_idx": 0, "sample_idx": 0,
                 "reward": float(i), "reward_components": None}
                for i in range(10)
            ])
            result = CliRunner().invoke(
                reward_summary_command, ["--metrics-log-dir", tmp, "--step", "5", "--json"],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            parsed = json.loads(result.output)
            self.assertEqual(parsed["sample_count"], 1)

    def test_cli_empty_dir_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = CliRunner().invoke(reward_summary_command, ["--metrics-log-dir", tmp])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No reward metrics", result.output)

    def test_cli_invalid_outlier_threshold(self):
        result = CliRunner().invoke(reward_summary_command, ["--outlier-threshold", "-1"])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_nonexistent_dir(self):
        result = CliRunner().invoke(reward_summary_command, ["--metrics-log-dir", "/nonexistent/path/xyz"])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_outlier_threshold_option(self):
        """Custom outlier threshold should be respected."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_jsonl(tmp, [
                {"step": 0, "epoch": 0, "prompt_idx": i, "sample_idx": 0,
                 "reward": float(v), "reward_components": None}
                for i, v in enumerate([0.0, 0.0, 0.0, 0.0, 100.0])
            ])
            # threshold=3 → outliers expected
            r1 = CliRunner().invoke(
                reward_summary_command, ["--metrics-log-dir", tmp, "--json", "--outlier-threshold", "3"],
            )
            self.assertEqual(r1.exit_code, 0)
            p1 = json.loads(r1.output)
            self.assertGreater(p1["total"]["outlier_fraction"], 0.0)
            # threshold=100 → no outliers
            r2 = CliRunner().invoke(
                reward_summary_command, ["--metrics-log-dir", tmp, "--json", "--outlier-threshold", "100"],
            )
            self.assertEqual(r2.exit_code, 0)
            p2 = json.loads(r2.output)
            self.assertAlmostEqual(p2["total"]["outlier_fraction"], 0.0)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class FormatTest(unittest.TestCase):
    """Table and JSON output structure validation."""

    def _make_report(self) -> RewardSummaryReport:
        total = RewardStatistics(
            count=10, mean=0.5, std=0.3, min=0.0, max=1.0,
            zero_fraction=0.2, missing_fraction=0.0, outlier_fraction=0.1,
        )
        comp = RewardStatistics(
            count=10, mean=0.3, std=0.2, min=0.0, max=0.5,
            zero_fraction=0.3, missing_fraction=0.1, outlier_fraction=0.0,
        )
        return RewardSummaryReport(total=total, components={"a": comp}, sample_count=10)

    def test_table_output_contains_all_columns(self):
        report = self._make_report()
        table = format_reward_table(report, use_color=False)
        for col in ("Mean", "Std", "Min", "Max", "Zero%", "Missing%", "Outlier%"):
            self.assertIn(col, table)
        self.assertIn("total", table)
        self.assertIn("a", table)

    def test_json_output_is_valid_json(self):
        report = self._make_report()
        output = format_reward_json(report)
        parsed = json.loads(output)
        self.assertIn("total", parsed)
        self.assertIn("components", parsed)
        self.assertIn("sample_count", parsed)
        self.assertEqual(parsed["sample_count"], 10)
        # Verify total fields
        for field in ("count", "mean", "std", "min", "max", "zero_fraction", "missing_fraction", "outlier_fraction"):
            self.assertIn(field, parsed["total"])
        # Verify component fields
        self.assertIn("a", parsed["components"])
        for field in ("count", "mean", "std", "min", "max", "zero_fraction", "missing_fraction", "outlier_fraction"):
            self.assertIn(field, parsed["components"]["a"])


if __name__ == "__main__":
    unittest.main()