from __future__ import annotations

import json
import tempfile
import unittest

from click.testing import CliRunner

from areno.cli import metrics as metrics_module
from areno.cli.main import main


def _write_test_events(log_dir: str, *, tags: dict[str, list[tuple[int, float]]]) -> None:
    """Write TensorBoard event files with the given scalar tags and values."""

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=log_dir)
    for tag, points in tags.items():
        for step, value in points:
            writer.add_scalar(tag, value, step)
    writer.flush()
    writer.close()


class CliMetricsTest(unittest.TestCase):
    def test_top_level_cli_lists_metrics(self):
        result = CliRunner().invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("metrics", result.output)

    def test_metrics_help_lists_options(self):
        result = CliRunner().invoke(main, ["metrics", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("--log-dir", result.output)
        self.assertIn("--name", result.output)
        self.assertIn("--json", result.output)
        self.assertIn("--limit", result.output)

    def test_list_tags_prints_available_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_events(tmp, tags={"train/loss": [(0, 1.0), (1, 0.5)], "rollout/reward": [(0, 0.1)]})
            result = CliRunner().invoke(main, ["metrics", "--log-dir", tmp])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("train/loss", result.output)
        self.assertIn("rollout/reward", result.output)

    def test_list_tags_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_events(tmp, tags={"train/loss": [(0, 1.0)]})
            result = CliRunner().invoke(main, ["metrics", "--log-dir", tmp, "--json"])

        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertIn("tags", parsed)
        self.assertIn("train/loss", parsed["tags"])

    def test_query_metric_prints_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_events(tmp, tags={"train/loss": [(0, 1.0), (1, 0.5), (2, 0.25)]})
            result = CliRunner().invoke(main, ["metrics", "--log-dir", tmp, "--name", "train/loss"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Metric: train/loss", result.output)
        self.assertIn("Min:", result.output)
        self.assertIn("Max:", result.output)
        self.assertIn("Last:", result.output)
        self.assertIn("step", result.output)

    def test_query_metric_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_events(tmp, tags={"train/loss": [(0, 1.0), (1, 0.5), (2, 0.25)]})
            result = CliRunner().invoke(main, ["metrics", "--log-dir", tmp, "--name", "train/loss", "--json"])

        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["name"], "train/loss")
        self.assertEqual(parsed["count"], 3)
        self.assertEqual(parsed["min_value"], 0.25)
        self.assertEqual(parsed["max_value"], 1.0)
        self.assertEqual(parsed["last_value"], 0.25)
        self.assertIsNotNone(parsed["first_step"])
        self.assertIsNotNone(parsed["last_step"])
        self.assertIn("sparkline", parsed)

    def test_metric_not_found_lists_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_events(tmp, tags={"train/loss": [(0, 1.0)]})
            result = CliRunner().invoke(main, ["metrics", "--log-dir", tmp, "--name", "nonexistent"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not found", result.output)
        self.assertIn("train/loss", result.output)

    def test_nonexistent_log_dir_errors(self):
        result = CliRunner().invoke(main, ["metrics", "--log-dir", "/definitely/nonexistent/path"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not exist", result.output)

    def test_empty_directory_prints_friendly_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = CliRunner().invoke(main, ["metrics", "--log-dir", tmp])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("No TensorBoard scalar tags", result.output)

    def test_nan_values_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_events(
                tmp,
                tags={
                    "train/loss": [
                        (0, float("nan")),
                        (1, 0.5),
                        (2, float("inf")),
                        (3, 0.25),
                    ]
                },
            )
            result = CliRunner().invoke(main, ["metrics", "--log-dir", tmp, "--name", "train/loss", "--json"])

        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        # Only the two finite values should appear.
        self.assertEqual(parsed["count"], 2)
        self.assertEqual(parsed["min_value"], 0.25)
        self.assertEqual(parsed["max_value"], 0.5)

    def test_limit_truncates_points(self):
        points = [(i, float(i)) for i in range(100)]
        with tempfile.TemporaryDirectory() as tmp:
            _write_test_events(tmp, tags={"train/loss": points})
            result = CliRunner().invoke(
                main, ["metrics", "--log-dir", tmp, "--name", "train/loss", "--limit", "10", "--json"]
            )

        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["count"], 10)
        self.assertEqual(parsed["last_step"], 99)

    def test_sparkline_renders_for_constant_values(self):
        self.assertEqual(metrics_module._sparkline([5.0, 5.0, 5.0]), "___")

    def test_sparkline_renders_for_varied_values(self):
        result = metrics_module._sparkline([0.0, 1.0])
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0], result[1])

    def test_fmt_handles_special_values(self):
        self.assertEqual(metrics_module._fmt(None), "n/a")
        self.assertEqual(metrics_module._fmt(float("nan")), "NaN")
        self.assertEqual(metrics_module._fmt(float("inf")), "Inf")
        self.assertEqual(metrics_module._fmt(float("-inf")), "-Inf")

    def test_fmt_preserves_precision(self):
        self.assertIn("e", metrics_module._fmt(1e-10))
        self.assertIn("e", metrics_module._fmt(10000.0))
        self.assertEqual(metrics_module._fmt(0.5), "0.5")


if __name__ == "__main__":
    unittest.main()
