"""CPU tests for the pure ``areno.api.metric_reader`` helpers (issue #254)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from areno.api import metric_reader as mr


def _write_fixture(metrics_dir: Path) -> int:
    """Write real tfevents covering normal / NaN / single-point series.

    Returns the writer process pid so pid-filter tests have a known match.
    Values use powers-of-two fractions to survive TensorBoard's float32
    round-trip, so assertions can compare for exact equality.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(metrics_dir))
    try:
        for step in range(5):
            writer.add_scalar("rollout/rewards_mean", 0.5 + 0.125 * step, step)
        writer.add_scalar("rollout/loss_with_nan", float("nan"), 0)
        writer.add_scalar("rollout/loss_with_nan", 1.5, 1)
        writer.add_scalar("rollout/single_point", 9.0, 7)
    finally:
        writer.close()
    return os.getpid()


class ReadScalarPointsTest(unittest.TestCase):
    def test_reads_points_skipping_nan_and_preserving_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp))
            points = mr.read_scalar_points(tmp)

            names_steps = [(p["name"], p["step"], p["value"]) for p in points]
            self.assertEqual(
                names_steps,
                [
                    ("rollout/rewards_mean", 0, 0.5),
                    ("rollout/rewards_mean", 1, 0.625),
                    ("rollout/rewards_mean", 2, 0.75),
                    ("rollout/rewards_mean", 3, 0.875),
                    ("rollout/rewards_mean", 4, 1.0),
                    ("rollout/loss_with_nan", 1, 1.5),
                    ("rollout/single_point", 7, 9.0),
                ],
            )
            self.assertTrue(all("time" in p and p["time"] for p in points))

    def test_empty_directory_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mr.read_scalar_points(tmp), [])
            self.assertEqual(mr.tensorboard_event_sources(Path(tmp), None), [Path(tmp)])

    def test_pid_filter_matches_filename_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = _write_fixture(Path(tmp))
            self.assertEqual(len(mr.read_scalar_points(tmp, pid=pid)), 7)
            self.assertEqual(mr.read_scalar_points(tmp, pid=99999999), [])

    def test_truncation_keeps_last_500(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(path))
            try:
                for step in range(600):
                    writer.add_scalar("rollout/big", float(step), step)
            finally:
                writer.close()
            points = mr.read_scalar_points(tmp, pid=None)
            self.assertEqual(len(points), 500)
            self.assertEqual(points[0]["step"], 100)
            self.assertEqual(points[-1]["step"], 599)


class SummarizeMetricTest(unittest.TestCase):
    @staticmethod
    def _points(name: str, values: list[float], *, start_step: int = 0) -> list[dict]:
        return [
            {"name": name, "value": value, "step": start_step + i, "time": "t"}
            for i, value in enumerate(values)
        ]

    def test_normal_series_streaming_min_max_and_trend(self):
        points = self._points("m", [1.0, 3.0, 2.0, 6.0, 4.0])
        summary = mr.summarize_metric(points, "m", recent_n=3)

        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["last"], 4.0)
        self.assertEqual(summary["last_step"], 4)
        self.assertEqual(summary["min_step"], 0)
        self.assertEqual(summary["max_step"], 4)
        self.assertEqual(summary["mean"], 3.2)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 6.0)
        self.assertEqual(summary["recent"], [2.0, 6.0, 4.0])
        self.assertEqual(summary["recent_steps"], [2, 3, 4])
        # trend shares the recent window, so --limit bounds the sparkline length
        self.assertEqual(len(summary["trend"]), 3)
        self.assertEqual(summary["trend"], [0.0, 1.0, 0.5])

    def test_trend_bounded_by_recent_n(self):
        points = self._points("m", [float(i) for i in range(10)])
        summary = mr.summarize_metric(points, "m", recent_n=5)

        self.assertEqual(len(summary["recent"]), 5)
        self.assertEqual(len(summary["trend"]), 5)
        self.assertEqual(summary["trend"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(summary["recent_steps"], [5, 6, 7, 8, 9])
        self.assertEqual(summary["max_step"], 9)
        self.assertEqual(summary["mean"], 4.5)

    def test_flat_series_trend_is_half(self):
        points = self._points("m", [2.0, 2.0, 2.0])
        summary = mr.summarize_metric(points, "m")
        self.assertEqual(summary["trend"], [0.5, 0.5, 0.5])
        self.assertEqual(summary["min"], 2.0)
        self.assertEqual(summary["max"], 2.0)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["recent_steps"], [0, 1, 2])

    def test_single_point_series(self):
        points = self._points("m", [9.0])
        summary = mr.summarize_metric(points, "m")
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["last"], 9.0)
        self.assertEqual(summary["last_step"], 0)
        self.assertEqual(summary["min_step"], 0)
        self.assertEqual(summary["max_step"], 0)
        self.assertEqual(summary["mean"], 9.0)
        self.assertEqual(summary["min"], 9.0)
        self.assertEqual(summary["max"], 9.0)
        self.assertEqual(summary["recent"], [9.0])
        self.assertEqual(summary["recent_steps"], [0])
        self.assertEqual(summary["trend"], [0.5])

    def test_unknown_name_yields_empty_summary(self):
        points = self._points("m", [1.0, 2.0])
        summary = mr.summarize_metric(points, "does/not/exist")
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["last"])
        self.assertIsNone(summary["last_step"])
        self.assertIsNone(summary["min_step"])
        self.assertIsNone(summary["max_step"])
        self.assertIsNone(summary["mean"])
        self.assertEqual(summary["recent"], [])
        self.assertEqual(summary["recent_steps"], [])
        self.assertEqual(summary["trend"], [])

    def test_nan_values_filtered_by_number_like(self):
        points = [
            {"name": "m", "value": 1.0, "step": 0, "time": "t"},
            {"name": "m", "value": float("nan"), "step": 1, "time": "t"},
            {"name": "m", "value": 3.0, "step": 2, "time": "t"},
        ]
        summary = mr.summarize_metric(points, "m")
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["min_step"], 0)
        self.assertEqual(summary["max_step"], 2)


class ListAndRenderTest(unittest.TestCase):
    def test_list_available_tags_sorted(self):
        points = [
            {"name": "c", "value": 1.0, "step": 0, "time": "t"},
            {"name": "a", "value": 1.0, "step": 0, "time": "t"},
            {"name": "b", "value": 1.0, "step": 0, "time": "t"},
            {"name": "", "value": 1.0, "step": 0, "time": "t"},
        ]
        self.assertEqual(mr.list_available_tags(points), ["a", "b", "c"])

    def test_render_table_contains_sparkline_and_fields(self):
        points = [
            {"name": "m", "value": 1.0, "step": 0, "time": "t"},
            {"name": "m", "value": 5.0, "step": 1, "time": "t"},
            {"name": "m", "value": 3.0, "step": 2, "time": "t"},
        ]
        summary = mr.summarize_metric(points, "m")
        rendered = mr.render_table(summary)

        self.assertIn("metric   m", rendered)
        self.assertIn("count    3", rendered)
        self.assertIn("steps    0 -> 2", rendered)
        self.assertIn("last     3", rendered)
        self.assertIn("(step 2)", rendered)
        self.assertIn("mean", rendered)
        self.assertIn("step 0: 1", rendered)
        self.assertIn("step 2: 3", rendered)
        sparkline = rendered.split("trend    ", 1)[1].splitlines()[0]
        self.assertEqual(len(sparkline), 3)
        for glyph in sparkline:
            self.assertIn(glyph, "▁▂▃▄▅▆▇█")

    def test_render_table_handles_empty_trend(self):
        rendered = mr.render_table(mr.summarize_metric([], "missing"))
        self.assertIn("count    0", rendered)
        self.assertIn("last     -", rendered)

    def test_render_json_round_trips_and_keeps_trend_array(self):
        points = [
            {"name": "m", "value": 1.0, "step": 0, "time": "t"},
            {"name": "m", "value": 3.0, "step": 1, "time": "t"},
        ]
        summary = mr.summarize_metric(points, "m")
        decoded = json.loads(json.dumps(mr.render_json(summary)))
        self.assertEqual(decoded["name"], "m")
        self.assertEqual(decoded["count"], 2)
        self.assertEqual(decoded["last"], 3.0)
        self.assertEqual(decoded["last_step"], 1)
        self.assertEqual(decoded["min_step"], 0)
        self.assertEqual(decoded["max_step"], 1)
        self.assertEqual(decoded["mean"], 2.0)
        self.assertEqual(decoded["recent_steps"], [0, 1])
        self.assertEqual(decoded["trend"], [0.0, 1.0])
        self.assertEqual(decoded["recent"], [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()