"""CPU tests for the ``areno metrics`` CLI (issue #254).

Drives the Click command end-to-end with ``CliRunner`` against real
``events.out.tfevents.*`` fixtures so the matrix runs through the same read
path users hit (option wiring, exit codes, available-tag listing, JSON shape).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from torch.utils.tensorboard import SummaryWriter

from areno.cli.metrics import metrics_command


def _writer(metrics_dir: Path) -> SummaryWriter:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(metrics_dir))


class MetricsCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def _invoke(self, args: list[str]):
        return self.runner.invoke(metrics_command, args)

    def test_success_table_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            writer = _writer(d)
            try:
                for step in range(5):
                    writer.add_scalar("rollout/rewards_mean", 0.5 + 0.125 * step, step)
                writer.add_scalar("train/loss", 2.0, 0)
            finally:
                writer.close()

            table = self._invoke(["--metrics-dir", str(d), "--name", "rollout/rewards_mean"])
            self.assertEqual(table.exit_code, 0)
            self.assertIn("metric   rollout/rewards_mean", table.output)
            self.assertIn("count    5", table.output)
            self.assertIn("steps    0 -> 4", table.output)
            self.assertIn("mean", table.output)
            self.assertIn("(step 4)", table.output)
            self.assertIn("step 0: 0.5", table.output)
            sparkline = table.output.split("trend    ", 1)[1].splitlines()[0]
            self.assertEqual(len(sparkline), 5)

            jres = self._invoke(["--metrics-dir", str(d), "--name", "rollout/rewards_mean", "--json"])
            self.assertEqual(jres.exit_code, 0)
            payload = json.loads(jres.output)
            self.assertEqual(payload["name"], "rollout/rewards_mean")
            self.assertEqual(payload["count"], 5)
            self.assertEqual(payload["min"], 0.5)
            self.assertEqual(payload["max"], 1.0)
            self.assertEqual(payload["max_step"], 4)
            self.assertEqual(payload["last_step"], 4)
            self.assertEqual(payload["mean"], 0.75)
            self.assertEqual(payload["recent_steps"], [0, 1, 2, 3, 4])
            self.assertEqual(len(payload["trend"]), 5)
            self.assertEqual(len(payload["recent"]), 5)

    def test_no_name_lists_available_tags_exit_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            writer = _writer(d)
            try:
                writer.add_scalar("rollout/a", 1.0, 0)
                writer.add_scalar("train/b", 2.0, 0)
            finally:
                writer.close()

            res = self._invoke(["--metrics-dir", str(d)])
            self.assertEqual(res.exit_code, 0)
            self.assertIn("available metric tags", res.output)
            self.assertIn("rollout/a", res.output)
            self.assertIn("train/b", res.output)

    def test_unknown_name_lists_available_tags_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            writer = _writer(d)
            try:
                writer.add_scalar("rollout/a", 1.0, 0)
            finally:
                writer.close()

            res = self._invoke(["--metrics-dir", str(d), "--name", "does/not/exist"])
            self.assertEqual(res.exit_code, 1)
            self.assertIn("metric 'does/not/exist' not found", res.output)
            self.assertIn("rollout/a", res.output)
            self.assertIn("Error:", res.output)

    def test_unknown_name_json_emits_missing_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            writer = _writer(d)
            try:
                writer.add_scalar("rollout/a", 1.0, 0)
            finally:
                writer.close()

            res = self._invoke(["--metrics-dir", str(d), "--name", "x", "--json"])
            # The JSON listing is emitted before the ClickException, which follows
            # as an "Error:" line. Assert the payload unconditionally so a dropped
            # listing fails the test instead of passing silently.
            payload = json.loads(res.output.split("\nError:")[0])
            self.assertEqual(payload["available_tags"], ["rollout/a"])
            self.assertEqual(payload["missing"], "x")
            self.assertEqual(res.exit_code, 1)

    def test_empty_directory_reports_none_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._invoke(["--metrics-dir", tmp, "--name", "anything"])
            self.assertEqual(res.exit_code, 1)
            self.assertIn("available tags: (none found)", res.output)

    def test_directory_with_only_rollout_samples_no_event(self):
        # rollout_samples.jsonl must NOT be read as metrics; with no event files
        # the run yields no metric points.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "rollout_samples.123.jsonl").write_text('{"name":"x","value":1}\n', encoding="utf-8")
            res = self._invoke(["--metrics-dir", str(d), "--name", "x"])
            self.assertEqual(res.exit_code, 1)
            self.assertIn("available tags: (none found)", res.output)

    def test_missing_directory_exits_1(self):
        res = self._invoke(["--metrics-dir", "/definitely/missing/areno/zzz", "--name", "x"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("metrics dir not found", res.output)

    def test_missing_tensorboard_reports_clear_error(self):
        # Simulate a missing tensorboard install by mapping the module to None,
        # which makes the CLI's `import tensorboard` probe raise ImportError.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"tensorboard": None}):
                res = self._invoke(["--metrics-dir", tmp, "--name", "train/loss"])
            self.assertEqual(res.exit_code, 1)
            self.assertIn("tensorboard is not installed", res.output)

    def test_single_point_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            writer = _writer(d)
            try:
                writer.add_scalar("rollout/one", 9.0, 7)
            finally:
                writer.close()

            res = self._invoke(["--metrics-dir", str(d), "--name", "rollout/one"])
            self.assertEqual(res.exit_code, 0)
            self.assertIn("count    1", res.output)
            self.assertIn("last     9", res.output)
            sparkline = res.output.split("trend    ", 1)[1].splitlines()[0]
            self.assertEqual(len(sparkline), 1)

    def test_all_nan_series_is_empty(self):
        # A tag whose every point is NaN must not appear as available at all.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            writer = _writer(d)
            try:
                writer.add_scalar("rollout/only_nan", float("nan"), 0)
                writer.add_scalar("rollout/only_nan", float("nan"), 1)
                writer.add_scalar("rollout/real", 1.0, 0)
            finally:
                writer.close()

            listing = self._invoke(["--metrics-dir", str(d)])
            self.assertNotIn("only_nan", listing.output)
            self.assertIn("rollout/real", listing.output)

            res = self._invoke(["--metrics-dir", str(d), "--name", "rollout/only_nan"])
            self.assertEqual(res.exit_code, 1)
            self.assertIn("rollout/only_nan", res.output)

    def test_truncation_caps_at_500_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            writer = _writer(d)
            try:
                for step in range(600):
                    writer.add_scalar("rollout/big", float(step), step)
            finally:
                writer.close()

            res = self._invoke(["--metrics-dir", str(d), "--name", "rollout/big"])
            self.assertEqual(res.exit_code, 0)
            self.assertIn("count    500", res.output)
            # --limit (default 20) bounds the sparkline to the recent window;
            # count is the full 500, but trend/recent stay bounded.
            sparkline = res.output.split("trend    ", 1)[1].splitlines()[0]
            self.assertLessEqual(len(sparkline), 20)

    def test_pid_filter_excludes_other_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            writer = _writer(d)
            try:
                writer.add_scalar("rollout/a", 1.0, 0)
            finally:
                writer.close()

            match = self._invoke(["--metrics-dir", str(d), "--pid", str(os.getpid()), "--name", "rollout/a"])
            self.assertEqual(match.exit_code, 0)

            miss = self._invoke(["--metrics-dir", str(d), "--pid", "99999999", "--name", "rollout/a"])
            self.assertEqual(miss.exit_code, 1)
            self.assertIn("available tags: (none found)", miss.output)


if __name__ == "__main__":
    unittest.main()