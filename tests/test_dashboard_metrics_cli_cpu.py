"""CPU tests for the multi-metric dashboard CLI (issue #265).

These run without a GPU and without external services. They cover:
- success path (human-readable + --json structured output, asserted fields)
- malformed input (missing job, unknown metric names, bad limit)
- boundary values (limit = 1, limit = 5000)
- default behavior unchanged (one metric reproduces the existing series)
- an integration-style check using a tiny local JSONL fixture

The CLI reuses the dashboard's existing metric_series contract; these tests
import areno.cli.dashboard directly, which does not pull torch.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from areno.cli.dashboard import metrics_command


def _write_jsonl_fixture(root: Path) -> tuple[str, str]:
    """Create a metrics dir + dashboard state with one job spanning 3 metrics.

    Returns (job_id, metrics_dir_relpath). Scales differ on purpose so dual-axis
    and normalization paths are exercisable elsewhere.
    """
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    # loss ~ [0.7, 2.8], lr ~ [0.0004, 0.001], reward ~ [-1, 5] -> different scales
    for step in range(10):
        lines.append(json.dumps({"name": "train/loss", "value": round(2.8 - step * 0.2, 4), "step": step}))
        lines.append(json.dumps({"name": "train/lr", "value": round(0.001 - step * 0.00006, 6), "step": step}))
        lines.append(json.dumps({"name": "train/reward", "value": round(-1 + step * 0.6, 4), "step": step}))
    (metrics_dir / "scalars.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Seed the dashboard state file so the CLI process can discover the job
    # without a running server.
    job_id = "testjob000001"
    state = {
        "jobs": [
            {
                "id": job_id,
                "kind": "train",
                "name": "fixture run",
                "command": [],
                "config": {},
                "launch": {},
                "metrics_dir": "metrics",
                "status": "succeeded",
                "stage": "done",
                "step": 9,
                "metrics_count": 30,
            }
        ]
    }
    (root / ".areno-dashboard-state.json").write_text(json.dumps(state), encoding="utf-8")
    return job_id, "metrics"


class MetricsCliCpuTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.job_id, _ = _write_jsonl_fixture(self.root)
        self._env_patch = patch.dict(os.environ, {"ARENO_DASHBOARD_ROOT": str(self.root)})
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()
        self._tmp.cleanup()

    def _run(self, *args: str):
        return CliRunner().invoke(metrics_command, args)

    # --- success path -------------------------------------------------------
    def test_json_output_has_expected_fields(self) -> None:
        result = self._run("--job", self.job_id, "--names", "train/loss,train/reward", "--limit", "5", "--json")
        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["job_id"], self.job_id)
        self.assertEqual(payload["limit"], 5)
        names = [m["name"] for m in payload["metrics"]]
        self.assertEqual(names, ["train/loss", "train/reward"])
        # Each metric reports point_count and points with step/value fields.
        for entry in payload["metrics"]:
            self.assertIn("point_count", entry)
            self.assertLessEqual(entry["point_count"], 5)
            self.assertTrue(all("step" in p and "value" in p for p in entry["points"]))

    def test_human_readable_lists_each_metric(self) -> None:
        result = self._run("--job", self.job_id, "--names", "train/loss,train/lr", "--limit", "3")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("# train/loss", result.output)
        self.assertIn("# train/lr", result.output)
        self.assertIn("step", result.output)

    # --- malformed input ----------------------------------------------------
    def test_missing_job_argument(self) -> None:
        result = self._run("--names", "train/loss")
        self.assertNotEqual(result.exit_code, 0)

    def test_unknown_job(self) -> None:
        result = self._run("--job", "does-not-exist", "--names", "train/loss")
        self.assertNotEqual(result.exit_code, 0)
        # Failure must identify the affected input (the job id), per the issue.
        self.assertIn("does-not-exist", result.output)

    def test_unknown_metric_name_is_reported(self) -> None:
        result = self._run("--job", self.job_id, "--names", "train/loss,typo/tag")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("typo/tag", result.output)

    def test_empty_names_rejected(self) -> None:
        result = self._run("--job", self.job_id, "--names", ", ,")
        self.assertNotEqual(result.exit_code, 0)

    # --- boundary values ----------------------------------------------------
    def test_limit_lower_boundary(self) -> None:
        result = self._run("--job", self.job_id, "--names", "train/loss", "--limit", "1", "--json")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["metrics"][0]["point_count"], 1)

    def test_limit_upper_boundary(self) -> None:
        result = self._run("--job", self.job_id, "--names", "train/loss", "--limit", "5000", "--json")
        self.assertEqual(result.exit_code, 0, result.output)

    def test_limit_out_of_range_rejected(self) -> None:
        result = self._run("--job", self.job_id, "--names", "train/loss", "--limit", "5001")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("limit", result.output.lower())

    # --- default behavior unchanged ----------------------------------------
    def test_single_metric_returns_same_series_as_existing_contract(self) -> None:
        """One metric via the CLI must match STATE.metric_series directly."""
        from areno.dashboard.server import STATE

        cli = self._run("--job", self.job_id, "--names", "train/loss", "--limit", "500", "--json")
        self.assertEqual(cli.exit_code, 0, cli.output)
        cli_points = json.loads(cli.output)["metrics"][0]["points"]
        direct = STATE.metric_series(self.job_id, "train/loss", limit=500)
        # Same step sequence and values -> the new CLI does not alter the
        # existing single-metric series.
        self.assertEqual([p["step"] for p in cli_points], [p["step"] for p in direct])
        self.assertEqual([p["value"] for p in cli_points], [p["value"] for p in direct])

    # --- integration: tiny local fixture, multi-metric ----------------------
    def test_multi_metric_fixture_end_to_end(self) -> None:
        result = self._run(
            "--job", self.job_id, "--names", "train/loss,train/lr,train/reward", "--limit", "10", "--json"
        )
        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(len(payload["metrics"]), 3)
        # All three metrics present and each carries the fixture's 10 steps.
        for entry in payload["metrics"]:
            self.assertEqual(entry["point_count"], 10)


if __name__ == "__main__":
    unittest.main()
