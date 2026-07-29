"""CPU tests for ``areno timing-summary`` (issue #256).

Two layers:

* Pure-logic tests (``_reconcile`` / ``_is_partial`` / ``format_*``) run
  everywhere — no torch or tensorboard needed.
* End-to-end tests write real TensorBoard scalars via ``SummaryWriter`` and
  read them back through the CLI; they are skipped when torch/tensorboard are
  absent so the suite still runs on a bare CPU machine.

All assertions target specific summary fields and error messages, not just
exit status.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner

from areno.dashboard import timeperf as tp


def _summary_writer_cls():
    """Return a ``SummaryWriter`` class from torch, or fall back to tensorboardX.

    ``areno.api.metrics`` writes via torch when available; for tests we only
    need to produce readable ``events.out.tfevents.*`` files, so either writer
    works. Falling back to tensorboardX lets the suite run on a CPU machine
    without torch installed (review finding 6 — broader CI coverage).
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        from tensorboardX import SummaryWriter
    return SummaryWriter


def _have_tensorboard() -> bool:
    # Read side needs the ``tensorboard`` package (EventAccumulator); write side
    # needs torch OR tensorboardX.
    try:
        import tensorboard  # noqa: F401
        _summary_writer_cls()
    except Exception:
        return False
    return True


# Step-level scalar tags a trainer actually emits (see areno/api/metrics.py and
# the algorithm trainers). ``total`` is the reported end-to-end wall time.
# A pid guaranteed never to correspond to a live process, so finished runs
# resolve deterministically to ``completed`` regardless of the stale
# ``status`` field the trainer leaves behind (review finding 1).
_DEAD_PID = 999_999


def _write_run(run_dir: Path, steps: list[dict[int, float]], *, status: str = "succeeded") -> None:
    """Write a tiny deterministic run into ``run_dir``.

    ``steps`` is a list of ``{tag: value}`` dicts, one per step. Each step's
    dict is written as scalars at that step index. A ``dashboard_state.<pid>
    .json`` snapshot is also written. ``status="running"`` uses the live test
    process pid so ``load_run_status`` reports ``active``; any other status uses
    a dead pid so it reports ``completed`` — independent of platform init pid.
    """
    import os

    SummaryWriter = _summary_writer_cls()

    writer = SummaryWriter(log_dir=str(run_dir))
    for index, scalars in enumerate(steps):
        for tag, value in scalars.items():
            writer.add_scalar(tag, float(value), index)
    writer.close()

    pid = os.getpid() if status == "running" else _DEAD_PID
    state = {"pid": pid, "stage": "train_end", "status": status, "updated_at": 0, "step": len(steps) - 1}
    (run_dir / f"dashboard_state.{pid}.json").write_text(json.dumps(state), encoding="utf-8")


class ReconcileLogicTest(unittest.TestCase):
    """No-dependency tests for the reconciliation primitives."""

    def test_reconcile_reported_matches_reconstructed(self):
        recon = tp._reconcile({"total": 10.0, "rollout": 4.0, "train": 6.0})
        self.assertEqual(recon["total_source"], "reported")
        self.assertAlmostEqual(recon["diff"], 0.0)
        self.assertAlmostEqual(recon["reported_total"], 10.0)
        self.assertAlmostEqual(recon["reconstructed_total"], 10.0)

    def test_reconcile_partial_step_falls_back_to_reconstructed(self):
        # A step with rollout recorded but no e2e/total: partial signature.
        values = {"rollout": 4.0}
        recon = tp._reconcile(values)
        self.assertEqual(recon["total_source"], "reconstructed")
        self.assertAlmostEqual(recon["reported_total"], 4.0)
        self.assertTrue(tp._is_partial(values))

    def test_reconcile_nonzero_diff_is_preserved(self):
        # Phases sum to 12 but reported total is 10 -> diff -2 (overlap/other).
        recon = tp._reconcile({"total": 10.0, "rollout": 4.0, "train": 8.0})
        self.assertAlmostEqual(recon["diff"], -2.0)

    def test_save_and_sync_weight_segments_are_not_phases_in_reconstruction(self):
        # reconstructed_total must not include the synthetic rollup 'total'.
        recon = tp._reconcile({"total": 5.0, "rollout": 5.0})
        self.assertAlmostEqual(recon["reconstructed_total"], 5.0)


class FormatTest(unittest.TestCase):
    """No-dependency tests for the two output renderers."""

    def _summary(self) -> dict:
        return {
            "run_status": "completed",
            "run_dir": "/tmp/x",
            "num_steps": 2,
            "latest_update": {
                "step": 1,
                "segments": {
                    "rollout": 4.0,
                    "make_sample": None,
                    "reward": None,
                    "old policy log probs": None,
                    "actor log probs": None,
                    "ref log probs": None,
                    "value": None,
                    "advantages": None,
                    "sync weight": None,
                    "train": 6.0,
                    "save": None,
                    "other": None,
                },
                "partial": False,
                "reported_total": 10.0,
                "reconstructed_total": 10.0,
                "diff": 0.0,
                "total_source": "reported",
            },
            "whole_run": {
                "segments": {"rollout": 8.0, "train": 12.0, "other": 0.0},
                "reported_total": 20.0,
                "reconstructed_total": 20.0,
                "diff": 0.0,
                "total_source": "reported",
            },
            "overlap": [],
            "overlap_note": "no overlapping sub-phase timers are recorded by current trainers; overlap is always empty",
            "missing": ["save", "sync weight"],
            "divergences": [],
        }

    def test_format_table_mentions_missing_phases_and_totals(self):
        text = tp.format_table(self._summary())
        self.assertIn("Run status: completed", text)
        self.assertIn("reported_total", text)
        self.assertIn("reconstructed_total", text)
        self.assertIn("Missing phases:", text)
        self.assertIn("save", text)
        self.assertIn("Overlap: none", text)

    def test_format_json_round_trips_with_expected_keys(self):
        payload = json.loads(tp.format_json(self._summary()))
        self.assertEqual(
            list(payload.keys()),
            [
                "run_status",
                "run_dir",
                "num_steps",
                "latest_update",
                "whole_run",
                "overlap",
                "overlap_note",
                "missing",
                "divergences",
            ],
        )
        self.assertEqual(payload["missing"], ["save", "sync weight"])
        self.assertEqual(payload["overlap"], [])


class RunStatusTest(unittest.TestCase):
    """No-dependency tests for liveness-based run_status (finding 1 + 6 reverse).

    The trainer never writes a terminal ``status``, so ``status == "running"``
    is stale for finished runs. ``load_run_status`` must resolve the pid and
    check process liveness rather than trust the field. These cases run without
    torch/tensorboard — exactly the bare-CI coverage the review asked for.
    """

    def _write_state(self, dir_path: Path, pid: int, status: str = "running") -> None:
        payload = {"pid": pid, "stage": "train_end", "status": status, "updated_at": 0, "step": 3}
        (dir_path / f"dashboard_state.{pid}.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_finished_run_reports_completed_despite_stale_running_status(self):
        """A dead pid with status=running must read as completed (review finding 1)."""
        import tempfile

        run_dir = Path(tempfile.mkdtemp())
        self._write_state(run_dir, pid=999_999, status="running")  # pid guaranteed not alive
        self.assertEqual(tp.load_run_status(run_dir), "completed")

    def test_live_process_reports_active(self):
        """A live pid (this test process) with status=running reads as active."""
        import os
        import tempfile

        run_dir = Path(tempfile.mkdtemp())
        self._write_state(run_dir, pid=os.getpid(), status="running")
        self.assertEqual(tp.load_run_status(run_dir), "active")

    def test_missing_state_file_defaults_completed(self):
        import tempfile

        run_dir = Path(tempfile.mkdtemp())
        self.assertEqual(tp.load_run_status(run_dir), "completed")

    def test_unreadable_state_file_defaults_completed(self):
        import tempfile

        run_dir = Path(tempfile.mkdtemp())
        (run_dir / "dashboard_state.123.json").write_text("not json", encoding="utf-8")
        self.assertEqual(tp.load_run_status(run_dir), "completed")


class AccumulateEventTest(unittest.TestCase):
    """No-dependency tests for ``_accumulate_event`` (findings 3 & 4).

    Exercises the tag->segment classification directly, so the PPO sub-phase
    folding and the rollup/echo divergence logic are covered on a bare machine
    without the ``tensorboard`` reader.
    """

    def _fresh(self):
        return {}, {}, {}

    def test_ppo_subphases_fold_into_other_not_dropped(self):
        """Out-of-vocab sub-phase tags fold into 'other' and stay in the sum (finding 3)."""
        bucket, rollup_seen, phase_seen = {}, {}, {}
        for tag, value in [
            ("train/step_e2e_time_s", 10.0),
            ("train/step_rollout_time_s", 4.0),
            ("train/step_train_time_s", 6.0),
            ("time/critic_value_forward_time_s", 1.5),
            ("time/critic_train_time_s", 0.8),
            ("time/reward_score_time_s", 0.3),
            ("time/ref_logprob_forward_time_s", 0.2),
        ]:
            tp._accumulate_event(bucket, 0, tag, value, rollup_seen, phase_seen)
        # rollout + train + ref log probs + other(critic value+critic train+reward score)
        self.assertAlmostEqual(bucket["other"], 2.6)
        self.assertAlmostEqual(bucket["rollout"], 4.0)
        self.assertAlmostEqual(bucket["train"], 6.0)
        self.assertAlmostEqual(bucket["ref log probs"], 0.2)
        recon = tp._reconcile(bucket)
        self.assertAlmostEqual(recon["reconstructed_total"], 12.8)
        self.assertAlmostEqual(recon["diff"], 10.0 - 12.8)

    def test_echo_matching_rollup_produces_no_divergence(self):
        """time/rollout echoing the step rollup value must not flag a divergence (finding 4)."""
        bucket, rollup_seen, phase_seen = {}, {}, {}
        tp._accumulate_event(bucket, 0, "train/step_rollout_time_s", 4.0, rollup_seen, phase_seen)
        msg = tp._accumulate_event(bucket, 0, "time/rollout", 4.0, rollup_seen, phase_seen)
        self.assertIsNone(msg)
        # Rollup value is authoritative and not overwritten by the echo.
        self.assertEqual(bucket["rollout"], 4.0)

    def test_echo_diverging_from_rollup_flags_divergence(self):
        """A time/* echo that disagrees with the rollup is surfaced, not silently overwritten (finding 4)."""
        bucket, rollup_seen, phase_seen = {}, {}, {}
        tp._accumulate_event(bucket, 0, "train/step_train_time_s", 6.0, rollup_seen, phase_seen)
        msg = tp._accumulate_event(bucket, 0, "time/train", 7.0, rollup_seen, phase_seen)
        self.assertIsNotNone(msg)
        self.assertIn("diverges", msg)
        # Authoritative rollup value retained.
        self.assertEqual(bucket["train"], 6.0)

    def test_e2e_total_tag_set_as_total_not_phase(self):
        bucket, rollup_seen, phase_seen = {}, {}, {}
        tp._accumulate_event(bucket, 0, "train/step_e2e_time_s", 10.0, rollup_seen, phase_seen)
        self.assertEqual(bucket["total"], 10.0)
        self.assertNotIn("total", [n for n in tp.TIME_SEGMENT_ORDER])  # total is rollup, not a phase


@unittest.skipUnless(_have_tensorboard(), "torch + tensorboard required for end-to-end timing-summary tests")
class EndToEndTest(unittest.TestCase):
    """Write a real run and read it back through the CLI."""

    def setUp(self) -> None:
        import tempfile

        self._cwd = tempfile.mkdtemp()
        self.run_dir = Path(self._cwd) / "run"
        self.run_dir.mkdir()

    def _invoke(self, *args: str):
        from areno.cli.main import main

        return CliRunner().invoke(main, ["timing-summary", str(self.run_dir), *args])

    def test_successful_run_sums_phases_and_reconciles(self):
        # Two steps: rollout 4 + train 6 = total 10 each, no overlap.
        _write_run(
            self.run_dir,
            [
                {"train/step_rollout_time_s": 4.0, "train/step_train_time_s": 6.0, "train/step_e2e_time_s": 10.0},
                {"train/step_rollout_time_s": 4.0, "train/step_train_time_s": 6.0, "train/step_e2e_time_s": 10.0},
            ],
        )
        res = self._invoke("--json")
        self.assertEqual(res.exit_code, 0, res.output)
        payload = json.loads(res.output)
        self.assertEqual(payload["num_steps"], 2)
        self.assertEqual(payload["whole_run"]["segments"]["rollout"], 8.0)
        self.assertEqual(payload["whole_run"]["segments"]["train"], 12.0)
        self.assertAlmostEqual(payload["whole_run"]["diff"], 0.0)
        self.assertEqual(payload["whole_run"]["total_source"], "reported")
        # save / sync weight are never emitted -> must show as missing.
        self.assertIn("save", payload["missing"])
        self.assertIn("sync weight", payload["missing"])

    def test_partial_latest_step_is_flagged(self):
        # Step 0 complete; step 1 only has rollout (run still in progress).
        _write_run(
            self.run_dir,
            [
                {"train/step_rollout_time_s": 4.0, "train/step_train_time_s": 6.0, "train/step_e2e_time_s": 10.0},
                {"train/step_rollout_time_s": 4.0},
            ],
            status="running",
        )
        res = self._invoke("--json")
        self.assertEqual(res.exit_code, 0, res.output)
        payload = json.loads(res.output)
        self.assertEqual(payload["run_status"], "active")
        self.assertTrue(payload["latest_update"]["partial"])
        self.assertEqual(payload["latest_update"]["total_source"], "reconstructed")

    def test_active_run_status_from_dashboard_state(self):
        _write_run(
            self.run_dir,
            [
                {"train/step_rollout_time_s": 4.0, "train/step_train_time_s": 6.0, "train/step_e2e_time_s": 10.0},
            ],
            status="running",
        )
        res = self._invoke("--json")
        self.assertEqual(json.loads(res.output)["run_status"], "active")

    def test_table_output_renders_for_real_run(self):
        _write_run(
            self.run_dir,
            [
                {"train/step_rollout_time_s": 4.0, "train/step_train_time_s": 6.0, "train/step_e2e_time_s": 10.0},
            ],
        )
        res = self._invoke()
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Run status:", res.output)
        self.assertIn("rollout", res.output)


class ValidationTest(unittest.TestCase):
    """Input validation runs before any TensorBoard reading (no deps needed)."""

    def _invoke(self, path: str):
        from areno.cli.main import main

        return CliRunner().invoke(main, ["timing-summary", path])

    def test_nonexistent_dir_exits_with_message(self):
        res = self._invoke("/no/such/areno/dir")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("does not exist", res.output)

    def test_empty_dir_exits_with_metrics_message(self):
        import tempfile

        empty = Path(tempfile.mkdtemp())
        res = self._invoke(str(empty))
        self.assertEqual(res.exit_code, 1)
        self.assertIn("no AReno metrics found", res.output)

    def test_command_is_registered(self):
        """The new command shows up in the top-level help (default behavior unchanged)."""
        from areno.cli.main import main

        res = CliRunner().invoke(main, ["--help"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("timing-summary", res.output)


if __name__ == "__main__":
    unittest.main()
