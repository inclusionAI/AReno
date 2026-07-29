"""CPU tests for the RL phase waterfall feature (issue #266).

All tests use inline deterministic fixtures — no GPU, no database, no
network.  The fixtures mirror the ``job.timeperf`` row structure produced
by ``DashboardState._append_timeperf_row`` in ``areno/dashboard/server.py``.
"""

from __future__ import annotations

import math
import unittest

from areno.api.dashboard import PHASE_GROUPS, PHASE_ORDER, phase_waterfall


# ---------------------------------------------------------------------------
# Inline fixtures
# ---------------------------------------------------------------------------

def _row(step, total, segments):
    """Build a timeperf row dict matching the real server.py schema."""
    return {
        "step": step,
        "segments": [{"name": n, "seconds": s} for n, s in segments],
        "total_s": total,
        "time": "2026-01-01T00:00:00Z",
        "source": "metrics",
    }


def _common_rows():
    """3 rows: normal, slow, missing-reward — covers the core scenarios."""
    return [
        # step 0: healthy row with 5 segment values
        _row(0, 100.0, [
            ("rollout", 50.0),
            ("reward", 5.0),
            ("train", 30.0),
            ("save", 5.0),
            ("sync weight", 3.0),
        ]),
        # step 1: slow row (total=200 > threshold)
        _row(1, 200.0, [
            ("rollout", 120.0),
            ("reward", 10.0),
            ("train", 50.0),
            ("sync weight", 5.0),
        ]),
        # step 2: missing reward, other segment absent (waiting = residual)
        _row(2, 80.0, [
            ("rollout", 40.0),
            ("train", 20.0),
        ]),
    ]


def _phase_durations(updates, step):
    """Extract {phase_name: duration_s} for a given step."""
    for u in updates:
        if u["step"] == step:
            return {p["name"]: p["duration_s"] for p in u["phases"]}
    return {}


def _phase_by_name(updates, step, name):
    """Extract a single phase dict for a given step."""
    for u in updates:
        if u["step"] == step:
            for p in u["phases"]:
                if p["name"] == name:
                    return p
    return None


def _sum_durations(updates, step):
    for u in updates:
        if u["step"] == step:
            return sum(p["duration_s"] for p in u["phases"])
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class PhaseWaterfallTest(unittest.TestCase):
    """Core waterfall function tests."""

    def test_waterfall_layout(self):
        """Two-step fixture: verify start/end/duration match hand calculation."""
        rows = [
            _row(0, 100.0, [
                ("rollout", 50.0),
                ("reward", 5.0),
                ("train", 30.0),
                ("save", 5.0),
                ("sync weight", 3.0),
            ]),
        ]
        updates, errors = phase_waterfall(rows)
        self.assertEqual(len(updates), 1)
        self.assertEqual(errors, [])

        u = updates[0]
        self.assertEqual(u["step"], 0)
        self.assertAlmostEqual(u["total_s"], 100.0)

        phases = {p["name"]: p for p in u["phases"]}
        # 5 phases always present
        self.assertEqual(set(phases.keys()), set(PHASE_ORDER))

        # save(5) + train(30) = 35 → training
        self.assertAlmostEqual(phases["training"]["duration_s"], 35.0)
        # sequential start/end
        self.assertAlmostEqual(phases["rollout"]["start_s"], 0.0)
        self.assertAlmostEqual(phases["rollout"]["end_s"], 50.0)
        self.assertAlmostEqual(phases["reward"]["start_s"], 50.0)
        self.assertAlmostEqual(phases["reward"]["end_s"], 55.0)
        self.assertAlmostEqual(phases["training"]["start_s"], 55.0)
        self.assertAlmostEqual(phases["training"]["end_s"], 90.0)
        self.assertAlmostEqual(phases["synchronization"]["start_s"], 90.0)
        self.assertAlmostEqual(phases["synchronization"]["end_s"], 93.0)
        self.assertAlmostEqual(phases["waiting"]["start_s"], 93.0)
        self.assertAlmostEqual(phases["waiting"]["end_s"], 100.0)

        # totals consistency
        total_dur = sum(p["duration_s"] for p in u["phases"])
        self.assertAlmostEqual(total_dur, u["total_s"])

    def test_threshold_filters_slow_updates(self):
        """slow_threshold=0 → none flagged; between two totals → one flagged."""
        rows = _common_rows()
        # threshold=0 → no slow flags
        updates, _ = phase_waterfall(rows, slow_threshold=0.0)
        self.assertFalse(any(u["is_slow"] for u in updates))

        # threshold=150 → step 1 (total=200) is slow, step 0 (100) and 2 (80) not
        updates, _ = phase_waterfall(rows, slow_threshold=150.0)
        slow_map = {u["step"]: u["is_slow"] for u in updates}
        self.assertFalse(slow_map[0])
        self.assertTrue(slow_map[1])
        self.assertFalse(slow_map[2])

        # boundary: total == threshold → NOT slow (strict >)
        rows2 = [_row(0, 100.0, [("rollout", 50.0), ("train", 50.0)])]
        updates2, _ = phase_waterfall(rows2, slow_threshold=100.0)
        self.assertFalse(updates2[0]["is_slow"])

    def test_gap_as_waiting_phase(self):
        """other segment present → waiting = other seconds; absent → residual."""
        # With 'other' segment explicitly present
        rows = [_row(0, 100.0, [
            ("rollout", 60.0), ("train", 30.0), ("other", 10.0),
        ])]
        updates, _ = phase_waterfall(rows)
        waiting = _phase_by_name(updates, 0, "waiting")
        self.assertIsNotNone(waiting)
        self.assertAlmostEqual(waiting["duration_s"], 10.0)

        # Without 'other' → waiting = total - sum(known segments) = 100 - 90 = 10
        rows2 = [_row(0, 100.0, [
            ("rollout", 60.0), ("train", 30.0),
        ])]
        updates2, _ = phase_waterfall(rows2)
        waiting2 = _phase_by_name(updates2, 0, "waiting")
        self.assertIsNotNone(waiting2)
        self.assertAlmostEqual(waiting2["duration_s"], 10.0)

        # No double counting: Σ durations == total
        self.assertAlmostEqual(_sum_durations(updates2, 0), 100.0)

    def test_missing_phase(self):
        """Missing reward → reward phase duration=0, still present; Σ == total."""
        rows = [_row(0, 80.0, [("rollout", 40.0), ("train", 20.0)])]
        updates, errors = phase_waterfall(rows)
        self.assertEqual(len(updates), 1)

        reward = _phase_by_name(updates, 0, "reward")
        self.assertIsNotNone(reward, "reward phase must exist even when missing")
        self.assertAlmostEqual(reward["duration_s"], 0.0)

        # Σ durations still equals total (waiting absorbs the residual)
        self.assertAlmostEqual(_sum_durations(updates, 0), 80.0)

    def test_in_progress_skipped(self):
        """total_s missing/≤0/<sum(segments) → row enters errors, not updates."""
        rows_bad = [
            {"step": 0, "segments": [], "total_s": 0},       # total <= 0
            {"step": 1, "segments": [{"name": "rollout", "seconds": 50}]},  # no total_s
            {"step": 2, "segments": [{"name": "rollout", "seconds": 50}], "total_s": 30},  # total < sum
        ]
        updates, errors = phase_waterfall(rows_bad)
        self.assertEqual(len(updates), 0)
        self.assertEqual(len(errors), 3)
        error_steps = {e["step"] for e in errors}
        self.assertEqual(error_steps, {0, 1, 2})
        # Each error identifies the affected step and has a reason
        for e in errors:
            self.assertIn("reason", e)
            self.assertTrue(e["reason"])

    def test_unmapped_segment_handling(self):
        """Unmapped segment name → enters errors, seconds don't vanish."""
        rows = [_row(0, 100.0, [
            ("rollout", 50.0),
            ("train", 30.0),
            ("unknown_phase", 10.0),
        ])]
        updates, errors = phase_waterfall(rows)
        self.assertEqual(len(updates), 1)
        # unmapped segment reported in errors
        unmapped = [e for e in errors if "unmapped" in e.get("reason", "")]
        self.assertEqual(len(unmapped), 1)
        self.assertIn("unknown_phase", unmapped[0]["reason"])
        # seconds not lost → Σ durations == total
        self.assertAlmostEqual(_sum_durations(updates, 0), 100.0)

    def test_empty_and_single_segment(self):
        """Empty input → ([], []); single segment → single non-zero phase."""
        updates, errors = phase_waterfall([])
        self.assertEqual(updates, [])
        self.assertEqual(errors, [])

        rows = [_row(0, 50.0, [("rollout", 50.0)])]
        updates, _ = phase_waterfall(rows)
        self.assertEqual(len(updates), 1)
        durations = _phase_durations(updates, 0)
        self.assertAlmostEqual(durations["rollout"], 50.0)
        self.assertAlmostEqual(durations["reward"], 0.0)
        self.assertAlmostEqual(durations["training"], 0.0)
        self.assertAlmostEqual(durations["synchronization"], 0.0)
        # waiting = 50 - 50 = 0
        self.assertAlmostEqual(durations["waiting"], 0.0)

    def test_malformed_input_validation(self):
        """Negative threshold → ValueError with clear message; bad rows → errors."""
        with self.assertRaises(ValueError) as ctx:
            phase_waterfall([], slow_threshold=-1.0)
        self.assertIn("slow_threshold", str(ctx.exception))
        self.assertIn("non-negative", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            phase_waterfall([], slow_threshold="abc")
        self.assertIn("slow_threshold", str(ctx.exception))

        # Non-dict rows enter errors
        updates, errors = phase_waterfall(["not a dict", None])
        self.assertEqual(len(updates), 0)
        self.assertEqual(len(errors), 2)

        # Row missing 'step'
        updates2, errors2 = phase_waterfall([{"total_s": 10, "segments": []}])
        self.assertEqual(len(updates2), 0)
        self.assertTrue(any("missing 'step'" in e["reason"] for e in errors2))

        # Non-numeric segment seconds → treated as 0, error logged
        rows = [_row(0, 10.0, [("rollout", "not_a_number")])]
        updates3, errors3 = phase_waterfall(rows)
        self.assertEqual(len(updates3), 1)
        self.assertTrue(any("non-numeric" in e["reason"] for e in errors3))

    def test_phase_groups_cover_all_segments(self):
        """Every name in TIME_SEGMENT_ORDER must map to a PHASE_GROUPS entry."""
        time_segment_order = [
            "rollout", "make_sample", "reward", "old policy log probs",
            "actor log probs", "ref log probs", "value", "advantages",
            "sync weight", "train", "save", "other",
        ]
        mapped = set()
        for segs in PHASE_GROUPS.values():
            mapped.update(segs)
        for name in time_segment_order:
            self.assertIn(name, mapped, f"'{name}' not mapped in PHASE_GROUPS")


class WaterfallAPITest(unittest.TestCase):
    """Integration-style test: DashboardState.waterfall with a fake Job."""

    def test_waterfall_method_returns_structured_data(self):
        """DashboardState.waterfall returns updates + errors + summary."""
        from areno.dashboard.server import DashboardState, Job

        ds = DashboardState()
        ds.jobs = {}  # clear loaded state
        job = Job(kind="train", name="test", command=[], config={}, metrics_dir=None)
        job.timeperf = [
            _row(0, 100.0, [("rollout", 50.0), ("train", 30.0), ("other", 20.0)]),
            _row(1, 200.0, [("rollout", 100.0), ("train", 80.0), ("other", 20.0)]),
        ]
        ds.jobs[job.id] = job

        result = ds.waterfall(job.id, slow_threshold=150.0)
        self.assertEqual(result["job_id"], job.id)
        self.assertEqual(result["slow_threshold"], 150.0)
        self.assertEqual(len(result["updates"]), 2)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["summary"]["count"], 2)
        self.assertEqual(result["summary"]["slow_count"], 1)
        self.assertAlmostEqual(result["summary"]["avg_total_s"], 150.0)

        # Verify each update has the 5 phases
        for u in result["updates"]:
            self.assertEqual(len(u["phases"]), 5)
            # Σ durations == total
            total_dur = sum(p["duration_s"] for p in u["phases"])
            self.assertAlmostEqual(total_dur, u["total_s"])

    def test_waterfall_method_missing_job(self):
        """Nonexistent job_id returns empty updates."""
        from areno.dashboard.server import DashboardState

        ds = DashboardState()
        ds.jobs = {}
        result = ds.waterfall("nonexistent", slow_threshold=0.0)
        self.assertIsNone(result["job_id"])
        self.assertEqual(result["updates"], [])
        self.assertEqual(result["summary"], {})


class WaterfallAcceptanceTest(unittest.TestCase):
    """Tests covering issue acceptance criteria gaps."""

    def test_no_overlap_between_phases(self):
        """Issue requires 'test overlapping' — assert phases never overlap.

        Under serial accumulation, each phase's start_s must equal the
        previous phase's end_s.  No two phases share the same time range.
        """
        rows = _common_rows()
        updates, _ = phase_waterfall(rows)
        self.assertGreater(len(updates), 0)
        for u in updates:
            phases = u["phases"]
            for i in range(1, len(phases)):
                self.assertAlmostEqual(
                    phases[i]["start_s"],
                    phases[i - 1]["end_s"],
                    places=5,
                    msg=(
                        f"step {u['step']}: phase '{phases[i]['name']}' starts at "
                        f"{phases[i]['start_s']} but previous phase "
                        f"'{phases[i-1]['name']}' ends at {phases[i-1]['end_s']} — overlap detected"
                    ),
                )
            # First phase starts at 0, last phase ends at total_s
            self.assertAlmostEqual(phases[0]["start_s"], 0.0)
            self.assertAlmostEqual(phases[-1]["end_s"], u["total_s"])

    def test_deterministic_output(self):
        """Issue requires 'deterministic output' — same input → same output."""
        rows = _common_rows()
        updates_a, errors_a = phase_waterfall(rows, slow_threshold=100.0)
        updates_b, errors_b = phase_waterfall(rows, slow_threshold=100.0)
        self.assertEqual(updates_a, updates_b)
        self.assertEqual(errors_a, errors_b)

    def test_existing_behavior_unchanged(self):
        """Issue requires 'verify existing behavior unchanged when not enabled'.

        The default (slow_threshold=0) must not alter the timeperf data
        returned by the existing GET /api/jobs/<id> endpoint.  The waterfall
        endpoint with default params should produce updates whose steps and
        totals match the raw timeperf rows 1:1.
        """
        from areno.dashboard.server import DashboardState, Job

        ds = DashboardState()
        ds.jobs = {}
        job = Job(kind="train", name="test", command=[], config={}, metrics_dir=None)
        raw_rows = [
            _row(0, 100.0, [("rollout", 60.0), ("train", 30.0), ("other", 10.0)]),
            _row(1, 50.0, [("rollout", 30.0), ("train", 15.0), ("other", 5.0)]),
        ]
        job.timeperf = raw_rows
        ds.jobs[job.id] = job

        # Existing endpoint returns timeperf unchanged
        job_detail = ds.get_job(job.id)
        self.assertEqual(len(job_detail.timeperf), 2)
        self.assertEqual(job_detail.timeperf[0]["step"], 0)
        self.assertAlmostEqual(job_detail.timeperf[0]["total_s"], 100.0)

        # Waterfall with default params (slow_threshold=0) doesn't flag slow
        result = ds.waterfall(job.id, slow_threshold=0.0)
        self.assertFalse(any(u["is_slow"] for u in result["updates"]))
        # Steps and totals match raw rows 1:1
        for raw, wf in zip(raw_rows, result["updates"]):
            self.assertEqual(wf["step"], raw["step"])
            self.assertAlmostEqual(wf["total_s"], raw["total_s"])


if __name__ == "__main__":
    unittest.main()