"""CPU tests for structured live progress (#276)."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# .agents is not a Python package (starts with dot); add to path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".agents" / "scripts"))
from progress import (  # noqa: E402
    ProgressDisplay,
    ProgressEvent,
    ProgressTracker,
    get_progress,
)


# ---------------------------------------------------------------------------
# ProgressEvent
# ---------------------------------------------------------------------------


class TestProgressEvent:
    def test_defaults(self):
        e = ProgressEvent(stage="test", status="started")
        assert e.stage == "test"
        assert e.status == "started"
        assert e.message == ""
        assert e.step is None
        assert e.total is None
        assert e.elapsed_s == 0.0
        assert e.data is None

    def test_full_fields(self):
        e = ProgressEvent(
            stage="rollout", status="running", message="step 5/10",
            step=5, total=10, elapsed_s=2.5, data={"tps": 120},
        )
        assert e.step == 5
        assert e.total == 10
        assert e.data == {"tps": 120}

    def test_as_dict_minimal(self):
        d = ProgressEvent(stage="s", status="ok").as_dict()
        assert d["stage"] == "s"
        assert d["status"] == "ok"
        assert "step" not in d
        assert "total" not in d
        assert "data" not in d

    def test_as_dict_full(self):
        d = ProgressEvent(
            stage="s", status="ok", step=1, total=10, data={"k": "v"},
        ).as_dict()
        assert d["step"] == 1
        assert d["total"] == 10
        assert d["data"] == {"k": "v"}

    def test_json_serializable(self):
        e = ProgressEvent(stage="s", status="ok", step=1, total=10)
        encoded = json.dumps(e.as_dict())
        decoded = json.loads(encoded)
        assert decoded["stage"] == "s"


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------


class TestProgressTracker:
    def test_begin_stage(self):
        t = ProgressTracker()
        e = t.begin_stage("rollout", total=100)
        assert e.stage == "rollout"
        assert e.status == "started"
        assert e.total == 100

    def test_begin_sets_active_stage(self):
        t = ProgressTracker()
        t.begin_stage("rollout")
        assert t.active_stage == "rollout"

    def test_advance(self):
        t = ProgressTracker()
        t.begin_stage("rollout", total=100)
        e = t.advance(50, "halfway")
        assert e is not None
        assert e.stage == "rollout"
        assert e.status == "running"
        assert e.step == 50
        assert e.total == 100

    def test_advance_no_active_stage(self):
        t = ProgressTracker()
        assert t.advance(1) is None

    def test_complete_stage(self):
        t = ProgressTracker()
        t.begin_stage("rollout", total=100)
        e = t.complete_stage("done")
        assert e.stage == "rollout"
        assert e.status == "completed"
        assert t.active_stage is None

    def test_complete_no_active_stage(self):
        t = ProgressTracker()
        e = t.complete_stage("nothing")
        assert e.status == "completed"

    def test_fail_stage(self):
        t = ProgressTracker()
        t.begin_stage("rollout")
        e = t.fail_stage("OOM")
        assert e.status == "failed"
        assert e.message == "OOM"

    def test_fail_stage_captures_last_completed(self):
        t = ProgressTracker()
        t.begin_stage("rollout")
        t.complete_stage("ok")
        t.begin_stage("train")
        e = t.fail_stage("crash")
        assert e.data is not None
        assert e.data["last_completed_stage"] == "rollout"

    def test_fail_stage_pops(self):
        t = ProgressTracker()
        t.begin_stage("rollout")
        t.fail_stage("error")
        assert t.active_stage is None

    def test_cancel(self):
        t = ProgressTracker()
        t.begin_stage("rollout")
        t.complete_stage("ok")
        t.begin_stage("train")
        e = t.cancel()
        assert e.status == "cancelled"
        assert e.data is not None
        assert e.data["last_completed_stage"] == "rollout"
        assert t.active_stage is None

    def test_cancel_clears_stack(self):
        t = ProgressTracker()
        t.begin_stage("a")
        t.begin_stage("b")
        t.cancel()
        assert t.active_stage is None

    def test_nested_stages(self):
        t = ProgressTracker()
        t.begin_stage("outer", total=10)
        t.advance(5)
        t.begin_stage("inner", total=20)
        assert t.active_stage == "inner"
        t.advance(10)
        t.complete_stage("inner done")
        assert t.active_stage == "outer"


# ---------------------------------------------------------------------------
# ProgressDisplay — line mode
# ---------------------------------------------------------------------------


class TestProgressDisplayLineMode:
    @contextmanager
    def _display(self, mode="line"):
        buf = StringIO()
        d = ProgressDisplay(mode=mode, file=buf)
        yield d, buf
        d.close()

    def test_started(self):
        with self._display() as (d, buf):
            d.render(ProgressEvent(stage="rollout", status="started", message="begin"))
        output = buf.getvalue().strip()
        assert "[rollout]" in output
        assert "started" in output
        assert "begin" in output

    def test_running_with_progress(self):
        with self._display() as (d, buf):
            d.render(ProgressEvent(stage="rollout", status="running",
                                    step=50, total=100, message="halfway"))
        output = buf.getvalue().strip()
        assert "50/100" in output
        assert "halfway" in output

    def test_completed(self):
        with self._display() as (d, buf):
            d.render(ProgressEvent(stage="rollout", status="completed", message="done"))
        output = buf.getvalue().strip()
        assert "completed" in output
        assert "done" in output


# ---------------------------------------------------------------------------
# ProgressDisplay — JSON Lines mode
# ---------------------------------------------------------------------------


class TestProgressDisplayJsonLinesMode:
    @contextmanager
    def _display(self):
        buf = StringIO()
        d = ProgressDisplay(mode="jsonl", file=buf)
        yield d, buf
        d.close()

    def test_jsonl_output(self):
        with self._display() as (d, buf):
            d.render(ProgressEvent(stage="rollout", status="started", total=100))
        line = buf.getvalue().strip()
        obj = json.loads(line)
        assert obj["stage"] == "rollout"
        assert obj["status"] == "started"
        assert obj["total"] == 100

    def test_jsonl_multiple_events(self):
        with self._display() as (d, buf):
            d.render(ProgressEvent(stage="a", status="started"))
            d.render(ProgressEvent(stage="a", status="completed"))
        lines = [json.loads(l) for l in buf.getvalue().strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["status"] == "started"
        assert lines[1]["status"] == "completed"

    def test_jsonl_non_ascii(self):
        with self._display() as (d, buf):
            d.render(ProgressEvent(stage="test", status="started", message="中文"))
        line = buf.getvalue().strip()
        obj = json.loads(line)
        assert obj["message"] == "中文"


# ---------------------------------------------------------------------------
# ProgressDisplay — mode detection
# ---------------------------------------------------------------------------


class TestProgressDisplayMode:
    def test_auto_tty(self):
        with patch.object(sys.stdout, "isatty", return_value=True):
            d = ProgressDisplay(mode="auto")
        assert d._mode == "tty"

    def test_auto_line(self):
        with patch.object(sys.stdout, "isatty", return_value=False):
            d = ProgressDisplay(mode="auto")
        assert d._mode == "line"

    def test_explicit_jsonl(self):
        d = ProgressDisplay(mode="jsonl")
        assert d._mode == "jsonl"

    def test_explicit_tty(self):
        d = ProgressDisplay(mode="tty")
        assert d._mode == "tty"


# ---------------------------------------------------------------------------
# ProgressDisplay — edge cases
# ---------------------------------------------------------------------------


class TestProgressDisplayEdgeCases:
    def test_render_none_noop(self):
        buf = StringIO()
        d = ProgressDisplay(mode="line", file=buf)
        d.render(None)
        d.close()
        assert buf.getvalue() == ""

    def test_close_idempotent(self):
        d = ProgressDisplay(mode="line")
        d.close()
        d.close()  # Should not raise


# ---------------------------------------------------------------------------
# get_progress singleton
# ---------------------------------------------------------------------------


class TestGetProgress:
    def test_returns_same_instance(self):
        t1 = get_progress()
        t2 = get_progress()
        assert t1 is t2

    def test_returns_tracker(self):
        t = get_progress()
        assert isinstance(t, ProgressTracker)