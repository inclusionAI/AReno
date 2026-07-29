"""Structured live progress for long-running skill operations.

Provides a ``ProgressTracker`` that emits typed stage events and a
``ProgressDisplay`` that renders them in three modes:

* **TTY** — ``rich.progress`` bars with spinners (auto-detected via ``isatty``)
* **Line** — ``[stage] status: message`` plain text
* **JSON Lines** — one JSON object per line (explicit opt-in for piped consumers)

``rich>=13`` and ``tqdm>=4.66`` are already project dependencies so no new
packages are required.

Usage::

    tracker = ProgressTracker()
    display = ProgressDisplay(mode="auto")

    tracker.begin_stage("rollout", total=500, message="generating completions")
    display.render(tracker.advance(234, "234/500"))

    tracker.complete_stage(message="rollout done (12.3s/step)")
    display.render(...)

    display.close()
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, TextIO


# ---------------------------------------------------------------------------
# Progress event
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProgressEvent:
    """Immutable structured event emitted by :class:`ProgressTracker`."""

    stage: str
    status: str  # "started" | "running" | "completed" | "failed" | "cancelled"
    message: str = ""
    step: int | None = None
    total: int | None = None
    elapsed_s: float = 0.0
    data: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
            "elapsed_s": self.elapsed_s,
        }
        if self.step is not None:
            d["step"] = self.step
        if self.total is not None:
            d["total"] = self.total
        if self.data:
            d["data"] = self.data
        return d


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Stage-aware event emitter with a simple stack model.

    Each ``begin_stage`` pushes onto an internal stack. ``complete_stage``
    and ``fail_stage`` pop the top of the stack. Nested stages are supported:
    when a new stage begins while another is active, the outer stage is
    paused and resumes when the inner stage completes.
    """

    def __init__(self) -> None:
        self._stack: list[dict[str, Any]] = []
        self._last_completed: str | None = None
        self._stage_start: float | None = None

    # -- public API ---------------------------------------------------------

    def begin_stage(
        self, stage: str, *, total: int | None = None, message: str = ""
    ) -> ProgressEvent:
        """Start a new stage, pushing it onto the stack."""
        self._stage_start = time.monotonic()
        entry = {"stage": stage, "total": total, "start": self._stage_start}
        self._stack.append(entry)
        return ProgressEvent(
            stage=stage,
            status="started",
            message=message,
            total=total,
        )

    def advance(
        self, step: int, message: str = "", data: dict[str, Any] | None = None
    ) -> ProgressEvent | None:
        """Report progress within the current stage."""
        if not self._stack:
            return None
        active = self._stack[-1]
        return ProgressEvent(
            stage=active["stage"],
            status="running",
            message=message,
            step=step,
            total=active.get("total"),
            elapsed_s=self._elapsed(),
            data=data,
        )

    def complete_stage(
        self, message: str = "", data: dict[str, Any] | None = None
    ) -> ProgressEvent:
        """Finish the current stage successfully and pop it from the stack."""
        if not self._stack:
            return ProgressEvent(stage="", status="completed", message=message)
        active = self._stack.pop()
        self._last_completed = active["stage"]
        return ProgressEvent(
            stage=active["stage"],
            status="completed",
            message=message,
            step=active.get("total"),
            total=active.get("total"),
            elapsed_s=self._elapsed(),
            data=data,
        )

    def fail_stage(self, error: str) -> ProgressEvent:
        """Fail the current stage, pop it, and record the error."""
        if not self._stack:
            return ProgressEvent(
                stage="",
                status="failed",
                message=error,
                data={"last_completed_stage": self._last_completed},
            )
        active = self._stack.pop()
        # Keep the last *successfully* completed stage, not the failed one.
        return ProgressEvent(
            stage=active["stage"],
            status="failed",
            message=error,
            total=active.get("total"),
            elapsed_s=self._elapsed(),
            data={"last_completed_stage": self._last_completed},
        )

    def cancel(self) -> ProgressEvent:
        """Cancel the current operation and pop all active stages."""
        last = self._last_completed
        self._stack.clear()
        return ProgressEvent(
            stage=last or "",
            status="cancelled",
            message="operation cancelled",
            data={"last_completed_stage": last},
        )

    @property
    def active_stage(self) -> str | None:
        """Name of the currently active stage, if any."""
        return self._stack[-1]["stage"] if self._stack else None

    # -- internal -----------------------------------------------------------

    def _elapsed(self) -> float:
        if self._stage_start is None:
            return 0.0
        return time.monotonic() - self._stage_start


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------


class ProgressDisplay:
    """Render :class:`ProgressEvent` objects to TTY, plain text, or JSON Lines.

    Parameters
    ----------
    mode:
        ``"auto"`` (default) detects TTY vs non-TTY automatically.
        ``"tty"`` forces rich progress bars.
        ``"line"`` forces plain-text output.
        ``"jsonl"`` outputs one JSON object per line.
    file:
        Target stream (defaults to ``sys.stdout``).
    """

    def __init__(self, mode: str = "auto", file: TextIO | None = None) -> None:
        self._mode = self._resolve_mode(mode)
        self._file = file or sys.stdout
        self._progress: Any = None
        self._task_ids: dict[str, Any] = {}

    # -- public API ---------------------------------------------------------

    def render(self, event: ProgressEvent) -> None:
        """Render a single progress event."""
        if event is None:
            return
        if self._mode == "jsonl":
            self._render_jsonl(event)
        elif self._mode == "tty":
            self._render_tty(event)
        else:
            self._render_line(event)

    def close(self) -> None:
        """Flush any residual output and reset terminal state."""
        if self._mode == "tty" and self._progress is not None:
            for task_id in list(self._task_ids.values()):
                self._progress.remove_task(task_id)
            self._progress.stop()
            self._progress = None
            self._task_ids.clear()

    # -- mode resolution ----------------------------------------------------

    @staticmethod
    def _resolve_mode(mode: str) -> str:
        if mode == "jsonl":
            return "jsonl"
        if mode == "tty":
            return "tty"
        if mode == "line":
            return "line"
        # "auto": detect
        return "tty" if sys.stdout.isatty() else "line"

    # -- renderers ----------------------------------------------------------

    def _render_jsonl(self, event: ProgressEvent) -> None:
        print(json.dumps(event.as_dict(), ensure_ascii=False), file=self._file)

    def _render_line(self, event: ProgressEvent) -> None:
        parts = [f"[{event.stage}]", event.status]
        if event.step is not None and event.total:
            parts.append(f"{event.step}/{event.total}")
        if event.message:
            parts.append(event.message)
        print(" ".join(parts), file=self._file)

    def _render_tty(self, event: ProgressEvent) -> None:
        try:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
        except ImportError:
            self._render_line(event)
            return

        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=None,
                transient=False,
                file=self._file,
            )
            self._progress.start()

        task_id = self._task_ids.get(event.stage)
        if event.status == "started":
            total = event.total or 1
            desc = event.stage
            task_id = self._progress.add_task(desc, total=total)
            self._task_ids[event.stage] = task_id
        elif event.status == "running" and task_id is not None:
            self._progress.update(task_id, completed=event.step or 0)
        elif event.status in ("completed", "failed", "cancelled") and task_id is not None:
            self._progress.update(task_id, completed=self._progress.tasks[task_id].total)
            self._progress.remove_task(task_id)
            self._task_ids.pop(event.stage, None)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


_tracker: ProgressTracker | None = None


def get_progress() -> ProgressTracker:
    """Return a module-level shared :class:`ProgressTracker` singleton.

    Skill scripts that do not want to wire a tracker through every function
    can call this to get a lazily-initialised instance.
    """
    global _tracker
    if _tracker is None:
        _tracker = ProgressTracker()
    return _tracker