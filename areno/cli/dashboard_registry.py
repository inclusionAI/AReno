"""Lightweight process registry consumed by the AReno dashboard and CLI.

Intentionally avoids importing areno engine, model, or accelerator modules
so that CLI commands and the dashboard agent stay fast and work without a GPU.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

GLOBAL_REGISTRY_FILE = Path.home() / ".areno" / "dashboard-jobs.json"


# ---------------------------------------------------------------------------
# Registry path
# ---------------------------------------------------------------------------


def dashboard_registry_path(cwd: str | Path | None = None) -> Path:
    return GLOBAL_REGISTRY_FILE


# ---------------------------------------------------------------------------
# Registry write (called at launch time by the CLI and SDK)
# ---------------------------------------------------------------------------


def register_dashboard_job(
    *,
    kind: str,
    name: str,
    command: list[str] | None = None,
    config: dict[str, Any] | None = None,
    metrics_dir: str | None = None,
    cwd: str | Path | None = None,
) -> None:
    item = {
        "id": uuid4().hex[:12],
        "kind": kind,
        "name": name,
        "pid": os.getpid(),
        "command": command or [],
        "config": config or {},
        "metrics_dir": metrics_dir,
        "cwd": str(Path.cwd()),
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    path = dashboard_registry_path(cwd)
    data = _read_registry_internal(path)
    jobs = [entry for entry in data.get("jobs", []) if entry.get("pid") != item["pid"]]
    jobs.append(item)
    _write_registry(path, {"jobs": jobs[-200:]})


def _read_registry_internal(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"jobs": []}


def _write_registry(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Registry read (public – used by CLI commands and dashboard server)
# ---------------------------------------------------------------------------


def read_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Return the deduplicated job list from the global dashboard registry.

    The registry is a JSON file whose top-level key ``"jobs"`` holds a list
    of per-launch dictionaries.  Entries sharing the same ``"pid"`` are
    deduplicated by keeping only the *last* occurrence (the most recent
    launch for that PID).
    """
    target = path or GLOBAL_REGISTRY_FILE
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_jobs: list[dict[str, Any]] = list(data.get("jobs", []))
    # Deduplicate by PID – keep the *last* entry for each PID.
    seen: dict[int, int] = {}
    for idx, entry in enumerate(raw_jobs):
        pid = entry.get("pid")
        if isinstance(pid, int):
            seen[pid] = idx
    deduped = [raw_jobs[idx] for idx in sorted(seen.values())]
    return deduped


# ---------------------------------------------------------------------------
# Process liveness
# ---------------------------------------------------------------------------


def pid_is_running(pid: int) -> bool:
    """Return ``True`` when the OS reports the PID as alive."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Dashboard state (per-run metadata written by MetricsRecorder)
# ---------------------------------------------------------------------------


def read_dashboard_state(metrics_dir: str | Path, pid: int) -> dict[str, Any] | None:
    """Read ``dashboard_state.<pid>.json`` inside *metrics_dir*."""
    state_file = Path(metrics_dir) / f"dashboard_state.{pid}.json"
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------


def derive_status(entry: dict[str, Any]) -> str:
    """Return a human-readable status for one registry *entry*.

    Rules (checked in order):

    1. PID alive + dashboard state has ``stage`` → use the stage name.
    2. PID alive, no stage info → ``"running"``.
    3. PID dead, state has ``status == "succeeded"`` → ``"succeeded"``.
    4. PID dead, state has ``status == "failed"`` → ``"failed"``.
    5. PID dead, state has ``status == "stopped"`` → ``"stopped"``.
    6. PID dead, no dashboard state or no explicit status → ``"exited"``.

    ``dashboard_state.{pid}.json`` is written by
    ``MetricsRecorder.record_dashboard_state`` (updates while running) and
    by the dashboard server's job watcher (sets final ``status`` on exit).
    """
    pid = entry.get("pid")
    if not isinstance(pid, int):
        return "unknown"

    alive = pid_is_running(pid)
    metrics_dir = entry.get("metrics_dir", "")
    state = read_dashboard_state(metrics_dir, pid) if metrics_dir else None

    if alive:
        stage = (state or {}).get("stage", "")
        return stage if stage else "running"

    # PID is dead — use the dashboard-state status field (written by the
    # dashboard server's _watch handler on process exit).
    if state is not None:
        st = state.get("status", "")
        if st in {"succeeded", "failed", "stopped"}:
            return st
        return "exited"

    return "exited"


# ---------------------------------------------------------------------------
# Human-readable age
# ---------------------------------------------------------------------------


def compute_age(created_at: float, now: float | None = None) -> str:
    """Return a compact age string like ``"3m 12s ago"``."""
    if now is None:
        now = time.time()
    seconds = max(0.0, now - created_at)
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m ago"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    return f"{days}d {hours}h ago"


# ---------------------------------------------------------------------------
# Duration (wall-clock span of the run)
# ---------------------------------------------------------------------------


def compute_duration(
    created_at: float,
    updated_at: float | None = None,
    now: float | None = None,
) -> str:
    """Return a compact duration string like ``"2m 34s"``.

    *created_at* is the Unix timestamp when the run was registered.

    *updated_at* is the last modification timestamp written to the registry
    by ``register_dashboard_job`` or ``scan_registered_jobs``.  Because it
    is initialised to ``created_at`` at registration time and may never be
    updated, values not strictly greater than ``created_at`` are treated as
    missing — the duration is measured up to *now* (default ``time.time()``).

    For still-running jobs *updated_at* is typically ``None`` or equal to
    ``created_at``, so the duration reflects wall-clock time since launch.
    """
    if now is None:
        now = time.time()
    if updated_at is not None and updated_at > created_at:
        end = updated_at
    else:
        end = now
    seconds = max(0.0, end - created_at)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    if seconds < 86400:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    return f"{days}d {hours}h"


# ---------------------------------------------------------------------------
# Table formatter (no external table library)
# ---------------------------------------------------------------------------


def format_table(columns: list[str], rows: list[list[str]]) -> str:
    """Return a plain-text table with space-aligned columns."""
    if not rows:
        return ""
    ncols = len(columns)
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            if i >= ncols:
                break
            widths[i] = max(widths[i], len(cell))
    header = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    lines = [header]
    for row in rows:
        padded = []
        for i, cell in enumerate(row):
            if i >= ncols:
                break
            padded.append(cell.ljust(widths[i]))
        lines.append("  ".join(padded))
    return "\n".join(lines)