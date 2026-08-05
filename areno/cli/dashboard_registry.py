"""Lightweight process registry consumed by the AReno dashboard."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

GLOBAL_REGISTRY_FILE = Path.home() / ".areno" / "dashboard-jobs.json"


def dashboard_registry_path(cwd: str | Path | None = None) -> Path:
    return GLOBAL_REGISTRY_FILE


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
        "command": command or sys.argv,
        "config": config or {},
        "metrics_dir": metrics_dir,
        "cwd": str(Path.cwd()),
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    path = dashboard_registry_path(cwd)
    data = _read_registry(path)
    jobs = [entry for entry in data.get("jobs", []) if entry.get("pid") != item["pid"]]
    jobs.append(item)
    _write_registry(path, {"jobs": jobs[-200:]})


def _read_registry(path: Path) -> dict[str, Any]:
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
