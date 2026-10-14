"""Atomic state tracking for restartable world-model VLA workflows."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areno.experimental.world_model_vla.config import WorldModelVLAConfig


class WorkflowStateStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir) / "workflow_state.json"
        self.lock_path = self.path.with_suffix(".lock")

    def initial_state(self, config: WorldModelVLAConfig) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backend": "rlinf_wan",
            "stage_ends": list(config.stage_ends),
            "jobs": {},
            "stages": {},
            "evaluation": {"status": "not_submitted"},
        }

    def ensure(self, config: WorldModelVLAConfig) -> dict[str, Any]:
        config.output_dir.mkdir(parents=True, exist_ok=True)

        def validate_or_initialize(state: dict[str, Any]) -> dict[str, Any]:
            if not state:
                return self.initial_state(config)
            if state.get("schema_version") != 1 or state.get("backend") != "rlinf_wan":
                raise RuntimeError(f"incompatible workflow state: {self.path}")
            if state.get("stage_ends") != list(config.stage_ends):
                raise RuntimeError(f"configured stage_ends do not match existing workflow state: {self.path}")
            return state

        return self.update(validate_or_initialize)

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = self.read()
            updated = transform(state)
            temporary = self.path.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(self.path)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return updated


__all__ = ["WorkflowStateStore"]
