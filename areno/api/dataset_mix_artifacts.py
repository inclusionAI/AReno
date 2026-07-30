"""Structured, sample-free artifacts for dataset-mixing plans."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_dataset_mix_plan(
    summary: Mapping[str, Any],
    metrics_log_dir: str | None,
    *,
    process_id: int | None = None,
) -> Path | None:
    """Write one deterministic plan artifact for the summary's epoch."""

    if not metrics_log_dir:
        return None

    epoch = summary.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("dataset mix summary epoch must be a non-negative integer")

    path = Path(metrics_log_dir)
    path.mkdir(parents=True, exist_ok=True)
    pid = os.getpid() if process_id is None else process_id
    artifact_path = path / f"dataset_mix_plan.{pid}.epoch-{epoch}.json"
    artifact_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact_path


__all__ = ["write_dataset_mix_plan"]
