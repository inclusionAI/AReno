"""Metric extraction for RLinf rich-text logs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STEP = re.compile(r"Global Step:\s*(\d+)\s*/\s*(\d+)")
_SUCCESS = re.compile(r"(?:^|[\s│])success_once=([-+0-9.eE]+)")


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    source: Path
    success_once: float | None
    global_step: int | None
    total_steps: int | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source"] = str(self.source)
        return result


def parse_metric_log(path: str | Path) -> MetricSnapshot:
    source = Path(path)
    if not source.is_file():
        return MetricSnapshot(source, None, None, None)
    text = _ANSI.sub("", source.read_text(encoding="utf-8", errors="replace"))
    steps = _STEP.findall(text)
    successes = _SUCCESS.findall(text)
    global_step = total_steps = None
    if steps:
        global_step, total_steps = (int(value) for value in steps[-1])
    success_once = float(successes[-1]) if successes else None
    return MetricSnapshot(source, success_once, global_step, total_steps)


__all__ = ["MetricSnapshot", "parse_metric_log"]
