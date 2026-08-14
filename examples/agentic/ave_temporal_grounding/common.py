"""Shared annotation, manifest, and scoring helpers for AVE temporal grounding."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLIP_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class EventAnnotation:
    event_class: str
    video_id: str
    quality: str
    start_seconds: float
    end_seconds: float

    @property
    def key(self) -> tuple[str, str, float, float]:
        return (self.event_class, self.video_id, self.start_seconds, self.end_seconds)


def read_annotations(path: str | Path, *, has_header: bool) -> list[EventAnnotation]:
    """Read ampersand-delimited AVE annotations."""

    rows = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if has_header:
        if not rows or rows[0].strip() != "Category&VideoID&Quality&StartTime&EndTime":
            raise ValueError(f"unexpected AVE annotation header in {path}")
        rows = rows[1:]
    result: list[EventAnnotation] = []
    for line_number, line in enumerate(rows, start=2 if has_header else 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("&")]
        if len(fields) != 5:
            raise ValueError(f"invalid AVE annotation row {line_number}: expected 5 fields")
        event_class, video_id, quality, start, end = fields
        annotation = EventAnnotation(event_class, video_id, quality, float(start), float(end))
        if annotation.start_seconds == annotation.end_seconds:
            continue
        if not 0 <= annotation.start_seconds < annotation.end_seconds <= CLIP_SECONDS:
            raise ValueError(f"invalid AVE interval at row {line_number}: {start}-{end}")
        result.append(annotation)
    return result


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            root = Path(str(record.get("dataset_root") or manifest.parent)).expanduser()
            for field in ("video_path", "audio_path"):
                media = Path(str(record[field])).expanduser()
                record[field] = str((media if media.is_absolute() else root / media).resolve())
            validate_record(record, line_number=line_number)
            records.append(record)
    return records


def validate_record(record: dict[str, Any], *, line_number: int | None = None) -> None:
    prefix = f"manifest row {line_number}" if line_number is not None else "record"
    if not str(record.get("event_class", "")).strip():
        raise ValueError(f"{prefix} has no event_class")
    start = float(record["start_seconds"])
    end = float(record["end_seconds"])
    if not 0 <= start < end <= CLIP_SECONDS:
        raise ValueError(f"{prefix} has invalid interval {start}-{end}")


def prompt_text(event_class: str) -> str:
    return (
        f'Locate the audiovisual event "{event_class}" in this 10-second clip. '
        "Use both visible action and synchronized sound, then report its start and end times in seconds."
    )


def timestamp_reward(predicted_start: Any, predicted_end: Any, expected_start: Any, expected_end: Any) -> float:
    """Score temporal IoU and boundary accuracy with a strict dense curve."""

    try:
        start = float(predicted_start)
        end = float(predicted_end)
        target_start = float(expected_start)
        target_end = float(expected_end)
    except (TypeError, ValueError):
        return -1.0
    if not all(math.isfinite(value) for value in (start, end, target_start, target_end)):
        return -1.0
    if start < 0 or end > CLIP_SECONDS or start >= end:
        return -1.0

    intersection = max(0.0, min(end, target_end) - max(start, target_start))
    union = max(end, target_end) - min(start, target_start)
    temporal_iou = intersection / union if union > 0 else 0.0
    boundary_error = (abs(start - target_start) + abs(end - target_end)) / (2 * CLIP_SECONDS)
    boundary_score = max(0.0, 1.0 - boundary_error)
    quality = 0.75 * temporal_iou**2 + 0.25 * boundary_score**2
    return round(2.0 * quality - 1.0, 6)


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
