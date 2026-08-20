"""Shared annotation and manifest helpers for AVE event recognition."""

from __future__ import annotations

import json
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
    labels = record.get("event_classes")
    if not isinstance(labels, list) or not labels or any(not str(label).strip() for label in labels):
        raise ValueError(f"{prefix} has no valid event_classes")
    start = float(record["start_seconds"])
    end = float(record["end_seconds"])
    if not 0 <= start < end <= CLIP_SECONDS:
        raise ValueError(f"{prefix} has invalid interval {start}-{end}")


def prompt_text(start_seconds: float, end_seconds: float) -> str:
    return (
        f"Which audiovisual events occur between {start_seconds:g} and {end_seconds:g} seconds in this clip? "
        "Use synchronized visual and audio evidence and report the concise event label list."
    )


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
