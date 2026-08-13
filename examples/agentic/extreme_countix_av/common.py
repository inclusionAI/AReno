"""Shared data and scoring helpers for the Extreme Countix-AV example."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Sample:
    youtube_id: str
    condition: str
    video_path: Path
    audio_path: Path
    action_class: str
    repetition_count: int
    repetition_start_frame: int
    repetition_end_frame: int
    start_crop_frame: int
    end_crop_frame: int

    def as_record(self, root: Path) -> dict[str, Any]:
        return {
            "id": f"{self.youtube_id}:{self.condition}",
            "youtube_id": self.youtube_id,
            "condition": self.condition,
            "video_path": _relative_or_absolute(self.video_path, root),
            "audio_path": _relative_or_absolute(self.audio_path, root),
            "action_class": self.action_class,
            "repetition_count": self.repetition_count,
            "repetition_start_frame": self.repetition_start_frame,
            "repetition_end_frame": self.repetition_end_frame,
            "start_crop_frame": self.start_crop_frame,
            "end_crop_frame": self.end_crop_frame,
        }


def discover_samples(dataset_root: str | Path) -> list[Sample]:
    """Index all labelled, paired audiovisual samples below ``dataset_root``."""

    root = Path(dataset_root).expanduser().resolve()
    labels = read_labels(root / "ExtremeLabels.csv")
    videos = _media_by_id(root / "Videos", ".mp4")
    audio = _media_by_id(root / "Audio", ".wav")
    samples: list[Sample] = []
    for youtube_id, label in labels.items():
        for video_path in videos.get(youtube_id, []):
            condition = video_path.parent.name
            audio_path = _matching_media(audio.get(youtube_id, []), condition)
            if audio_path is None:
                continue
            samples.append(
                Sample(
                    youtube_id=youtube_id,
                    condition=condition,
                    video_path=video_path,
                    audio_path=audio_path,
                    action_class=label["action_class"],
                    repetition_count=label["repetition_count"],
                    repetition_start_frame=label["repetition_start_frame"],
                    repetition_end_frame=label["repetition_end_frame"],
                    start_crop_frame=label["start_crop_frame"],
                    end_crop_frame=label["end_crop_frame"],
                )
            )
    return sorted(samples, key=lambda sample: (sample.condition.lower(), sample.youtube_id))


def read_labels(csv_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read the official CSV and omit its unlabelled VGGSound appendix rows."""

    labels: dict[str, dict[str, Any]] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        header = next(rows, None)
        if not header or header[:2] != ["youtube_id", "repetition_start_frame"]:
            raise ValueError(f"unexpected ExtremeLabels.csv header: {header}")
        for line_number, row in enumerate(rows, start=2):
            if len(row) == 6:
                # The official file appends unlabelled VGGSound rows. They are
                # useful media, but cannot supervise action classification.
                continue
            if len(row) != 7:
                raise ValueError(f"invalid ExtremeLabels.csv row {line_number}: expected 6 or 7 columns")
            youtube_id, rep_start, rep_end, crop_start, crop_end, repetitions, action_class = row
            action_class = action_class.strip()
            if not action_class:
                continue
            labels[youtube_id] = {
                "repetition_start_frame": int(float(rep_start)),
                "repetition_end_frame": int(float(rep_end)),
                "start_crop_frame": int(float(crop_start)),
                "end_crop_frame": int(float(crop_end)),
                "repetition_count": int(round(float(repetitions))),
                "action_class": action_class,
            }
    return labels


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            media_root = Path(str(record.get("dataset_root") or manifest.parent)).expanduser()
            for field in ("video_path", "audio_path"):
                media_path = Path(str(record[field])).expanduser()
                if not media_path.is_absolute():
                    media_path = media_root / media_path
                record[field] = str(media_path.resolve())
            if not record.get("action_class"):
                raise ValueError(f"manifest row {line_number} has no action_class")
            records.append(record)
    return records


def normalize_action(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def action_similarity(predicted: Any, expected: Any) -> float:
    """Combine token overlap and edit similarity for short action labels."""

    predicted_norm = normalize_action(predicted)
    expected_norm = normalize_action(expected)
    if not predicted_norm or not expected_norm:
        return 0.0
    if predicted_norm == expected_norm:
        return 1.0
    predicted_tokens = set(predicted_norm.split())
    expected_tokens = set(expected_norm.split())
    token_f1 = 2 * len(predicted_tokens & expected_tokens) / (len(predicted_tokens) + len(expected_tokens))
    sequence = SequenceMatcher(None, predicted_norm, expected_norm).ratio()
    return min(1.0, 0.65 * token_f1 + 0.35 * sequence)


def count_similarity(predicted: Any, expected: Any) -> float:
    try:
        predicted_count = int(round(float(predicted)))
        expected_count = int(round(float(expected)))
    except (TypeError, ValueError):
        return 0.0
    if predicted_count < 0:
        return 0.0
    error = abs(predicted_count - expected_count)
    return max(0.0, 1.0 - error / max(expected_count, 2))


def prompt_text() -> str:
    return (
        "Watch the video and listen to its audio. Identify the repeated physical activity and count the completed "
        "repetitions in the labelled activity interval. Use both modalities when one is ambiguous, then call "
        "report_repetitions exactly once."
    )


def _media_by_id(root: Path, suffix: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in root.rglob(f"*{suffix}"):
        result.setdefault(path.stem, []).append(path.resolve())
    return result


def _matching_media(paths: Iterable[Path], condition: str) -> Path | None:
    candidates = list(paths)
    return next((path for path in candidates if path.parent.name == condition), candidates[0] if candidates else None)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
