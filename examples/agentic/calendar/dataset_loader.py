"""Dataset loader for the calendar scheduling agentic example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL calendar scenarios and convert them to Areno prompt records."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "calendar.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    state = game.record_to_state(raw)
    meeting_id = raw.get("target_meeting_id", state.meetings[0].id if state.meetings else "")
    prompt = game.format_prompt(state, meeting_id)
    record = dict(raw)
    record["id"] = raw.get("id", f"calendar-{index:05d}")
    record["prompt"] = prompt
    record["target_meeting_id"] = meeting_id
    return record