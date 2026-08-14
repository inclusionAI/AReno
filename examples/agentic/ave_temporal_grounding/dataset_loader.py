"""Dataset loader for AVE audiovisual temporal grounding."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_manifest, prompt_text  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load an event-level JSONL manifest produced by dataset_generator.py."""

    del default_loader
    records = load_manifest(dataset_path)
    return [_format_record(record) for record in records]


def _format_record(record: dict) -> dict:
    start = float(record["start_seconds"])
    end = float(record["end_seconds"])
    response = f"{start:g}-{end:g} seconds"
    return {
        **record,
        "prompt": prompt_text(str(record["event_class"])),
        "response": response,
        "reference": response,
        "solutions": [response],
    }
