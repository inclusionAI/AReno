"""Dataset loader for AVE audiovisual event recognition."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_manifest, prompt_text  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load an event-level JSONL manifest produced by dataset_generator.py."""

    del default_loader
    return [_format_record(record) for record in load_manifest(dataset_path)]


def _format_record(record: dict) -> dict:
    labels = [str(label) for label in record["event_classes"]]
    response = json.dumps({"events": labels}, ensure_ascii=False)
    return {
        **record,
        "prompt": prompt_text(float(record["start_seconds"]), float(record["end_seconds"])),
        "response": response,
        "reference": labels,
        "solutions": [response],
    }
