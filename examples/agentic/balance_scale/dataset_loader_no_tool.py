"""Dataset loader for the balance-scale XML no-tool example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import _format_record, _load_records  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL puzzles and format XML action prompts."""

    del default_loader
    return [_format_record_xml(raw) for raw in _load_records(dataset_path)]


def _format_record_xml(raw: dict) -> dict:
    """Convert one raw JSONL record into an Areno training record with an XML prompt."""

    record = _format_record(raw)
    record["prompt"] = game.format_xml_prompt(
        int(raw["num_balls"]), int(raw["max_weighings"])
    )
    return record