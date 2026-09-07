"""Dataset loader for the balance-scale agentic example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL puzzles and convert them to Areno prompt records."""

    # AReno passes a default dataset loader we do not need for this example.
    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw) for raw in records]


def _load_records(dataset_path: str) -> list[dict]:
    """Read raw puzzle records from a JSONL file; fall back to the generator when the file is missing."""

    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "puzzles.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict) -> dict:
    """Convert one raw JSONL record into an Areno training record with a prompt."""

    num_balls = int(raw["num_balls"])
    max_weighings = int(raw["max_weighings"])
    return {
        "id": raw.get("id", f"puzzle-{raw['odd_ball_index']:05d}"),
        "prompt": game.format_prompt(num_balls, max_weighings),
        "num_balls": num_balls,
        "odd_ball_index": int(raw["odd_ball_index"]),
        "odd_ball_direction": raw["odd_ball_direction"],
        "max_weighings": max_weighings,
    }
