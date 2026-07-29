"""Dataset loader for the maze agentic example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL mazes and convert them to AReno prompt records."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "mazes.jsonl"
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
    state = game.deserialize_maze(raw)
    return {
        "id": raw.get("id", f"maze-{index:05d}"),
        "prompt": game.format_prompt(state),
        "maze": raw["maze"],
        "width": raw["width"],
        "height": raw["height"],
        "start": raw["start"],
        "goal": raw["goal"],
        "keys": raw.get("keys", []),
        "doors": raw.get("doors", []),
        "vision_radius": raw.get("vision_radius", 1),
        "max_steps": raw["max_steps"],
        "shortest_path_len": raw.get("shortest_path_len", 0),
    }