"""Dataset loader for the Countdown arithmetic tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL puzzles and convert them to Areno prompt records."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "countdown.jsonl"
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
    numbers = game.normalize_numbers(raw["numbers"])
    target = int(raw["target"])
    return {
        "id": raw.get("id", f"countdown-{index:05d}"),
        "prompt": game.format_prompt(numbers, target),
        "numbers": numbers,
        "target": target,
        "best_score": game.best_score(numbers, target),
    }