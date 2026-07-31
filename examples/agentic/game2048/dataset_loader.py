"""Dataset loader for the 2048 agentic example.

Bridges the JSONL board file and AReno's training pipeline:
  - load_training_dataset(): entry point called by AReno trainer
  - Reads board records from JSONL, converts each to a prompt-style dict
    with system prompt, user message (rendered board), and tool schema
  - If the JSONL file is missing, falls back to dataset_generator to
    create boards on-the-fly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL initial boards and convert them to Areno prompt records."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "boards.jsonl"
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
    board = game.normalize_board(raw["board"])
    return {
        "id": raw.get("id", f"game2048-{index:05d}"),
        "seed": int(raw.get("seed", index)),
        "board": board,
        "prompt": game.format_prompt(board),
        "max_moves": int(raw.get("max_moves", game.DEFAULT_MAX_MOVES)),
    }