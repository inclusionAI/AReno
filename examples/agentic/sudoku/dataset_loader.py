"""Dataset loader for Sudoku agentic training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import DEFAULT_DIFFICULTY, DEFAULT_MAX_ACTIONS, make_prompt, normalize_board  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize puzzle records into Areno prompt records."""

    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        record["puzzle"] = normalize_board(record["puzzle"])
        record["difficulty"] = str(record.get("difficulty", DEFAULT_DIFFICULTY))
        record["max_actions"] = int(record.get("max_actions", DEFAULT_MAX_ACTIONS))
        record["id"] = str(record.get("id", f"sudoku-{index:05d}"))
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records