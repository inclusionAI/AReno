"""Dataset loader for Wordle agentic training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import DEFAULT_MAX_GUESSES, WORDLE_LENGTH, make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize records while remaining tokenizer independent."""

    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        record["word_length"] = int(record.get("word_length", WORDLE_LENGTH))
        record["max_guesses"] = min(
            max(int(record.get("max_guesses", DEFAULT_MAX_GUESSES)), 1), 6
        )
        record["secret"] = str(record["secret"]).lower()
        record["id"] = str(record.get("id", f"wordle-{index:05d}"))
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records

