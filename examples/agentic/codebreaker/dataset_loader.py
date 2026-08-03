"""Dataset loader for Codebreaker agentic training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import DEFAULT_CODE_LENGTH, DEFAULT_MAX_GUESSES, make_prompt, normalize_code  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize records while remaining tokenizer and processor independent."""

    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        record["code_length"] = int(record.get("code_length", DEFAULT_CODE_LENGTH))
        if record["code_length"] != DEFAULT_CODE_LENGTH:
            raise ValueError(f"Codebreaker code_length must be {DEFAULT_CODE_LENGTH}")
        record["max_guesses"] = min(max(int(record.get("max_guesses", DEFAULT_MAX_GUESSES)), 1), 6)
        record["secret"] = normalize_code(record["secret"], code_length=record["code_length"])
        record["id"] = str(record.get("id", f"codebreaker-{index:05d}"))
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records
