"""Dataset loader for the multi-tool agentic example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize JSONL rows into prompt-bearing records."""

    rows = default_loader(dataset_path)
    records = []
    for row in rows:
        record = dict(row)
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records