"""Dataset loader for 2048 SFT warmup data."""

from __future__ import annotations

import json
from pathlib import Path


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load SFT JSONL records with ``prompt`` and ``response`` fields."""

    del default_loader
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records