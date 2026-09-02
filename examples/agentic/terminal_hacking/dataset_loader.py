"""Dataset loader for the terminal-hacking agentic example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import make_filter_prompt, normalize_record  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        try:
            record = normalize_record(dict(row))
            record["prompt"] = make_filter_prompt(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid terminal-hacking record {index}: {exc}") from exc
        records.append(record)
    return records
