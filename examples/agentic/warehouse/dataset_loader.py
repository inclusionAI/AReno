"""Dataset loader for the warehouse-picking agentic RL example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import DIFFICULTY_CONFIGS, build_state, make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize JSONL warehouse rows into prompt-bearing records."""

    rows = default_loader(dataset_path)
    records: list[dict] = []
    for index, row in enumerate(rows):
        record = dict(row)
        difficulty = record.get("difficulty")
        defaults = DIFFICULTY_CONFIGS.get(str(difficulty), {})
        for key, value in defaults.items():
            if key not in record:
                record[key] = list(value) if isinstance(value, list) else value
        record["sku_pool"] = [str(s) for s in record.get("sku_pool", [])]
        record["start_shelf"] = str(record.get("start_shelf", "A1"))
        try:
            build_state(record)
        except ValueError as exc:
            raise ValueError(f"warehouse dataset row {index}: {exc}") from exc
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records
