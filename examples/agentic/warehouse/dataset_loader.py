"""Dataset loader for the warehouse-picking agentic RL example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import make_prompt  # noqa: E402

_TASK_DEFAULTS = {
    "small": {
        "rows": 2,
        "cols": 2,
        "sku_pool": ["S1", "S2", "S3", "S4"],
        "max_stock": 5,
        "order_size": 2,
        "stockout": False,
    },
    "medium": {
        "rows": 3,
        "cols": 3,
        "sku_pool": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "max_stock": 4,
        "order_size": 3,
        "stockout": False,
    },
    "hard": {
        "rows": 4,
        "cols": 3,
        "sku_pool": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        "max_stock": 3,
        "order_size": 5,
        "stockout": True,
    },
}


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize JSONL warehouse rows into prompt-bearing records."""

    rows = default_loader(dataset_path)
    records: list[dict] = []
    for row in rows:
        record = dict(row)
        defaults = _TASK_DEFAULTS.get(str(record.get("difficulty", "")), {})
        for key, value in defaults.items():
            if key not in record:
                record[key] = value
        record["sku_pool"] = [str(s) for s in record.get("sku_pool", [])]
        record["start_shelf"] = str(record.get("start_shelf", "A1"))
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records