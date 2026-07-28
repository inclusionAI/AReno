"""Generate JSONL warehouse-picking tasks for the agentic RL example."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import generate_layout, generate_order  # noqa: E402

TASKS = [
    {
        "difficulty": "small",
        "rows": 2,
        "cols": 2,
        "sku_pool": ["S1", "S2", "S3", "S4"],
        "max_stock": 5,
        "order_size": 2,
        "stockout": False,
    },
    {
        "difficulty": "medium",
        "rows": 3,
        "cols": 3,
        "sku_pool": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "max_stock": 4,
        "order_size": 3,
        "stockout": False,
    },
    {
        "difficulty": "hard",
        "rows": 4,
        "cols": 3,
        "sku_pool": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        "max_stock": 3,
        "order_size": 5,
        "stockout": True,
    },
]


def generate_records(count: int, *, seed: int = 2026) -> list[dict]:
    """Deterministically generate warehouse picking task records."""

    rng = random.Random(seed)
    records: list[dict] = []
    for idx in range(count):
        task = dict(rng.choice(TASKS))
        task["id"] = idx
        task["seed"] = seed + idx
        task["start_shelf"] = "A1"
        record_rng = random.Random(task["seed"])
        layout = generate_layout(
            rows=task["rows"],
            cols=task["cols"],
            sku_pool=task["sku_pool"],
            max_stock_per_sku=task["max_stock"],
            rng=record_rng,
        )
        task["order"] = generate_order(layout["shelves"], task["order_size"], record_rng)
        records.append(task)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL tasks for the Areno warehouse agentic example."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--count", type=int, default=256, help="Number of records.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()