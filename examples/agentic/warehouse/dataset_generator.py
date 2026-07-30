"""Generate JSONL warehouse-picking tasks for the agentic RL example."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import DIFFICULTY_CONFIGS, generate_layout, generate_order  # noqa: E402


def generate_records(count: int, *, seed: int = 2026) -> list[dict]:
    """Deterministically generate warehouse picking task records."""

    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    difficulties = list(DIFFICULTY_CONFIGS)
    records: list[dict] = []
    for idx in range(count):
        difficulty = difficulties[idx % len(difficulties)]
        task = {
            "difficulty": difficulty,
            **DIFFICULTY_CONFIGS[difficulty],
        }
        task["sku_pool"] = list(task["sku_pool"])
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
        task["order"] = generate_order(layout["shelves"], task["order_size"], record_rng, exclude_shelf=task["start_shelf"])
        records.append(task)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL tasks for the Areno warehouse agentic example.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--count", type=int, default=256, help="Number of records.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    try:
        records = generate_records(args.count, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "ok",
        "output": str(output),
        "count": len(records),
        "difficulties": dict(Counter(record["difficulty"] for record in records)),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
