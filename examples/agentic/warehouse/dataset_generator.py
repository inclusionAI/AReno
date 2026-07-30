"""Generate JSONL warehouse-picking tasks for the agentic example.

Each task record specifies a difficulty level (small/medium/hard), a
random seed for deterministic warehouse generation, and optionally an
explicit order. Run as a script to produce a JSONL file:

    python dataset_generator.py --output warehouse_tasks.jsonl --count 256
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Difficulty presets with default order sizes.
DIFFICULTY_PRESETS = {
    "small": {"order_size": 1, "seed_range": (1, 1000)},
    "medium": {"order_size": 2, "seed_range": (1, 1000)},
    "hard": {"order_size": 3, "seed_range": (1, 1000)},
}


def generate_records(
    count: int = 256,
    *,
    seed: int = 2026,
    difficulties: list[str] | None = None,
) -> list[dict]:
    """Generate deterministic warehouse-picking task records.

    Records are distributed across difficulty levels. Each record has a
    unique seed so the warehouse is deterministic and reproducible.
    """

    if difficulties is None:
        difficulties = ["small", "medium", "hard"]

    rng = random.Random(seed)
    records = []

    for idx in range(count):
        difficulty = difficulties[idx % len(difficulties)]
        preset = DIFFICULTY_PRESETS[difficulty]
        task_seed = rng.randint(preset["seed_range"][0], preset["seed_range"][1])

        record = {
            "id": f"warehouse-{idx:05d}",
            "difficulty": difficulty,
            "seed": task_seed,
        }
        records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL tasks for the Areno warehouse-picking agentic example."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--count", type=int, default=256, help="Number of records.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for generation.")
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()