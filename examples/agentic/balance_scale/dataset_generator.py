"""Generate reproducible JSONL puzzle records for the balance-scale example."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026
DEFAULT_NUM_BALLS = 9
DEFAULT_MAX_WEIGHINGS = 3


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    num_balls: int = DEFAULT_NUM_BALLS,
    max_weighings: int = DEFAULT_MAX_WEIGHINGS,
) -> list[dict]:
    """Generate *count* reproducible puzzle records with a seeded RNG.

    Records are not required to be unique — the same (odd_ball_index,
    odd_ball_direction) combination may appear multiple times across
    a large dataset, which is the expected behavior for RL training.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if num_balls < 3:
        raise ValueError("num_balls must be at least 3")

    rng = random.Random(seed)
    records: list[dict] = []
    for i in range(count):
        odd_index = rng.randint(0, num_balls - 1)
        direction = rng.choice(game.ODD_DIRECTIONS)
        records.append(
            {
                "id": f"puzzle-{i:05d}",
                "num_balls": num_balls,
                "odd_ball_index": odd_index,
                "odd_ball_direction": direction,
                "max_weighings": max_weighings,
            }
        )
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL puzzles for the AReno balance-scale agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of puzzles to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--num-balls", type=int, default=DEFAULT_NUM_BALLS, help="Number of balls per puzzle.")
    parser.add_argument(
        "--max-weighings", type=int, default=DEFAULT_MAX_WEIGHINGS, help="Maximum weighings per puzzle."
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.num_balls < 3:
        raise ValueError("--num-balls must be at least 3")
    if args.max_weighings < 1:
        raise ValueError("--max-weighings must be at least 1")

    records = generate_records(
        args.count,
        seed=args.seed,
        num_balls=args.num_balls,
        max_weighings=args.max_weighings,
    )
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
