"""Generate odd-ball balance-scale puzzles for the agentic example."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026
DEFAULT_NUM_BALLS = 12
DEFAULT_MAX_WEIGHINGS = 0  # 0 = auto (2x information-theoretic minimum)


def _auto_max_weighings(num_balls: int) -> int:
    """Compute soft upper bound: 2x information-theoretic minimum."""

    min_w = max(1, math.ceil(math.log(num_balls * 2, 3)))
    return min_w * 2


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    num_balls: int = DEFAULT_NUM_BALLS,
    max_weighings: int = DEFAULT_MAX_WEIGHINGS,
    num_balls_range: tuple[int, int] | None = None,
) -> list[dict]:
    """Generate reproducible odd-ball puzzle records.

    Each record contains ``num_balls``, ``odd_ball_index``, ``direction``
    (``"heavier"`` or ``"lighter"``), and ``max_weighings``. When
    ``max_weighings`` is 0, it is set to 2x the information-theoretic
    minimum (ceil(log3(num_balls*2))), which is a soft upper bound to
    prevent infinite loops while allowing the agent flexibility.

    When ``num_balls_range`` is provided (e.g. ``(3, 12)``), each puzzle
    is generated with a random number of balls within that range. This
    improves generalisation by preventing the model from memorising a
    single ball count.
    """

    rng = random.Random(seed)
    records: list[dict] = []

    for i in range(count):
        # Determine num_balls for this record
        if num_balls_range is not None:
            nb = rng.randint(num_balls_range[0], num_balls_range[1])
        else:
            nb = num_balls

        odd_ball_index = rng.randint(0, nb - 1)
        direction = rng.choice(game.DIRECTIONS)

        # Compute max_weighings for this record
        if max_weighings <= 0:
            mw = _auto_max_weighings(nb)
        else:
            mw = max_weighings

        records.append(
            {
                "id": f"generated-{i:05d}",
                "num_balls": nb,
                "odd_ball_index": odd_ball_index,
                "direction": direction,
                "max_weighings": mw,
            }
        )
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def split_records(records: list[dict], test_ratio: float) -> tuple[list[dict], list[dict]]:
    """Split records into train and test sets.

    Returns (train_records, test_records).
    """

    n_test = max(1, int(len(records) * test_ratio))
    return records[:-n_test], records[-n_test:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL puzzles for the Areno odd-ball balance-scale agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of puzzles to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--num-balls", type=int, default=DEFAULT_NUM_BALLS, help="Number of balls per puzzle (fixed).")
    parser.add_argument(
        "--num-balls-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="Random number of balls per puzzle within [MIN, MAX]. Overrides --num-balls.",
    )
    parser.add_argument(
        "--max-weighings", type=int, default=DEFAULT_MAX_WEIGHINGS, help="Maximum weighings allowed (0 = auto)."
    )
    parser.add_argument(
        "--split", type=float, default=0.0, metavar="RATIO",
        help="Split into train/test sets. RATIO is the test fraction (e.g. 0.33). "
             "Generates <output> (train) and <outputstem>_test.jsonl (test).",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.max_weighings < 0:
        raise ValueError("--max-weighings must be >= 0 (0 = auto)")

    # Determine num_balls configuration
    num_balls_range = None
    num_balls = args.num_balls
    if args.num_balls_range is not None:
        lo, hi = args.num_balls_range
        if lo < 2 or hi < lo:
            raise ValueError("--num-balls-range must be MIN>=2 and MAX>=MIN")
        num_balls_range = (lo, hi)
    elif args.num_balls < 2:
        raise ValueError("--num-balls must be at least 2")

    records = generate_records(
        args.count,
        seed=args.seed,
        num_balls=num_balls,
        max_weighings=args.max_weighings,
        num_balls_range=num_balls_range,
    )

    if args.split > 0:
        train_records, test_records = split_records(records, args.split)
        if args.output == "-":
            print("=== TRAIN ===")
            write_jsonl(train_records, sys.stdout)
            print("\n=== TEST ===")
            write_jsonl(test_records, sys.stdout)
        else:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            test_path = output_path.parent / f"{output_path.stem}_test{output_path.suffix}"
            with output_path.open("w", encoding="utf-8") as handle:
                write_jsonl(train_records, handle)
            with test_path.open("w", encoding="utf-8") as handle:
                write_jsonl(test_records, handle)
            print(f"Train: {output_path} ({len(train_records)} records)")
            print(f"Test:  {test_path} ({len(test_records)} records)")
    else:
        if args.output == "-":
            write_jsonl(records, sys.stdout)
        else:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                write_jsonl(records, handle)


if __name__ == "__main__":
    main()
