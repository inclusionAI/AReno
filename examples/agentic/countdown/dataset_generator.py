"""Generate Countdown number puzzles for the agentic example."""

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

# Classic Countdown number pools (from the TV show).
LARGE_NUMBERS = [25, 50, 75, 100]
SMALL_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_NUM_COUNT = 4


def generate_records(
    count: int = DEFAULT_COUNT, *, seed: int = DEFAULT_SEED
) -> list[dict]:
    """Generate reproducible Countdown puzzles.

    Each puzzle has a list of numbers and a target. The target is derived
    from a random valid two-number operation so that at least one good
    move always exists.
    """

    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[tuple[tuple[int, ...], int]] = set()

    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not generate enough unique Countdown puzzles")

        numbers, target = _random_puzzle(rng)
        key = (tuple(sorted(numbers)), target)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "id": f"generated-{len(records):05d}",
                "numbers": numbers,
                "target": target,
            }
        )
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _random_puzzle(rng: random.Random) -> tuple[list[int], int]:
    """Generate one puzzle: numbers + a solvable target."""

    num_count = rng.randint(4, 6)

    # Pick 0-2 large numbers, rest small.
    large_count = rng.randint(0, min(2, num_count - 2))
    small_count = num_count - large_count

    numbers: list[int] = []
    large_pool = list(LARGE_NUMBERS)
    rng.shuffle(large_pool)
    numbers.extend(large_pool[:large_count])

    small_pool = list(SMALL_NUMBERS) + list(SMALL_NUMBERS)  # allow duplicates
    rng.shuffle(small_pool)
    numbers.extend(small_pool[:small_count])

    rng.shuffle(numbers)

    # Derive target from a random valid operation on two numbers.
    # This guarantees at least one move can hit the target exactly.
    idx_a, idx_b = rng.sample(range(num_count), 2)
    a, b = numbers[idx_a], numbers[idx_b]
    op = rng.choice(game.OPERATIONS)

    result = game.calculate(a, b, op)
    if result is None or result <= 0:
        # Fallback: use a + b as target (always valid and positive).
        target = a + b
    else:
        target = int(result)

    return numbers, target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL puzzles for the Areno Countdown agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of puzzles to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    records = generate_records(args.count, seed=args.seed)
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()