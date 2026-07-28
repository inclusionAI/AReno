"""Generate reproducible Towers of Hanoi tasks for the agentic example."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = game.DEFAULT_COUNT
DEFAULT_SEED = game.DEFAULT_SEED
MIN_DISKS = game.MIN_DISKS
MAX_DISKS = game.MAX_DISKS


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    min_disks: int = MIN_DISKS,
    max_disks: int = MAX_DISKS,
) -> list[dict]:
    """Return deterministic Towers of Hanoi tasks spread across the disk range.

    Each task records the disk count ``n`` and a generous move cap derived from
    the known optimum. The optimum itself is never stored on the record so the
    reward path must independently recompute it via ``game.optimal_steps``.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    min_disks = game.validate_disks(min_disks)
    max_disks = game.validate_disks(max_disks)
    if min_disks > max_disks:
        raise ValueError("min_disks must be <= max_disks")
    rng = random.Random(seed)
    records: list[dict] = []
    for index in range(count):
        n = rng.randint(min_disks, max_disks)
        records.append(
            {
                "id": f"hanoi-{index + 1:05d}",
                "n": n,
                "max_moves": game.default_max_moves(n),
            }
        )
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL tasks for the Areno Towers of Hanoi agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of tasks to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--min-disks", type=int, default=MIN_DISKS, help="Smallest number of disks per task.")
    parser.add_argument("--max-disks", type=int, default=MAX_DISKS, help="Largest number of disks per task.")
    args = parser.parse_args()

    records = generate_records(
        args.count,
        seed=args.seed,
        min_disks=args.min_disks,
        max_disks=args.max_disks,
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
