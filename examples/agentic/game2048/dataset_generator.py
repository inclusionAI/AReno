"""Generate reproducible 2048 mid-game boards for agentic training.

Produces boards with a specified max tile (64/128/256/512) and layout
pattern (corner or scattered), so the agent learns from diverse mid-game
positions instead of always starting from a fresh board.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026

# (max_tile, max_moves) mapping: bigger tiles → board is fuller → fewer moves needed
TIER_CONFIG = [
    (64, 50),
    (128, 40),
    (256, 35),
    (512, 30),
]


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Return deterministic mid-game boards across tile tiers and patterns.

    Distribution: 75% corner-locked, 25% scattered.
    Each tier gets count // len(TIER_CONFIG) records.
    """

    per_tier = max(count // len(TIER_CONFIG), 1)
    records: list[dict] = []
    index = 0

    for max_tile, max_moves in TIER_CONFIG:
        corner_n = int(per_tier * 0.75)
        scattered_n = per_tier - corner_n
        for pattern, n in [("corner", corner_n), ("scattered", scattered_n)]:
            for _ in range(n):
                if len(records) >= count:
                    break
                board_seed = seed + index
                board = game.generate_board(max_tile, pattern, board_seed)
                records.append(
                    {
                        "id": f"game2048-{index + 1:05d}",
                        "seed": board_seed,
                        "board": board,
                        "max_moves": max_moves,
                    }
                )
                index += 1

    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL mid-game boards for the Areno 2048 agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of boards to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
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