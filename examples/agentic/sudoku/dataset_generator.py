"""Generate reproducible Sudoku puzzles for agentic training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game import DEFAULT_DIFFICULTY, DEFAULT_MAX_ACTIONS, DIFFICULTY_EMPTY, generate_puzzle

DEFAULT_COUNT = 64
DEFAULT_SEED = 2026


def generate_records(
    count: int = DEFAULT_COUNT,
    difficulty: str = DEFAULT_DIFFICULTY,
    *,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Return deterministic uniquely-solvable Sudoku puzzles."""

    records: list[dict] = []
    for i in range(count):
        puzzle = generate_puzzle(difficulty, seed=seed + i)
        records.append(
            {
                "id": f"sudoku-{i + 1:05d}",
                "puzzle": puzzle["puzzle"],
                "difficulty": difficulty,
                "max_actions": DEFAULT_MAX_ACTIONS,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--difficulty", choices=list(DIFFICULTY_EMPTY), default=DEFAULT_DIFFICULTY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    records = generate_records(args.count, args.difficulty, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()