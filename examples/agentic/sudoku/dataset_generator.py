"""Generate reproducible uniquely-solvable Sudoku puzzles as JSONL.

Each record stores the initial puzzle (agent-visible board, 0 = empty), the
difficulty band, the seed used, and the action budget. The solution is **not**
stored: grading uses the visible-board invariant (filled + no conflict), so
the answer never needs to leave the generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sudoku  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026
# Curriculum order: records are emitted in this difficulty order (tutorial
# first, extreme last). AReno's trainer consumes the dataset list sequentially,
# so an ordered dataset becomes an automatic easy->hard curriculum in pass 1.
# `tutorial` (~15 empty cells) gives short episodes so the RL loop can be
# verified quickly on limited GPUs before scaling to fuller boards.
DEFAULT_DIFFICULTIES = "tutorial,easy,medium,hard,extreme"
DIFFICULTY_ORDER: tuple[str, ...] = ("tutorial", "easy", "medium", "hard", "extreme")


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    difficulties: str = DEFAULT_DIFFICULTIES,
    action_budget: int = sudoku.DEFAULT_ACTION_BUDGET,
) -> list[dict]:
    """Generate reproducible puzzle records across the requested difficulties."""

    bands = _parse_difficulties(difficulties)
    records: list[dict] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    base_rng_seed = seed
    idx = 0
    attempts = 0
    per_band = max(1, count // len(bands))
    for band in bands:
        produced = 0
        while produced < per_band and len(records) < count:
            attempts += 1
            if attempts > count * 50:
                raise RuntimeError("could not generate enough unique Sudoku puzzles")
            env_seed = base_rng_seed + idx
            env = sudoku.SudokuEnv.generate(difficulty=band, seed=env_seed, action_budget=action_budget)
            key = tuple(tuple(row) for row in env.puzzle)
            if key in seen:
                idx += 1
                continue
            seen.add(key)
            records.append(
                {
                    "id": f"generated-{idx:05d}",
                    "difficulty": band,
                    "seed": env_seed,
                    "action_budget": action_budget,
                    "puzzle": [row[:] for row in env.puzzle],
                }
            )
            produced += 1
            idx += 1
    # Re-sort into canonical curriculum order (easy -> extreme) regardless of
    # the order bands were requested in, so pass-1 consumption is always a
    # clean easy->hard curriculum. Stable on (difficulty, seed).
    order = {band: i for i, band in enumerate(DIFFICULTY_ORDER)}
    records.sort(key=lambda rec: (order.get(rec["difficulty"], len(order)), rec["seed"]))
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _parse_difficulties(difficulties: str) -> list[str]:
    bands = [d.strip().lower() for d in difficulties.split(",") if d.strip()]
    if not bands:
        bands = [sudoku.DEFAULT_DIFFICULTY]
    for band in bands:
        if band not in sudoku.DIFFICULTY_CLUES:
            raise ValueError(f"unknown difficulty {band!r}; choose from {sorted(sudoku.DIFFICULTY_CLUES)}")
    return bands


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL Sudoku puzzles for the AReno agentic example.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of puzzles to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument(
        "--difficulties",
        default=DEFAULT_DIFFICULTIES,
        help="Comma-separated difficulty bands, e.g. 'easy,medium,hard,extreme'.",
    )
    parser.add_argument(
        "--action-budget",
        type=int,
        default=sudoku.DEFAULT_ACTION_BUDGET,
        help="Per-episode action budget.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    records = generate_records(
        args.count,
        seed=args.seed,
        difficulties=args.difficulties,
        action_budget=args.action_budget,
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