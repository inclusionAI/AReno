"""Generate seeded 2048 starting boards for the agentic example."""

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
DEFAULT_SPAWNS = 2
DEFAULT_CAP = game.DEFAULT_EPISODE_CAP
DEFAULT_TRIALS = 8


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    spawns: int = DEFAULT_SPAWNS,
    cap: int = DEFAULT_CAP,
    trials: int = DEFAULT_TRIALS,
) -> list[dict]:
    """Generate reproducible 2048 starting boards with a random baseline."""

    rng = random.Random(seed)
    records: list[dict] = []
    # Dedup on the replay seed (equivalently (board, seed)): a starting board is
    # a function of its seed, and the seed also drives episode spawns at reward
    # time, so two records sharing a board but differing in seed are distinct
    # training samples. This avoids the ~480-board ceiling that 2-tile spawns
    # would impose if we deduped on the board alone.
    seen_seeds: set[int] = set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not generate enough unique 2048 boards")
        board_seed = rng.randrange(0, 2**31)
        board = _spawn_board(board_seed, spawns)
        if board_seed in seen_seeds or game.is_terminal(board):
            continue
        seen_seeds.add(board_seed)
        baseline = game.random_episode(board, seed=board_seed, cap=cap, trials=trials)
        records.append(
            {
                "id": f"generated-{len(records):05d}",
                "board": board,
                "seed": board_seed,
                "random_baseline": baseline,
            }
        )
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _spawn_board(board_seed: int, spawns: int) -> game.Board:
    """Build a starting board by spawning ``spawns`` tiles under a seeded RNG."""

    board: game.Board = [[0 for _ in range(game.SIZE)] for _ in range(game.SIZE)]
    rng = random.Random(board_seed)
    for _ in range(spawns):
        board = game.spawn_tile(board, rng)
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL boards for the Areno 2048 agentic example.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of boards to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for board generation.")
    parser.add_argument("--spawns", type=int, default=DEFAULT_SPAWNS, help="Initial tiles to spawn per board.")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="Episode length cap for the baseline.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Random baseline trials per board.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.spawns <= 0:
        raise ValueError("--spawns must be positive")

    records = generate_records(
        args.count, seed=args.seed, spawns=args.spawns, cap=args.cap, trials=args.trials
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