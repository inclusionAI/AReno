"""Generate 6x6 Othello openings for the agentic example.

Produces deterministic, *reachable* opening positions by playing only legal
moves from the standard opening. Never emits arbitrary/invalid boards.
"""

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
DEFAULT_MAX_PLIES = 8


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    max_plies: int = DEFAULT_MAX_PLIES,
) -> list[dict]:
    """Generate reproducible reachable opening positions where Black is to move.

    Each board is built by playing a random *legal* move sequence from the
    standard opening, alternating players, for a small random number of plies.
    The resulting board is kept only if the game is non-terminal and it is
    Black's turn; otherwise more attempts are tried.
    """

    if count <= 0:
        raise ValueError("--count must be positive")
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    if max_plies < 0:
        raise ValueError("--max-plies must be non-negative")

    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not generate enough unique Othello openings")
        board = _random_opening(rng, max_plies)
        if game.is_terminal(board):
            continue
        if game.next_player(board) != "B":
            continue
        key = tuple(tuple(row) for row in board)
        if key in seen:
            continue
        seen.add(key)
        records.append({"id": f"generated-{len(records):05d}", "board": board})
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _random_opening(rng: random.Random, max_plies: int) -> game.Board:
    """Play a random legal-move prefix of an Othello game from the opening."""

    board = game.new_board()
    player = "B"
    plies = rng.randint(0, max_plies)
    for _ in range(plies):
        moves = game.legal_moves(board, player)
        if not moves:
            # Forced pass: the other side keeps the turn.
            player = game.opponent(player)
            moves = game.legal_moves(board, player)
            if not moves:
                break
        move = rng.choice(moves)
        board = game.apply_move(board, move[0], move[1], player)
        player = game.opponent(player)
    return board


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL boards for the Areno 6x6 Othello agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of boards to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES, help="Max opening plies.")
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed, max_plies=args.max_plies)
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()