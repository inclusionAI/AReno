"""Generate reachable 6x6 Othello boards for the agentic example."""

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


def generate_records(count: int = DEFAULT_COUNT, *, seed: int = DEFAULT_SEED) -> list[dict]:
    """Generate reproducible reachable boards where B is to move."""

    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not generate enough unique Othello boards")
        board = _random_board(rng)
        key = tuple(tuple(row) for row in board)
        if key in seen or game.is_terminal(board) or game.next_player(board) != "B":
            continue
        if not game.has_legal_move(board, "B"):
            continue
        seen.add(key)
        records.append(
            {
                "id": f"othello-{len(records):05d}",
                "board": board,
                "player": "B",
            }
        )
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _random_board(rng: random.Random) -> game.Board:
    """Play random legal moves from the initial board to get a reachable position."""

    board = game.initial_board()
    player = "B"
    num_moves = rng.randint(0, 8)
    for _ in range(num_moves):
        if game.is_terminal(board):
            break
        if not game.has_legal_move(board, player):
            player = "W" if player == "B" else "B"
            if not game.has_legal_move(board, player):
                break
        moves = game.legal_moves(board, player)
        move = rng.choice(moves)
        board = game.apply_move(board, move[0], move[1], player)
        player = "W" if player == "B" else "B"
    # Ensure it is B's turn: if W is to move, play one more W move
    if game.next_player(board) == "W" and game.has_legal_move(board, "W"):
        moves = game.legal_moves(board, "W")
        move = rng.choice(moves)
        board = game.apply_move(board, move[0], move[1], "W")
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL boards for the Areno 6x6 Othello agentic example.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of boards to generate.")
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
