"""Generate SFT training data for Tic-Tac-Toe using minimax-optimal moves.

Produces a JSONL file where each line is:
    {"prompt": "<board text>", "response": "<move>N</move>"}

Usage:
    python gen_sft_data.py --output dataset.jsonl --num-samples 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

EMPTY = game.EMPTY


def random_board(max_moves: int = 8) -> game.Board | None:
    """Generate a random intermediate board where X is to move.

    Plays random moves from empty up to max_moves plies, then returns the
    board if the game is still ongoing and it's X's turn.
    """
    board = [[EMPTY] * 3 for _ in range(3)]
    for ply in range(max_moves):
        if game.is_terminal(board):
            return None
        player = "X" if ply % 2 == 0 else "O"
        moves = game.legal_moves(board)
        move = random.choice(moves)
        board = game.apply_move(board, move, player)
    # Need: game not terminal, and it's X's turn (X count == O count)
    if game.is_terminal(board):
        return None
    flat = [cell for row in board for cell in row]
    if flat.count("X") != flat.count("O"):
        return None
    return board


def board_to_prompt(board: game.Board) -> str:
    """Build the same prompt format the inference code uses."""
    return game.format_xml_prompt(board)


def move_to_response(move: int) -> str:
    """Format the target response."""
    return f"<move>{move}</move>"


def generate_dataset(num_samples: int, seed: int = 42) -> list[dict]:
    """Generate unique (prompt, response) pairs via minimax."""
    random.seed(seed)
    seen = set()
    samples = []
    attempts = 0
    max_attempts = num_samples * 50

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        board = random_board(max_moves=8)
        if board is None:
            continue
        prompt = board_to_prompt(board)
        if prompt in seen:
            continue
        seen.add(prompt)

        best = game.best_moves(board)
        if not best:
            continue

        # Use the first minimax-optimal move (deterministic)
        move = best[0]
        samples.append({
            "prompt": prompt,
            "response": move_to_response(move),
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate Tic-Tac-Toe SFT data")
    parser.add_argument("--output", type=str, default="dataset.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    samples = generate_dataset(args.num_samples, args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} samples -> {out_path}")

    # Print a few examples
    print("\n--- Examples ---")
    for s in samples[:5]:
        print(f"prompt:\n{s['prompt']}")
        print(f"response: {s['response']}")
        print()


if __name__ == "__main__":
    main()
