"""Generate SFT warmup data for 2048 tool-call training.

For each board, generates valid tool-call responses using a simple random-search
heuristic. The responses are NOT optimal — they just need to be valid format
and better than random, so the model learns the <tool_call> format before GRPO.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def generate(records: list[dict], num_examples: int = 200, rng_seed: int = 42) -> list[dict]:
    """Generate SFT examples from board records."""

    rng = random.Random(rng_seed)
    examples: list[dict] = []
    per_board = max(1, num_examples // max(len(records), 1))

    for rec in records:
        board = game.normalize_board(rec["board"])
        seed_base = int(rec["seed"])
        for j in range(per_board):
            seed = seed_base + j * 997
            moves = _best_of_random(board, seed, rng, trials=8)
            if not moves:
                moves = [rng.choice(list(game.ACTIONS))]
            response = (
                '<tool_call>'
                + json.dumps({"name": "choose_moves", "arguments": {"moves": moves}}, ensure_ascii=False)
                + '</tool_call>'
            )
            examples.append({"prompt": game.format_prompt(board), "response": response})
            if len(examples) >= num_examples:
                return examples
    return examples


def _best_of_random(board: game.Board, seed: int, rng: random.Random, trials: int) -> list[str]:
    """Pick the best of several random move sequences."""

    best_moves: list[str] = []
    best_score = float("-inf")
    for t in range(trials):
        trial_rng = random.Random(seed + t * 1009)
        moves = _random_moves(board, trial_rng)
        result = game.play_episode(board, moves, seed=seed + t * 2003)
        score = game.score_episode(result)
        if score > best_score:
            best_score = score
            best_moves = moves
    return best_moves


def _random_moves(board: game.Board, rng: random.Random) -> list[str]:
    """Generate 1–8 random moves, preferring legal directions."""

    num = rng.randint(1, 8)
    moves: list[str] = []
    current = board
    for _ in range(num):
        legal = game.legal_moves(current)
        if not legal:
            break
        move = rng.choice(legal)
        moves.append(move)
        current, _, _ = game.slide(current, move)
    return moves


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT data for 2048 tool-call warmup")
    parser.add_argument("--input", required=True, help="Path to boards JSONL")
    parser.add_argument("--output", required=True, help="Path to write SFT JSONL")
    parser.add_argument("--num-examples", type=int, default=200, help="Number of SFT examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    examples = generate(records, args.num_examples, args.seed)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} SFT examples to {args.output}")