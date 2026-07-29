"""Deterministic Wordle evaluation script (Issue #189).

Usage:
    python -m evaluate --dataset /tmp/areno-wordle.jsonl
    python -m evaluate --dataset /tmp/areno-wordle.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game


def evaluate_dataset(
    records: list[dict],
    *,
    strategy: str = "random",
    seed: int | None = None,
) -> list[dict]:
    """
    Evaluate a dataset by playing each game using a deterministic strategy.

    Args:
        records: List of game records (each has 'target' and 'max_guesses')
        strategy: Guessing strategy ('random' for random valid words)
        seed: Random seed for reproducibility

    Returns:
        List of result dicts with 'target', 'solved', 'guesses' fields.
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    results: list[dict] = []
    for record in records:
        target = record["target"]
        max_guesses = record.get("max_guesses", game.MAX_GUESSES)

        g = game.create_new_game(target)
        solved = False
        guesses_used = 0

        for attempt in range(max_guesses):
            guesses_used = attempt + 1

            if strategy == "random":
                # Pick a random valid word as guess
                guess = rng.choice(sorted(game.WORD_LIST))
            else:
                guess = target  # Cheat: always guess correctly for testing

            try:
                g = game.apply_guess(g, guess)
            except ValueError:
                # Invalid guess - still counts as an attempt
                continue

            if game.is_terminal(g):
                result = game.game_result(g)
                if result is True:
                    solved = True
                break

        results.append({
            "target": target,
            "solved": solved,
            "guesses": guesses_used,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Wordle game dataset and report solve statistics."
    )
    parser.add_argument(
        "--dataset", "-d",
        default="/tmp/areno-wordle.jsonl",
        help="Path to JSONL dataset file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic evaluation.",
    )
    parser.add_argument(
        "--strategy",
        default="random",
        choices=["random", "perfect"],
        help="Guessing strategy: 'random' or 'perfect'.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for structured stats JSON.",
    )
    args = parser.parse_args()

    # Validate dataset path
    is_valid, error_msg = game.validate_dataset_path(args.dataset)
    if not is_valid:
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)

    # Load records
    path = Path(args.dataset).expanduser()
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    if not records:
        print("Error: Dataset is empty or contains no valid records.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} games from {args.dataset}")
    print(f"Strategy: {args.strategy}, Seed: {args.seed or 'default'}")

    # Run evaluation
    results = evaluate_dataset(records, strategy=args.strategy, seed=args.seed)

    # Compute statistics
    stats = game.compute_stats(results)
    print()
    print(game.format_stats(stats, human_readable=True))

    # Output structured format if requested
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(game.format_stats(stats, human_readable=False))
        print(f"\nStructured stats written to: {args.output}")


if __name__ == "__main__":
    main()