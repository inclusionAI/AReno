"""Generate reproducible Wordle tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from game import DEFAULT_MAX_GUESSES, WORDLE_LENGTH, WORDLE_WORDS


def generate_records(count: int = 256, *, seed: int = 2026) -> list[dict]:
    """Return deterministic Wordle secrets drawn from the bundled word list."""

    rng = random.Random(seed)
    records = []
    pool = rng.sample(WORDLE_WORDS, min(count, len(WORDLE_WORDS)))
    while len(records) < count:
        idx = len(records) % len(pool)
        records.append({
            "id": f"wordle-{len(records) + 1:05d}",
            "secret": pool[idx],
            "word_length": WORDLE_LENGTH,
            "max_guesses": DEFAULT_MAX_GUESSES,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    records = generate_records(args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()

