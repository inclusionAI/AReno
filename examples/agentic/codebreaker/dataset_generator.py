"""Generate reproducible Codebreaker tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from game import DEFAULT_CODE_LENGTH, DEFAULT_MAX_GUESSES, normalize_code


def generate_records(count: int = 256, *, seed: int = 2026) -> list[dict]:
    """Return deterministic unique-digit secrets."""

    rng = random.Random(seed)
    records = []
    seen: set[str] = set()
    while len(records) < count:
        secret = "".join(rng.sample("0123456789", DEFAULT_CODE_LENGTH))
        if secret in seen:
            continue
        seen.add(secret)
        records.append(
            {
                "id": f"codebreaker-{len(records) + 1:05d}",
                "secret": normalize_code(secret),
                "code_length": DEFAULT_CODE_LENGTH,
                "max_guesses": DEFAULT_MAX_GUESSES,
            }
        )
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
