"""Generate reproducible Gymnasium episode records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROMPT = "Play the episode and maximize the cumulative reward."


def generate_records(count: int = 256, *, seed: int = 2026, prompt: str = DEFAULT_PROMPT) -> list[dict]:
    """Return one record per episode; the record seed fully fixes the episode."""

    return [
        {"id": f"gymnasium-{index:05d}", "seed": seed + index, "prompt": prompt}
        for index in range(1, count + 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed, prompt=args.prompt)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} episode records to {args.output}")


if __name__ == "__main__":
    main()
