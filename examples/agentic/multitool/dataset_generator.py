"""Generate JSONL multi-tool tasks for the agentic example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import generate_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL tasks for the Areno multi-tool agentic example."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--count", type=int, default=256, help="Number of records.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()