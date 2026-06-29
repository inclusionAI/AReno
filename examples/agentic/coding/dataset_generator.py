"""Generate JSONL coding tasks for the agentic coding example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_records(source_path: str | Path, *, count: int | None = None) -> list[dict]:
    """Load the bundled coding tasks and optionally keep the first ``count``."""

    source = Path(source_path).expanduser()
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if count is not None:
        if count <= 0:
            raise ValueError("count must be positive")
        records = records[:count]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL tasks for the AReno coding agentic example.")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--source",
        default=str(Path(__file__).with_name("dataset.jsonl")),
        help="Source JSONL task pool.",
    )
    parser.add_argument("--count", type=int, default=None, help="Optional number of tasks to emit.")
    args = parser.parse_args()

    records = generate_records(args.source, count=args.count)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
