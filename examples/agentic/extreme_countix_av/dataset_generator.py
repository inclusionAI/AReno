"""Create a compact JSONL manifest for Extreme Countix-AV."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import discover_samples


def generate_manifest(dataset_root: str | Path, output: str | Path, *, seed: int = 42) -> list[dict]:
    root = Path(dataset_root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    records = [{**sample.as_record(root), "dataset_root": str(root)} for sample in discover_samples(root)]
    random.Random(seed).shuffle(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", required=True, help="Directory containing ExtremeLabels.csv, Videos, and Audio"
    )
    parser.add_argument("--output", required=True, help="Output JSONL manifest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = generate_manifest(args.dataset_root, args.output, seed=args.seed)
    print(f"wrote {len(records)} labelled audiovisual records to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
