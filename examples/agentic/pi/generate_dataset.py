"""Download ModelScope SWE-bench issues and convert them to pi/DinD records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Lite"
REVISION = "7162590844b7b94414849b7a4565e070db9780f2"
FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "problem_statement",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
)


def convert_row(row: dict, *, dataset: str, revision: str, split: str, source_sha256: str) -> dict:
    instance = {key: row[key] for key in FIELDS if key in row}
    for key in ("instance_id", "repo", "base_commit", "version", "problem_statement", "test_patch"):
        if not isinstance(instance.get(key), str) or not instance[key].strip():
            raise ValueError(f"SWE-bench row requires {key}")
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = instance.get(key, [])
        value = json.loads(value) if isinstance(value, str) else value
        if not isinstance(value, list) or not all(isinstance(test, str) for test in value):
            raise ValueError(f"{key} must be a list of test names")
        instance[key] = value
    if not instance["FAIL_TO_PASS"]:
        raise ValueError("SWE-bench task has no FAIL_TO_PASS tests")
    # Never copy the gold patch, hints, test patch or target test names into pi's prompt/files.
    return {
        "prompt": (
            f"Fix this issue in the repository at /testbed ({instance['repo']}).\n\n"
            f"{instance['problem_statement']}\n\n"
            "Inspect the code, implement the fix and run relevant tests. Do not commit your changes."
        ),
        "swebench": instance,
        "source": {"dataset": dataset, "revision": revision, "split": split, "sha256": source_sha256},
        "max_turns": 64,
        "timeout": 1800,
        "verify_timeout": 1800,
    }


def generate(snapshot: Path, output: Path, *, dataset: str, revision: str, split: str, limit: int | None = None) -> int:
    import pyarrow.parquet as pq

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    files = sorted((snapshot / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise ValueError(f"no parquet files for split {split}")
    rows, seen = [], set()
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for raw in batch.to_pylist():
                row = convert_row(raw, dataset=dataset, revision=revision, split=split, source_sha256=digest)
                instance_id = row["swebench"]["instance_id"]
                if instance_id in seen:
                    raise ValueError(f"duplicate instance_id: {instance_id}")
                seen.add(instance_id)
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("selected split is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--revision", default=None, help="ModelScope revision; Lite defaults to a pinned snapshot")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    from modelscope import snapshot_download

    revision = args.revision or (REVISION if args.dataset == DATASET else "master")
    snapshot = snapshot_download(
        args.dataset,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"data/{args.split}-*.parquet", "README.md"],
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    count = generate(
        Path(snapshot), args.output, dataset=args.dataset, revision=revision, split=args.split, limit=args.limit
    )
    print(f"Wrote {count} {args.split} repository tasks to {args.output}")


if __name__ == "__main__":
    main()
