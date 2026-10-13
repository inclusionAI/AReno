"""Revalidate real issue tasks in the shared task environment before training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

from generate_dataset import check_task, write_snapshot
from generate_swebench import apply, source_rows
from swebench_verify import unpack


def validate(row, raw, python):
    source = row["source"]
    if (
        source["base_commit"] != raw["base_commit"]
        or source["repo"] != raw["repo"]
        or source["problem_statement_sha256"] != hashlib.sha256(raw["problem_statement"].encode()).hexdigest()
    ):
        raise ValueError("task provenance does not match the pinned source")
    with tempfile.TemporaryDirectory(prefix="areno-pi-revalidate-") as directory:
        root = Path(directory)
        unpack(row["workspace_archive"], root)
        write_snapshot(root, {k: v.encode() for k, v in row["files"].items()})
        broken = check_task(root, row["verify"], python)
        f2p = row["source"]["fail_to_pass"]
        p2p = row["source"]["pass_to_pass"]
        if (
            not broken["complete"]
            or any(broken["tests"].get(t) != "failed" for t in f2p)
            or any(broken["tests"].get(t) != "passed" for t in p2p)
        ):
            raise ValueError("base mismatch: " + broken["failure_output"][-2000:])
        apply(root, raw["patch"])
        gold = check_task(root, row["verify"], python)
        if gold["returncode"] or not gold["complete"]:
            raise ValueError("reference mismatch: " + gold["failure_output"][-2000:])
    return {"task_id": row["source"]["task_id"], "broken": broken, "reference": gold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    raw = source_rows(args.snapshot)
    report = {"accepted": [], "rejected": []}
    manifest = []
    seen = set()
    opener = gzip.open if args.input.suffix == ".gz" else open
    pending = args.output.with_suffix(".pending.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with opener(args.input, "rt") as source, pending.open("w") as output:
        for line in source:
            row = json.loads(line)
            task_id = row["source"]["task_id"]
            try:
                if row["source"]["issue_url"] in seen:
                    raise ValueError("duplicate upstream issue")
                evidence = validate(row, raw[task_id], args.python)
                report["accepted"].append(evidence)
                manifest.append(row["source"])
                seen.add(row["source"]["issue_url"])
                output.write(line)
                output.flush()
                print("ACCEPT", len(manifest), task_id, flush=True)
            except Exception as exc:
                report["rejected"].append({"task_id": task_id, "reason": str(exc)})
                print("REJECT", task_id, str(exc)[-300:].replace("\n", " "), flush=True)
            args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
            if len(manifest) == args.limit:
                break
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    if len(manifest) != args.limit:
        raise SystemExit(f"Only {len(manifest)}/{args.limit} validated; training dataset not published")
    pending.replace(args.output)
    print(f"Validated {len(manifest)} real issues: {args.output}")


if __name__ == "__main__":
    main()
