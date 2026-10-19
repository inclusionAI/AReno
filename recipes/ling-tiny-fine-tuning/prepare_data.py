"""Offline dataset preparation for maintainers; never called by the Notebook.

Reuse existing SFT inputs and oracle answers. No synthetic requests or reasoning
are generated. Run as a module from the repository root.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from recipes.payment.payment_service.models import PaymentRequest
from recipes.payment.payment_service.service import PaymentCalculationService


RECIPE_DIR = Path(__file__).resolve().parent


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_fingerprint(request: dict[str, Any]) -> str:
    payload = {
        **request,
        "order": {key: value for key, value in request["order"].items() if key != "order_id"},
    }
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def read_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    return [
        (line_number, json.loads(line))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip()
    ]


def prepare(source: Path, evaluation: Path, output: Path, source_revision: str, val_count: int, seed: int) -> dict:
    source_rows = read_rows(source)
    eval_rows = read_rows(evaluation)
    eval_keys = {request_fingerprint(row["input"]) for _, row in eval_rows}
    system_prompt = (RECIPE_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()
    seen: set[str] = set()
    rows = []
    excluded = []
    for line_number, row in source_rows:
        request = row["input"]
        answer = PaymentCalculationService.calculate(PaymentRequest.from_dict(request)).to_dict()
        if canonical(answer) != canonical(row["ground_truth"]):
            raise ValueError(f"Oracle mismatch at source line {line_number}: {row['case_id']}")
        fingerprint = request_fingerprint(request)
        reason = "evaluation_overlap" if fingerprint in eval_keys else "duplicate_request" if fingerprint in seen else None
        if reason:
            excluded.append({"source_line": line_number, "case_id": row["case_id"], "reason": reason})
            continue
        seen.add(fingerprint)
        rows.append({
            "case_id": row["case_id"],
            "source_line": line_number,
            "bucket": row["bucket"],
            "request_fingerprint": fingerprint,
            "input": request,
            "ground_truth": row["ground_truth"],
            "prompt": f"{system_prompt}\n\nPayment request:\n{canonical(request)}",
            "response": canonical(row["ground_truth"]),
        })
    if not 0 < val_count < len(rows):
        raise ValueError("Validation count must be positive and smaller than the retained dataset")

    # Proportional bucket quotas with largest remainders; hashes give stable
    # selection/order independently of Python's random implementation.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["bucket"]].append(row)
    quotas = {bucket: len(group) * val_count // len(rows) for bucket, group in groups.items()}
    priority = sorted(groups, key=lambda bucket: (-(len(groups[bucket]) * val_count % len(rows)), bucket))
    for bucket in priority[:val_count - sum(quotas.values())]:
        quotas[bucket] += 1

    def selection_key(row: dict) -> str:
        return hashlib.sha256(f"{seed}:{row['request_fingerprint']}".encode("utf-8")).hexdigest()

    train, valid = [], []
    for bucket in sorted(groups):
        group = sorted(groups[bucket], key=selection_key)
        valid.extend(group[:quotas[bucket]])
        train.extend(group[quotas[bucket]:])

    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, split in (("train.jsonl", train), ("valid.jsonl", valid)):
        split.sort(key=selection_key)
        path = output / name
        # Refuse to overwrite an existing published split.
        with path.open("x", encoding="utf-8") as stream:
            for row in split:
                stream.write(canonical(row) + "\n")
        files[name] = {
            "rows": len(split), "sha256": sha256(path), "bytes": path.stat().st_size,
            "buckets": dict(sorted(Counter(row["bucket"] for row in split).items())),
            "statuses": dict(sorted(Counter(row["ground_truth"]["status"] for row in split).items())),
        }

    manifest = {
        "dataset_version": "ling-tiny-fine-tuning-json-v1",
        "source": {"repository": "mlx-finetune-microsystem", "revision": source_revision,
                   "file": source.name, "sha256": sha256(source), "rows": len(source_rows)},
        "evaluation_overlap_reference": {"file": evaluation.name, "sha256": sha256(evaluation),
                                         "rows": len(eval_rows), "used_for_training": False},
        "preparation": {
            "seed": seed, "validation_count": val_count,
            "split_method": "bucket-proportional largest remainder; SHA256(seed:fingerprint) selection and order",
            "fingerprint": "SHA256 of canonical input JSON excluding order.order_id; all arrays and other IDs preserved",
            "target": "canonical ground_truth JSON; no solution or reasoning text",
            "oracle_checked_rows": len(source_rows), "oracle_mismatches": 0,
            "excluded": excluded, "token_length_checked": False, "token_length_filtered_rows": 0,
            "code_sha256": {name: sha256(RECIPE_DIR / name) for name in (
                "prepare_data.py", "system_prompt.txt", "payment_service/models.py",
                "payment_service/service.py", "payment_service/promotion_engine.py",
            )},
        },
        "files": files,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True, help="Existing test file used only for overlap exclusion")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.source, args.evaluation, args.output_dir, args.source_revision, args.val_count, args.seed)
    print(json.dumps({"files": manifest["files"], "excluded": manifest["preparation"]["excluded"]}, indent=2))


if __name__ == "__main__":
    main()
