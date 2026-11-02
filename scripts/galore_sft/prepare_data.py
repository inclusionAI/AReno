"""Pinned Dolly preparation; grouping precedes fixed splits and token statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

UTC = timezone.utc


REPO = "AI-ModelScope/databricks-dolly-15k"
REVISION = "40789acc12fc986e9145f87d159f19e2f1d75c6a"
RAW_FILES = {
    "databricks-dolly-15k.jsonl": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
    "README.md": "db06c60090a11b20e79d1933b2f316f781605f07761ff2ae2795d6523e3e960b",
}
SALT = "areno-dolly-group-split-v1:"
SPLITS = ("train", "validation", "test")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_example(row):
    return "\0".join(normalize(row[key]) for key in ("instruction", "context", "response"))


def format_prompt(instruction, context):
    prompt = f"### Instruction:\n{instruction}\n\n"
    if context:
        prompt += f"### Context:\n{context}\n\n"
    return prompt + "### Response:\n"


def split_for_group(group_id):
    bucket = int(sha(SALT + group_id)[:16], 16) % 10000
    return "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"


def prepare_rows(raw_rows):
    output = {split: [] for split in SPLITS}
    manifests = {split: [] for split in SPLITS}
    seen, dropped = set(), []
    for source_index, row in enumerate(raw_rows):
        required = ("instruction", "context", "response")
        if any(not isinstance(row.get(key), str) for key in required):
            dropped.append({"source_index": source_index, "reason": "invalid_required_string"})
            continue
        row = dict(row)
        for key in required:
            row[key] = row[key].strip()
        if not row["instruction"] or not row["response"]:
            dropped.append({"source_index": source_index, "reason": "empty_instruction_or_response"})
            continue
        full_hash = sha(canonical_example(row))
        if full_hash in seen:
            dropped.append({"source_index": source_index, "reason": "normalized_full_example_duplicate"})
            continue
        seen.add(full_hash)
        group_kind = "context" if row["context"] else "instruction"
        group_id = sha(group_kind + ":" + normalize(row[group_kind]))
        split = split_for_group(group_id)
        example = {"prompt": format_prompt(row["instruction"], row["context"]), "response": row["response"]}
        output[split].append(example)
        manifests[split].append(
            {
                "index": len(output[split]) - 1,
                "source_index": source_index,
                "group_id": group_id,
                "group_kind": group_kind,
                "category": row.get("category", "unknown"),
                "normalized_prompt_sha256": sha(normalize(example["prompt"])),
                "normalized_full_example_sha256": full_hash,
            }
        )
    return output, manifests, dropped


def stats(values):
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": len(values),
        "total": int(array.sum()),
        "mean": float(array.mean()),
        "min": int(array.min()),
        "max": int(array.max()),
        "quantiles": {str(q): float(np.quantile(array, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", dest="tokenizer_dir", type=Path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="Fetch the pinned ModelScope snapshot; otherwise verify local raw files"
    )
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.download:
        if args.cache_dir is None:
            parser.error("--download requires an explicit --cache-dir")
        from modelscope.hub.snapshot_download import dataset_snapshot_download

        dataset_snapshot_download(
            dataset_id=REPO,
            revision=REVISION,
            cache_dir=str(args.cache_dir),
            local_dir=str(args.raw_dir),
            allow_patterns=list(RAW_FILES),
            max_workers=2,
        )
    for name, expected in RAW_FILES.items():
        actual = hashlib.sha256((args.raw_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned source mismatch for {name}: {actual}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir / "databricks-dolly-15k.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
        == "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
    )
    output, manifests, dropped = prepare_rows(raw_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    split_statistics = {}
    for split in SPLITS:
        rows = output[split]
        prompts = tokenizer([row["prompt"] for row in rows], add_special_tokens=False, return_attention_mask=False)[
            "input_ids"
        ]
        responses = tokenizer([row["response"] for row in rows], add_special_tokens=False, return_attention_mask=False)[
            "input_ids"
        ]
        prompt_lengths, response_lengths = [len(x) for x in prompts], [len(x) for x in responses]
        whole_lengths = [a + b for a, b in zip(prompt_lengths, response_lengths, strict=True)]
        assert all(length > 0 for length in response_lengths)
        for record, prompt_length, response_length in zip(
            manifests[split], prompt_lengths, response_lengths, strict=True
        ):
            record.update(
                prompt_tokens=prompt_length,
                response_tokens=response_length,
                total_tokens=prompt_length + response_length,
            )
        caps = {}
        for cap in (512, 1024):
            target_lengths = [
                max(0, min(response, cap - prompt))
                for prompt, response in zip(prompt_lengths, response_lengths, strict=True)
            ]
            caps[str(cap)] = {
                "naive_prefix_capped_rows_with_zero_targets": sum(length == 0 for length in target_lengths),
                "naive_prefix_capped_target_tokens": sum(target_lengths),
                "whole_rows_over_cap": sum(length > cap for length in whole_lengths),
                "note": "diagnostic only; no stored row is truncated and no row is filtered by length",
            }
        split_statistics[split] = {
            "rows": len(rows),
            "groups": len({record["group_id"] for record in manifests[split]}),
            "categories": dict(sorted(Counter(record["category"] for record in manifests[split]).items())),
            "prompt_tokens": stats(prompt_lengths),
            "response_tokens": stats(response_lengths),
            "total_tokens": stats(whole_lengths),
            "context_cap_diagnostics": caps,
        }
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        write_jsonl(args.output_dir / f"{split}.manifest.jsonl", manifests[split])
    for field in ("group_id", "normalized_prompt_sha256", "normalized_full_example_sha256"):
        sets = [{record[field] for record in manifests[split]} for split in SPLITS]
        assert all(not sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3)), field
    assert sum(len(rows) for rows in output.values()) + len(dropped) == len(raw_rows)
    (args.output_dir / "orders").mkdir(exist_ok=True)
    for seed in (0, 1, 2):
        order = list(range(len(output["train"])))
        random.Random(seed).shuffle(order)
        (args.output_dir / "orders" / f"seed{seed}.json").write_text(json.dumps(order, separators=(",", ":")) + "\n")
    file_hashes = {
        str(path.relative_to(args.output_dir)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "summary.json"
    }
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "raw_rows": len(raw_rows),
        "retained_rows": sum(len(rows) for rows in output.values()),
        "dropped_rows": dropped,
        "dropped_reason_counts": dict(Counter(row["reason"] for row in dropped)),
        "split_rule": "fixed SHA256 group split,90/5/5; seeds0/1/2 only permute training order",
        "normalization_for_keys": "NFKC-whitespace-collapse-casefold; output strings preserved except outer whitespace",
        "grouping_limit": "exact normalized reference context is a document proxy; original Wikipedia article IDs absent, so no true article-level or near-duplicate guarantee",
        "stored_schema": ["prompt", "response"],
        "truncated": False,
        "eos_added": False,
        "tokenization": "local pinned Qwen3-0.6B tokenizer; add_special_tokens=False; prompt/response encoded separately and concatenated for counts",
        "tokenizer_sha256": {
            name: hashlib.sha256((args.tokenizer_dir / name).read_bytes()).hexdigest()
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "splits": split_statistics,
        "files": file_hashes,
        "integrity": {
            "group_disjoint": True,
            "normalized_prompt_disjoint": True,
            "full_example_disjoint": True,
            "all_response_tokens_positive": True,
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "LICENSE-ATTRIBUTION.md").write_text(
        (Path(__file__).resolve().parent / "DATASET-LICENSE.md").read_text()
    )
    (args.output_dir / "source.json").write_text(
        json.dumps(
            {
                "source_hub": "ModelScope",
                "repo_id": REPO,
                "revision": REVISION,
                "raw_files_sha256": RAW_FILES,
                "fallback_used": False,
                "license": "CC-BY-SA-3.0",
                "author": "Databricks, Inc.; Wikipedia contributors",
            },
            indent=2,
        )
        + "\n"
    )
    expected_files = json.loads((Path(__file__).resolve().parent / "data-manifest.json").read_text())["files"]
    if file_hashes != expected_files:
        raise ValueError("prepared files differ from the pinned tokenizer/data contract")
    print(
        json.dumps({key: value for key, value in summary.items() if key not in ("files", "dropped_rows")}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
