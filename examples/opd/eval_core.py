"""Shared, dependency-light held-out OPD evaluation and paired statistics.

These are task-based measurements, NOT the training OPD loss. The held-out
answer is never supplied as part of the generation prompt.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
from pathlib import Path


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("answer must be a string")
    return " ".join(text.casefold().strip().translate(str.maketrans("", "", string.punctuation)).split())


def load_eval_set(path: Path) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    dataset_sha256 = hashlib.sha256(raw).hexdigest()
    records: list[dict] = []
    used: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"evaluation row {line_number} must be an object")
        for key in ("id", "prompt", "answer"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"row {line_number} requires nonempty {key!r}")
        if record["id"] in used:
            raise ValueError(f"duplicate evaluation id {record['id']!r}")
        used.add(record["id"])
        aliases = record.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(x, str) or not x.strip() for x in aliases):
            raise ValueError(f"row {line_number} aliases must be strings")
        if any(normalize_answer(x) == "" for x in [record["answer"], *aliases]):
            raise ValueError("normalized answer must be nonempty")
        records.append({"id": record["id"], "prompt": record["prompt"],
                        "answer": record["answer"], "aliases": aliases})
    if not records:
        raise ValueError("evaluation dataset is empty")
    return records, dataset_sha256


def validate_train_holdout_disjoint(training_path: Path, eval_rows: list[dict]) -> None:
    """Guard against *exact* prompt leakage; paraphrase leakage is not detected."""
    train_prompts = set()
    for line in Path(training_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
                raise ValueError("training dataset row requires a prompt")
            train_prompts.add(" ".join(item["prompt"].casefold().split()))
    overlap = {row["id"] for row in eval_rows if " ".join(row["prompt"].casefold().split()) in train_prompts}
    if overlap:
        raise ValueError(f"held-out prompts overlap training prompts: {sorted(overlap)}")


def exact_match(completion: str, row: dict) -> bool:
    correct = {normalize_answer(row["answer"]), *(normalize_answer(x) for x in row.get("aliases", []))}
    return normalize_answer(completion) in correct


def summary_from_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no prediction records")
    for row in rows:
        if not isinstance(row.get("correct"), bool):
            raise ValueError("prediction row missing boolean correct")
        if not isinstance(row.get("answer_nll_sum"), (int, float)) or not math.isfinite(row["answer_nll_sum"]) or row["answer_nll_sum"] < 0:
            raise ValueError("prediction row missing nonnegative finite NLL")
        if not isinstance(row.get("answer_tokens"), int) or row["answer_tokens"] <= 0:
            raise ValueError("prediction row missing positive answer token count")
    nll_sum = sum(float(row["answer_nll_sum"]) for row in rows)
    n_tokens = sum(row["answer_tokens"] for row in rows)
    return {
        "count": len(rows), "exact_match": sum(row["correct"] for row in rows) / len(rows),
        "answer_nll_per_token": nll_sum / n_tokens,
        "answer_perplexity": math.exp(min(nll_sum / n_tokens, 700.0)),
        "answer_tokens": n_tokens,
        "mean_generation_wall_time_s": sum(float(row["generation_wall_time_s"]) for row in rows) / len(rows)
            if all(isinstance(row.get("generation_wall_time_s"), (int, float)) for row in rows) else None,
    }


def paired_bootstrap_deltas(before: list[dict], after: list[dict], *, replicates: int = 2000, seed: int = 17) -> dict:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    n = len(before)
    if n < 1 or n != len(after):
        raise ValueError("paired bootstrap requires nonempty equal length data")
    rng = random.Random(seed)
    diffs = [float(a["correct"]) - float(b["correct"]) for b, a in zip(before, after, strict=True)]
    samples = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates)]
    samples.sort()
    return {
        "delta_exact_match": sum(diffs) / n,
        "bootstrap_95pct_ci": [samples[int(0.025 * (replicates - 1))], samples[int(0.975 * (replicates - 1))]],
        "bootstrap_replicates": replicates, "seed": seed,
    }


def read_prediction_file(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("rows"), list):
        raise ValueError(f"invalid Stage 7 evaluation file: {path}")
    if not isinstance(data.get("dataset_sha256"), str):
        raise ValueError("missing dataset fingerprint")
    records = data["rows"]
    ids = [r.get("id") for r in records]
    if not records or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("prediction rows have missing or duplicate ids")
    if any(not isinstance(r.get("prompt"), str) or not isinstance(r.get("answer"), str) or
           not isinstance(r.get("completion"), str) or r.get("correct") != exact_match(r["completion"], r)
           for r in records):
        raise ValueError("prediction row missing fields or incorrectly scored")
    summary_from_rows(records)
    return data


def aligned_rows(before: dict, after: dict) -> tuple[list[dict], list[dict]]:
    if before["dataset_sha256"] != after["dataset_sha256"]:
        raise ValueError("dataset fingerprints differ; cannot compare different held-out sets")
    a = {r["id"]: r for r in before["rows"]}
    b = {r["id"]: r for r in after["rows"]}
    if a.keys() != b.keys():
        raise ValueError("evaluation IDs differ")
    ordered = sorted(a)
    for key in ordered:
        if a[key]["prompt"] != b[key]["prompt"] or a[key]["answer"] != b[key]["answer"] or a[key].get("aliases", []) != b[key].get("aliases", []):
            raise ValueError(f"evaluation content differs for id {key}")
    return [a[k] for k in ordered], [b[k] for k in ordered]
