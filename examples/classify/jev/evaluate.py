"""Evaluate a classify checkpoint on JevForge records.

Scores every question with `SequenceScorer` (AReno's own model, packed
forward, same encoding as training) and reports, per question type and per
source group:

- accuracy: argmax of the prediction equals argmax of the gold distribution
- ce:       cross-entropy of the prediction against the gold distribution
- kl:       KL(gold || prediction)
- tv:       total variation, 0.5 * sum |p - gold|
- brier:    sum over options of (p - gold)^2
- conf:     mean max probability; overconf = conf - accuracy
- ece:      10 equal-width bins over max probability vs accuracy (our own
            definition; the typed-decisions card does not publish its ECE code)

KL / TV / Brier reproduce the typed-decisions card's `Uniform` row exactly
(0.444 / 0.381 / 0.238), so those columns compare directly with its
leaderboard. Accuracy matches for any model without argmax ties.

    python examples/classify/jev/evaluate.py --checkpoint ~/areno-runs/ling-3.0-tiny-jev \
        --records ~/data/jev-records/typed-decisions --split test --output eval.json

`--dump questions.jsonl` also writes each question's raw (T=1) logits and gold
target, so `calibrate.py` can fit a temperature or re-score at any temperature
without running the model again.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import load_questions  # noqa: E402


def question_metrics(probs: list[float], gold: list[float]) -> dict[str, float]:
    eps = 1e-12
    predicted = max(range(len(probs)), key=probs.__getitem__)
    expected = max(range(len(gold)), key=gold.__getitem__)
    return {
        "accuracy": float(predicted == expected),
        "ce": -sum(g * math.log(max(p, eps)) for p, g in zip(probs, gold, strict=True)),
        "kl": sum(g * math.log(max(g, eps) / max(p, eps)) for p, g in zip(probs, gold, strict=True) if g > 0),
        "tv": 0.5 * sum(abs(p - g) for p, g in zip(probs, gold, strict=True)),
        "brier": sum((p - g) ** 2 for p, g in zip(probs, gold, strict=True)),
        "confidence": max(probs),
    }


def expected_calibration_error(rows: list[dict], bins: int = 10) -> float:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[min(int(row["confidence"] * bins), bins - 1)].append(row)
    total = 0.0
    for bucket in buckets.values():
        confidence = sum(r["confidence"] for r in bucket) / len(bucket)
        accuracy = sum(r["accuracy"] for r in bucket) / len(bucket)
        total += len(bucket) / len(rows) * abs(confidence - accuracy)
    return total


def summarize(rows: list[dict]) -> dict[str, float]:
    keys = ("accuracy", "ce", "kl", "tv", "brier", "confidence")
    out = {"n": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}
    out["overconfidence"] = out["confidence"] - out["accuracy"]
    out["ece"] = expected_calibration_error(rows)
    return out


def report(results: list[dict], meta: dict) -> dict:
    """Summaries overall / by type / by group (<= 20 groups); prints a table."""

    summary = dict(meta)
    summary["all"] = summarize(results)
    by_type, by_group = defaultdict(list), defaultdict(list)
    for result in results:
        by_type[result["type"]].append(result)
        by_group[result["group"]].append(result)
    summary["by_type"] = {key: summarize(value) for key, value in sorted(by_type.items())}
    if len(by_group) <= 20:
        summary["by_group"] = {key: summarize(value) for key, value in sorted(by_group.items())}

    def line(name: str, s: dict) -> str:
        return (
            f"{name:>28}: n={s['n']:6d} acc={s['accuracy']:.3f} ce={s['ce']:.4f} kl={s['kl']:.4f} "
            f"tv={s['tv']:.4f} brier={s['brier']:.4f} conf={s['confidence']:.3f} "
            f"overconf={s['overconfidence']:+.3f} ece={s['ece']:.3f}"
        )

    print(line("all", summary["all"]))
    for key, value in summary["by_type"].items():
        print(line(key, value))
    for key, value in summary.get("by_group", {}).items():
        print(line(key, value))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="evaluate a seeded random subset (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--attn-backend", choices=["flash", "native"], default="flash")
    parser.add_argument("--questions-per-forward", type=int, default=32)
    parser.add_argument("--output", default=None, help="write summary JSON here")
    parser.add_argument("--dump", default=None, help="write per-question raw logits + gold (jsonl) here")
    args = parser.parse_args()

    import torch

    from areno.experimental.classify.scorer import SequenceScorer

    rows = load_questions(args.records, args.split)
    if args.limit and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)
    scorer = SequenceScorer(args.checkpoint, attn_backend=args.attn_backend, max_tokens=16384)
    tokenizer = scorer.tokenizer

    def encode(text: str) -> list[int]:
        return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]

    results, dumped, skipped = [], [], 0
    batch: list[tuple[dict, list[list[int]]]] = []

    def flush() -> None:
        if not batch:
            return
        scores = scorer.score([leaf for _, leaves in batch for leaf in leaves]).float()
        offset = 0
        for row, leaves in batch:
            raw = scores[offset : offset + len(leaves)]
            offset += len(leaves)
            probs = torch.softmax(raw / args.temperature, dim=0).tolist()
            metrics = question_metrics(probs, row["target"])
            results.append({"type": row["type"], "group": row["source_group"], **metrics})
            if args.dump:
                dumped.append(
                    {"type": row["type"], "group": row["source_group"], "logits": raw.tolist(), "target": row["target"]}
                )
        batch.clear()

    for index, row in enumerate(rows):
        prefix = encode(row["prompt"])
        leaves = [prefix + encode(candidate) for candidate in row["candidates"]]
        if max(len(leaf) for leaf in leaves) > args.max_seq_len:
            skipped += 1
            continue
        batch.append((row, leaves))
        if len(batch) >= args.questions_per_forward:
            flush()
        if (index + 1) % 1000 == 0:
            print(f"  {index + 1}/{len(rows)} questions", flush=True)
    flush()

    meta = {"checkpoint": args.checkpoint, "records": args.records, "split": args.split}
    summary = report(results, {**meta, "temperature": args.temperature, "skipped_too_long": skipped})
    if skipped:
        print(f"skipped {skipped} questions longer than --max-seq-len={args.max_seq_len}")
    if args.output:
        Path(args.output).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.dump:
        with Path(args.dump).expanduser().open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"meta": meta}) + "\n")
            for item in dumped:
                handle.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
