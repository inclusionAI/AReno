"""Evaluate a classify checkpoint on JevForge records.

Scores every question with `SequenceScorer` (AReno's own model, packed
forward, same encoding as training) and reports, per question type and per
source group:

- accuracy: argmax of the prediction equals argmax of the gold distribution
- ce:       cross-entropy of the prediction against the gold distribution
- kl:       KL(gold || prediction)
- brier:    sum over options of (p - gold)^2

The typed-decisions leaderboard reports accuracy / KL / Brier on its `test`
split; its exact metric code is not published in the dataset card, so treat
the comparison as approximate.

    python examples/classify/jev/evaluate.py --checkpoint ~/areno-runs/ling-3.0-tiny-jev \
        --records ~/data/jev-records/typed-decisions --split test --output eval.json
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
        "brier": sum((p - g) ** 2 for p, g in zip(probs, gold, strict=True)),
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    keys = ("accuracy", "ce", "kl", "brier")
    return {"n": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}


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

    results, skipped = [], 0
    batch: list[tuple[dict, list[list[int]]]] = []

    def flush() -> None:
        if not batch:
            return
        scores = scorer.score([leaf for _, leaves in batch for leaf in leaves]).float()
        offset = 0
        for row, leaves in batch:
            logits = scores[offset : offset + len(leaves)] / args.temperature
            offset += len(leaves)
            probs = torch.softmax(logits, dim=0).tolist()
            metrics = question_metrics(probs, row["target"])
            results.append({"type": row["type"], "group": row["source_group"], **metrics})
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

    summary = {"checkpoint": args.checkpoint, "records": args.records, "split": args.split, "skipped_too_long": skipped}
    summary["all"] = summarize(results)
    by_type, by_group = defaultdict(list), defaultdict(list)
    for result in results:
        by_type[result["type"]].append(result)
        by_group[result["group"]].append(result)
    summary["by_type"] = {key: summarize(value) for key, value in sorted(by_type.items())}
    if len(by_group) <= 20:
        summary["by_group"] = {key: summarize(value) for key, value in sorted(by_group.items())}

    def line(name: str, s: dict) -> str:
        return f"{name:>28}: n={s['n']:6d} acc={s['accuracy']:.3f} ce={s['ce']:.4f} kl={s['kl']:.4f} brier={s['brier']:.4f}"

    print(line("all", summary["all"]))
    for key, value in summary["by_type"].items():
        print(line(key, value))
    for key, value in summary.get("by_group", {}).items():
        print(line(key, value))
    if skipped:
        print(f"skipped {skipped} questions longer than --max-seq-len={args.max_seq_len}")
    if args.output:
        Path(args.output).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
