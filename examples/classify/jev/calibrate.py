"""Fit / apply a softmax temperature on logits dumped by `evaluate.py --dump`.

No model is loaded: dumps hold each question's raw (T=1) logits and gold.

    # fit on a calibration dump (JevForge-style grid + refine on mean CE)
    python examples/classify/jev/calibrate.py fit calibration.jsonl [--per-type] --output temperature.json

    # re-score other dumps with that temperature (same metrics as evaluate.py)
    python examples/classify/jev/calibrate.py apply typed-decisions.jsonl --temperature-file temperature.json

Fit the temperature on a split of the *training* distribution (Open-Jev
calibration) only; fitting it on the evaluation set would leak test labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import question_metrics, report  # noqa: E402


def load_dump(path: str) -> tuple[dict, list[dict]]:
    lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0]).get("meta", {})
    return meta, [json.loads(line) for line in lines[1:] if line.strip()]


def softmax(logits: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def mean_ce(items: list[dict], temperature: float) -> float:
    total = 0.0
    for item in items:
        probs = softmax(item["logits"], temperature)
        total -= sum(g * math.log(max(p, 1e-12)) for p, g in zip(probs, item["target"], strict=True))
    return total / max(len(items), 1)


def fit_temperature(items: list[dict]) -> float:
    """Log-spaced grid over [0.05, 20], then a +-5% fine grid around the best point."""

    coarse = [0.05 * (400 ** (i / 159)) for i in range(160)]
    best = min(coarse, key=lambda t: mean_ce(items, t))
    fine = [best * (0.95 + 0.1 * i / 100) for i in range(101)]
    return round(min(fine, key=lambda t: mean_ce(items, t)), 4)


def temperature_for(temperatures: float | dict, kind: str) -> float:
    if isinstance(temperatures, dict):
        return float(temperatures.get(kind, temperatures.get("all", 1.0)))
    return float(temperatures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("dump")
    fit.add_argument("--per-type", action="store_true", help="also fit one temperature per question type")
    fit.add_argument("--output", required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("dump")
    group = apply.add_mutually_exclusive_group(required=True)
    group.add_argument("--temperature", type=float)
    group.add_argument("--temperature-file")
    apply.add_argument("--output", default=None)
    args = parser.parse_args()

    meta, items = load_dump(args.dump)
    if args.command == "fit":
        result: dict = {"fitted_on": args.dump, "n": len(items), "all": fit_temperature(items)}
        result["ce_at_1"] = mean_ce(items, 1.0)
        result["ce_at_fit"] = mean_ce(items, result["all"])
        if args.per_type:
            for kind in sorted({item["type"] for item in items}):
                result[kind] = fit_temperature([item for item in items if item["type"] == kind])
        Path(args.output).expanduser().write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return

    if args.temperature is not None:
        temperatures: float | dict = args.temperature
    else:
        loaded = json.loads(Path(args.temperature_file).expanduser().read_text(encoding="utf-8"))
        skip = {"ce_at_1", "ce_at_fit"}
        temperatures = {key: value for key, value in loaded.items() if isinstance(value, float) and key not in skip}
    results = []
    for item in items:
        probs = softmax(item["logits"], temperature_for(temperatures, item["type"]))
        results.append({"type": item["type"], "group": item["group"], **question_metrics(probs, item["target"])})
    summary = report(results, {**meta, "temperature": temperatures, "dump": args.dump})
    if args.output:
        Path(args.output).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
