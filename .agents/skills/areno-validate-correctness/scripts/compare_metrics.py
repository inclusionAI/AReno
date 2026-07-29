#!/usr/bin/env python3
"""Compare TensorBoard scalar series by step."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import Result, build_parser, skill_main


def load(path: str, metric: str) -> dict[int, float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(path, size_guidance={"scalars": 0})
    accumulator.Reload()
    return {item.step: item.value for item in accumulator.Scalars(metric)}


@skill_main
def main() -> Result:
    parser = build_parser("Compare TensorBoard scalar series by step.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    reports = {}
    ok = True
    for metric in args.metric:
        baseline = load(args.baseline, metric)
        candidate = load(args.candidate, metric)
        common = sorted(set(baseline) & set(candidate))
        rows = []
        for step in common:
            left, right = baseline[step], candidate[step]
            delta = abs(left - right)
            passed = delta <= args.atol + args.rtol * abs(left)
            ok = ok and passed
            rows.append({"step": step, "baseline": left, "candidate": right, "abs_diff": delta, "ok": passed})
        if not common:
            ok = False
        reports[metric] = {"common_steps": len(common), "rows": rows}
    return Result(ok=ok, data={"metrics": reports})


if __name__ == "__main__":
    raise SystemExit(main())