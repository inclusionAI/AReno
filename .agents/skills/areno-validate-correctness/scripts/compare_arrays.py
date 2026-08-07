#!/usr/bin/env python3
"""Compare numeric JSON arrays with explicit tolerances."""

from __future__ import annotations

import json
import math
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import Result, build_parser, skill_main


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, (int, float)):
        yield float(value)
    else:
        raise TypeError(f"non-numeric value {value!r}")


@skill_main
def main() -> Result:
    parser = build_parser("Compare numeric JSON arrays with explicit tolerances.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    left = list(flatten(json.loads(args.baseline.read_text(encoding="utf-8"))))
    right = list(flatten(json.loads(args.candidate.read_text(encoding="utf-8"))))
    diffs = [abs(a - b) for a, b in zip(left, right)]
    finite = all(math.isfinite(value) for value in left + right)
    passed = (
        len(left) == len(right)
        and finite
        and all(diff <= args.atol + args.rtol * abs(a) for a, diff in zip(left, diffs))
    )
    return Result(
        ok=passed,
        data={
            "baseline_count": len(left),
            "candidate_count": len(right),
            "finite": finite,
            "max_abs_diff": max(diffs, default=0.0),
            "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
