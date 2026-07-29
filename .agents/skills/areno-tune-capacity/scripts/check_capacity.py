#!/usr/bin/env python3
"""Check AReno capacity relationships without allocating a model."""

from __future__ import annotations

import argparse
import json
import math


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--max-running-prompts", type=int, required=True)
    parser.add_argument("--mini-bs", type=int, required=True)
    parser.add_argument("--score-micro-bs", type=int, default=8)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    args = parser.parse_args()
    values = vars(args)
    errors = [f"{key} must be positive" for key, value in values.items() if key != "memory_fraction" and value <= 0]
    if not 0 < args.memory_fraction <= 0.9:
        errors.append("memory_fraction must be in (0, 0.9]")
    if args.world_size % args.tp_size:
        errors.append("world_size must be divisible by tp_size")
    demand = args.batch_size * args.n_samples
    waves = math.ceil(demand / args.max_running_prompts) if args.max_running_prompts > 0 else None
    result = {
        "ok": not errors,
        "errors": errors,
        "rollout_demand": demand,
        "minimum_admission_waves": waves,
        "data_parallel_size": args.world_size // args.tp_size if not args.world_size % args.tp_size else None,
        "settings": values,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
