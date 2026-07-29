#!/usr/bin/env python3
"""Summarize AReno TensorBoard time/* metrics and selected throughput keys."""

from __future__ import annotations

import argparse
import json
import statistics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--drop-first", type=int, default=1)
    args = parser.parse_args()
    result: dict = {"ok": False, "log_dir": args.log_dir}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(args.log_dir, size_guidance={"scalars": 0})
        accumulator.Reload()
        keys = sorted(accumulator.Tags().get("scalars", []))
        selected = [key for key in keys if key.startswith("time/") or "throughput" in key or "tokens_per_second" in key]
        summaries = {}
        for key in selected:
            values = [event.value for event in accumulator.Scalars(key)][max(args.drop_first, 0) :]
            if values:
                summaries[key] = {
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                }
        result.update(ok=bool(summaries), metrics=summaries, available_keys=keys)
        if not summaries:
            result["error"] = "no time or throughput metrics found"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
