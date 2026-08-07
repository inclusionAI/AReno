#!/usr/bin/env python3
"""Summarize AReno TensorBoard time/* metrics and selected throughput keys."""

from __future__ import annotations

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import build_parser, skill_main


@skill_main
def main() -> dict:
    parser = build_parser("Summarize AReno TensorBoard time/* metrics and selected throughput keys.")
    parser.add_argument("log_dir")
    parser.add_argument("--drop-first", type=int, default=1)
    args = parser.parse_args()

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
    result: dict = {"ok": bool(summaries), "metrics": summaries, "available_keys": keys, "log_dir": args.log_dir}
    if not summaries:
        result["error"] = "no time or throughput metrics found"
    return result


if __name__ == "__main__":
    raise SystemExit(main())
