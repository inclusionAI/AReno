#!/usr/bin/env python3
"""List or summarize selected scalar series from TensorBoard event files."""

from __future__ import annotations

import argparse
import fnmatch
import json
import statistics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--pattern", action="append", default=[], help="Scalar glob; may be repeated")
    parser.add_argument("--drop-first", type=int, default=0)
    parser.add_argument("--last", type=int, default=0, help="Keep only the last N selected events")
    parser.add_argument("--list", action="store_true", help="Only list available scalar names")
    args = parser.parse_args()
    result: dict = {"ok": False, "log_dir": args.log_dir}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(args.log_dir, size_guidance={"scalars": 0})
        accumulator.Reload()
        keys = sorted(accumulator.Tags().get("scalars", []))
        patterns = args.pattern or ["*"]
        selected = [key for key in keys if any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)]
        if args.list:
            result.update(ok=True, available_keys=keys, selected_keys=selected)
        else:
            summaries = {}
            for key in selected:
                events = accumulator.Scalars(key)[max(args.drop_first, 0) :]
                if args.last > 0:
                    events = events[-args.last :]
                values = [event.value for event in events]
                if values:
                    summaries[key] = {
                        "count": len(values),
                        "first_step": events[0].step,
                        "last_step": events[-1].step,
                        "mean": statistics.fmean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "last": values[-1],
                    }
            result.update(ok=bool(summaries), metrics=summaries, available_keys=keys)
            if not summaries:
                result["error"] = "no matching scalar events found"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
