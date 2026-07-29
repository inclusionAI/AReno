#!/usr/bin/env python3
"""List or read scalar series from AReno TensorBoard event files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--last", type=int, default=20)
    args = parser.parse_args()
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(args.log_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        keys = sorted(accumulator.Tags().get("scalars", []))
        selected = args.metric or keys
        unknown = sorted(set(selected) - set(keys))
        series = {
            key: [
                {"step": item.step, "value": item.value, "wall_time": item.wall_time}
                for item in accumulator.Scalars(key)[-max(args.last, 0) :]
            ]
            for key in selected
            if key in keys
        }
        result = {"ok": not unknown, "path": str(args.log_dir), "keys": keys, "unknown": unknown, "series": series}
    except Exception as exc:
        result = {"ok": False, "path": str(args.log_dir), "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
