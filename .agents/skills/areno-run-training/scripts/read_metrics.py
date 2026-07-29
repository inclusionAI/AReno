#!/usr/bin/env python3
"""List or read scalar series from AReno TensorBoard event files."""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import build_parser, skill_main


@skill_main
def main() -> dict:
    parser = build_parser("List or read scalar series from AReno TensorBoard event files.")
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--last", type=int, default=20)
    args = parser.parse_args()

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
    return {"ok": not unknown, "path": str(args.log_dir), "keys": keys, "unknown": unknown, "series": series}


if __name__ == "__main__":
    raise SystemExit(main())