#!/usr/bin/env python3
"""Summarise a training run into a compact report (no network, no GPU).

Used by both example workflows. Accepts upstream JSON fields via CLI flags
and prints a structured summary.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise a run or probe.")
    parser.add_argument("--algo", help="Algorithm name.")
    parser.add_argument("--model", help="Model checkpoint.")
    parser.add_argument("--dataset", help="Dataset reference.")
    parser.add_argument("--steps-completed", type=int, help="Steps completed.")
    parser.add_argument("--final-reward", type=float, help="Final reward value.")
    parser.add_argument("--final-loss", type=float, help="Final loss value.")
    parser.add_argument("--source", default="run", help="Source type (run or probe).")
    parser.add_argument(
        "--probe-ok", dest="probe_ok", action="store_true", help="Probe succeeded."
    )
    parser.add_argument(
        "--model-advertised", help="Model advertised by the serving endpoint."
    )
    args = parser.parse_args()

    summary: dict[str, object] = {"ok": True, "source": args.source}
    if args.algo:
        summary["algo"] = args.algo
    if args.model:
        summary["model"] = args.model
    if args.dataset:
        summary["dataset"] = args.dataset
    if args.steps_completed is not None:
        summary["steps_completed"] = args.steps_completed
    if args.final_reward is not None:
        summary["final_reward"] = args.final_reward
    if args.final_loss is not None:
        summary["final_loss"] = args.final_loss
    if args.source == "probe":
        summary["probe_ok"] = args.probe_ok
        if args.model_advertised:
            summary["model_advertised"] = args.model_advertised
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())