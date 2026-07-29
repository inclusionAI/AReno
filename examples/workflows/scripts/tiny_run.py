#!/usr/bin/env python3
"""Simulate a tiny deterministic training run (no network, no GPU).

Part of the ``recipe_to_summary`` example workflow. Accepts recipe fields
from :doc:`make_recipe.py` and produces a fake step/metric snapshot.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate a tiny training run.")
    parser.add_argument("--algo", required=True, help="Algorithm name.")
    parser.add_argument("--model", required=True, help="Model checkpoint.")
    parser.add_argument("--dataset", required=True, help="Dataset reference.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--n-samples", type=int, default=4, help="Sample count.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens.")
    args = parser.parse_args()

    run = {
        "ok": True,
        "algo": args.algo,
        "model": args.model,
        "dataset": args.dataset,
        "steps_completed": 3,
        "final_reward": 0.75,
        "final_loss": 1.2,
        "batch_size": args.batch_size,
        "n_samples": args.n_samples,
    }
    print(json.dumps(run, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())