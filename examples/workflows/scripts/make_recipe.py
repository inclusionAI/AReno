#!/usr/bin/env python3
"""Generate a tiny deterministic training recipe (no network, no GPU).

Part of the ``recipe_to_summary`` example workflow.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Produce a tiny training recipe.")
    parser.add_argument("--algo", default="gspo", help="Algorithm name.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model checkpoint.")
    parser.add_argument("--dataset", default="gsm8k:main", help="Dataset reference.")
    args = parser.parse_args()

    recipe = {
        "ok": True,
        "algo": args.algo,
        "model": args.model,
        "dataset": args.dataset,
        "batch_size": 2,
        "n_samples": 4,
        "max_new_tokens": 64,
    }
    print(json.dumps(recipe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())