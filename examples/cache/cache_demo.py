#!/usr/bin/env python3
"""Reproducible demo for the AReno dataset tokenization cache (Issue #206).

Runs ``areno train`` on a tiny math dataset with ``--dataset-cache-path`` set,
then prints the cache events from stderr -- they are emitted by the ``areno``
logger (INFO level) to **stderr**, not stdout, so a naive ``stdout`` capture
sees nothing. The first epoch always logs ``stage=dataset_cache_miss`` and
writes the artifact; a subsequent run with the same fingerprint logs
``stage=dataset_cache_hit``. Adjust verbosity with the ``ARENO_LOG_LEVEL``
environment variable (default ``INFO``).

Heavy knobs are environment-overridable so the same script scales from a
smoke test to the multi-GPU run it was written to reproduce::

    ARENO_CKPT=Qwen/Qwen3-0.8B ARENO_WORLD_SIZE=2 python examples/cache/cache_demo.py

This script does not edit any source tree; it only generates a small dataset
under a temp directory and shells out to the ``areno`` CLI.

Run::

    python examples/cache/cache_demo.py               # one-GPU smoke (world_size=1)
    ARENO_WORLD_SIZE=2 python examples/cache/cache_demo.py   # reproduce the dp=2 case
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# Resolve the repo root so --reward-fn-path is absolute and not CWD-dependent.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CKPT = os.environ.get("ARENO_CKPT", "Qwen/Qwen3-0.6B")
_TP_SIZE = int(os.environ.get("ARENO_TP_SIZE", "1"))
_WORLD_SIZE = int(os.environ.get("ARENO_WORLD_SIZE", "1"))

# A few one-step arithmetic rows. The default dataset loader passes JSONL
# through untouched, so each row needs a `prompt` (for rollout) and a
# `solutions` list: the rollout scorer maps item.solutions -> reward record's
# `answer`, which examples/math/math_verify_reward.py:reward_fn reads.
_MINIMAL_ROWS = [
    {"prompt": "What is 7 + 5? Put the answer in \\boxed{}.", "solutions": ["12"]},
    {"prompt": "What is 9 - 4? Put the answer in \\boxed{}.", "solutions": ["5"]},
    {"prompt": "What is 3 * 8? Put the answer in \\boxed{}.", "solutions": ["24"]},
    {"prompt": "What is 20 / 5? Put the answer in \\boxed{}.", "solutions": ["4"]},
]


def write_minimal_dataset(path: str) -> None:
    """Write a tiny math JSONL if it does not already exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in _MINIMAL_ROWS:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    test_dir = os.environ.get("ARENO_CACHE_DEMO_DIR", "/tmp/areno-cache-demo")
    os.makedirs(test_dir, exist_ok=True)

    dataset_path = os.path.join(test_dir, "data.jsonl")
    cache_path = os.path.join(test_dir, "cache")
    reward_fn_path = os.path.join(_REPO_ROOT, "examples", "math", "math_verify_reward.py")

    if not os.path.exists(dataset_path):
        write_minimal_dataset(dataset_path)
        print(f"[demo] wrote minimal dataset -> {dataset_path}")

    cmd = [
        "areno", "train",
        "--ckpt", _CKPT,
        "--dataset-path", dataset_path,
        "--reward-fn-path", reward_fn_path,
        "--algo", "gspo",
        "--tp-size", str(_TP_SIZE),
        "--world-size", str(_WORLD_SIZE),
        "--batch-size", "2",
        "--n-samples", "2",
        "--mini-bs", "1",
        "--epochs", "2",
        "--dataset-cache-path", cache_path,
    ]
    print("[demo] running:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Config panel is echoed to stdout; cache events go to stderr.
    print("=== returncode:", result.returncode)
    print("=== stdout (tail 3000) ===")
    print(result.stdout[-3000:])

    print("=== cache events (from stderr) ===")
    events = [line for line in result.stderr.split("\n") if "stage=dataset_cache_" in line]
    for line in events:
        print(line)

    if not events:
        # No events usually means the run died before reaching rollout (e.g.
        # model hub resolution failure) -- surface the stderr tail to debug.
        print("=== no cache events; stderr (tail 3000) below ===")
        print(result.stderr[-3000:])

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())