#!/usr/bin/env python3
"""Seed the AReno dashboard state file with synthetic jobs for testing.

Usage:
    python scripts/seed_dashboard.py [--count 50] [--output .areno-dashboard-state.json]

The generated jobs cover a mix of states, types, algorithms, models, and
datasets so that search, filtering, and sorting can be exercised without
running real training tasks.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from uuid import uuid4

from areno.dashboard.server import _launch_value


STATES = ["running", "succeeded", "failed", "stopped", "exited"]
ALGOS = ["sft", "dpo", "gspo", "grpo", "ppo"]
MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-0.5B",
]
DATASETS = ["gsm8k:main", "yahma/alpaca-cleaned:train", "openai/gsm8k", "math", "tictactoe.jsonl"]

DEFAULT_OUTPUT = Path(".areno-dashboard-state.json")


def _make_sections_launch(launch_items):
    """Build a sections-format launch config matching real CLI output."""
    sections = []
    current_title = "Basic"
    current_items = []
    for key, value in launch_items:
        if key in ("world_size", "tp_size", "attn_backend"):
            if current_items:
                sections.append({"title": current_title, "items": current_items})
            current_title = "Runtime"
            current_items = []
        elif key in ("batch_size", "n_samples", "max_running_prompts", "max_new_tokens"):
            if current_items:
                sections.append({"title": current_title, "items": current_items})
            current_title = "Rollout"
            current_items = []
        current_items.append({"key": key, "value": value})
    if current_items:
        sections.append({"title": current_title, "items": current_items})
    return {"sections": sections}


def make_job(index: int) -> dict:
    kind = random.choice(["train", "serve"])
    algo = random.choice(ALGOS)
    model = random.choice(MODELS)
    dataset = random.choice(DATASETS)
    status = random.choice(STATES)
    now_ts = time.time()
    created_ts = now_ts - random.randint(60, 86400 * 7)
    updated_ts = created_ts + random.randint(10, 3600)
    step = random.randint(0, 200)
    # Half the jobs use sections format (like real CLI), half use flat dict (like dashboard launcher)
    use_sections = random.random() < 0.5

    if kind == "serve":
        if use_sections:
            launch = _make_sections_launch([("model_path", model), ("host", "0.0.0.0"), ("port", 8000)])
        else:
            launch = {"model_path": model, "host": "0.0.0.0", "port": 8000}
        name = f"serve {model}"
    else:
        if use_sections:
            launch = _make_sections_launch([
                ("algo", algo), ("ckpt", model), ("dataset_path", dataset),
                ("model_hub", "modelscope"),
                ("world_size", random.choice([1, 2, 4])),
                ("tp_size", random.choice([1, 2])),
                ("attn_backend", "flash"),
            ])
        else:
            launch = {"algo": algo, "ckpt": model, "dataset_path": dataset, "model_hub": "modelscope"}
        name = f"train {algo} {model}"

    return {
        "id": uuid4().hex[:12],
        "kind": kind,
        "name": name,
        "command": ["areno", kind, "--ckpt", model],
        "config": {},
        "config_text": "",
        "launch": launch,
        "metrics_dir": f"/tmp/areno/tfevent/run-{index}",
        "status": status,
        "stage": "registered" if status == "running" else "exited",
        "role": "",
        "step": step,
        "created_at": created_ts,
        "updated_at": updated_ts,
        "returncode": 0 if status == "succeeded" else (1 if status == "failed" else None),
        "pid": None,
        "logs": [f"registered AReno command: areno {kind}"],
        "metrics_count": random.randint(0, 50),
        "samples": [],
        "timeperf": [],
        "perf": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed dashboard state with synthetic jobs.")
    parser.add_argument("--count", type=int, default=50, help="Number of synthetic jobs to generate.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output state file path.")
    args = parser.parse_args()

    jobs = [make_job(i) for i in range(args.count)]
    payload = {"jobs": jobs}

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.count} synthetic jobs to {args.output}")
    print(f"States: {', '.join(sorted(set(j['status'] for j in jobs)))}")
    algos = sorted(set(filter(None, (_launch_value(j['launch'], 'algo') for j in jobs))))
    print(f"Algorithms: {', '.join(algos)}")
    print(f"Types: {', '.join(sorted(set(j['kind'] for j in jobs)))}")


if __name__ == "__main__":
    main()