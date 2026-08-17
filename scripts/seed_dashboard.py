#!/usr/bin/env python3
"""Seed the AReno dashboard state file with synthetic jobs for testing.

Usage:
    python scripts/seed_dashboard.py [--count 50] [--output <path>] [--replace]

The generated jobs cover a mix of states, types, algorithms, models, and
datasets so that search, filtering, and sorting can be exercised without
running real training tasks.

By default the script **merges** synthetic jobs into the existing state
file, preserving any real jobs already present. Use ``--replace`` to
discard existing jobs and write only synthetic data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
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


def _iso(ts: float) -> str:
    """Convert epoch seconds to ISO-8601 string matching dashboard format."""
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).replace(microsecond=0).isoformat()


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
    now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
    created_ts = now_ts - random.randint(60, 86400 * 7)
    updated_ts = created_ts + random.randint(10, 3600)
    step = random.randint(0, 200)
    use_sections = random.random() < 0.5

    if kind == "serve":
        if use_sections:
            launch = _make_sections_launch([("model_path", model), ("host", "0.0.0.0"), ("port", 8000)])
        else:
            launch = {"model_path": model, "host": "0.0.0.0", "port": 8000}
        name = f"serve {model}"
    else:
        if use_sections:
            launch = _make_sections_launch(
                [
                    ("algo", algo),
                    ("ckpt", model),
                    ("dataset_path", dataset),
                    ("model_hub", "modelscope"),
                    ("world_size", random.choice([1, 2, 4])),
                    ("tp_size", random.choice([1, 2])),
                    ("attn_backend", "flash"),
                ]
            )
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
        "created_at": _iso(created_ts),
        "updated_at": _iso(updated_ts),
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
    parser.add_argument("--replace", action="store_true", help="Replace existing jobs instead of merging.")
    args = parser.parse_args()

    existing_jobs = []
    if args.output.exists() and not args.replace:
        try:
            data = json.loads(args.output.read_text(encoding="utf-8"))
            existing_jobs = data.get("jobs", [])
            if not isinstance(existing_jobs, list):
                existing_jobs = []
        except Exception:
            existing_jobs = []

    new_jobs = [make_job(i) for i in range(args.count)]
    all_jobs = existing_jobs + new_jobs

    payload = {"jobs": all_jobs}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    action = "Replaced" if args.replace else "Merged"
    print(f"{action} {args.count} synthetic jobs into {args.output} ({len(all_jobs)} total)")
    print(f"States: {', '.join(sorted(set(j['status'] for j in all_jobs)))}")
    algos = sorted(set(filter(None, (_launch_value(j.get("launch", {}), "algo") for j in all_jobs))))
    print(f"Algorithms: {', '.join(algos)}")
    print(f"Types: {', '.join(sorted(set(j['kind'] for j in all_jobs)))}")


if __name__ == "__main__":
    main()
