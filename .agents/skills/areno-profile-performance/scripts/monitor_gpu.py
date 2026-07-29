#!/usr/bin/env python3
"""Sample NVIDIA GPU utilization, memory, and target-process residency."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time
from pathlib import Path

GPU_QUERY = "index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw"
PROCESS_QUERY = "gpu_uuid,pid,process_name,used_gpu_memory"


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def process_tree_pids(root_pids: list[int], include_children: bool) -> set[int]:
    selected = set(root_pids)
    if not include_children:
        return selected
    try:
        import psutil
    except ImportError:
        return selected
    for pid in root_pids:
        try:
            selected.update(child.pid for child in psutil.Process(pid).children(recursive=True))
        except psutil.Error:
            continue
    return selected


def query_csv(arguments: list[str]) -> list[list[str]]:
    process = subprocess.run(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return [row for row in csv.reader(io.StringIO(process.stdout), skipinitialspace=True) if row]


def sample(target_pids: set[int]) -> dict:
    gpu_rows = query_csv([f"--query-gpu={GPU_QUERY}"])
    try:
        process_rows = query_csv([f"--query-compute-apps={PROCESS_QUERY}"])
    except subprocess.CalledProcessError:
        process_rows = []
    processes = [
        {
            "gpu_uuid": row[0],
            "pid": int(row[1]),
            "name": row[2],
            "memory_used_mib": number(row[3]),
        }
        for row in process_rows
        if len(row) >= 4 and (not target_pids or int(row[1]) in target_pids)
    ]
    return {
        "timestamp": time.time(),
        "gpus": [
            {
                "index": int(row[0]),
                "uuid": row[1],
                "name": row[2],
                "utilization_percent": number(row[3]),
                "memory_used_mib": number(row[4]),
                "memory_total_mib": number(row[5]),
                "power_watts": number(row[6]),
            }
            for row in gpu_rows
        ],
        "target_processes": processes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, action="append", default=[], help="Target PID; may be repeated")
    parser.add_argument("--no-children", action="store_true", help="Do not include descendants of target PIDs")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Write JSON Lines here instead of stdout")
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")

    output = args.output.open("w", encoding="utf-8") if args.output else None
    deadline = time.monotonic() + args.duration
    count = 0
    try:
        while time.monotonic() < deadline:
            record = sample(process_tree_pids(args.pid, not args.no_children))
            line = json.dumps(record, sort_keys=True)
            print(line, file=output, flush=True)
            count += 1
            time.sleep(min(args.interval, max(deadline - time.monotonic(), 0)))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    finally:
        if output:
            output.close()
    if args.output:
        print(json.dumps({"ok": True, "samples": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
