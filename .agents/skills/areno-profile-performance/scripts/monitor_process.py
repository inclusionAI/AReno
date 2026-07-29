#!/usr/bin/env python3
"""Sample wall time, CPU, RSS, threads, and I/O for a process tree."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def snapshot(root_pid: int) -> dict:
    import psutil

    root = psutil.Process(root_pid)
    processes = [root, *root.children(recursive=True)]
    rows = []
    for process in processes:
        try:
            memory = process.memory_info()
            io = process.io_counters()
            rows.append(
                {
                    "pid": process.pid,
                    "name": process.name(),
                    "status": process.status(),
                    "cpu_percent": process.cpu_percent(interval=None),
                    "rss_bytes": memory.rss,
                    "threads": process.num_threads(),
                    "read_bytes": io.read_bytes,
                    "write_bytes": io.write_bytes,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {"timestamp": time.time(), "root_pid": root_pid, "processes": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Write JSON Lines here instead of stdout")
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")

    try:
        import psutil
    except ImportError as exc:
        print(json.dumps({"ok": False, "error": f"ImportError: {exc}; install psutil"}))
        return 1
    try:
        root = psutil.Process(args.pid)
        for process in [root, *root.children(recursive=True)]:
            process.cpu_percent(interval=None)
    except psutil.Error as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1

    output = args.output.open("w", encoding="utf-8") if args.output else None
    deadline = time.monotonic() + args.duration
    count = 0
    try:
        while time.monotonic() < deadline and root.is_running():
            print(json.dumps(snapshot(args.pid), sort_keys=True), file=output, flush=True)
            count += 1
            time.sleep(min(args.interval, max(deadline - time.monotonic(), 0)))
    except psutil.Error as exc:
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
