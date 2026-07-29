#!/usr/bin/env python3
"""Summarize JSONL emitted by the AReno GPU or process monitors."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def stats(values: list[float]) -> dict[str, float] | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
        "last": finite[-1],
    }


def summarize_gpu(records: list[dict]) -> dict:
    by_gpu: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    process_memory: dict[int, list[float]] = defaultdict(list)
    for record in records:
        for gpu in record.get("gpus", []):
            index = gpu["index"]
            for key in ("utilization_percent", "memory_used_mib", "memory_total_mib", "power_watts"):
                if gpu.get(key) is not None:
                    by_gpu[index][key].append(gpu[key])
        for process in record.get("target_processes", []):
            if process.get("memory_used_mib") is not None:
                process_memory[process["pid"]].append(process["memory_used_mib"])
    return {
        "kind": "gpu",
        "samples": len(records),
        "gpus": {
            str(index): {key: stats(values) for key, values in fields.items()}
            for index, fields in sorted(by_gpu.items())
        },
        "target_process_memory_mib": {str(pid): stats(values) for pid, values in sorted(process_memory.items())},
    }


def summarize_process(records: list[dict]) -> dict:
    totals: dict[str, list[float]] = defaultdict(list)
    for record in records:
        processes = record.get("processes", [])
        totals["process_count"].append(len(processes))
        for key in ("cpu_percent", "rss_bytes", "threads", "read_bytes", "write_bytes"):
            totals[key].append(sum(process.get(key, 0) for process in processes))
    return {
        "kind": "process",
        "samples": len(records),
        "process_tree": {key: stats(values) for key, values in totals.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError("monitor file contains no records")
        if "gpus" in records[0]:
            result = summarize_gpu(records)
        elif "processes" in records[0]:
            result = summarize_process(records)
        else:
            raise ValueError("unrecognized monitor record type")
        result["ok"] = True
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
