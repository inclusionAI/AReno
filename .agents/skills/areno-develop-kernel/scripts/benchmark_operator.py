#!/usr/bin/env python3
"""Benchmark a unary tensor operator with synchronized timings."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time


def resolve(spec: str):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("callable must be module:function")
    return getattr(importlib.import_module(module), name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--callable", required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    result: dict = {"ok": False}
    try:
        import torch

        function = resolve(args.callable)
        shape = tuple(int(item) for item in args.shape.split(","))
        tensor = torch.randn(shape, device=args.device, dtype=getattr(torch, args.dtype))
        synchronize = torch.cuda.synchronize if tensor.is_cuda else lambda: None
        for _ in range(args.warmup):
            function(tensor)
        synchronize()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            function(tensor)
            synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e3)
        ordered = sorted(samples)
        result = {
            "ok": True,
            "shape": shape,
            "dtype": args.dtype,
            "device": args.device,
            "iterations": len(samples),
            "mean_us": statistics.fmean(samples),
            "median_us": statistics.median(samples),
            "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
