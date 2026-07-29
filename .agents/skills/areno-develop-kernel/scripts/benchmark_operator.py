#!/usr/bin/env python3
"""Benchmark a unary tensor operator with synchronized timings."""

from __future__ import annotations

import importlib
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import Result, build_parser, skill_main


def resolve(spec: str):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("callable must be module:function")
    return getattr(importlib.import_module(module), name)


@skill_main
def main() -> Result:
    parser = build_parser("Benchmark a unary tensor operator with synchronized timings.")
    parser.add_argument("--callable", required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

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
    return Result(
        ok=True,
        data={
            "shape": shape,
            "dtype": args.dtype,
            "device": args.device,
            "iterations": len(samples),
            "mean_us": statistics.fmean(samples),
            "median_us": statistics.median(samples),
            "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())