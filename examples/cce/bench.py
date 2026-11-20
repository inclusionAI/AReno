"""Reproducible baseline for the opt-in cut cross-entropy path.

Compares ``F.cross_entropy(hidden @ weight.T, targets)`` (materialized logits)
with ``areno.accel.cut_cross_entropy`` for the same shape, reporting loss
agreement, wall time for forward+backward, and the peak resident-set growth of
the process.

Peak RSS is process-lifetime, so run one mode per process when you want a
clean memory number:

    python examples/cce/bench.py --mode materialized
    python examples/cce/bench.py --mode cce

Usage:
    python examples/cce/bench.py [--rows 2048] [--hidden 1024] [--vocab 65536]
                                 [--chunk 4096] [--iters 5] [--mode both]
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from areno.accel.cut_cross_entropy import cut_cross_entropy


def _peak_rss_mb() -> float | None:
    """Peak resident set size of this process, or None when unavailable."""

    try:
        import resource
    except ModuleNotFoundError:
        # Windows: getrusage is unavailable; psutil exposes the peak working set.
        try:
            import psutil
        except ModuleNotFoundError:
            return None
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return peak / (1024 * 1024)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def _parameters(rows: int, hidden: int, vocab: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(0)
    hidden_states = torch.randn(rows, hidden, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(vocab, hidden, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(vocab, (rows,), device=device)
    return hidden_states, weight, targets


def _run(mode: str, hidden_states, weight, targets, chunk: int) -> float:
    if mode == "materialized":
        return F.cross_entropy(hidden_states @ weight.T, targets)
    return cut_cross_entropy(hidden_states, weight, targets, vocab_chunk_size=chunk)


def _measure(mode: str, args, dtype: torch.dtype, device: torch.device) -> dict:
    hidden_states, weight, targets = _parameters(args.rows, args.hidden, args.vocab, dtype, device)
    start_rss = _peak_rss_mb()
    times = []
    loss = None
    for _ in range(args.iters):
        hidden_states.grad = None
        weight.grad = None
        started = time.perf_counter()
        loss = _run(mode, hidden_states, weight, targets, args.chunk)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    end_rss = _peak_rss_mb()
    growth = None if start_rss is None or end_rss is None else end_rss - start_rss
    return {
        "mode": mode,
        "loss": float(loss),
        "median_ms": statistics.median(times) * 1e3,
        "peak_rss_growth_mb": growth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--chunk", type=int, default=4096, help="vocabulary rows projected per CCE chunk")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["both", "materialized", "cce"], default="both")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    modes = ["materialized", "cce"] if args.mode == "both" else [args.mode]

    print(
        f"rows={args.rows} hidden={args.hidden} vocab={args.vocab} "
        f"chunk={args.chunk} dtype={args.dtype} device={args.device}"
    )
    # Peak RSS is process-lifetime: `--mode both` under-reports the second row,
    # so run one mode per process for memory numbers that can be compared.
    print("| path | loss | fwd+bwd median ms | peak RSS growth MB |")
    print("| --- | --- | --- | --- |")
    for mode in modes:
        result = _measure(mode, args, dtype, device)
        growth = "n/a" if result["peak_rss_growth_mb"] is None else f"{result['peak_rss_growth_mb']:.1f}"
        print(f"| {mode} | {result['loss']:.6f} | {result['median_ms']:.2f} | {growth} |")


if __name__ == "__main__":
    main()
