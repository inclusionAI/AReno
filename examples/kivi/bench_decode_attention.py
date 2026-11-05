"""Kernel-level decode benchmark: KIVI paged cache vs flash-attn BF16 paged cache.

One decode step of one attention layer, at fixed rows (concurrency) and cached
context. The KIVI step includes the residual-group flush, the ring append and
the fused dequantize + split-K attention, i.e. everything the runtime launches
per layer. The BF16 step is ``flash_attn_with_kvcache`` with the in-place append
used by AReno's flash backend. Reports CUDA-event step times (2 warmup + N
measured iterations), cache bytes and the peak allocation including workspace.

    python examples/kivi/bench_decode_attention.py --rows 1 8 32 128 --context 1024 4096 8192 32768
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

BLOCK_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--context", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtypes", nargs="+", default=["bf16", "kivi2", "kivi4"])
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture each step in a CUDA graph and time replays, as AReno's decode loop runs.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _time(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def bench_one(args, dtype: str, rows: int, context: int) -> dict:
    from flash_attn import flash_attn_with_kvcache

    from areno.engine.layers.attention_backend.kivi import KIVI_BITS, KiviInferBackend, KiviKVCache, KiviSpec
    from areno.engine.runtime.metadata import InferMeta

    device = torch.device("cuda")
    blocks_per_row = -(-(context + args.warmup + args.iters + 1) // BLOCK_SIZE)
    block_table = torch.arange(rows * blocks_per_row, device=device, dtype=torch.int32).view(rows, blocks_per_row)
    q = torch.randn(rows, 1, args.q_heads, args.head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(rows, 1, args.kv_heads, args.head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # Every iteration decodes at the same cached length so steps are comparable.
    seqlens = torch.full((rows,), context, device=device, dtype=torch.int32)
    torch.cuda.empty_cache()
    if dtype == "bf16":
        shape = (rows * blocks_per_row, BLOCK_SIZE, args.kv_heads, args.head_dim)
        k_cache = torch.randn(shape, device=device, dtype=torch.bfloat16)
        v_cache = torch.randn(shape, device=device, dtype=torch.bfloat16)
        cache_bytes = 2 * k_cache.numel() * k_cache.element_size()

        def step():
            return flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v, cache_seqlens=seqlens, block_table=block_table, causal=True
            )

    else:
        spec = KiviSpec(bits=KIVI_BITS[dtype], group_size=args.group_size, residual_length=args.residual_length)
        cache = KiviKVCache.allocate(
            spec,
            num_blocks=rows * blocks_per_row,
            block_size=BLOCK_SIZE,
            num_slots=rows,
            num_kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        for tensor in cache.quantized_tensors:
            tensor.copy_(torch.randint(0, 255, tensor.shape, device=device).to(tensor.dtype))
        cache.k_scale.uniform_(0.01, 0.1)
        cache.v_scale.uniform_(0.01, 0.1)
        cache.k_ring.normal_()
        cache.v_ring.normal_()
        cache_bytes = cache.nbytes()
        backend = KiviInferBackend("flash")
        meta = InferMeta(
            mode="decode",
            cache_seqlens=seqlens,
            block_table=block_table,
            recurrent_slots=torch.arange(rows, device=device),
        )

        def step():
            return backend(q, k, v, cache, meta)

    # Setup temporaries are gone; measure only what the step itself adds.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    if args.cuda_graph:
        step()  # compile kernels outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        times = _time(graph.replay, args.warmup, args.iters)
    else:
        times = _time(step, args.warmup, args.iters)
    peak_allocated = torch.cuda.max_memory_allocated()
    tokens_read = rows * (context + 1)
    return {
        "dtype": dtype,
        "rows": rows,
        "context": context,
        "median_ms": statistics.median(times),
        "p10_ms": sorted(times)[len(times) // 10],
        "p90_ms": sorted(times)[(9 * len(times)) // 10],
        "cache_bytes": cache_bytes,
        "step_workspace_bytes": peak_allocated - base,
        "total_bytes": cache_bytes + peak_allocated - base,
        # Process-wide allocator peaks while stepping (cache, inputs, workspace).
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "kv_tokens_per_ms": tokens_read / statistics.median(times),
    }


def main() -> None:
    args = parse_args()
    results = []
    for context in args.context:
        for rows in args.rows:
            baseline = None
            for dtype in args.dtypes:
                row = bench_one(args, dtype, rows, context)
                if dtype == "bf16":
                    baseline = row["median_ms"]
                if baseline is not None:
                    row["speedup_vs_bf16"] = baseline / row["median_ms"]
                results.append(row)
                print(json.dumps(row))
                torch.cuda.empty_cache()
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "args": vars(args),
                    "results": results,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
