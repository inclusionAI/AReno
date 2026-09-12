"""GPU bench + correctness for the Hopper FP8 decode-linear (torch._scaled_mm).

Validates the integrated A8W8 path (``quantized_fp8_scaled_mm`` over the uint8
E4M3 payload) against the bf16 reference, then times it vs the bf16
``areno_linear`` cuBLAS path at Qwen3-8B shapes under CUDA-graph replay —
decode runs under graphs, and without them launch overhead hides the win at
small M. Requires Hopper/Ada (cc >= 8.9):

    CUDA_VISIBLE_DEVICES=0 python scripts/bench/fp8_scaled_mm_bench.py
"""

from __future__ import annotations

import torch

from areno.accel import areno_linear
from areno.accel.kernels.fp8_scaled_mm import quantized_fp8_scaled_mm, scaled_mm_available


def quantize_weight_e4m3(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 quantize; matches areno.engine.quantization semantics."""
    amax = w.abs().amax()
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax)).to(torch.float32)
    q = (w.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


_SHAPES = [
    (1, 6144, 4096),  # qkv proj
    (1, 4096, 4096),  # o proj
    (1, 24576, 4096),  # gate_up proj
    (1, 4096, 12288),  # down proj
    (4, 24576, 4096),  # batch decode
    (64, 24576, 4096),  # prefill-ish
]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs().max().item() + 1e-6
    return d / denom


def bench_graphed(fn, reps: int = 200, warm: int = 50) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    for _ in range(warm):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def main() -> None:
    torch.manual_seed(0)
    if not scaled_mm_available():
        raise RuntimeError("torch._scaled_mm (FP8 A8W8) requires Hopper/Ada (cc >= 8.9); this bench is Hopper-only")

    print("=== correctness vs bf16 reference ===")
    for M, N, K in [(1, 4096, 4096), (4, 24576, 4096)]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * (4.0 / (K**0.5))
        w_u8, scale = quantize_weight_e4m3(w)
        out = quantized_fp8_scaled_mm(x, w_u8, scale)
        ref = x @ w.T
        print(
            f"  M={M:3d} N={N:5d} K={K:5d}: shape={tuple(out.shape)} rel_vs_bf16={rel_err(out, ref):.4f} "
            f"finite={bool(torch.isfinite(out).all())}"
        )

    print("\n=== decode throughput vs bf16 areno_linear (CUDA-graph replay) ===")
    print(f"{'M':>4} {'N':>6} {'K':>6} {'bf16(ms)':>10} {'fp8(ms)':>10} {'speedup':>8} {'fp8GB/s':>8}")
    for M, N, K in _SHAPES:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * (4.0 / (K**0.5))
        w_u8, scale = quantize_weight_e4m3(w)
        ybuf = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

        def bf16_call():
            ybuf.copy_(areno_linear(x, w, None))

        def fp8_call():
            ybuf.copy_(quantized_fp8_scaled_mm(x, w_u8, scale))

        t_bf16 = bench_graphed(bf16_call)
        t_fp8 = bench_graphed(fp8_call)
        mb = N * K / 1e6
        gbs = mb * 1e6 / (t_fp8 * 1e-3) / 1e9
        print(f"{M:4d} {N:6d} {K:6d} {t_bf16:10.4f} {t_fp8:10.4f} {t_bf16 / t_fp8:8.3f}x {gbs:8.1f}")


if __name__ == "__main__":
    main()
