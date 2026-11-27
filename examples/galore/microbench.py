"""Measure GaLore optimizer state, step cost and projection-refresh workspace on one GPU.

Two measurements back the GaLore report:

* a 1024 x 1024 weight at rank 128: persistent optimizer bytes of GaLoreAdamW
  versus the FP32 AdamW in ``AdamWFP32Master`` (FP32 and BF16 weights), plus the
  median time and allocator peak of steady and projection-refresh steps after
  two warm-up steps;
* every distinct projection block shape of Qwen3-0.6B and Qwen3-1.7B: time and
  allocator workspace of one basis refresh (``top_singular_basis``) next to a
  full ``torch.linalg.svd`` of the same gradient.

    python examples/galore/microbench.py --output microbench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from areno.engine.layers.linear import mark_linear_weight
from areno.engine.optim import AdamWFP32Master, GaLoreAdamW
from areno.engine.optim.galore import top_singular_basis

BLOCK_SHAPES = {
    "Qwen3-0.6B": {
        "q_proj": (2048, 1024),
        "k_proj/v_proj": (1024, 1024),
        "o_proj": (1024, 2048),
        "gate_proj/up_proj": (3072, 1024),
        "down_proj": (1024, 3072),
    },
    "Qwen3-1.7B": {
        "q_proj/o_proj": (2048, 2048),
        "k_proj/v_proj": (1024, 2048),
        "gate_proj/up_proj": (6144, 2048),
        "down_proj": (2048, 6144),
    },
}


def _cuda_timed(fn) -> tuple[float, int]:
    """Return seconds and peak allocated bytes above the pre-call level."""

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, torch.cuda.max_memory_allocated() - base


def square_matrix(dtype: torch.dtype, steps: int, rank: int, gap: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(0)
    result = {}
    for name in ("adamw", "galore"):
        param = torch.nn.Parameter((0.02 * torch.randn(1024, 1024, generator=generator, device="cuda")).to(dtype))
        mark_linear_weight(param, (1024,))
        kwargs = {"lr": 1.0e-4, "betas": (0.9, 0.999), "weight_decay": 0.0}
        if name == "galore":
            optimizer = GaLoreAdamW([param], rank=rank, update_proj_gap=gap, **kwargs)
        else:
            optimizer = AdamWFP32Master([param], **kwargs)
        steady, refresh = [], []
        for step in range(1, steps + 1):
            # Training hands optimizers an accumulated FP32 main_grad.
            param.main_grad = torch.randn(1024, 1024, generator=generator, device="cuda")
            seconds, peak = _cuda_timed(optimizer.step)
            if step > 2:
                is_refresh = name == "galore" and (step - 1) % gap == 0
                (refresh if is_refresh else steady).append((seconds, peak))
        if name == "galore":
            state = optimizer.state_memory_metrics()
            persistent = {
                "projection_bytes": state["projection_bytes"],
                "moment_bytes": state["moment_bytes"],
                "master_bytes": state["master_bytes"],
            }
        else:
            bucket = optimizer.buckets[0]
            persistent = {
                "moment_bytes": bucket.exp_avg.nbytes + bucket.exp_avg_sq.nbytes,
                # FP32 weights are their own master; BF16 weights keep low bits.
                "master_bytes": bucket.master_storage.nbytes if dtype == torch.bfloat16 else 0,
            }
        persistent["state_bytes"] = sum(persistent.values())
        result[name] = {
            "persistent": persistent,
            "steady_median_s": statistics.median(seconds for seconds, _ in steady),
            "steady_max_step_peak_bytes": max(peak for _, peak in steady),
            "refresh_median_s": statistics.median(seconds for seconds, _ in refresh) if refresh else None,
            "refresh_max_step_peak_bytes": max(peak for _, peak in refresh) if refresh else None,
        }
    result["state_ratio"] = result["galore"]["persistent"]["state_bytes"] / result["adamw"]["persistent"]["state_bytes"]
    return result


def refresh_workspace(rank: int, repeats: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(1)
    report = {}
    for model, shapes in BLOCK_SHAPES.items():
        for name, (rows, cols) in shapes.items():
            gradient = torch.randn(rows, cols, generator=generator, device="cuda")
            side = "right" if rows >= cols else "left"
            entry = {"shape": [rows, cols], "side": side, "gradient_bytes": gradient.nbytes}
            for label, fn in (
                ("galore_basis", lambda: top_singular_basis(gradient, rank, side)),
                ("full_svd", lambda: torch.linalg.svd(gradient, full_matrices=False)),
            ):
                fn()
                runs = [_cuda_timed(fn) for _ in range(repeats)]
                entry[f"{label}_median_s"] = statistics.median(seconds for seconds, _ in runs)
                entry[f"{label}_workspace_bytes"] = max(peak for _, peak in runs)
            report[f"{model} {name}"] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update-proj-gap", type=int, default=4)
    parser.add_argument("--steps", type=int, default=22)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "rank": args.rank,
        "square_1024": {
            str(dtype).removeprefix("torch."): square_matrix(dtype, args.steps, args.rank, args.update_proj_gap)
            for dtype in (torch.float32, torch.bfloat16)
        },
        "refresh_workspace": refresh_workspace(args.rank, args.repeats),
    }
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
