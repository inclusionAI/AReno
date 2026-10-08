"""Conservative planning estimates, not a measured CUDA memory probe."""

from __future__ import annotations

import math
import re

# Modal's documented GPU capacities. Reserve 20% for allocator/engine headroom.
GPU_GIB = {"L4": 24, "A10G": 24, "L40S": 48, "A100-40GB": 40, "A100-80GB": 80, "H100": 80, "H200": 141, "B200": 192}


def positive(value, default):
    if isinstance(value, bool):
        raise ValueError("Memory estimation parameters must be numeric")
    number = float(default if value in (None, "") else value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Memory estimation parameters must be finite and positive")
    return number


def enabled(value):
    return value is True or str(value).lower() == "true"


def recommend(request, rates):
    model = request.get("model", {})
    checkpoint = str(model.get("checkpoint", ""))
    billions = model.get("parameters_billion")
    source = "User-provided parameter count"
    if billions in (None, ""):
        sizes = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)([BM])(?=$|[^a-z])", checkpoint, re.I)
        mixture = re.search(r"(\d+)x(\d+(?:\.\d+)?)B", checkpoint, re.I)
        if not sizes and not mixture:
            return {
                "available": False,
                "reason": "Model size is unknown. Enter total parameters (billions), including all MoE experts.",
            }
        billions = max((float(value) / (1000 if unit.lower() == "m" else 1) for value, unit in sizes), default=0)
        if mixture:
            billions = max(billions, float(mixture[1]) * float(mixture[2]))
        source = "Approximate total parameter count from model name; verify for custom or MoE models"
    billions = positive(billions, 1)
    deployment = request.get("kind") == "deployment"
    stages = [{"algo": "serve", "params": request.get("serve", {})}] if deployment else request.get("stages", [])
    if not stages:
        return {"available": False, "reason": "Select a training algorithm or serving task."}
    estimates = []
    count = 1
    for stage in stages:
        params = stage.get("params", {})
        tp = positive(params.get("tp_size"), 1)
        world = positive(params.get("world_size"), tp)
        if not tp.is_integer() or not world.is_integer():
            raise ValueError("TP and world size must be integers")
        tp, world = int(tp), int(world)
        if not 1 <= tp <= world <= 8 or world % tp:
            raise ValueError("Use world_size divisible by tp_size, between 1 and 8")
        count = max(count, world)
        weights = billions * 1e9 * 2 / 2**30 / tp
        rl = stage.get("algo") in ("grpo", "gspo", "ppo")
        sequence = positive(
            params.get("max_context_len") or params.get("max_cache_len") or params.get("max_seq_len"), 4096
        )
        microbatch = positive(params.get("mini_bs"), 1)
        concurrency = positive(
            params.get("max_running_prompts") or params.get("max_running_requests"), 32 if rl or deployment else 1
        )
        adam4 = not enabled(params.get("adam_8bit")) and params.get("adam_4bit") not in (False, "false", "False")
        optimizer_bytes = 0.625 if adam4 else 2.125 if enabled(params.get("adam_8bit")) else 8
        components = {"weights": weights, "workspace": 4.0}
        if not deployment:
            components.update(
                gradients=weights,
                master_weights=weights * (2.125 / 2),
                optimizer=weights * optimizer_bytes / 2,
                activations=max(2, weights * 0.25 * sequence / 2048 * microbatch),
            )
            if stage.get("algo") == "dpo" or rl:
                components["reference_weights"] = weights
            if stage.get("algo") == "ppo":
                # Critic training state plus reward model; conservative same-size assumption.
                components["critic_and_reward"] = weights * (2 + (2.125 + optimizer_bytes) / 2) + weights
        if deployment or rl:
            components["kv_cache"] = max(2, weights * 0.125 * sequence / 2048 * concurrency)
        estimates.append(
            {
                "algo": stage.get("algo"),
                "optimizer": "none"
                if deployment
                else "adam_4bit"
                if adam4
                else "adam_8bit"
                if enabled(params.get("adam_8bit"))
                else "adam_fp32",
                "components_gib": components,
                "required_gib_per_gpu": sum(components.values()) / 0.8,
            }
        )
    peak = max(estimates, key=lambda item: item["required_gib_per_gpu"])
    candidates = [
        {
            "gpu": gpu,
            "count": count,
            "memory_gib_per_gpu": capacity,
            "gpu_hourly_cost": float(rates["gpu_per_second"][gpu]) * 3600 * count,
        }
        for gpu, capacity in GPU_GIB.items()
        if capacity >= peak["required_gib_per_gpu"] and gpu in rates["gpu_per_second"] and (gpu != "A10G" or count <= 4)
    ]
    candidates.sort(key=lambda item: (item["gpu_hourly_cost"], item["memory_gib_per_gpu"]))
    return {
        "available": bool(candidates),
        "recommended": candidates[0] if candidates else None,
        "reason": None
        if candidates
        else "No supported GPU fits this estimate at the selected tensor parallel size. Increase TP/world size or reduce sequence length/concurrency.",
        "parameters_billion": billions,
        "parameter_source": source,
        "required_gib_per_gpu": round(peak["required_gib_per_gpu"], 2),
        "stages": estimates,
        "assumptions": [
            "BF16 weights and gradients; Adam 4-bit uses packed first moments, factored variance and compact master metadata.",
            "20% headroom; activation and KV-cache allowances are conservative heuristics, not an architecture-specific CUDA probe.",
            "DP optimizer sharding and LoRA savings are not deducted. Separate RL models are assumed the same size. Check actual GPU memory during the run.",
        ],
        "candidates": candidates,
        "pricing_source": rates.get("source"),
    }
