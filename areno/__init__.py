"""Top-level areno package.

Sets process-wide knobs that must be in place before any CUDA/Triton kernel or
torch.compile call runs: a single CUDA stream for collectives and a generous
TorchDynamo cache so the engine can compile many specialized graphs (per shape
bucket, prefill vs decode, train vs infer) without thrashing.

Exposes the user-facing surface: configuration dataclasses, the rollout
output container, and the `ArenoEngine` coordinator.
"""

from __future__ import annotations

import os

# A single CUDA stream connection keeps NCCL collectives ordered with compute,
# which is what areno's TP/DP all-reduce + all-gather patterns assume.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

try:
    import torch._dynamo as _dynamo
except ModuleNotFoundError:
    _dynamo = None

if _dynamo is not None:
    # Train, prefill, decode, scoring and multiple shape buckets all produce
    # distinct compiled artifacts; raise the cache limits so recompilation does
    # not evict graphs that will be replayed across RL steps.
    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
    try:
        _dynamo.config.accumulated_cache_size_limit = max(_dynamo.config.accumulated_cache_size_limit, 256)
    except AttributeError:
        pass

from areno.engine.log import configure_default_logging

configure_default_logging()


def __getattr__(name: str):
    """Lazily expose engine symbols without importing kernel-heavy modules."""

    if name == "ArenoEngine":
        from areno.engine import ArenoEngine

        return ArenoEngine
    if name in {"EngineConfig", "ModelConfig", "OptimizerConfig", "RuntimeConfig"}:
        from areno.engine import config

        return getattr(config, name)
    if name in {"RolloutOutput", "SamplingParams", "TrainStats"}:
        from areno.engine import data

        return getattr(data, name)
    if name == "Trainer":
        from areno.api.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArenoEngine",
    "EngineConfig",
    "ModelConfig",
    "OptimizerConfig",
    "RolloutOutput",
    "RuntimeConfig",
    "SamplingParams",
    "TrainStats",
    "Trainer",
]
