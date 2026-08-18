"""Data containers and helpers shared across engine, runtime and serving.

`batch` defines the dataclasses returned to the user (rollouts, train stats,
sampling parameters) plus tree-walking helpers to move them between devices.
Submodules `rollout_state`, `sampling`, and `tokenizer` are imported on demand
by the runtime and worker layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areno.engine.data.batch import RolloutOutput, SamplingParams, TrainStats, to_cpu, to_device


def __getattr__(name: str):
    """Load Torch-backed containers only when the CUDA engine requests them."""

    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module

    batch = import_module("areno.engine.data.batch")
    value = getattr(batch, name)
    globals()[name] = value
    return value


__all__ = ["RolloutOutput", "SamplingParams", "TrainStats", "to_cpu", "to_device"]
