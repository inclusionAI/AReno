"""Top-level batch dataclasses returned by the engine API.

Defines the small, picklable Python containers that the engine sends back to
the user: training stats, sampling parameters, and the padded rollout output.
The `to_device` / `to_cpu` helpers walk arbitrary container trees so engine
internals can move whole payloads (often containing nested dicts and tensors)
between devices in a single call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class TrainStats:
    """Metrics returned by one worker train step."""

    loss: float
    stepped: bool = True
    metrics: dict[str, float] | None = None


@dataclass(slots=True)
class SamplingParams:
    """Sampling controls consumed by areno rollout."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_new_tokens: int = 0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    suppress_token_ids: tuple[int, ...] = ()
    suppress_special_tokens: bool = True


@dataclass(slots=True)
class RolloutOutput:
    """Padded rollout tensors plus per-sequence Python token lists."""

    prompt_ids: list[list[int]]
    response_ids: list[list[int]]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    logprobs: torch.Tensor
    finish_reason: list[str]
    metrics: dict[str, float] | None = None


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors inside Python containers to a device."""

    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


def to_cpu(obj: Any, share_memory: bool = False) -> Any:
    """Recursively detach tensors to CPU, optionally using shared memory."""

    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
        if share_memory:
            tensor.share_memory_()
        return tensor
    if isinstance(obj, dict):
        return {k: to_cpu(v, share_memory=share_memory) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v, share_memory=share_memory) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v, share_memory=share_memory) for v in obj)
    return obj
