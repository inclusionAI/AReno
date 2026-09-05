"""Lightweight acceleration helpers that do not import optional kernels."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)
_LOGGED: set[str] = set()


def log_once(key: str, message: str, *, level: int = logging.DEBUG) -> None:
    """Log ``message`` at most once per process for the given ``key``."""

    if key in _LOGGED:
        return
    logger.log(level, message)
    _LOGGED.add(key)


def warn_once(key: str, message: str) -> None:
    """Emit a warning at most once per process for the given ``key``."""

    log_once(key, message, level=logging.WARNING)


@torch._dynamo.disable
def is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    """True if the tensor lives on CUDA and we are inside a graph capture."""

    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


@torch._dynamo.disable
def can_use_cuda_kernel(tensor: torch.Tensor, name: str, *, allow_sm121: bool = False) -> bool:
    """Return whether a fused CUDA kernel can run for ``tensor``."""

    if not tensor.is_cuda:
        return False
    return True
