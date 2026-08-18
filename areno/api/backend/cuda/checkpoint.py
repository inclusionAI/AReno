"""CUDA checkpoint adapter."""

from __future__ import annotations

from typing import Any


def save_checkpoint(engine: Any, path: str) -> str:
    """Persist a CUDA engine checkpoint using its native format."""

    return engine.save_checkpoint(path)


__all__ = ["save_checkpoint"]
