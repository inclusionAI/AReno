"""Shared trainer-side shutdown helpers."""

from __future__ import annotations

from typing import Any


def close_trainer(instance: Any, shutdown: Any | None) -> None:
    """Close a trainer instance and propagate the initial event to worker ranks."""

    payload = shutdown.worker_payload() if shutdown is not None else None
    if payload is None:
        instance.close()
    else:
        instance.close(shutdown_info=payload)
