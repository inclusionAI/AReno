"""PPO role declarations used by the trainer/backend boundary.

Algorithms like PPO need several model instances at once (actor/ref/reward/
critic). The trainer names these roles and supplies checkpoint paths; the
backend is responsible for actually loading them, deciding how to colocate or
offload weights, and exposing the score/train operations. Keeping the role
description as a small dataclass keeps the contract minimal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelRole:
    """A PPO model role owned by the backend.

    The public trainer names roles and checkpoints, but never calls
    onload/offload. Backends decide how actor/ref/reward/critic are colocated
    and how memory is moved between role operations. `optimizer_lr` is only
    meaningful when `trainable=True`.
    """

    name: str
    path: str
    trainable: bool
    optimizer_lr: float | None = None


class MissingRoleCapability(RuntimeError):
    """Raised when a backend cannot execute a required role operation."""

