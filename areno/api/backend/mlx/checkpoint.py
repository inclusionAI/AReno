"""MLX checkpoint persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def save_checkpoint(
    provider: Any,
    optimizer: Any,
    *,
    source_path: str,
    destination_path: str,
    policy_version: int,
    global_step: int,
) -> str:
    """Save provider weights, processors, and AReno resume metadata."""

    import mlx.core as mx

    destination = Path(destination_path)
    destination.mkdir(parents=True, exist_ok=True)
    mx.eval(provider.model.parameters(), optimizer.state)
    provider.save(destination)
    source = Path(source_path)
    for filename in ("generation_config.json",):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, destination / filename)
    (destination / "areno_mlx_state.json").write_text(
        json.dumps(
            {
                "checkpoint_format": "mlx",
                "format_version": 1,
                "policy_version": policy_version,
                "global_step": global_step,
            }
        ),
        encoding="utf-8",
    )
    return str(destination)


__all__ = ["save_checkpoint"]


def save_full_jev_checkpoint(backend: Any, ctx: Any, path: str) -> str:
    """Save a loadable backbone separately from the head and optimizer state."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from areno.api.backend.mlx.decision import save_head

    # MLX-LM strict loading must not encounter score_head weights before AReno
    # attaches the head. Restore the live module even if checkpoint I/O fails.
    head = backend.model.pop("score_head")
    try:
        result = save_checkpoint(
            backend.provider,
            backend.optimizer,
            source_path=backend.config.model_path or ctx.model_path,
            destination_path=path,
            policy_version=backend._policy_version,
            global_step=ctx.global_step,
        )
    finally:
        backend.model.score_head = head
    save_head(backend, path)
    leaves = tree_flatten(backend.optimizer.state)
    arrays = {key: value for key, value in leaves if isinstance(value, mx.array)}
    scalars = {key: value for key, value in leaves if not isinstance(value, mx.array)}
    mx.save_safetensors(str(Path(path) / "jev_optimizer.safetensors"), arrays)
    (Path(path) / "jev_optimizer.json").write_text(json.dumps(scalars) + "\n")
    return result


def restore_jev_optimizer(backend: Any, ctx: Any) -> None:
    """Resume moments and the schedule counter after restoring backbone/head."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    checkpoint = backend.config.decision_checkpoint
    if checkpoint is None:
        return
    root = Path(checkpoint)
    state_path = root / "areno_mlx_state.json"
    optimizer_path = root / "jev_optimizer.safetensors"
    if optimizer_path.exists():
        arrays = mx.load(str(optimizer_path))
        scalars = json.loads((root / "jev_optimizer.json").read_text())
        backend.optimizer.state = tree_unflatten(list({**scalars, **arrays}.items()))
    if state_path.exists():
        state = json.loads(state_path.read_text())
        ctx.global_step = int(state["global_step"])
        backend._policy_version = int(state["policy_version"])
