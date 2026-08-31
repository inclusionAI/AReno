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
