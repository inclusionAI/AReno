"""MLX checkpoint persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    *,
    model_config: dict[str, Any],
    source_path: str,
    destination_path: str,
    policy_version: int,
    global_step: int,
) -> str:
    """Save MLX-LM weights, tokenizer files, and AReno resume metadata."""

    import mlx.core as mx
    from mlx_lm.utils import save_config, save_model

    destination = Path(destination_path)
    destination.mkdir(parents=True, exist_ok=True)
    mx.eval(model.parameters(), optimizer.state)
    save_model(destination, model)
    save_config(model_config, config_path=destination / "config.json")
    tokenizer.save_pretrained(destination)
    source = Path(source_path)
    for filename in ("generation_config.json",):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, destination / filename)
    (destination / "areno_mlx_state.json").write_text(
        json.dumps({"format_version": 1, "policy_version": policy_version, "global_step": global_step}),
        encoding="utf-8",
    )
    return str(destination)


__all__ = ["save_checkpoint"]
