"""Checkpoint attention contract shared by label training and serving."""

import json
from pathlib import Path


def resolve_bi_attn(path, requested=None):
    if requested is not None and type(requested) is not bool:
        raise ValueError("bi_attn must be a boolean or None")
    file = Path(path) / "label_jev_config.json"
    metadata = json.loads(file.read_text()) if file.exists() else {}
    config_file = Path(path) / "config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    text_config = config.get("text_config", config)
    causal = text_config.get("is_causal", True)
    if type(causal) is not bool:
        raise ValueError("checkpoint is_causal must be a boolean")
    saved = metadata.get("bi_attn", not causal)
    if metadata.get("bi_attn") is False and not causal:
        raise ValueError("checkpoint protocol and model attention modes disagree")
    if type(saved) is not bool:
        raise ValueError("checkpoint bi_attn must be a boolean")
    if requested is False and saved:
        raise ValueError("cannot serve or resume a bidirectional checkpoint with causal attention")
    return saved if requested is None else requested
