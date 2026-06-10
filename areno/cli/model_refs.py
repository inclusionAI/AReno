"""Shared CLI helpers for local checkpoint paths and Hugging Face model ids."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

ConfigT = TypeVar("ConfigT")


def resolve_model_ref(model_ref: str, cache: dict[str, str] | None = None) -> str:
    """Resolve a local path or Hugging Face repo id to a local checkpoint directory."""

    path = Path(model_ref)
    if path.exists():
        return str(path)
    if cache is not None and model_ref in cache:
        return cache[model_ref]
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"{model_ref!r} is not a local checkpoint path and huggingface_hub is unavailable"
        ) from exc
    resolved = snapshot_download(model_ref)
    if cache is not None:
        cache[model_ref] = resolved
    return resolved


def resolve_model_refs_for_config(config: ConfigT) -> ConfigT:
    """Resolve all model references in a trainer config, sharing duplicate downloads."""

    cache: dict[str, str] = {}
    config.ckpt = resolve_model_ref(config.ckpt, cache)
    algo = str(getattr(config, "algo", "")).lower()
    if algo == "dpo" and getattr(config, "ref_ckpt", None) is not None:
        config.ref_ckpt = resolve_model_ref(config.ref_ckpt, cache)
    if algo == "ppo":
        if getattr(config, "ref_ckpt", None) is not None:
            config.ref_ckpt = resolve_model_ref(config.ref_ckpt, cache)
        if getattr(config, "reward_ckpt", None) is not None:
            config.reward_ckpt = resolve_model_ref(config.reward_ckpt, cache)
        if getattr(config, "critic_ckpt", None) is not None:
            config.critic_ckpt = resolve_model_ref(config.critic_ckpt, cache)
    return config
