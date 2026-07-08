"""Shared CLI helpers for local checkpoint paths and remote model ids."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

ConfigT = TypeVar("ConfigT")


def resolve_model_ref(model_ref: str, cache: dict[str, str] | None = None, *, model_hub: str = "modelscope") -> str:
    """Resolve a local path or remote model id to a local checkpoint directory."""

    path = Path(model_ref)
    if path.exists():
        return str(path)
    cache_key = f"{model_hub}:{model_ref}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    resolved = _snapshot_download(model_ref, model_hub=model_hub)
    if cache is not None:
        cache[cache_key] = resolved
    return resolved


def _snapshot_download(model_ref: str, *, model_hub: str) -> str:
    if model_hub == "hf":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                f"{model_ref!r} is not a local checkpoint path and huggingface_hub is unavailable"
            ) from exc
        return snapshot_download(model_ref)
    if model_hub == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError(f"{model_ref!r} is not a local checkpoint path and modelscope is unavailable") from exc
        return snapshot_download(model_ref)
    raise ValueError("model_hub must be one of: hf, modelscope")


def resolve_model_refs_for_config(config: ConfigT) -> ConfigT:
    """Resolve all model references in a trainer config, sharing duplicate downloads."""

    cache: dict[str, str] = {}
    model_hub = str(getattr(config, "model_hub", "modelscope"))
    config.ckpt = resolve_model_ref(config.ckpt, cache, model_hub=model_hub)
    algo = str(getattr(config, "algo", "")).lower()
    if algo == "dpo" and getattr(config, "ref_ckpt", None) is not None:
        config.ref_ckpt = resolve_model_ref(config.ref_ckpt, cache, model_hub=model_hub)
    if algo == "ppo":
        if getattr(config, "ref_ckpt", None) is not None:
            config.ref_ckpt = resolve_model_ref(config.ref_ckpt, cache, model_hub=model_hub)
        if getattr(config, "reward_ckpt", None) is not None:
            config.reward_ckpt = resolve_model_ref(config.reward_ckpt, cache, model_hub=model_hub)
        if getattr(config, "critic_ckpt", None) is not None:
            config.critic_ckpt = resolve_model_ref(config.critic_ckpt, cache, model_hub=model_hub)
    return config
