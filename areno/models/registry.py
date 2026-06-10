"""Model adapter registry.

Maintains a name -> ``ModelAdapter`` table that the engine consults to map
HuggingFace configurations to internal model classes. Adapters are
registered by the ``areno.models`` plugin package on first use via
``load_model_plugins``; once loaded the registry is queried by either the
areno ``model_type`` (set during ``config_from_hf``) or by matching the
raw HF config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock

from areno.engine.config import ModelConfig
from areno.engine.parallel.context import get_tp_context
from areno.models.base import ModelAdapter


# Registered adapters keyed by their declared ``name`` field.
_ADAPTERS: dict[str, ModelAdapter] = {}
# Tracks whether the areno.models plugin pack has been imported yet so
# we only pay the registration cost once.
_PLUGINS_LOADED = False
_REGISTRY_LOCK = RLock()


def register_adapter(adapter: ModelAdapter) -> None:
    """Add ``adapter`` to the registry. Duplicate names are an error."""

    with _REGISTRY_LOCK:
        if adapter.name in _ADAPTERS:
            raise ValueError(f"duplicate model adapter: {adapter.name}")
        _ADAPTERS[adapter.name] = adapter


def _adapter(name: str) -> ModelAdapter:
    """Look up a registered adapter by ``name`` (loading plugins on first use)."""

    load_model_plugins()
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unknown model adapter {name!r}; registered: {known}") from exc


def adapter_from_hf(model_path: str | Path) -> ModelAdapter:
    """Pick the adapter that claims the HF config at ``model_path``."""

    load_model_plugins()
    hf_config = read_hf_config(model_path)
    # Adapters are queried in registration order; the first to match wins.
    for adapter in _ADAPTERS.values():
        if adapter.match_hf_config(hf_config):
            return adapter
    model_type = hf_config.get("model_type")
    known = ", ".join(sorted(_ADAPTERS))
    raise ValueError(f"no adapter matched HF model_type={model_type!r}; registered: {known}")


def config_from_hf(model_path: str | Path) -> ModelConfig:
    """Resolve adapter and translate the HF config into a ``ModelConfig``."""

    adapter = adapter_from_hf(model_path)
    return adapter.config_from_hf(read_hf_config(model_path))


def build_model(config: ModelConfig):
    """Instantiate the nn.Module for ``config`` using its model_type adapter."""

    return _adapter(config.model_type).build(config)


def load_model_weights(model, config: ModelConfig, model_path: str | Path) -> None:
    """Load weights via the model's adapter; only rank 0 prints progress."""

    adapter = _adapter(config.model_type)
    # The checkpoint loader reads this env var to gate progress output; restore
    # the original value (or unset it) when the load finishes.
    old_progress = os.environ.get("ARENO_CKPT_PROGRESS")
    os.environ["ARENO_CKPT_PROGRESS"] = "1" if get_tp_context().rank == 0 else "0"
    try:
        adapter.load_weights(model, model_path)
    finally:
        if old_progress is None:
            os.environ.pop("ARENO_CKPT_PROGRESS", None)
        else:
            os.environ["ARENO_CKPT_PROGRESS"] = old_progress


def save_model_weights(model, config: ModelConfig, output_path: str | Path, source_path: str | Path | None) -> str | None:
    """Save weights via the model's adapter, unwrapping any torch.compile wrapper."""

    return _adapter(config.model_type).save_weights(_unwrap_compiled_model(model), output_path, source_path)


def _unwrap_compiled_model(model):
    """Strip any number of ``torch.compile`` wrappers to reveal the raw module."""

    original = getattr(model, "_orig_mod", None)
    if original is None:
        return model
    return _unwrap_compiled_model(original)


def read_hf_config(model_path: str | Path) -> dict:
    """Read the HuggingFace ``config.json`` from ``model_path``."""

    with (Path(model_path) / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_plugins() -> None:
    """Load model plugin adapters from the repository's areno.models package."""

    global _PLUGINS_LOADED
    with _REGISTRY_LOCK:
        if _PLUGINS_LOADED:
            return
        # Importing the plugin pack triggers its `register_models` side effect,
        # which calls `register_adapter` for every concrete model class.
        import areno.models

        areno.models.register_models()
        _PLUGINS_LOADED = True
