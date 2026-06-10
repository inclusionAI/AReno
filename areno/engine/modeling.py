"""Shared model construction and optimizer helpers for worker components."""

from __future__ import annotations

from pathlib import Path

import torch

from areno.engine.config import EngineConfig
from areno.models.registry import build_model
from areno.engine.optim import AdamW8bit, AdamWFP32Master


def param_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    """Return `main_grad` when available, otherwise autograd `.grad`."""

    main_grad = getattr(param, "main_grad", None)
    if isinstance(main_grad, torch.Tensor):
        return main_grad
    return param.grad


def build_model_on_device(config: EngineConfig, device: torch.device) -> torch.nn.Module:
    """Construct the model directly on `device` under the configured dtype."""

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(config.model.dtype)
        with torch.device(device), skip_torch_init(enabled=config.model_path is not None and not config.dummy_load):
            return build_model(config.model)
    finally:
        torch.set_default_dtype(old_dtype)


class skip_torch_init:
    """Temporarily replace expensive torch init functions with no-ops."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._saved = {}

    def __enter__(self):
        if not self.enabled:
            return self
        self._saved = {
            "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
            "uniform_": torch.nn.init.uniform_,
            "normal_": torch.nn.init.normal_,
        }
        torch.nn.init.kaiming_uniform_ = _noop_init
        torch.nn.init.uniform_ = _noop_init
        torch.nn.init.normal_ = _noop_init
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        for name, fn in self._saved.items():
            setattr(torch.nn.init, name, fn)


def _noop_init(tensor: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
    """Replacement for `torch.nn.init.*` that returns the tensor unchanged."""

    return tensor


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Recursively unwrap `torch.compile`'s `_orig_mod` indirection."""

    original = getattr(model, "_orig_mod", None)
    if original is None:
        return model
    return unwrap_model(original)


def canonical_model_path(path: str | None) -> str | None:
    """Resolve `path` to an absolute filesystem path for cache-key comparisons."""

    if path is None:
        return None
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError):
        return str(path)


def build_optimizer(params, optimizer_config, ctx, *, lr: float | None = None):
    """Construct the configured DP-sharded optimizer implementation."""

    optimizer_cls = AdamW8bit if optimizer_config.adam_8bit else AdamWFP32Master
    return optimizer_cls(
        params,
        lr=optimizer_config.lr if lr is None else float(lr),
        betas=optimizer_config.betas,
        weight_decay=optimizer_config.weight_decay,
        bucket_numel=optimizer_config.fp32_master_bucket_numel,
        dp_rank=ctx.dp_rank,
        dp_size=ctx.dp_size,
        dp_group=ctx.dp_group,
    )
