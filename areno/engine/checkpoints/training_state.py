"""Rank-local state for a complete, fixed-layout training snapshot."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from areno.engine.modeling import unwrap_model
from areno.engine.optim import AdamWFP32Master


def _layout(worker: Any) -> dict[str, Any]:
    model = unwrap_model(worker.model)
    names = {id(param): name for name, param in model.named_parameters()}
    return {
        "parameters": [
            (name, list(param.shape), str(param.dtype), param.requires_grad) for name, param in model.named_parameters()
        ],
        "buckets": [
            {
                "numel": bucket.numel,
                "shard_numel": bucket.shard_numel,
                "refs": [
                    (
                        names[id(ref.model_param)],
                        ref.param_start,
                        ref.numel,
                        ref.bucket_start,
                        ref.shard_start,
                        ref.shard_numel,
                        ref.shard_bucket_start,
                    )
                    for ref in bucket.refs
                ],
            }
            for bucket in worker.optimizer.buckets
        ],
        "optimizer": asdict(worker.config.optimizer),
    }


def save_worker_state(worker: Any, path: str) -> None:
    """Save one worker's state at a completed optimizer boundary."""

    if type(worker.optimizer) is not AdamWFP32Master:
        raise ValueError("training snapshots currently require AdamWFP32Master")
    if worker.optimizer.has_gradients() or any(
        p.grad is not None or isinstance(getattr(p, "main_grad", None), torch.Tensor)
        for p in worker.optimizer.model_params
    ):
        raise ValueError("cannot save training state with uncommitted gradients")
    numpy_rng = np.random.get_state()
    payload = {
        "version": 1,
        "layout": _layout(worker),
        "model": {name: tensor.detach().cpu() for name, tensor in unwrap_model(worker.model).state_dict().items()},
        "optimizer": worker.optimizer.state_dict(),
        "global_step": worker._global_step,
        "adapter_version": worker.adapter_registry.version if worker.adapter_registry is not None else None,
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(worker.device) if worker.device.type == "cuda" else None,
    }
    torch.save(payload, Path(path))


def load_worker_state(worker: Any, path: str) -> None:
    """Validate and restore the model, optimizer, update count and RNG state."""

    if type(worker.optimizer) is not AdamWFP32Master:
        raise ValueError("training snapshots currently require AdamWFP32Master")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("version") != 1 or payload.get("layout") != _layout(worker):
        raise ValueError("training snapshot worker layout or optimizer configuration does not match")
    model = unwrap_model(worker.model)
    current = model.state_dict()
    saved = payload["model"]
    if saved.keys() != current.keys() or any(
        not isinstance(saved[name], torch.Tensor)
        or saved[name].shape != current[name].shape
        or saved[name].dtype != current[name].dtype
        for name in current
    ):
        raise ValueError("training snapshot model layout does not match")
    optimizer = payload["optimizer"]
    buckets = worker.optimizer.buckets
    if (
        optimizer["dp_rank"] != worker.optimizer.dp_rank
        or optimizer["dp_size"] != worker.optimizer.dp_size
        or len(optimizer["master_params"]) != len(buckets)
        or len(optimizer["state"]) != len(buckets)
    ):
        raise ValueError("training snapshot optimizer shard layout does not match")
    for bucket, master, state in zip(buckets, optimizer["master_params"], optimizer["state"], strict=True):
        if not isinstance(master, torch.Tensor) or master.numel() != bucket.shard_numel:
            raise ValueError("training snapshot optimizer master shard does not match")
        if state is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state[key]
                if tensor is not None and (
                    not isinstance(tensor, torch.Tensor) or tensor.numel() != bucket.shard_numel
                ):
                    raise ValueError("training snapshot optimizer moment shard does not match")
    model.load_state_dict(saved, strict=True)
    worker.optimizer.load_state_dict(optimizer)
    worker.optimizer.lr = optimizer["lr"]
    worker._global_step = int(payload["global_step"])
    if worker.adapter_registry is not None:
        worker.adapter_registry.version = int(payload["adapter_version"])
    random.setstate(payload["python_rng"])
    numpy_rng = payload["numpy_rng"]
    np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state(payload["cuda_rng"], worker.device)
