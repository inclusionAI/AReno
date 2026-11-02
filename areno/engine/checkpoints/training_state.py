"""Explicit single-device actor optimizer/RNG checkpoint sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import torch

_SCHEDULE_FIELDS = ("base_lr", "min_lr", "lr_decay_steps", "lr_warmup_steps", "lr_decay_style")


def save_training_state(worker, path: str) -> None:
    directory = Path(path)
    payload = {
        "format": "areno-training-v1",
        "optimizer_class": type(worker.optimizer).__name__,
        "parameter_names": [name for name, param in worker.model.named_parameters() if param.requires_grad],
        "optimizer": worker.optimizer.state_dict(),
        "global_step": worker._global_step,
        "schedule": {name: getattr(worker, name) for name in _SCHEDULE_FIELDS},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(worker.device) if worker.device.type == "cuda" else None,
    }
    pending = directory / "training_state.pt.tmp"
    torch.save(payload, pending)
    pending.replace(directory / "training_state.pt")
    metadata = {key: payload[key] for key in ("format", "optimizer_class", "global_step")}
    pending_json = directory / "training_state.json.tmp"
    pending_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pending_json.replace(directory / "training_state.json")


def load_training_state(worker, path: str) -> None:
    payload = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    if payload.get("format") != "areno-training-v1":
        raise ValueError("unsupported training-state checkpoint format")
    if payload["optimizer_class"] != type(worker.optimizer).__name__:
        raise ValueError("training-state checkpoint optimizer does not match")
    names = [name for name, param in worker.model.named_parameters() if param.requires_grad]
    if names != payload["parameter_names"]:
        raise ValueError("training-state checkpoint parameter names/order do not match")
    worker.optimizer.load_state_dict(payload["optimizer"])
    # Existing bucket optimizers restore tensors but leave scalar settings to callers.
    for name in ("lr", "betas", "weight_decay", "eps"):
        if name in payload["optimizer"]:
            setattr(worker.optimizer, name, payload["optimizer"][name])
    for name in _SCHEDULE_FIELDS:
        setattr(worker, name, payload["schedule"][name])
    worker._global_step = int(payload["global_step"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None and worker.device.type == "cuda":
        torch.cuda.set_rng_state(payload["cuda_rng"], worker.device)
