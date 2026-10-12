"""Additional policy state needed to preserve existing FFT training options."""

from __future__ import annotations

import json
import weakref
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn


class AuxiliaryPolicyState:
    """Live media parameters and router buffers, with an immutable base view.

    Tensor paths are resolved against the model on every access because moving a
    model between CPU and CUDA can replace its buffer objects. Step-local routing
    counters are deliberately excluded from policy publication and checkpoints.
    """

    def __init__(self, model: nn.Module | None = None, adapter_path: str | None = None) -> None:
        self.parameter_names = ()
        self.buffer_names = ()
        self.base = {}
        self.multimodal_unfreeze = {}
        if model is None:
            return
        self._model = weakref.ref(model)
        metadata = read_policy_metadata(adapter_path)
        self.parameter_names = tuple(
            name for name, parameter in model.named_parameters() if getattr(parameter, "_areno_policy_sync", False)
        )
        buffer_names = []
        routing_buffers = set()
        for name, module in model.named_modules():
            if hasattr(module, "bias_update_rate"):
                for component in ("expert_bias", "local_expert_bias"):
                    if hasattr(module, component):
                        path = f"{name}.{component}" if name else component
                        routing_buffers.add(path)
                        if module.bias_update_rate != 0.0:
                            buffer_names.append(path)
        if not set(metadata.get("buffers", ())).issubset(routing_buffers):
            raise ValueError("adapter routing state does not match this model")
        # Imported routing state remains part of the policy even when updates
        # are disabled for inference or a later training configuration.
        self.buffer_names = tuple(sorted(set(buffer_names) | set(metadata.get("buffers", ()))))
        saved_parameters = tuple(metadata.get("parameters", ()))
        if saved_parameters and set(saved_parameters) != set(self.parameter_names):
            raise ValueError("adapter media parameters do not match the configured multimodal unfreeze options")
        available_buffers = dict(model.named_buffers())
        if any(name not in available_buffers for name in self.buffer_names):
            raise ValueError("adapter routing state does not match this model")
        self.base = {name: tensor.detach().cpu().clone() for name, tensor in self.named_tensors()}
        self.multimodal_unfreeze: dict[str, bool] = {}

    @property
    def active(self) -> bool:
        return bool(self.parameter_names or self.buffer_names)

    def named_parameters(self):
        if not self.parameter_names:
            return
        parameters = dict(self._model().named_parameters())
        for name in self.parameter_names:
            yield name, parameters[name]

    def named_tensors(self):
        yield from self.named_parameters()
        if not self.buffer_names:
            return
        buffers = dict(self._model().named_buffers())
        for name in self.buffer_names:
            yield name, buffers[name]

    def metadata(self) -> dict:
        return {
            "version": 1,
            "parameters": list(self.parameter_names),
            "buffers": list(self.buffer_names),
            "multimodal_unfreeze": self.multimodal_unfreeze,
        }

    @contextmanager
    def base_only(self):
        """Restore the original checkpoint's media/routing state temporarily."""

        current = {name: tensor.detach().clone() for name, tensor in self.named_tensors()}
        with torch.no_grad():
            for name, tensor in self.named_tensors():
                tensor.copy_(self.base[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, tensor in self.named_tensors():
                    tensor.copy_(current[name])


def read_policy_metadata(adapter_path: str | None) -> dict:
    if adapter_path is None:
        return {}
    metadata = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    policy = metadata.get("areno_policy_state", {})
    if str(metadata.get("peft_type", "")).upper() == "ARENO_LORA_POLICY" and not policy:
        raise ValueError("AReno LoRA policy adapter requires policy-state metadata")
    if policy and policy.get("version") != 1:
        raise ValueError("unsupported AReno LoRA policy-state version")
    return policy
