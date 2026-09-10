"""Execution wrappers for native LoRA bindings.

The wrappers in this module deliberately do not own base weights.  Native
projection modules keep their original identity and checkpoint keys while the
wrapper owns only the adapter slots and the execution rule for combining their
delta with an already-computed base result.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


class MergedLoraBinding(nn.ModuleDict):
    """LoRA slots bound to independently sharded components of one GEMM."""

    def __init__(self) -> None:
        super().__init__()
        self._component_indices: dict[str, int] = {}

    def bind(self, component: str, component_index: int, slot: nn.Module) -> None:
        if component in self:
            raise ValueError(f"LoRA component {component!r} is already bound")
        self[component] = slot
        self._component_indices[component] = int(component_index)

    def apply(self, x: torch.Tensor, output: torch.Tensor, output_sizes: Iterable[int]) -> torch.Tensor:
        if not self:
            return output
        parts = list(output.split(tuple(output_sizes), dim=-1))
        changed = False
        for component, slot in self.items():
            if not slot.enabled:
                continue
            index = self._component_indices[component]
            parts[index] = parts[index] + slot(x)
            changed = True
        return torch.cat(parts, dim=-1) if changed else output


@dataclass(frozen=True, slots=True)
class RoutedLoraTarget:
    """Static layout for one logical target in a grouped expert owner."""

    component: str
    weight_path: str
    in_features: int
    out_features: int


class RoutedExpertLoraBinding(nn.ModuleDict):
    """Grouped-expert LoRA execution shared by model-family adapters."""

    def __init__(self, *, intermediate_size: int, targets: Iterable[RoutedLoraTarget]) -> None:
        super().__init__()
        self.intermediate_size = int(intermediate_size)
        self.targets = tuple(targets)

    def bind(self, component: str, slot: nn.Module) -> None:
        if component in self:
            raise ValueError(f"LoRA component {component!r} is already bound")
        self[component] = slot

    @property
    def active(self) -> bool:
        return bool(self) and next(iter(self.values())).enabled

    def apply_gate_up(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if not self.active:
            return output
        if "linear_fc1" in self:
            output = output + self["linear_fc1"](x, tokens_per_expert)
        gate, up = output.chunk(2, dim=-1)
        if "gate_proj" in self:
            gate = gate + self["gate_proj"](x, tokens_per_expert)
        if "up_proj" in self:
            up = up + self["up_proj"](x, tokens_per_expert)
        return torch.cat((gate, up), dim=-1)

    def apply_down(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if not self.active:
            return output
        if "linear_fc2" in self:
            output = output + self["linear_fc2"](x, tokens_per_expert)
        if "down_proj" in self:
            output = output + self["down_proj"](x, tokens_per_expert)
        return output

    def zero_grad_edge(self, reference: torch.Tensor) -> torch.Tensor:
        """Keep every bound A/B parameter in an empty-route autograd graph."""

        zero = reference.new_zeros(())
        if not self.active:
            return zero
        for slot in self.values():
            zero = zero + slot.lora_A.reshape(-1)[0] * 0 + slot.lora_B.reshape(-1)[0] * 0
        return zero

    @torch.no_grad()
    def merge_inference_weights_(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        """Merge active adapter deltas directly into final fused rollout tiles."""

        if not self.active:
            return
        if "linear_fc1" in self:
            self._add_delta_(w1, self["linear_fc1"])
        if "linear_fc2" in self:
            self._add_delta_(w2, self["linear_fc2"])
        if "gate_proj" in self:
            self._add_delta_(w1[:, : self.intermediate_size], self["gate_proj"])
        if "up_proj" in self:
            self._add_delta_(w1[:, self.intermediate_size :], self["up_proj"])
        if "down_proj" in self:
            self._add_delta_(w2, self["down_proj"])

    @staticmethod
    @torch.no_grad()
    def _add_delta_(weight: torch.Tensor, slot: nn.Module) -> None:
        weight.baddbmm_(slot.lora_B, slot.lora_A, beta=1.0, alpha=float(slot.scale.item()))
