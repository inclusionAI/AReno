"""TP-aware native LoRA slots for Qwen3 dense and routed-expert projections."""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F
from torch import nn

from areno.accel import areno_grouped_linear
from areno.adapters.config import LoraConfig
from areno.engine.layers.linear import RowParallelLinear, mark_tensor_parallel_parameter
from areno.engine.parallel.context import get_tp_context


class LoraSlot(nn.Module):
    """One canonical LoRA A/B pair owned by its native projection module."""

    def __init__(
        self,
        *,
        logical_name: str,
        base_weight: nn.Parameter,
        global_in_features: int,
        global_out_features: int,
        local_in_features: int,
        local_out_features: int,
        row_parallel: bool,
        config: LoraConfig,
        seed: int,
    ) -> None:
        super().__init__()
        ctx = get_tp_context()
        self.logical_name = logical_name
        self.rank = int(config.rank)
        self.global_in_features = int(global_in_features)
        self.global_out_features = int(global_out_features)
        self.local_in_features = int(local_in_features)
        self.local_out_features = int(local_out_features)
        self.row_parallel = bool(row_parallel)
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.local_in_features, device=base_weight.device, dtype=base_weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(self.local_out_features, self.rank, device=base_weight.device, dtype=base_weight.dtype)
        )
        self.register_buffer("scale", torch.tensor(config.scale, device=base_weight.device, dtype=torch.float32))
        if row_parallel:
            mark_tensor_parallel_parameter(self.lora_A, True, sequence_parallel=True)
            mark_tensor_parallel_parameter(self.lora_B, False, sequence_parallel=True, tp_grad_allreduce=True)
        else:
            mark_tensor_parallel_parameter(self.lora_A, False, sequence_parallel=True, tp_grad_allreduce=True)
            mark_tensor_parallel_parameter(self.lora_B, True, sequence_parallel=True)
        self._reset_parameters(seed, ctx.rank, ctx.world_size)

    @torch.no_grad()
    def _reset_parameters(self, seed: int, tp_rank: int, tp_size: int) -> None:
        material = f"{int(seed)}:{self.logical_name}".encode()
        target_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(target_seed)
        canonical_A = torch.empty(self.rank, self.global_in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(canonical_A, a=math.sqrt(5), generator=generator)
        if self.row_parallel:
            shard = self.global_in_features // tp_size
            canonical_A = canonical_A[:, tp_rank * shard : (tp_rank + 1) * shard]
        self.lora_A.copy_(canonical_A.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
        self.lora_B.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class RoutedExpertLoraSlot(nn.Module):
    """One expert-sharded canonical LoRA A/B pair for grouped Qwen3-MoE GEMMs."""

    def __init__(
        self,
        *,
        logical_name: str,
        base_weight: nn.Parameter,
        local_num_experts: int,
        local_expert_start: int,
        in_features: int,
        out_features: int,
        config: LoraConfig,
        seed: int,
    ) -> None:
        super().__init__()
        self.logical_name = logical_name
        self.rank = int(config.rank)
        self.local_num_experts = int(local_num_experts)
        self.local_expert_start = int(local_expert_start)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_A = nn.Parameter(
            torch.empty(
                self.local_num_experts,
                self.rank,
                self.in_features,
                device=base_weight.device,
                dtype=base_weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.empty(
                self.local_num_experts,
                self.out_features,
                self.rank,
                device=base_weight.device,
                dtype=base_weight.dtype,
            )
        )
        self.register_buffer("scale", torch.tensor(config.scale, device=base_weight.device, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.lora_A, True, sequence_parallel=False, tp_grad_allreduce=False)
        mark_tensor_parallel_parameter(self.lora_B, True, sequence_parallel=False, tp_grad_allreduce=False)
        self._reset_parameters(seed)

    @torch.no_grad()
    def _reset_parameters(self, seed: int) -> None:
        for local_expert_id in range(self.local_num_experts):
            expert_id = self.local_expert_start + local_expert_id
            material = f"{int(seed)}:{self.logical_name}:expert={expert_id}".encode()
            target_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(target_seed)
            initial_A = torch.empty(self.rank, self.in_features, dtype=torch.float32)
            nn.init.kaiming_uniform_(initial_A, a=math.sqrt(5), generator=generator)
            self.lora_A[local_expert_id].copy_(initial_A.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
        self.lora_B.zero_()

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        hidden = areno_grouped_linear(x.contiguous(), self.lora_A, tokens_per_expert)
        return areno_grouped_linear(hidden, self.lora_B, tokens_per_expert) * self.scale


class AdapterRegistry:
    """Non-owning index over LoRA slots; projection modules remain sole owners."""

    def __init__(self, slots: dict[str, LoraSlot | RoutedExpertLoraSlot], config: LoraConfig) -> None:
        self.slots = slots
        self.config = config
        self.version = 0

    def named_parameters(self):
        for name, slot in self.slots.items():
            yield f"{name}.lora_A.weight", slot.lora_A
            yield f"{name}.lora_B.weight", slot.lora_B

    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for _, parameter in self.named_parameters())

    def increment_version(self) -> int:
        self.version += 1
        return self.version


def initialize_lora(model: nn.Module, config: LoraConfig, *, seed: int) -> AdapterRegistry:
    """Freeze a Qwen3 dense or MoE base and attach canonical targets."""

    model_config = getattr(model, "config", None)
    if getattr(model_config, "model_type", None) not in {"qwen3", "qwen3_moe"}:
        raise ValueError("native LoRA currently supports Qwen3 models only")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    requested = set(config.target_modules)
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot] = {}
    for layer_index, layer in enumerate(model.layers):
        prefix = f"layers.{layer_index}"
        qkv = layer.self_attn.qkv_proj
        for component_index, component in enumerate(("q_proj", "k_proj", "v_proj")):
            if component not in requested:
                continue
            logical_name = f"{prefix}.self_attn.{component}"
            slot = LoraSlot(
                logical_name=logical_name,
                base_weight=qkv.weight,
                global_in_features=qkv.in_features,
                global_out_features=qkv.out_features[component_index],
                local_in_features=qkv.in_features,
                local_out_features=qkv.local_out_features[component_index],
                row_parallel=False,
                config=config,
                seed=seed,
            )
            qkv.install_lora_component(component, component_index, slot)
            slots[logical_name] = slot

        if "o_proj" in requested:
            owner = layer.self_attn.o_proj
            logical_name = f"{prefix}.self_attn.o_proj"
            slot = _row_slot(logical_name, owner, config, seed)
            owner.install_lora(slot)
            slots[logical_name] = slot

        if getattr(model_config, "enable_moe_block", False):
            _install_moe_slots(layer.mlp.experts, prefix, requested, config, seed, slots)
        else:
            gate_up = layer.mlp.gate_up_proj
            for component_index, component in enumerate(("gate_proj", "up_proj")):
                if component not in requested:
                    continue
                logical_name = f"{prefix}.mlp.{component}"
                slot = LoraSlot(
                    logical_name=logical_name,
                    base_weight=gate_up.weight,
                    global_in_features=gate_up.in_features,
                    global_out_features=gate_up.out_features[component_index],
                    local_in_features=gate_up.in_features,
                    local_out_features=gate_up.local_out_features[component_index],
                    row_parallel=False,
                    config=config,
                    seed=seed,
                )
                gate_up.install_lora_component(component, component_index, slot)
                slots[logical_name] = slot

            if "down_proj" in requested:
                owner = layer.mlp.down_proj
                logical_name = f"{prefix}.mlp.down_proj"
                slot = _row_slot(logical_name, owner, config, seed)
                owner.install_lora(slot)
                slots[logical_name] = slot

    return AdapterRegistry(slots, config)


def _row_slot(logical_name: str, owner: RowParallelLinear, config: LoraConfig, seed: int) -> LoraSlot:
    return LoraSlot(
        logical_name=logical_name,
        base_weight=owner.weight,
        global_in_features=owner.in_features,
        global_out_features=owner.out_features,
        local_in_features=owner.local_in_features,
        local_out_features=owner.out_features,
        row_parallel=True,
        config=config,
        seed=seed,
    )


def _install_moe_slots(
    experts: nn.Module,
    prefix: str,
    requested: set[str],
    config: LoraConfig,
    seed: int,
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot],
) -> None:
    components = (
        ("gate_proj", experts.hidden_size, experts.intermediate_size, experts.gate_up_weight),
        ("up_proj", experts.hidden_size, experts.intermediate_size, experts.gate_up_weight),
        ("down_proj", experts.intermediate_size, experts.hidden_size, experts.down_weight),
    )
    for component, in_features, out_features, base_weight in components:
        if component not in requested:
            continue
        logical_name = f"{prefix}.mlp.experts.{{expert}}.{component}"
        slot = RoutedExpertLoraSlot(
            logical_name=logical_name,
            base_weight=base_weight,
            local_num_experts=experts.local_num_experts,
            local_expert_start=experts.local_expert_start,
            in_features=in_features,
            out_features=out_features,
            config=config,
            seed=seed,
        )
        experts.install_lora_component(component, slot)
        slots[logical_name] = slot
