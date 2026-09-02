"""TP-aware native LoRA slots for dense and routed-expert projections."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from areno.accel import areno_grouped_linear
from areno.adapters.config import LoraConfig
from areno.engine.layers.linear import ColumnParallelLinear, RowParallelLinear, mark_tensor_parallel_parameter
from areno.engine.parallel.context import get_tp_context


class _AdapterRuntimeState:
    """Shared control state for one model's adapter view."""

    def __init__(self) -> None:
        self.base_only_depth = 0

    @property
    def enabled(self) -> bool:
        return self.base_only_depth == 0


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
        runtime_state: _AdapterRuntimeState,
        output_range: tuple[int, int] | None = None,
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
        if output_range is None:
            output_range = (
                (0, self.global_out_features)
                if self.row_parallel
                else (ctx.rank * self.local_out_features, (ctx.rank + 1) * self.local_out_features)
            )
        self.output_start, self.output_end = (int(value) for value in output_range)
        self.output_replicated = (
            not self.row_parallel and self.local_out_features * ctx.world_size > self.global_out_features
        )
        self._runtime_state = runtime_state
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
            if self.output_replicated:
                setattr(
                    self.lora_B,
                    "tp_replicated_output_range",
                    (self.output_start, self.output_end, self.global_out_features),
                )
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

    @property
    def enabled(self) -> bool:
        return self._runtime_state.enabled


class RoutedExpertLoraSlot(nn.Module):
    """One expert-sharded canonical LoRA A/B pair for grouped MoE GEMMs."""

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
        runtime_state: _AdapterRuntimeState,
    ) -> None:
        super().__init__()
        self.logical_name = logical_name
        self.rank = int(config.rank)
        self.local_num_experts = int(local_num_experts)
        self.local_expert_start = int(local_expert_start)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._runtime_state = runtime_state
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

    @property
    def enabled(self) -> bool:
        return self._runtime_state.enabled


class AdapterRegistry:
    """Non-owning index over LoRA slots; projection modules remain sole owners."""

    def __init__(
        self,
        slots: dict[str, LoraSlot | RoutedExpertLoraSlot],
        config: LoraConfig,
        runtime_state: _AdapterRuntimeState,
    ) -> None:
        self.slots = slots
        self.config = config
        self._runtime_state = runtime_state
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

    @contextmanager
    def base_only(self) -> Iterator[None]:
        """Temporarily expose the frozen base policy without evaluating A/B."""

        self._runtime_state.base_only_depth += 1
        try:
            yield
        finally:
            self._runtime_state.base_only_depth -= 1


def initialize_lora(model: nn.Module, config: LoraConfig, *, seed: int) -> AdapterRegistry:
    """Freeze one supported native base and attach its canonical targets."""

    model_config = getattr(model, "config", None)
    model_type = getattr(model_config, "model_type", None)
    if model_type not in {"qwen3", "qwen3_moe", "bailing_moe_v3"}:
        raise ValueError("native LoRA currently supports Qwen3 and Bailing-MoE V3 models only")
    if model_type == "bailing_moe_v3" and not bool(getattr(model_config, "no_kda_lora", False)):
        raise ValueError("Bailing-MoE V3 native LoRA currently requires no_kda_lora=true")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    requested = set(config.target_modules)
    runtime_state = _AdapterRuntimeState()
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot] = {}
    if model_type == "bailing_moe_v3":
        matched = _initialize_bailing_v3_lora(model, requested, config, seed, runtime_state, slots)
    else:
        matched = _initialize_qwen3_lora(model, requested, config, seed, runtime_state, slots)
    missing = requested - matched
    if missing:
        raise ValueError(f"target_modules are not present in {model_type}: {', '.join(sorted(missing))}")
    return AdapterRegistry(slots, config, runtime_state)


def _initialize_qwen3_lora(
    model: nn.Module,
    requested: set[str],
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot],
) -> set[str]:
    matched: set[str] = set()
    model_config = model.config
    for layer_index, layer in enumerate(model.layers):
        prefix = f"layers.{layer_index}"
        qkv = layer.self_attn.qkv_proj
        for component_index, component in enumerate(("q_proj", "k_proj", "v_proj")):
            if component not in requested:
                continue
            matched.add(component)
            logical_name = f"{prefix}.self_attn.{component}"
            slot = LoraSlot(
                logical_name=logical_name,
                base_weight=qkv.weight,
                global_in_features=qkv.in_features,
                global_out_features=qkv.out_features[component_index],
                local_in_features=qkv.in_features,
                local_out_features=qkv.local_out_features[component_index],
                row_parallel=False,
                output_range=qkv.shard_ranges[component_index],
                config=config,
                seed=seed,
                runtime_state=runtime_state,
            )
            qkv.install_lora_component(component, component_index, slot)
            slots[logical_name] = slot

        if "o_proj" in requested:
            matched.add("o_proj")
            owner = layer.self_attn.o_proj
            logical_name = f"{prefix}.self_attn.o_proj"
            slot = _row_slot(logical_name, owner, config, seed, runtime_state)
            owner.install_lora(slot)
            slots[logical_name] = slot

        if getattr(model_config, "enable_moe_block", False):
            matched.update(_install_moe_slots(layer.mlp.experts, prefix, requested, config, seed, runtime_state, slots))
        else:
            gate_up = layer.mlp.gate_up_proj
            for component_index, component in enumerate(("gate_proj", "up_proj")):
                if component not in requested:
                    continue
                matched.add(component)
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
                    runtime_state=runtime_state,
                )
                gate_up.install_lora_component(component, component_index, slot)
                slots[logical_name] = slot

            if "down_proj" in requested:
                matched.add("down_proj")
                owner = layer.mlp.down_proj
                logical_name = f"{prefix}.mlp.down_proj"
                slot = _row_slot(logical_name, owner, config, seed, runtime_state)
                owner.install_lora(slot)
                slots[logical_name] = slot

    return matched


def _initialize_bailing_v3_lora(
    model: nn.Module,
    requested: set[str],
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot],
) -> set[str]:
    matched: set[str] = set()
    for layer_index, layer in enumerate(model.layers):
        prefix = f"layers.{layer_index}"
        attention = layer.attention
        attention_prefix = f"{prefix}.attention"
        if hasattr(attention, "q_conv1d_weight"):
            for component in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
                if component in requested:
                    matched.add(component)
                    _install_column_slot(
                        f"{attention_prefix}.{component}",
                        getattr(attention, component),
                        config,
                        seed,
                        runtime_state,
                        slots,
                    )
            if "o_proj" in requested:
                matched.add("o_proj")
                _install_row_slot(
                    f"{attention_prefix}.o_proj",
                    attention.o_proj,
                    config,
                    seed,
                    runtime_state,
                    slots,
                )
        else:
            if "q_proj" in requested and attention.q_proj is not None:
                matched.add("q_proj")
                _install_column_slot(
                    f"{attention_prefix}.q_proj",
                    attention.q_proj,
                    config,
                    seed,
                    runtime_state,
                    slots,
                )
            for component in ("q_a_proj", "kv_a_proj_with_mqa"):
                owner = getattr(attention, component, None)
                if component in requested and owner is not None:
                    matched.add(component)
                    slot = _replicated_slot(f"{attention_prefix}.{component}", owner, config, seed, runtime_state)
                    attention.install_lora_component(component, slot)
                    slots[slot.logical_name] = slot
            for component in ("q_b_proj", "kv_b_proj"):
                owner = getattr(attention, component, None)
                if component in requested and owner is not None:
                    matched.add(component)
                    _install_column_slot(f"{attention_prefix}.{component}", owner, config, seed, runtime_state, slots)
            if "dense" in requested:
                matched.add("dense")
                _install_row_slot(
                    f"{attention_prefix}.dense",
                    attention.dense,
                    config,
                    seed,
                    runtime_state,
                    slots,
                )

        mlp_prefix = f"{prefix}.mlp"
        if hasattr(layer.mlp, "experts"):
            matched.update(_install_moe_slots(layer.mlp.experts, prefix, requested, config, seed, runtime_state, slots))
            if layer.mlp.shared_experts is not None:
                matched.update(
                    _install_dense_mlp_slots(
                        layer.mlp.shared_experts,
                        f"{mlp_prefix}.shared_experts",
                        requested,
                        config,
                        seed,
                        runtime_state,
                        slots,
                    )
                )
        else:
            matched.update(
                _install_dense_mlp_slots(layer.mlp, mlp_prefix, requested, config, seed, runtime_state, slots)
            )
    return matched


def _install_column_slot(
    logical_name: str,
    owner: ColumnParallelLinear,
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot],
) -> None:
    slot = LoraSlot(
        logical_name=logical_name,
        base_weight=owner.weight,
        global_in_features=owner.in_features,
        global_out_features=owner.out_features,
        local_in_features=owner.in_features,
        local_out_features=owner.local_out_features,
        row_parallel=False,
        config=config,
        seed=seed,
        runtime_state=runtime_state,
    )
    owner.install_lora(slot)
    slots[logical_name] = slot


def _install_row_slot(
    logical_name: str,
    owner: RowParallelLinear,
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot],
) -> None:
    slot = _row_slot(logical_name, owner, config, seed, runtime_state)
    owner.install_lora(slot)
    slots[logical_name] = slot


def _replicated_slot(
    logical_name: str,
    owner: nn.Linear,
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
) -> LoraSlot:
    return LoraSlot(
        logical_name=logical_name,
        base_weight=owner.weight,
        global_in_features=owner.in_features,
        global_out_features=owner.out_features,
        local_in_features=owner.in_features,
        local_out_features=owner.out_features,
        row_parallel=False,
        output_range=(0, owner.out_features),
        config=config,
        seed=seed,
        runtime_state=runtime_state,
    )


def _install_dense_mlp_slots(
    mlp: nn.Module,
    prefix: str,
    requested: set[str],
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot],
) -> set[str]:
    matched: set[str] = set()
    for component in ("gate_proj", "up_proj"):
        if component in requested:
            matched.add(component)
            _install_column_slot(f"{prefix}.{component}", getattr(mlp, component), config, seed, runtime_state, slots)
    if "down_proj" in requested:
        matched.add("down_proj")
        _install_row_slot(f"{prefix}.down_proj", mlp.down_proj, config, seed, runtime_state, slots)
    return matched


def _row_slot(
    logical_name: str,
    owner: RowParallelLinear,
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
) -> LoraSlot:
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
        runtime_state=runtime_state,
    )


def _install_moe_slots(
    experts: nn.Module,
    prefix: str,
    requested: set[str],
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
    slots: dict[str, LoraSlot | RoutedExpertLoraSlot],
) -> set[str]:
    gate_up_weight = experts.gate_up_weight if hasattr(experts, "gate_up_weight") else experts.linear_fc1.weight
    down_weight = experts.down_weight if hasattr(experts, "down_weight") else experts.linear_fc2.weight
    components = (
        ("gate_proj", experts.hidden_size, experts.intermediate_size, gate_up_weight),
        ("up_proj", experts.hidden_size, experts.intermediate_size, gate_up_weight),
        ("down_proj", experts.intermediate_size, experts.hidden_size, down_weight),
    )
    matched: set[str] = set()
    for component, in_features, out_features, base_weight in components:
        if component not in requested:
            continue
        matched.add(component)
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
            runtime_state=runtime_state,
        )
        experts.install_lora_component(component, slot)
        slots[logical_name] = slot
    return matched
