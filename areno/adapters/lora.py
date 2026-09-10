"""TP-aware native LoRA slots for dense and routed-expert projections."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from areno.accel import areno_grouped_linear
from areno.adapters.config import LoraConfig
from areno.engine.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    mark_tensor_parallel_parameter,
)
from areno.engine.layers.lora import RoutedExpertLoraBinding
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


class LoraExecutionPattern(str, Enum):
    COLUMN_PARALLEL = "column_parallel"
    ROW_PARALLEL = "row_parallel"
    MERGED_COMPONENT = "merged_component"
    REPLICATED = "replicated"
    ROUTED_GROUPED_EXPERT = "routed_grouped_expert"


@dataclass(frozen=True, slots=True)
class LoraTargetSpec:
    """Resolved physical owner for one logical PEFT target."""

    logical_name: str
    component: str
    execution_pattern: LoraExecutionPattern
    owner: nn.Module
    base_weight: nn.Parameter
    global_in_features: int
    global_out_features: int
    local_in_features: int
    local_out_features: int
    component_index: int | None = None
    output_range: tuple[int, int] | None = None


def initialize_lora(model: nn.Module, config: LoraConfig, *, seed: int) -> AdapterRegistry:
    """Resolve, validate and atomically bind native LoRA wrappers."""

    model_config = getattr(model, "config", None)
    model_type = getattr(model_config, "model_type", None)
    if model_type not in {"qwen3", "qwen3_moe", "bailing_moe_v3"}:
        raise ValueError("native LoRA currently supports Qwen3 and Bailing-MoE V3 models only")
    if model_type == "bailing_moe_v3" and not bool(getattr(model_config, "no_kda_lora", False)):
        raise ValueError("Bailing-MoE V3 native LoRA currently requires no_kda_lora=true")

    requested = set(config.target_modules)
    resolved: list[LoraTargetSpec] = []
    matched: set[str] = set()
    for spec in _iter_lora_targets(model):
        selected = _matching_targets(requested, spec.component, spec.logical_name)
        if selected:
            resolved.append(spec)
            matched.update(selected)
    missing = requested - matched
    if missing:
        raise ValueError(f"target_modules are not present in {model_type}: {', '.join(sorted(missing))}")
    _validate_resolved_targets(resolved)

    runtime_state = _AdapterRuntimeState()
    pending = [(spec, _new_slot(spec, config, seed, runtime_state)) for spec in resolved]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    slots: dict[str, LoraSlot | RoutedExpertLoraSlot] = {}
    for spec, slot in pending:
        _bind_slot(spec, slot)
        slots[spec.logical_name] = slot
    return AdapterRegistry(slots, config, runtime_state)


def _iter_lora_targets(model: nn.Module) -> Iterator[LoraTargetSpec]:
    for module_name, owner in model.named_modules():
        if not module_name:
            continue
        if isinstance(owner, QKVParallelLinear):
            prefix = module_name.rsplit(".", 1)[0]
            for index, component in enumerate(owner.lora_components):
                yield LoraTargetSpec(
                    logical_name=f"{prefix}.{component}",
                    component=component,
                    execution_pattern=LoraExecutionPattern.MERGED_COMPONENT,
                    owner=owner,
                    base_weight=owner.weight,
                    global_in_features=owner.in_features,
                    global_out_features=owner.out_features[index],
                    local_in_features=owner.in_features,
                    local_out_features=owner.local_out_features[index],
                    component_index=index,
                    output_range=owner.shard_ranges[index],
                )
            continue
        if isinstance(owner, MergedColumnParallelLinear):
            prefix = module_name.rsplit(".", 1)[0]
            for index, component in enumerate(owner.lora_components):
                yield LoraTargetSpec(
                    logical_name=f"{prefix}.{component}",
                    component=component,
                    execution_pattern=LoraExecutionPattern.MERGED_COMPONENT,
                    owner=owner,
                    base_weight=owner.weight,
                    global_in_features=owner.in_features,
                    global_out_features=owner.out_features[index],
                    local_in_features=owner.in_features,
                    local_out_features=owner.local_out_features[index],
                    component_index=index,
                )
            continue
        if isinstance(owner, ReplicatedLinear):
            component = module_name.rsplit(".", 1)[-1]
            yield LoraTargetSpec(
                logical_name=module_name,
                component=component,
                execution_pattern=LoraExecutionPattern.REPLICATED,
                owner=owner,
                base_weight=owner.weight,
                global_in_features=owner.in_features,
                global_out_features=owner.out_features,
                local_in_features=owner.in_features,
                local_out_features=owner.out_features,
                output_range=(0, owner.out_features),
            )
            continue
        if isinstance(owner, ColumnParallelLinear):
            component = module_name.rsplit(".", 1)[-1]
            yield LoraTargetSpec(
                logical_name=module_name,
                component=component,
                execution_pattern=LoraExecutionPattern.COLUMN_PARALLEL,
                owner=owner,
                base_weight=owner.weight,
                global_in_features=owner.in_features,
                global_out_features=owner.out_features,
                local_in_features=owner.in_features,
                local_out_features=owner.local_out_features,
            )
            continue
        if isinstance(owner, RowParallelLinear):
            component = module_name.rsplit(".", 1)[-1]
            yield LoraTargetSpec(
                logical_name=module_name,
                component=component,
                execution_pattern=LoraExecutionPattern.ROW_PARALLEL,
                owner=owner,
                base_weight=owner.weight,
                global_in_features=owner.in_features,
                global_out_features=owner.out_features,
                local_in_features=owner.local_in_features,
                local_out_features=owner.out_features,
            )
            continue
        binding = getattr(owner, "lora_slots", None)
        if isinstance(binding, RoutedExpertLoraBinding):
            for target in binding.targets:
                yield LoraTargetSpec(
                    logical_name=f"{module_name}.{{expert}}.{target.component}",
                    component=target.component,
                    execution_pattern=LoraExecutionPattern.ROUTED_GROUPED_EXPERT,
                    owner=owner,
                    base_weight=_resolve_parameter(owner, target.weight_path),
                    global_in_features=target.in_features,
                    global_out_features=target.out_features,
                    local_in_features=target.in_features,
                    local_out_features=target.out_features,
                )


def _resolve_parameter(owner: nn.Module, path: str) -> nn.Parameter:
    value: object = owner
    for part in path.split("."):
        value = getattr(value, part)
    if not isinstance(value, nn.Parameter):
        raise TypeError(f"{type(owner).__name__}.{path} is not a Parameter")
    return value


def _validate_resolved_targets(specs: list[LoraTargetSpec]) -> None:
    names: set[str] = set()
    for spec in specs:
        if spec.logical_name in names:
            raise ValueError(f"duplicate native LoRA target {spec.logical_name}")
        names.add(spec.logical_name)
        if spec.execution_pattern in {
            LoraExecutionPattern.MERGED_COMPONENT,
            LoraExecutionPattern.ROUTED_GROUPED_EXPERT,
        }:
            if spec.component in spec.owner.lora_slots:
                raise ValueError(f"native LoRA target {spec.logical_name} is already bound")
        elif spec.owner.lora_slot is not None:
            raise ValueError(f"native LoRA target {spec.logical_name} is already bound")


def _new_slot(
    spec: LoraTargetSpec,
    config: LoraConfig,
    seed: int,
    runtime_state: _AdapterRuntimeState,
) -> LoraSlot | RoutedExpertLoraSlot:
    if spec.execution_pattern is LoraExecutionPattern.ROUTED_GROUPED_EXPERT:
        return RoutedExpertLoraSlot(
            logical_name=spec.logical_name,
            base_weight=spec.base_weight,
            local_num_experts=spec.owner.local_num_experts,
            local_expert_start=spec.owner.local_expert_start,
            in_features=spec.global_in_features,
            out_features=spec.global_out_features,
            config=config,
            seed=seed,
            runtime_state=runtime_state,
        )
    return LoraSlot(
        logical_name=spec.logical_name,
        base_weight=spec.base_weight,
        global_in_features=spec.global_in_features,
        global_out_features=spec.global_out_features,
        local_in_features=spec.local_in_features,
        local_out_features=spec.local_out_features,
        row_parallel=spec.execution_pattern is LoraExecutionPattern.ROW_PARALLEL,
        output_range=spec.output_range,
        config=config,
        seed=seed,
        runtime_state=runtime_state,
    )


def _bind_slot(spec: LoraTargetSpec, slot: LoraSlot | RoutedExpertLoraSlot) -> None:
    if spec.execution_pattern is LoraExecutionPattern.MERGED_COMPONENT:
        assert spec.component_index is not None
        spec.owner.install_lora_component(spec.component, spec.component_index, slot)
    elif spec.execution_pattern is LoraExecutionPattern.ROUTED_GROUPED_EXPERT:
        spec.owner.install_lora_component(spec.component, slot)
    else:
        spec.owner.install_lora(slot)


def _matching_targets(requested: set[str], component: str, logical_name: str) -> set[str]:
    aliases = {component, logical_name}
    if ".{expert}." in logical_name:
        aliases.add(logical_name.replace(".{expert}.", "."))
    return requested & aliases
