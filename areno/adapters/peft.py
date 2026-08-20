"""Standard PEFT safetensors import/export for native TP LoRA slots."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from areno.adapters.lora import AdapterRegistry, LoraSlot, RoutedExpertLoraSlot
from areno.engine.parallel.context import get_tp_context

_PREFIX = "base_model.model.model."


@torch.no_grad()
def load_peft_adapter(registry: AdapterRegistry, path: str | Path) -> None:
    """Copy one supported PEFT adapter into the registry's stable A/B storage."""

    input_path = Path(path)
    tensors = load_file(input_path / "adapter_model.safetensors", device="cpu")
    ctx = get_tp_context()
    expected_shapes = _expected_peft_shapes(registry, ctx.world_size)
    actual_keys = set(tensors)
    expected_keys = set(expected_shapes)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "PEFT adapter tensor keys do not match the native LoRA registry: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(tensors[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"PEFT adapter tensor {key!r} has shape {actual_shape}, expected {expected_shape}")

    for logical_name, slot in registry.slots.items():
        if isinstance(slot, RoutedExpertLoraSlot):
            for local_expert_id in range(slot.local_num_experts):
                expert_id = slot.local_expert_start + local_expert_id
                expert_name = logical_name.format(expert=expert_id)
                slot.lora_A[local_expert_id].copy_(
                    tensors[_key(expert_name, "A")].to(device=slot.lora_A.device, dtype=slot.lora_A.dtype)
                )
                slot.lora_B[local_expert_id].copy_(
                    tensors[_key(expert_name, "B")].to(device=slot.lora_B.device, dtype=slot.lora_B.dtype)
                )
            continue
        canonical_A = tensors[_key(logical_name, "A")]
        canonical_B = tensors[_key(logical_name, "B")]
        if slot.row_parallel:
            width = slot.local_in_features
            local_A = canonical_A[:, ctx.rank * width : (ctx.rank + 1) * width]
            local_B = canonical_B
        else:
            local_A = canonical_A
            local_B = canonical_B[slot.output_start : slot.output_end]
        slot.lora_A.copy_(local_A.to(device=slot.lora_A.device, dtype=slot.lora_A.dtype))
        slot.lora_B.copy_(local_B.to(device=slot.lora_B.device, dtype=slot.lora_B.dtype))


@torch.no_grad()
def export_peft_adapter(
    registry: AdapterRegistry,
    path: str | Path,
    *,
    base_model_name_or_path: str | None,
) -> str | None:
    """Gather the authoritative DP0 TP shards and write a PEFT adapter."""

    ctx = get_tp_context()
    if ctx.dp_rank != 0:
        return None
    state: dict[str, torch.Tensor] = {}
    for logical_name, slot in registry.slots.items():
        if isinstance(slot, RoutedExpertLoraSlot):
            gathered_A = _all_gather(slot.lora_A.detach(), ctx.world_size, ctx.group)
            gathered_B = _all_gather(slot.lora_B.detach(), ctx.world_size, ctx.group)
            if ctx.rank != 0:
                continue
            canonical_A = torch.cat(gathered_A, dim=0)
            canonical_B = torch.cat(gathered_B, dim=0)
            for expert_id in range(canonical_A.shape[0]):
                expert_name = logical_name.format(expert=expert_id)
                state[_key(expert_name, "A")] = canonical_A[expert_id].float().cpu().contiguous()
                state[_key(expert_name, "B")] = canonical_B[expert_id].float().cpu().contiguous()
            continue
        if slot.row_parallel:
            gathered_A = _all_gather(slot.lora_A.detach(), ctx.world_size, ctx.group)
            if ctx.rank != 0:
                continue
            canonical_A = torch.cat(gathered_A, dim=1)
            canonical_B = slot.lora_B.detach()
        else:
            gathered_B = _all_gather(slot.lora_B.detach(), ctx.world_size, ctx.group)
            if ctx.rank != 0:
                continue
            canonical_A = slot.lora_A.detach()
            canonical_B = (
                _gather_replicated_column(slot, gathered_B, ctx.world_size)
                if slot.output_replicated
                else torch.cat(gathered_B, dim=0)
            )
        state[_key(logical_name, "A")] = canonical_A.float().cpu().contiguous()
        state[_key(logical_name, "B")] = canonical_B.float().cpu().contiguous()
    if ctx.rank != 0:
        return None

    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": registry.config.alpha,
        "lora_dropout": registry.config.dropout,
        "peft_type": "LORA",
        "r": registry.config.rank,
        "target_modules": list(registry.config.target_modules),
        "task_type": "CAUSAL_LM",
    }
    (output_path / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_file(state, output_path / "adapter_model.safetensors")
    return str(output_path)


def _all_gather(tensor: torch.Tensor, world_size: int, group) -> list[torch.Tensor]:
    if world_size == 1:
        return [tensor]
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return gathered


def _gather_replicated_column(slot: LoraSlot, gathered: list[torch.Tensor], world_size: int) -> torch.Tensor:
    """Keep one authoritative copy of every unique replicated output range."""

    local_rows = slot.local_out_features
    unique_shards = slot.global_out_features // local_rows
    ranks_per_shard = world_size // unique_shards
    output = torch.empty(
        (slot.global_out_features, slot.rank),
        device=gathered[0].device,
        dtype=gathered[0].dtype,
    )
    for shard_index in range(unique_shards):
        start = shard_index * local_rows
        output[start : start + local_rows].copy_(gathered[shard_index * ranks_per_shard])
    return output


def _key(logical_name: str, component: str) -> str:
    return f"{_PREFIX}{logical_name}.lora_{component}.weight"


def _expected_peft_shapes(registry: AdapterRegistry, tp_size: int) -> dict[str, tuple[int, ...]]:
    """Return the one canonical PEFT tensor contract represented by a registry."""

    shapes: dict[str, tuple[int, ...]] = {}
    for logical_name, slot in registry.slots.items():
        if isinstance(slot, RoutedExpertLoraSlot):
            for expert_id in range(slot.local_num_experts * tp_size):
                expert_name = logical_name.format(expert=expert_id)
                shapes[_key(expert_name, "A")] = (slot.rank, slot.in_features)
                shapes[_key(expert_name, "B")] = (slot.out_features, slot.rank)
            continue
        shapes[_key(logical_name, "A")] = (slot.rank, slot.global_in_features)
        shapes[_key(logical_name, "B")] = (slot.global_out_features, slot.rank)
    return shapes
