"""Bounded TP collection of live weights for native GGUF export."""

from __future__ import annotations

import math
import re

import torch
import torch.distributed as dist

from areno.engine.parallel.context import get_tp_context
from areno.models.registry import build_policy_weight_plan


def _collect(layout, ctx, *, start: int = 0, shape: tuple[int, ...] | None = None) -> torch.Tensor | None:
    """Restore canonical pieces without the HF writer's per-tensor owner map."""
    shape = layout.shape if shape is None else shape
    count = math.prod(shape)
    output = torch.empty(count, dtype=torch.float32, device="cpu") if ctx.rank == 0 else None
    chunk_count = max(1, (32 * 1024 * 1024) // torch.empty((), dtype=layout.dtype).element_size())
    for offset in range(0, count, chunk_count):
        chunk = torch.empty(min(chunk_count, count - offset), dtype=layout.dtype, device=ctx.device)
        layout.read_chunk(start + offset, chunk, include_replicated=ctx.rank == 0)
        if ctx.world_size > 1:
            dist.all_reduce(chunk, group=ctx.group)
        if output is not None:
            output[offset : offset + chunk.numel()].copy_(chunk.float().cpu())
    return output.reshape(shape) if output is not None else None


def _merge_lora(registry, adapters, key, tensor, expert=None) -> set[str]:
    if registry is None:
        return set()
    logical = key.removeprefix("model.").removesuffix(".weight")
    match = re.search(r"\.experts\.(\d+)\.", logical)
    if match:
        expert = int(match.group(1))
        logical = logical[: match.start(1)] + "{expert}" + logical[match.end(1) :]
    candidates = [(logical, tensor)]
    if expert is not None and not match:
        if logical.endswith(".experts.gate_up_proj"):
            gate, up = tensor.chunk(2, dim=0)
            prefix = logical.removesuffix("gate_up_proj") + "{expert}."
            candidates = [(prefix + "gate_proj", gate), (prefix + "up_proj", up)]
        elif logical.endswith(".experts.down_proj"):
            candidates = [(logical.removesuffix("down_proj") + "{expert}.down_proj", tensor)]
    used = set()
    for name, value in candidates:
        if name not in registry.slots:
            continue
        a, b = adapters[f"{name}.lora_A.weight"], adapters[f"{name}.lora_B.weight"]
        if expert is not None:
            a, b = a[expert], b[expert]
        if tuple(value.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"LoRA merge shape mismatch: {key}")
        for row in range(0, value.shape[0], 1024):
            value[row : row + 1024].addmm_(b[row : row + 1024], a, beta=1, alpha=float(registry.config.scale))
        used.add(name)
    return used


def _leader_call(ctx, function, *args):
    """Publish writer errors before peers enter another collective."""
    result, error = None, None
    if ctx.rank == 0:
        try:
            result = function(*args)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    status = [error]
    if ctx.world_size > 1:
        dist.broadcast_object_list(status, src=ctx.tp_global_rank(0), group=ctx.group)
    if status[0] is not None:
        raise RuntimeError(f"GGUF export failed: {status[0]}")
    return result


@torch.no_grad()
def export_worker(worker, output_dir: str, file_type: str) -> str | None:
    from areno.export.architectures import collection_blocks
    from areno.export.gguf import GGUFPackage

    ctx = get_tp_context()
    if ctx.dp_rank != 0:
        return None
    if worker.config.lora is not None and worker.config.lora.qlora:
        raise ValueError("GGUF export of packed QLoRA bases is not supported")
    plan = build_policy_weight_plan(worker.model, worker.config.model)
    shapes = {key: task.policy_layout().shape for key, task in plan.items()}
    registry = worker.adapter_registry
    package = _leader_call(
        ctx,
        lambda: GGUFPackage(
            worker.config.model_path,
            output_dir,
            shapes,
            file_type=file_type,
            lora_merged=registry is not None,
            model_type=worker.config.model.model_type,
        ),
    )
    try:
        adapters = {}
        if registry is not None:
            from areno.engine.policy_sync import build_adapter_policy_plan

            for key, task in build_adapter_policy_plan(registry).items():
                value = _collect(task.policy_layout(), ctx)
                if ctx.rank == 0:
                    adapters[key] = value
        used_slots = set()
        for key, task in plan.items():
            layout = task.policy_layout()
            for expert, start, shape in collection_blocks(key, layout.shape):
                tensor = _collect(layout, ctx, start=start, shape=shape)

                def write(key: str, tensor: torch.Tensor, expert: int | None) -> None:
                    used_slots.update(_merge_lora(registry, adapters, key, tensor, expert))
                    package.write(key, tensor.numpy(), expert=expert)

                _leader_call(ctx, write, key, tensor, expert)
                del tensor

        def check_adapters():
            if registry is not None and registry.slots.keys() - used_slots:
                raise ValueError(f"unmerged LoRA targets: {sorted(registry.slots.keys() - used_slots)}")

        _leader_call(ctx, check_adapters)
        return _leader_call(ctx, package.finish if package is not None else lambda: None)
    finally:
        if package is not None:
            package.abort()
