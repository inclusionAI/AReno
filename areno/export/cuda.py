"""Bounded TP collection of live weights for native GGUF export."""

from __future__ import annotations

import torch
import torch.distributed as dist

from areno.engine.parallel.context import get_tp_context
from areno.models.registry import build_policy_weight_plan


def _collect(layout, ctx) -> torch.Tensor | None:
    """Restore canonical pieces without the HF writer's per-tensor owner map."""
    count = layout.numel
    output = torch.empty(count, dtype=torch.float32, device="cpu") if ctx.rank == 0 else None
    chunk_count = max(1, (32 * 1024 * 1024) // torch.empty((), dtype=layout.dtype).element_size())
    for offset in range(0, count, chunk_count):
        chunk = torch.empty(min(chunk_count, count - offset), dtype=layout.dtype, device=ctx.device)
        layout.read_chunk(offset, chunk, include_replicated=ctx.rank == 0)
        if ctx.world_size > 1:
            dist.all_reduce(chunk, group=ctx.group)
        if output is not None:
            output[offset : offset + chunk.numel()].copy_(chunk.float().cpu())
    return output.reshape(layout.shape) if output is not None else None


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
            worker.config.model_path, output_dir, shapes, file_type=file_type, lora_merged=registry is not None
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
        for key, task in plan.items():
            tensor = _collect(task.policy_layout(), ctx)

            def write(key: str, tensor: torch.Tensor) -> None:
                logical_name = key.removeprefix("model.").removesuffix(".weight")
                slot = registry.slots.get(logical_name) if registry is not None else None
                if slot is not None:
                    a = adapters[f"{logical_name}.lora_A.weight"]
                    b = adapters[f"{logical_name}.lora_B.weight"]
                    if tuple(tensor.shape) != (b.shape[0], a.shape[1]):
                        raise ValueError(f"LoRA merge shape mismatch: {key}")
                    # Small row blocks avoid retaining a second full delta matrix.
                    scale = float(registry.config.scale)
                    for row in range(0, tensor.shape[0], 1024):
                        tensor[row : row + 1024].addmm_(b[row : row + 1024], a, beta=1, alpha=scale)
                package.write(key, tensor.numpy())

            _leader_call(ctx, write, key, tensor)
            del tensor
        return _leader_call(ctx, package.finish if package is not None else lambda: None)
    finally:
        if package is not None:
            package.abort()
