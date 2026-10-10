"""Replace frozen native projections with packed NF4 after loading LoRA slots."""

from __future__ import annotations

import torch

from areno.accel.nf4 import NF4Weight
from areno.engine.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear


@torch.no_grad()
def initialize_qlora(model):
    from areno.models.bailing_v3.model import ArenoGroupedLinear

    if model.config.model_type not in {"qwen3", "bailing_moe_v3"}:
        raise ValueError("QLoRA currently supports native Qwen3 dense and Bailing-MoE V3")
    total_original = total_quantized = count = 0
    for module in list(model.modules()):
        if not isinstance(
            module, (ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ArenoGroupedLinear)
        ):
            continue
        weight = module.weight
        if weight.requires_grad:
            raise ValueError("initialize native LoRA before quantizing the frozen base")
        total_original += weight.numel() * weight.element_size()
        module.quantized_weight = NF4Weight(weight)
        module.weight = None
        total_quantized += module.quantized_weight.storage_bytes
        count += 1
    if not count:
        raise ValueError("QLoRA found no supported linear projections")
    model.qlora_memory = {
        "projections": count,
        "original_weight_bytes": total_original,
        "quantized_weight_bytes": total_quantized,
    }
    return model.qlora_memory
