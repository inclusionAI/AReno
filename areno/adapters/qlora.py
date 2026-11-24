"""Replace frozen native projections with packed NF4 after loading LoRA slots."""

from __future__ import annotations

import torch

from areno.accel.nf4 import NF4Weight
from areno.engine.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear


@torch.no_grad()
def initialize_qlora(model):
    from areno.models.bailing_v3.model import ArenoGroupedLinear
    from areno.models.qwen3.model import Qwen3MoeExperts
    from areno.models.registry import _adapter

    if not _adapter(model.config.model_type).supports_qlora:
        raise ValueError(f"QLoRA is not supported for {model.config.model_type}")
    total_original = total_quantized = count = 0
    for module in list(model.modules()):
        if isinstance(module, Qwen3MoeExperts):
            weights = (("gate_up_weight", "quantized_gate_up"), ("down_weight", "quantized_down"))
        elif isinstance(
            module, ColumnParallelLinear | MergedColumnParallelLinear | RowParallelLinear | ArenoGroupedLinear
        ):
            weights = (("weight", "quantized_weight"),)
        else:
            continue
        for name, packed_name in weights:
            weight = getattr(module, name)
            if weight.requires_grad:
                raise ValueError("initialize native LoRA before quantizing the frozen base")
            packed = NF4Weight(weight)
            setattr(module, packed_name, packed)
            setattr(module, name, None)
            total_original += weight.numel() * weight.element_size()
            total_quantized += packed.storage_bytes
            count += 1
    if not count:
        raise ValueError("QLoRA found no supported linear projections")
    model.qlora_memory = {
        "projections": count,
        "original_weight_bytes": total_original,
        "quantized_weight_bytes": total_quantized,
    }
    return model.qlora_memory
