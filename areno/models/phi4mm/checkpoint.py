"""Strict text-only checkpoint mapping for Phi-4-Multimodal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from areno.engine.checkpoints.common import (
    CheckpointSpec,
    LayerSpec,
    PackedSectionColumnSpec,
    ParallelTensorSpec,
    ReplicatedTensorSpec,
    TopLevelSpec,
    load_checkpoint_weights,
    save_checkpoint_weights,
)
from areno.engine.checkpoints.io import SafetensorsIndex
from areno.engine.parallel.context import get_tp_context

TOP_LEVEL_SPEC = TopLevelSpec(
    embedding_key="model.embed_tokens.weight",
    embedding_attr="model.embed_tokens",
    norm_key="model.norm.weight",
    norm_attr="model.norm.weight",
)
LAYER_NORM_SPECS = (
    ReplicatedTensorSpec("{prefix}.input_layernorm.weight", "input_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.post_attention_layernorm.weight", "post_attention_layernorm.weight"),
)
QKV_SPEC = PackedSectionColumnSpec(
    key="{prefix}.self_attn.qkv_proj.base_layer.weight",
    tensor_attr="self_attn.qkv_proj.weight",
    global_sizes_attr="self_attn.qkv_proj.out_features",
    local_sizes_attr="self_attn.qkv_proj.local_out_features",
)
ATTN_OUT_SPEC = ParallelTensorSpec(
    "{prefix}.self_attn.o_proj.base_layer.weight",
    "self_attn.o_proj.weight",
    1,
)
GATE_UP_SPEC = PackedSectionColumnSpec(
    key="{prefix}.mlp.gate_up_proj.base_layer.weight",
    tensor_attr="mlp.gate_up_proj.weight",
    global_sizes_attr="mlp.gate_up_proj.out_features",
    local_sizes_attr="mlp.gate_up_proj.local_out_features",
)
MLP_DOWN_SPEC = ParallelTensorSpec(
    "{prefix}.mlp.down_proj.base_layer.weight",
    "mlp.down_proj.weight",
    1,
)
LAYER_SPEC = LayerSpec(
    prefix="model.layers.{layer}",
    replicated=LAYER_NORM_SPECS,
    load_ops=(QKV_SPEC, ATTN_OUT_SPEC, GATE_UP_SPEC, MLP_DOWN_SPEC),
    save_ops=(QKV_SPEC, ATTN_OUT_SPEC, GATE_UP_SPEC, MLP_DOWN_SPEC),
)
CHECKPOINT_SPEC = CheckpointSpec(top_level=TOP_LEVEL_SPEC, layer=LAYER_SPEC)

_LAYER_BASE_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.qkv_proj.base_layer.weight",
    "self_attn.o_proj.base_layer.weight",
    "mlp.gate_up_proj.base_layer.weight",
    "mlp.down_proj.base_layer.weight",
)
_LORA_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj))\."
    r"lora_[AB]\.(vision|speech)\.weight$"
)


@dataclass(frozen=True, slots=True)
class Phi4MMCheckpointAudit:
    total: int
    consumed: int
    vision_lora_skipped: int
    speech_lora_skipped: int
    vision_skipped: int
    audio_skipped: int
    unknown: int


def _required_base_keys(num_hidden_layers: int) -> set[str]:
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    for layer in range(num_hidden_layers):
        required.update(f"model.layers.{layer}.{suffix}" for suffix in _LAYER_BASE_SUFFIXES)
    return required


def audit_phi4mm_checkpoint(model_path: str | Path, num_hidden_layers: int) -> Phi4MMCheckpointAudit:
    """Classify every checkpoint key and reject missing or unknown tensors."""

    index = SafetensorsIndex(model_path, progress=False)
    try:
        checkpoint_keys = set(index.weight_map)
    finally:
        index.close()
    required = _required_base_keys(num_hidden_layers)
    missing = sorted(required - checkpoint_keys)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Phi4MM checkpoint is missing {len(missing)} required base-language tensors: {preview}")

    counts = {"vision_lora": 0, "speech_lora": 0, "vision": 0, "audio": 0}
    unknown = []
    for tensor_key in checkpoint_keys - required:
        lora_match = _LORA_PATTERN.fullmatch(tensor_key)
        if lora_match is not None and int(lora_match.group(1)) < num_hidden_layers:
            counts[f"{lora_match.group(2)}_lora"] += 1
        elif tensor_key.startswith("model.embed_tokens_extend.image_embed."):
            counts["vision"] += 1
        elif tensor_key.startswith("model.embed_tokens_extend.audio_embed."):
            counts["audio"] += 1
        else:
            unknown.append(tensor_key)
    if unknown:
        preview = ", ".join(sorted(unknown)[:5])
        raise ValueError(f"Phi4MM checkpoint contains {len(unknown)} unknown tensors: {preview}")
    return Phi4MMCheckpointAudit(
        total=len(checkpoint_keys),
        consumed=len(required),
        vision_lora_skipped=counts["vision_lora"],
        speech_lora_skipped=counts["speech_lora"],
        vision_skipped=counts["vision"],
        audio_skipped=counts["audio"],
        unknown=0,
    )


def load_phi4mm_weights(model: nn.Module, model_path: str | Path) -> Phi4MMCheckpointAudit:
    """Audit and load the supported Phi-4 base-language tensors."""

    model.config.validate_tp(get_tp_context().world_size)
    audit = audit_phi4mm_checkpoint(model_path, len(model.layers))
    load_checkpoint_weights(model, str(model_path), CHECKPOINT_SPEC)
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise RuntimeError("Phi4MM embedding and LM head weight tying was lost during checkpoint loading")
    return audit


def save_phi4mm_weights(
    model: nn.Module,
    output_path: str | Path,
    source_path: str | Path | None,
) -> str | None:
    """Save only Phi-4 base-language weights in official HF key layout."""

    model.config.validate_tp(get_tp_context().world_size)
    return save_checkpoint_weights(
        model,
        str(output_path),
        None if source_path is None else str(source_path),
        CHECKPOINT_SPEC,
        copy_passthrough=False,
    )
