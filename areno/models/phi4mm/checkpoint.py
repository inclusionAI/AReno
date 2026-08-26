"""Strict checkpoint mapping for the supported Phi-4-Multimodal paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from areno.engine.checkpoints.common import (
    CheckpointSpec,
    CheckpointTensorStore,
    LayerSpec,
    PackedSectionColumnSpec,
    ParallelTensorSpec,
    PolicyTensorStore,
    ReplicatedTensorSpec,
    TopLevelSpec,
    copy_merged_column,
    gather_tensor_parallel_split_column_tensor,
    gather_tensor_parallel_tensor,
    load_checkpoint_weights,
    rank0_tensor,
    save_checkpoint_weights,
)
from areno.engine.checkpoints.io import SafetensorsIndex, _copy_row
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


def audit_phi4mm_checkpoint(
    model_path: str | Path,
    num_hidden_layers: int,
    vision_keys: set[str] | None = None,
    vision_lora_keys: set[str] | None = None,
) -> Phi4MMCheckpointAudit:
    """Classify every checkpoint key and reject missing or unknown tensors."""

    index = SafetensorsIndex(model_path, progress=False)
    try:
        checkpoint_keys = set(index.weight_map)
    finally:
        index.close()
    base_keys = _required_base_keys(num_hidden_layers)
    required = base_keys | (vision_keys or set()) | (vision_lora_keys or set())
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
    """Audit and load the supported Phi-4 language and multimodal tensors."""

    model.config.validate_tp(get_tp_context().world_size)
    multimodal_keys = _multimodal_checkpoint_keys(model)
    lora_keys = _lora_checkpoint_keys(model)
    audit = audit_phi4mm_checkpoint(model_path, len(model.layers), multimodal_keys, lora_keys)
    load_checkpoint_weights(model, str(model_path), CHECKPOINT_SPEC)
    if multimodal_keys:
        _load_multimodal_weights(model, model_path, multimodal_keys)
    if lora_keys:
        _load_lora_weights(model, model_path)
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise RuntimeError("Phi4MM embedding and LM head weight tying was lost during checkpoint loading")
    return audit


def _multimodal_checkpoint_keys(model: nn.Module) -> set[str]:
    extended = getattr(model.model, "embed_tokens_extend", None)
    if extended is None:
        return set()
    names = {name for name, _ in extended.named_parameters()}
    names.update(name for name, _ in extended.named_buffers())
    return {f"model.embed_tokens_extend.{name}" for name in names}


def _lora_checkpoint_keys(model: nn.Module) -> set[str]:
    return {
        f"model.layers.{layer_idx}.{name}"
        for layer_idx, layer in enumerate(model.layers)
        for name, _ in layer.named_parameters()
        if any(f".lora_{side}.{adapter}.weight" in name for side in "AB" for adapter in ("vision", "speech"))
    }


@torch.no_grad()
def _load_multimodal_weights(model: nn.Module, model_path: str | Path, keys: set[str] | None = None) -> None:
    extended = getattr(model.model, "embed_tokens_extend", None)
    if extended is None:
        return
    expected = keys if keys is not None else _multimodal_checkpoint_keys(model)
    index = SafetensorsIndex(model_path)
    try:
        missing = sorted(expected - set(index.weight_map))
        if missing:
            raise KeyError(f"missing Phi4MM multimodal weight {missing[0]}")
        index.prefetch(sorted(expected))
        for name, parameter in list(extended.named_parameters()) + list(extended.named_buffers()):
            key = f"model.embed_tokens_extend.{name}"
            source = index.get_tensor(key)
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"checkpoint tensor {key} shape {tuple(source.shape)} does not match {tuple(parameter.shape)}"
                )
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
    finally:
        index.close()


@torch.no_grad()
def _load_lora_weights(model: nn.Module, model_path: str | Path) -> None:
    context = get_tp_context()
    index = SafetensorsIndex(model_path)
    try:
        for layer_idx, layer in enumerate(model.layers):
            prefix = f"model.layers.{layer_idx}"
            for adapter in ("vision", "speech"):
                for name, module, sections in (
                    ("self_attn.qkv_proj", layer.self_attn.qkv_proj, layer.self_attn.qkv_proj.out_features),
                    ("mlp.gate_up_proj", layer.mlp.gate_up_proj, layer.mlp.gate_up_proj.out_features),
                ):
                    if adapter not in module.lora_A:
                        continue
                    lora_a = f"{prefix}.{name}.lora_A.{adapter}.weight"
                    lora_b = f"{prefix}.{name}.lora_B.{adapter}.weight"
                    module.lora_A[adapter].weight.copy_(index.get_tensor(lora_a).to(dtype=module.weight.dtype))
                    copy_merged_column(
                        module.lora_B[adapter].weight,
                        list(index.get_tensor(lora_b).split(tuple(sections), dim=0)),
                        context.rank,
                        context.world_size,
                    )
                for name, module in (
                    ("self_attn.o_proj", layer.self_attn.o_proj),
                    ("mlp.down_proj", layer.mlp.down_proj),
                ):
                    if adapter not in module.lora_A:
                        continue
                    lora_a = f"{prefix}.{name}.lora_A.{adapter}.weight"
                    lora_b = f"{prefix}.{name}.lora_B.{adapter}.weight"
                    _copy_row(
                        module.lora_A[adapter].weight,
                        index.get_tensor(lora_a),
                        context.rank,
                        context.world_size,
                    )
                    module.lora_B[adapter].weight.copy_(
                        index.get_tensor(lora_b).to(device=module.weight.device, dtype=module.weight.dtype)
                    )
    finally:
        index.close()


def save_phi4mm_weights(
    model: nn.Module,
    output_path: str | Path,
    source_path: str | Path | None,
) -> str | None:
    """Save Phi-4 language and multimodal weights in the official HF key layout."""

    model.config.validate_tp(get_tp_context().world_size)
    return save_checkpoint_weights(
        model,
        str(output_path),
        None if source_path is None else str(source_path),
        CHECKPOINT_SPEC,
        extra_tensors_fn=lambda tensors: _save_multimodal_weights(tensors, model),
        copy_passthrough=False,
    )


def _save_multimodal_weights(tensors: CheckpointTensorStore | PolicyTensorStore, model: nn.Module) -> None:
    """Stage replicated modality tensors and TP-aware LoRA tensors."""

    extended = getattr(model.model, "embed_tokens_extend", None)
    if extended is None:
        return
    for name, parameter in extended.named_parameters():
        tensors[f"model.embed_tokens_extend.{name}"] = rank0_tensor(parameter)
    for name, buffer in extended.named_buffers():
        tensors[f"model.embed_tokens_extend.{name}"] = rank0_tensor(buffer)

    for layer_idx, layer in enumerate(model.layers):
        prefix = f"model.layers.{layer_idx}"
        for name, module in (
            ("self_attn.qkv_proj", layer.self_attn.qkv_proj),
            ("mlp.gate_up_proj", layer.mlp.gate_up_proj),
        ):
            if not hasattr(module, "lora_A"):
                continue
            for adapter in module.lora_A:
                tensors[f"{prefix}.{name}.lora_A.{adapter}.weight"] = rank0_tensor(module.lora_A[adapter].weight)
                tensors[f"{prefix}.{name}.lora_B.{adapter}.weight"] = gather_tensor_parallel_split_column_tensor(
                    module.lora_B[adapter].weight,
                    list(module.local_out_features),
                )
        for name, module in (
            ("self_attn.o_proj", layer.self_attn.o_proj),
            ("mlp.down_proj", layer.mlp.down_proj),
        ):
            if not hasattr(module, "lora_A"):
                continue
            for adapter in module.lora_A:
                tensors[f"{prefix}.{name}.lora_A.{adapter}.weight"] = gather_tensor_parallel_tensor(
                    module.lora_A[adapter].weight,
                    dim=1,
                )
                tensors[f"{prefix}.{name}.lora_B.{adapter}.weight"] = rank0_tensor(module.lora_B[adapter].weight)


# Retain the PR2 private names for downstream tests and integrations that imported
# them before audio support generalized the checkpoint path.
_vision_checkpoint_keys = _multimodal_checkpoint_keys
_vision_lora_checkpoint_keys = _lora_checkpoint_keys
_load_vision_weights = _load_multimodal_weights
_load_vision_lora_weights = _load_lora_weights
_save_vision_weights = _save_multimodal_weights
