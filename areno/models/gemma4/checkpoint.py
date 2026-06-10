"""Gemma4 HF safetensors load/save specs.

Gemma4 places the text trunk under ``model`` (plain causal LM) or
``model.language_model`` (Gemma4 multimodal); the prefix is supplied at runtime
via ``checkpoint_spec(prefix)``. Peculiarities handled here:
    * KSharedQKVColumnSpec — Gemma4 sometimes stores K as a TP-replicated row
      shared across query groups; the loader knows how to broadcast it.
    * Optional per-layer input embeddings (PLE): per-layer projection,
      per-layer input gate, residual scaling, and a global tying projection
      that feeds the per-token-per-layer signal into each block.
    * Q/K RMSNorm and Q/K/V/O projection bias are all optional depending on
      sub-variant; everything is wrapped in OptionalXxxSpec.
"""

from __future__ import annotations

import torch

from areno.engine.checkpoints.common import (
    CheckpointSpec,
    KSharedQKVColumnSpec,
    LayerSpec,
    MergedColumnSpec,
    OptionalMergedColumnSpec,
    OptionalRangedSplitColumnSpec,
    OptionalReplicatedTensorSpec,
    ParallelTensorSpec,
    RangedSplitColumnSpec,
    ReplicatedTensorSpec,
    SplitColumnSpec,
    TopLevelSpec,
    _copy_parallel_tensor_from_index,
    _gather_ranged_column_tensors,
    attr_path,
    copy_merged_column_from_index,
    gather_moe_expert_weights,
    key,
    rank0_tensor,
    save_parallel_tensors,
    save_split_column_spec,
    split_local_tensors,
)
from areno.engine.checkpoints.io import _owns_checkpoint_tensor, _tensor_to_cpu


def top_level_spec(prefix: str) -> TopLevelSpec:
    """Build the top-level spec for either Gemma4 LM or Gemma4 multimodal trunks.

    The optional PLE tensors live next to the standard embed/norm tensors only
    when the variant ships per-layer-input embeddings, so they are declared as
    optional and silently skipped otherwise.
    """
    return TopLevelSpec(
        embedding_key=f"{prefix}.embed_tokens.weight",
        embedding_attr="embed_tokens",
        norm_key=f"{prefix}.norm.weight",
        norm_attr="norm.weight",
        optional_vocab=(
            # Secondary embedding table, indexed by token id, that feeds the
            # per-layer-input pathway alongside the main vocab embedding.
            ReplicatedTensorSpec(f"{prefix}.embed_tokens_per_layer.weight", "embed_tokens_per_layer.weight"),
        ),
        optional_replicated=(
            # Projects hidden states into the per-layer input subspace and
            # normalizes the result before splitting into per-layer slots.
            ReplicatedTensorSpec(f"{prefix}.per_layer_model_projection.weight", "per_layer_model_projection.weight"),
            ReplicatedTensorSpec(f"{prefix}.per_layer_projection_norm.weight", "per_layer_projection_norm.weight"),
        ),
    )


# Four RMSNorm tensors per layer (pre-attn, post-attn, pre-MLP, post-MLP) match
# Gemma's "sandwich norm" structure.
LAYER_NORM_SPECS = (
    ReplicatedTensorSpec("{prefix}.input_layernorm.weight", "input_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.post_attention_layernorm.weight", "post_attention_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.pre_feedforward_layernorm.weight", "pre_feedforward_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.post_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"),
)
# KSharedQKVColumnSpec handles the case where K is broadcast across query groups
# (TP world replicates K rather than sharding) while Q and V shard normally.
QKV_WEIGHT_SPEC = KSharedQKVColumnSpec(
    dst_attr="self_attn.qkv_proj.weight",
    q_key="{prefix}.self_attn.q_proj.weight",
    k_key="{prefix}.self_attn.k_proj.weight",
    v_key="{prefix}.self_attn.v_proj.weight",
)
# QKV bias only exists when attention_bias=True in the HF config.
QKV_BIAS_LOAD_SPEC = OptionalMergedColumnSpec(
    dst_attr="self_attn.qkv_proj.bias",
    keys=(
        "{prefix}.self_attn.q_proj.bias",
        "{prefix}.self_attn.k_proj.bias",
        "{prefix}.self_attn.v_proj.bias",
    ),
)
# Save side splits the local QKV slab back into per-key tensors using the
# (q,k,v) section sizes recorded on the fused linear.
QKV_SAVE_SPEC = RangedSplitColumnSpec(
    src_attr="self_attn.qkv_proj.weight",
    size_attr="self_attn.qkv_proj.local_out_features",
    keys=(
        "{prefix}.self_attn.q_proj.weight",
        "{prefix}.self_attn.k_proj.weight",
        "{prefix}.self_attn.v_proj.weight",
    ),
)
QKV_BIAS_SAVE_SPEC = OptionalRangedSplitColumnSpec(
    src_attr="self_attn.qkv_proj.bias",
    size_attr="self_attn.qkv_proj.local_out_features",
    keys=QKV_BIAS_LOAD_SPEC.keys,
)
# Per-head Q/K RMSNorm scales (Gemma4 attention applies them prior to RoPE).
Q_NORM_SPEC = OptionalReplicatedTensorSpec("{prefix}.self_attn.q_norm.weight", "self_attn.q_norm.weight")
K_NORM_SPEC = OptionalReplicatedTensorSpec("{prefix}.self_attn.k_norm.weight", "self_attn.k_norm.weight")
ATTN_ROW_SPEC = ParallelTensorSpec("{prefix}.self_attn.o_proj.weight", "self_attn.o_proj.weight", 1)
ATTN_BIAS_SPEC = OptionalReplicatedTensorSpec("{prefix}.self_attn.o_proj.bias", "self_attn.o_proj.bias")
# Fused SwiGLU gate+up. Loader concatenates the two HF tensors before sharding.
GATE_UP_WEIGHT_SPEC = MergedColumnSpec(
    dst_attr="mlp.gate_up_proj.weight",
    keys=("{prefix}.mlp.gate_proj.weight", "{prefix}.mlp.up_proj.weight"),
)
GATE_UP_SAVE_SPEC = SplitColumnSpec(
    src_attr="mlp.gate_up_proj.weight",
    size_attr="mlp.gate_up_proj.local_out_features",
    keys=GATE_UP_WEIGHT_SPEC.keys,
)
MLP_ROW_SPEC = ParallelTensorSpec("{prefix}.mlp.down_proj.weight", "mlp.down_proj.weight", 1)
# Per-layer PLE tensors: input gate, projection back to hidden_size, post-norm,
# and a scalar multiplier applied to the layer output. Same shape on load/save
# so the same optional specs are reused for both directions.
PLE_LOAD_SAVE_SPECS = (
    OptionalReplicatedTensorSpec("{prefix}.per_layer_input_gate.weight", "per_layer_input_gate.weight"),
    OptionalReplicatedTensorSpec("{prefix}.per_layer_projection.weight", "per_layer_projection.weight"),
    OptionalReplicatedTensorSpec("{prefix}.post_per_layer_input_norm.weight", "post_per_layer_input_norm.weight"),
    OptionalReplicatedTensorSpec("{prefix}.layer_scalar", "layer_scalar"),
)
MOE_NORM_SPECS = (
    OptionalReplicatedTensorSpec("{prefix}.pre_feedforward_layernorm_2.weight", "pre_feedforward_layernorm_2.weight"),
    OptionalReplicatedTensorSpec("{prefix}.post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_1.weight"),
    OptionalReplicatedTensorSpec("{prefix}.post_feedforward_layernorm_2.weight", "post_feedforward_layernorm_2.weight"),
)
LOAD_OPS = (
    QKV_WEIGHT_SPEC,
    QKV_BIAS_LOAD_SPEC,
    Q_NORM_SPEC,
    K_NORM_SPEC,
    ATTN_ROW_SPEC,
    ATTN_BIAS_SPEC,
    *PLE_LOAD_SAVE_SPECS,
    *MOE_NORM_SPECS,
)
SAVE_OPS = (
    Q_NORM_SPEC,
    K_NORM_SPEC,
    ATTN_ROW_SPEC,
    ATTN_BIAS_SPEC,
    *PLE_LOAD_SAVE_SPECS,
    *MOE_NORM_SPECS,
)


def load_gemma4_mlp(layer, index, prefix: str, rank: int, world_size: int) -> None:
    """Load either fused dense Gemma4 MLP or routed Gemma4 MoE MLP."""

    mlp = layer.mlp
    copy_merged_column_from_index(
        attr_path(mlp, GATE_UP_WEIGHT_SPEC.dst_attr.removeprefix("mlp.")),
        index,
        [template.format(prefix=prefix) for template in GATE_UP_WEIGHT_SPEC.keys],
        rank,
        world_size,
    )
    _copy_parallel_tensor_from_index(
        attr_path(mlp, MLP_ROW_SPEC.attr.removeprefix("mlp.")),
        index,
        MLP_ROW_SPEC.key.format(prefix=prefix),
        MLP_ROW_SPEC.dim,
        rank,
        world_size,
    )
    if getattr(layer, "moe", None) is not None:
        _load_gemma4_moe(layer, index, prefix, rank, world_size)


def save_gemma4_mlp(tensors, prefix: str, layer, context) -> None:
    """Save either fused dense Gemma4 MLP or routed Gemma4 MoE MLP."""

    del context
    mlp = layer.mlp
    save_split_column_spec(tensors, mlp, prefix, SplitColumnSpec("gate_up_proj.weight", "gate_up_proj.local_out_features", GATE_UP_WEIGHT_SPEC.keys))
    save_parallel_tensors(tensors, mlp, prefix, (ParallelTensorSpec(MLP_ROW_SPEC.key, "down_proj.weight", MLP_ROW_SPEC.dim),))
    if getattr(layer, "moe", None) is not None:
        _save_gemma4_moe(tensors, prefix, layer)


def save_gemma4_attention(tensors, prefix: str, layer, context) -> None:
    """Save Gemma4 QKV while preserving full-attention K=V HF layouts."""

    del context
    attention = layer.self_attn
    qkv = attention.qkv_proj
    parts = split_local_tensors(qkv.weight, qkv.local_out_features)
    gathered = _gather_ranged_column_tensors(list(parts), qkv.shard_ranges, qkv.out_features)
    templates = QKV_SAVE_SPEC.keys
    if attention.attention_k_eq_v and attention.layer_type == "full_attention":
        templates = templates[:2]
        gathered = gathered[:2]
    for template, tensor in zip(templates, gathered, strict=True):
        tensors[key(template, prefix)] = tensor
    if qkv.bias is not None:
        bias_parts = split_local_tensors(qkv.bias, qkv.local_out_features)
        bias_gathered = _gather_ranged_column_tensors(list(bias_parts), qkv.shard_ranges, qkv.out_features)
        bias_templates = QKV_BIAS_SAVE_SPEC.keys
        if attention.attention_k_eq_v and attention.layer_type == "full_attention":
            bias_templates = bias_templates[:2]
            bias_gathered = bias_gathered[:2]
        for template, tensor in zip(bias_templates, bias_gathered, strict=True):
            tensors[key(template, prefix)] = tensor


def _get_existing_tensor(index, keys: tuple[str, ...]) -> torch.Tensor:
    for tensor_key in keys:
        if tensor_key in index.weight_map:
            return index.get_tensor(tensor_key)
    raise KeyError(f"missing any of: {', '.join(keys)}")


def _load_gemma4_moe(layer, index, prefix: str, rank: int, world_size: int) -> None:
    router = layer.router
    moe = layer.moe
    if router is None or moe is None:
        return
    router.scale.copy_(_get_existing_tensor(index, (f"{prefix}.router.scale", f"{prefix}.router.scale.weight")).to(dtype=router.scale.dtype))
    router.proj.weight.copy_(_get_existing_tensor(index, (f"{prefix}.router.proj.weight", f"{prefix}.router.weight")).to(dtype=router.proj.weight.dtype))
    scale_keys = (f"{prefix}.router.per_expert_scale", f"{prefix}.moe.per_expert_scale")
    if any(k in index.weight_map for k in scale_keys):
        moe.per_expert_scale.copy_(_get_existing_tensor(index, scale_keys).to(dtype=moe.per_expert_scale.dtype))
    _load_gemma4_experts(index, moe.experts, prefix, rank, world_size)


def _load_gemma4_experts(index, experts, prefix: str, rank: int, world_size: int) -> None:
    del rank, world_size
    gate_up_key = next((k for k in (f"{prefix}.experts.gate_up_proj", f"{prefix}.experts.gate_up_proj.weight") if k in index.weight_map), None)
    down_key = next((k for k in (f"{prefix}.experts.down_proj", f"{prefix}.experts.down_proj.weight") if k in index.weight_map), None)
    if gate_up_key is not None and down_key is not None:
        gate_up = index.get_tensor(gate_up_key)
        down = index.get_tensor(down_key)
        for expert_id in range(experts.local_expert_start, experts.local_expert_end):
            gate, up = gate_up[expert_id].chunk(2, dim=0)
            experts.copy_expert(expert_id, gate, up, down[expert_id], 0, 1)
        return
    for expert_id in range(experts.local_expert_start, experts.local_expert_end):
        experts.copy_expert(
            expert_id,
            index.get_tensor(f"{prefix}.experts.{expert_id}.gate_proj.weight"),
            index.get_tensor(f"{prefix}.experts.{expert_id}.up_proj.weight"),
            index.get_tensor(f"{prefix}.experts.{expert_id}.down_proj.weight"),
            0,
            1,
        )


def _save_gemma4_moe(tensors, prefix: str, layer) -> None:
    router = layer.router
    moe = layer.moe
    if router is None or moe is None:
        return
    tensors[f"{prefix}.router.scale"] = rank0_tensor(router.scale)
    tensors[f"{prefix}.router.proj.weight"] = rank0_tensor(router.proj.weight)
    tensors[f"{prefix}.router.per_expert_scale"] = rank0_tensor(moe.per_expert_scale)
    full_weights = gather_moe_expert_weights(moe.experts)
    gate_up_key = f"{prefix}.experts.gate_up_proj"
    down_key = f"{prefix}.experts.down_proj"
    if full_weights is None:
        tensors[gate_up_key] = None
        tensors[down_key] = None
        return
    gate_weights, up_weights, down_weights = full_weights
    gate_up = torch.cat((gate_weights, up_weights), dim=1).contiguous()
    tensors[gate_up_key] = _tensor_to_cpu(gate_up) if _owns_checkpoint_tensor(gate_up_key) else None
    tensors[down_key] = _tensor_to_cpu(down_weights.contiguous()) if _owns_checkpoint_tensor(down_key) else None


def checkpoint_spec(prefix: str) -> CheckpointSpec:
    """Assemble the load/save spec around a runtime-chosen HF prefix."""
    layer = LayerSpec(
        prefix=f"{prefix}.layers.{{layer}}",
        replicated=LAYER_NORM_SPECS,
        load_ops=LOAD_OPS,
        save_ops=SAVE_OPS,
        load_handlers=(load_gemma4_mlp,),
        save_handlers=(save_gemma4_attention, save_gemma4_mlp),
    )
    return CheckpointSpec(top_level=top_level_spec(prefix), layer=layer)
