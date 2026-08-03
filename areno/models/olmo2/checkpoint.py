"""HF safetensors load/save spec for OLMo 2 checkpoints."""

from __future__ import annotations

from areno.engine.checkpoints.common import (
    CheckpointSpec,
    LayerSpec,
    MergedColumnSpec,
    ParallelTensorSpec,
    RangedSplitColumnSpec,
    ReplicatedTensorSpec,
    SplitColumnSpec,
    TopLevelSpec,
)

TOP_LEVEL_SPEC = TopLevelSpec(embedding_key="model.embed_tokens.weight", embedding_attr="embed_tokens")
LAYER_NORM_SPECS = (
    ReplicatedTensorSpec("{prefix}.post_attention_layernorm.weight", "post_attention_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.post_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"),
)
QKV_WEIGHT_SPEC = MergedColumnSpec(
    dst_attr="self_attn.qkv_proj.weight",
    keys=(
        "{prefix}.self_attn.q_proj.weight",
        "{prefix}.self_attn.k_proj.weight",
        "{prefix}.self_attn.v_proj.weight",
    ),
)
QKV_SAVE_SPEC = RangedSplitColumnSpec(
    src_attr="self_attn.qkv_proj.weight",
    size_attr="self_attn.qkv_proj.local_out_features",
    keys=QKV_WEIGHT_SPEC.keys,
)
Q_NORM_SPEC = ParallelTensorSpec("{prefix}.self_attn.q_norm.weight", "self_attn.q_norm.weight", 0)
K_NORM_SPEC = ParallelTensorSpec("{prefix}.self_attn.k_norm.weight", "self_attn.k_norm.weight", 0)
ATTN_ROW_SPEC = ParallelTensorSpec("{prefix}.self_attn.o_proj.weight", "self_attn.o_proj.weight", 1)
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

OLMO2_LAYER_SPEC = LayerSpec(
    prefix="model.layers.{layer}",
    replicated=LAYER_NORM_SPECS,
    load_ops=(QKV_WEIGHT_SPEC, Q_NORM_SPEC, K_NORM_SPEC, ATTN_ROW_SPEC, GATE_UP_WEIGHT_SPEC, MLP_ROW_SPEC),
    save_ops=(QKV_SAVE_SPEC, Q_NORM_SPEC, K_NORM_SPEC, ATTN_ROW_SPEC, GATE_UP_SAVE_SPEC, MLP_ROW_SPEC),
)
CHECKPOINT_SPEC = CheckpointSpec(top_level=TOP_LEVEL_SPEC, layer=OLMO2_LAYER_SPEC)
