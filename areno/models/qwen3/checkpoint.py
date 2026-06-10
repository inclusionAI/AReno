"""Qwen3 HF safetensors load/save specs.

Maps HF parameter names (``model.layers.{i}.self_attn.q_proj.weight`` etc.) to
areno's fused parallel layers (``self_attn.qkv_proj.weight`` for the merged
QKV projection, ``mlp.gate_up_proj.weight`` for the fused SwiGLU gate+up).
QKV and gate/up are stored split on disk and merged in memory; the save specs
reverse that by splitting the local column-parallel slabs back out per HF key.
"""

from __future__ import annotations

from areno.engine.checkpoints.common import (
    CheckpointSpec,
    DenseOrMoeSpec,
    LayerSpec,
    MergedColumnSpec,
    MoeSpec,
    OptionalMergedColumnSpec,
    OptionalRangedSplitColumnSpec,
    OptionalReplicatedTensorSpec,
    ParallelTensorSpec,
    RangedSplitColumnSpec,
    ReplicatedTensorSpec,
    SplitColumnSpec,
    TopLevelSpec,
)


# Vocab embedding lives directly under model.embed_tokens (no extra wrapper).
TOP_LEVEL_SPEC = TopLevelSpec(embedding_key="model.embed_tokens.weight", embedding_attr="embed_tokens")
# Per-layer RMSNorm weights are replicated across TP ranks.
LAYER_NORM_SPECS = (
    ReplicatedTensorSpec("{prefix}.input_layernorm.weight", "input_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.post_attention_layernorm.weight", "post_attention_layernorm.weight"),
)
# QKV is loaded by concatenating the three HF tensors along the output dim
# before sharding across TP, matching MergedColumnParallelLinear's layout.
QKV_WEIGHT_SPEC = MergedColumnSpec(
    dst_attr="self_attn.qkv_proj.weight",
    keys=(
        "{prefix}.self_attn.q_proj.weight",
        "{prefix}.self_attn.k_proj.weight",
        "{prefix}.self_attn.v_proj.weight",
    ),
)
# Bias only present when attention_bias=True in the HF config; treat as optional.
QKV_BIAS_SPEC = OptionalMergedColumnSpec(
    dst_attr="self_attn.qkv_proj.bias",
    keys=(
        "{prefix}.self_attn.q_proj.bias",
        "{prefix}.self_attn.k_proj.bias",
        "{prefix}.self_attn.v_proj.bias",
    ),
)
# Save side: reverse the merge using the per-output-section sizes stored on the
# linear layer so we recover the original q/k/v split widths.
QKV_SAVE_SPEC = RangedSplitColumnSpec(src_attr="self_attn.qkv_proj.weight", size_attr="self_attn.qkv_proj.local_out_features", keys=QKV_WEIGHT_SPEC.keys)
QKV_BIAS_SAVE_SPEC = OptionalRangedSplitColumnSpec(src_attr="self_attn.qkv_proj.bias", size_attr="self_attn.qkv_proj.local_out_features", keys=QKV_BIAS_SPEC.keys)
# Output projection is RowParallelLinear -> dim=1 sharded (input dim).
ATTN_ROW_SPEC = ParallelTensorSpec("{prefix}.self_attn.o_proj.weight", "self_attn.o_proj.weight", 1)
# Qwen3 adds optional per-head RMSNorm on Q and K (qk_layernorm=True).
Q_NORM_SPEC = OptionalReplicatedTensorSpec("{prefix}.self_attn.q_norm.weight", "self_attn.q_norm.weight")
K_NORM_SPEC = OptionalReplicatedTensorSpec("{prefix}.self_attn.k_norm.weight", "self_attn.k_norm.weight")
# Fused SwiGLU gate+up projection, identical pattern to QKV.
GATE_UP_WEIGHT_SPEC = MergedColumnSpec(
    dst_attr="mlp.gate_up_proj.weight",
    keys=("{prefix}.mlp.gate_proj.weight", "{prefix}.mlp.up_proj.weight"),
)
GATE_UP_SAVE_SPEC = SplitColumnSpec(src_attr="mlp.gate_up_proj.weight", size_attr="mlp.gate_up_proj.local_out_features", keys=GATE_UP_WEIGHT_SPEC.keys)
MLP_ROW_SPEC = ParallelTensorSpec("{prefix}.mlp.down_proj.weight", "mlp.down_proj.weight", 1)
MOE_SPEC = MoeSpec(
    gate_weight_key="{prefix}.gate.weight",
    gate_weight_attr="gate",
    expert_bias_key=None,
    expert_bias_attr=None,
    local_expert_bias_attr=None,
    experts_attr="experts",
    num_experts_attr="num_experts",
    expert_gate_key="{prefix}.experts.{expert}.gate_proj.weight",
    expert_up_key="{prefix}.experts.{expert}.up_proj.weight",
    expert_down_key="{prefix}.experts.{expert}.down_proj.weight",
    shared_experts_attr=None,
    shared_experts_prefix=None,
)
MOE_MLP_SPEC = DenseOrMoeSpec(attr="mlp", moe=MOE_SPEC)


QWEN_LAYER_SPEC = LayerSpec(
    prefix="model.layers.{layer}",
    replicated=LAYER_NORM_SPECS,
    load_ops=(
        QKV_WEIGHT_SPEC,
        QKV_BIAS_SPEC,
        Q_NORM_SPEC,
        K_NORM_SPEC,
        ATTN_ROW_SPEC,
        GATE_UP_WEIGHT_SPEC,
        MLP_ROW_SPEC,
    ),
    save_ops=(
        QKV_SAVE_SPEC,
        QKV_BIAS_SAVE_SPEC,
        Q_NORM_SPEC,
        K_NORM_SPEC,
        ATTN_ROW_SPEC,
        GATE_UP_SAVE_SPEC,
        MLP_ROW_SPEC,
    ),
)
CHECKPOINT_SPEC = CheckpointSpec(top_level=TOP_LEVEL_SPEC, layer=QWEN_LAYER_SPEC)
QWEN3_MOE_LAYER_SPEC = LayerSpec(
    prefix="model.layers.{layer}",
    replicated=LAYER_NORM_SPECS,
    load_ops=(
        QKV_WEIGHT_SPEC,
        QKV_BIAS_SPEC,
        Q_NORM_SPEC,
        K_NORM_SPEC,
        ATTN_ROW_SPEC,
        MOE_MLP_SPEC,
    ),
    save_ops=(
        QKV_SAVE_SPEC,
        QKV_BIAS_SAVE_SPEC,
        Q_NORM_SPEC,
        K_NORM_SPEC,
        ATTN_ROW_SPEC,
        MOE_MLP_SPEC,
    ),
)
QWEN3_MOE_CHECKPOINT_SPEC = CheckpointSpec(top_level=TOP_LEVEL_SPEC, layer=QWEN3_MOE_LAYER_SPEC)
