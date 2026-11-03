"""GPT-OSS checkpoint layouts, including expert-axis sharding and MXFP4."""

from areno.engine.checkpoints.common import (
    CheckpointSpec,
    LayerSpec,
    MergedColumnSpec,
    PackedExpertTensorSpec,
    ParallelTensorSpec,
    RangedSplitColumnSpec,
    ReplicatedTensorSpec,
    TopLevelSpec,
)

TOP = TopLevelSpec("model.embed_tokens.weight", "embed_tokens")
NORMS = tuple(
    ReplicatedTensorSpec("{prefix}." + name + ".weight", name + ".weight")
    for name in ("input_layernorm", "post_attention_layernorm")
)
QKV = tuple(
    MergedColumnSpec(
        "self_attn.qkv_proj." + suffix,
        tuple("{prefix}.self_attn." + proj + "." + suffix for proj in ("q_proj", "k_proj", "v_proj")),
    )
    for suffix in ("weight", "bias")
)
QKV_SAVE = tuple(RangedSplitColumnSpec(op.dst_attr, "self_attn.qkv_proj.local_out_features", op.keys) for op in QKV)
SHARDS = (
    ParallelTensorSpec("{prefix}.self_attn.o_proj.weight", "self_attn.o_proj.weight", 1),
    ParallelTensorSpec("{prefix}.self_attn.sinks", "self_attn.sinks", 0),
    ParallelTensorSpec("{prefix}.mlp.experts.gate_up_proj_bias", "mlp.experts.gate_up_proj_bias", 0),
    ParallelTensorSpec("{prefix}.mlp.experts.down_proj_bias", "mlp.experts.down_proj_bias", 0),
)
REPLICAS = tuple(
    ReplicatedTensorSpec("{prefix}." + name, name)
    for name in ("self_attn.o_proj.bias", "mlp.router.weight", "mlp.router.bias")
)
EXPERTS = tuple(
    PackedExpertTensorSpec("{prefix}.mlp.experts." + name, "mlp.experts." + name)
    for name in ("gate_up_proj", "down_proj")
)
SPEC = CheckpointSpec(
    TOP,
    LayerSpec(
        "model.layers.{layer}",
        NORMS,
        load_ops=QKV + SHARDS + REPLICAS + EXPERTS,
        save_ops=QKV_SAVE + SHARDS + REPLICAS + EXPERTS,
    ),
)
