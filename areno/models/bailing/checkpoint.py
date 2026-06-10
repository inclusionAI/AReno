"""Bailing-MoE-Linear-V2 HF safetensors load/save specs.

Bailing's HF checkpoint stores each routed expert as a standalone
``experts.{i}.gate_proj`` / ``up_proj`` / ``down_proj`` triple plus a router
``gate.weight`` and a (sigmoid-biased grouped) router expert-bias buffer.
``DenseOrMoeSpec`` defers to ``MoeSpec`` for the MoE layers and falls back to a
dense gate/up/down loader on the leading ``first_k_dense_replace`` layers.

Sharding-wise the experts use expert-parallelism that piggy-backs on the TP
process group: each rank owns ``num_experts / world_size`` consecutive experts
and the loader copies only those expert tensors into ``BailingGroupedExperts``'
fused 3D weight buffers. Both attention pathways (softmax MLA and linear
attention) share the same ``self_attn``/``attention`` HF prefix alias and the
same ``dense.weight`` row-parallel output projection, so a single
``AttentionSpec`` covers both.
"""

from __future__ import annotations

from areno.engine.checkpoints.common import (
    AttentionSpec,
    CheckpointSpec,
    DenseOrMoeSpec,
    LayerSpec,
    MoeSpec,
    ParallelTensorSpec,
    ReplicatedTensorSpec,
    TopLevelSpec,
)


# Bailing wraps the embedding under ``model.word_embeddings`` (not the more
# common ``model.embed_tokens``); the LM head is tied by default in the HF
# checkpoint and so is omitted from this spec.
TOP_LEVEL_SPEC = TopLevelSpec(embedding_key="model.word_embeddings.weight", embedding_attr="word_embeddings")
# Pre-attn and pre-MLP RMSNorm scales — replicated across TP ranks.
LAYER_NORM_SPECS = (
    ReplicatedTensorSpec("{prefix}.input_layernorm.weight", "input_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.post_attention_layernorm.weight", "post_attention_layernorm.weight"),
)
# Per-head Q/K RMSNorm scales (replicated; same dim on every rank).
ATTN_QK_NORM_SPECS = (
    ReplicatedTensorSpec("{prefix}.query_layernorm.weight", "query_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.key_layernorm.weight", "key_layernorm.weight"),
)
# Output projection is RowParallelLinear -> shard the input dim (dim=1).
ATTN_DENSE_ROW_SPEC = ParallelTensorSpec("{prefix}.dense.weight", "dense.weight", 1)
ATTN_DENSE_BIAS_SPEC = ReplicatedTensorSpec("{prefix}.dense.bias", "dense.bias")
# MoE block: a sigmoid-scored router with per-expert bias plus per-expert gate
# / up / down projections. ``local_expert_bias`` is the rank-local copy used by
# the biased grouped top-k kernel; ``num_experts_attr`` lets the loader build
# the right number of expert keys (``experts.{expert}.*``) per layer.
MOE_SPEC = MoeSpec(
    gate_weight_key="{prefix}.gate.weight",
    gate_weight_attr="gate.weight",
    expert_bias_key="{prefix}.gate.expert_bias",
    expert_bias_attr="gate.expert_bias",
    local_expert_bias_attr="gate.local_expert_bias",
    experts_attr="experts",
    num_experts_attr="num_experts",
    expert_gate_key="{prefix}.experts.{expert}.gate_proj.weight",
    expert_up_key="{prefix}.experts.{expert}.up_proj.weight",
    expert_down_key="{prefix}.experts.{expert}.down_proj.weight",
    # Bailing also keeps a small dense "shared expert" MLP that runs on every
    # token regardless of routing; it lives under the same layer prefix.
    shared_experts_attr="shared_experts",
    shared_experts_prefix="{prefix}.shared_experts",
)
# Bailing's HF checkpoint uses both ``attention`` and ``self_attn`` as the
# attention prefix depending on variant — list both for robust matching.
ATTENTION_SPEC = AttentionSpec(
    attr="attention",
    checkpoint_names=("attention", "self_attn"),
    qk_norms=ATTN_QK_NORM_SPECS,
    dense_rows=(ATTN_DENSE_ROW_SPEC,),
    dense_bias=ATTN_DENSE_BIAS_SPEC,
    # Linear-attention variant reuses the same dense output projection layout.
    linear_dense_rows=(ATTN_DENSE_ROW_SPEC,),
    linear_dense_bias=ATTN_DENSE_BIAS_SPEC,
)
# DenseOrMoeSpec picks ``moe`` for MoE layers and falls back to a dense MLP
# for layers strictly below ``first_k_dense_replace`` (handled in the loader).
MLP_SPEC = DenseOrMoeSpec(attr="mlp", moe=MOE_SPEC)


BAILING_LAYER_SPEC = LayerSpec(
    prefix="model.layers.{layer}",
    replicated=LAYER_NORM_SPECS,
    load_ops=(ATTENTION_SPEC, MLP_SPEC),
    save_ops=(ATTENTION_SPEC, MLP_SPEC),
)
CHECKPOINT_SPEC = CheckpointSpec(top_level=TOP_LEVEL_SPEC, layer=BAILING_LAYER_SPEC)
