"""Bailing-MoE V3 HF safetensors load/save specs.

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

import torch

from areno.engine.checkpoints.common import (
    CheckpointSpec,
    DenseOrMoeSpec,
    LayerSpec,
    MoeSpec,
    ReplicatedTensorSpec,
    TopLevelSpec,
)
from areno.engine.checkpoints.io import (
    _copy_column,
    _copy_row,
    gather_tensor_parallel_tensor,
    rank0_tensor,
)
from areno.engine.layers.linear import _shard_range

# Bailing wraps the embedding under ``model.word_embeddings`` (not the more
# common ``model.embed_tokens``). V3 checkpoints can ship an untied
# ``lm_head.weight``; ``load_embedding_norm_head`` copies it only when
# ``tie_word_embeddings`` is false.
TOP_LEVEL_SPEC = TopLevelSpec(
    embedding_key="model.word_embeddings.weight",
    embedding_attr="word_embeddings",
    lm_head_key="lm_head.weight",
)
# Pre-attn and pre-MLP RMSNorm scales — replicated across TP ranks.
LAYER_NORM_SPECS = (
    ReplicatedTensorSpec("{prefix}.input_layernorm.weight", "input_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.post_attention_layernorm.weight", "post_attention_layernorm.weight"),
)
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
# DenseOrMoeSpec picks ``moe`` for MoE layers and falls back to a dense MLP
# for layers strictly below ``first_k_dense_replace`` (handled in the loader).
MLP_SPEC = DenseOrMoeSpec(attr="mlp", moe=MOE_SPEC)


@torch.no_grad()
def load_bailing_v3_attention(module, index, prefix: str, rank: int, world_size: int) -> None:
    attn = module.attention
    attn_prefix = f"{prefix}.attention"
    if hasattr(attn, "q_conv1d_weight"):
        _load_kda_attention(attn, index, attn_prefix, rank, world_size)
        return
    _load_mla_attention(attn, index, attn_prefix, rank, world_size)


@torch.no_grad()
def _load_kda_attention(attn, index, prefix: str, rank: int, world_size: int) -> None:
    for name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "b_proj"):
        _copy_column(getattr(attn, name).weight, index.get_tensor(f"{prefix}.{name}.weight"), rank, world_size)
    for name in ("q", "k", "v"):
        _copy_column(
            getattr(attn, f"{name}_conv1d_weight"),
            index.get_tensor(f"{prefix}.{name}_conv1d.weight"),
            rank,
            world_size,
        )
    start, end = _shard_range(attn.proj_dim, rank, world_size)
    attn.dt_bias.copy_(index.get_tensor(f"{prefix}.dt_bias")[start:end].to(dtype=attn.dt_bias.dtype))
    start, end = _shard_range(attn.num_heads, rank, world_size)
    attn.A_log.copy_(index.get_tensor(f"{prefix}.A_log")[start:end].to(dtype=attn.A_log.dtype))
    attn.o_norm_weight.copy_(index.get_tensor(f"{prefix}.o_norm.weight").to(dtype=attn.o_norm_weight.dtype))
    _copy_row(attn.o_proj.weight, index.get_tensor(f"{prefix}.o_proj.weight"), rank, world_size)


@torch.no_grad()
def _load_mla_attention(attn, index, prefix: str, rank: int, world_size: int) -> None:
    if getattr(attn, "q_lora_rank", None) is None:
        _copy_column(attn.q_proj.weight, index.get_tensor(f"{prefix}.q_proj.weight"), rank, world_size)
    else:
        attn.q_a_proj.weight.copy_(index.get_tensor(f"{prefix}.q_a_proj.weight").to(dtype=attn.q_a_proj.weight.dtype))
        attn.q_a_layernorm.weight.copy_(
            index.get_tensor(f"{prefix}.q_a_layernorm.weight").to(dtype=attn.q_a_layernorm.weight.dtype)
        )
        _copy_column(attn.q_b_proj.weight, index.get_tensor(f"{prefix}.q_b_proj.weight"), rank, world_size)
    attn.kv_a_proj_with_mqa.weight.copy_(
        index.get_tensor(f"{prefix}.kv_a_proj_with_mqa.weight").to(dtype=attn.kv_a_proj_with_mqa.weight.dtype)
    )
    attn.kv_a_layernorm.weight.copy_(
        index.get_tensor(f"{prefix}.kv_a_layernorm.weight").to(dtype=attn.kv_a_layernorm.weight.dtype)
    )
    _copy_column(attn.kv_b_proj.weight, index.get_tensor(f"{prefix}.kv_b_proj.weight"), rank, world_size)
    if getattr(attn, "g_proj", None) is not None:
        _copy_column(attn.g_proj.weight, index.get_tensor(f"{prefix}.g_proj.weight"), rank, world_size)
    _copy_row(attn.dense.weight, index.get_tensor(f"{prefix}.dense.weight"), rank, world_size)


@torch.no_grad()
def save_bailing_v3_attention(tensors: dict[str, torch.Tensor | None], prefix: str, module, context) -> None:
    del context
    attn = module.attention
    attn_prefix = f"{prefix}.attention"
    if hasattr(attn, "q_conv1d_weight"):
        _save_kda_attention(tensors, attn_prefix, attn)
        return
    _save_mla_attention(tensors, attn_prefix, attn)


def _save_kda_attention(tensors: dict[str, torch.Tensor | None], prefix: str, attn) -> None:
    for name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "b_proj"):
        tensors[f"{prefix}.{name}.weight"] = gather_tensor_parallel_tensor(getattr(attn, name).weight, dim=0)
    for name in ("q", "k", "v"):
        tensors[f"{prefix}.{name}_conv1d.weight"] = gather_tensor_parallel_tensor(
            getattr(attn, f"{name}_conv1d_weight"), dim=0
        )
    tensors[f"{prefix}.dt_bias"] = gather_tensor_parallel_tensor(attn.dt_bias, dim=0)
    tensors[f"{prefix}.A_log"] = gather_tensor_parallel_tensor(attn.A_log, dim=0)
    tensors[f"{prefix}.o_norm.weight"] = rank0_tensor(attn.o_norm_weight)
    tensors[f"{prefix}.o_proj.weight"] = gather_tensor_parallel_tensor(attn.o_proj.weight, dim=1)


def _save_mla_attention(tensors: dict[str, torch.Tensor | None], prefix: str, attn) -> None:
    if getattr(attn, "q_lora_rank", None) is None:
        tensors[f"{prefix}.q_proj.weight"] = gather_tensor_parallel_tensor(attn.q_proj.weight, dim=0)
    else:
        tensors[f"{prefix}.q_a_proj.weight"] = rank0_tensor(attn.q_a_proj.weight)
        tensors[f"{prefix}.q_a_layernorm.weight"] = rank0_tensor(attn.q_a_layernorm.weight)
        tensors[f"{prefix}.q_b_proj.weight"] = gather_tensor_parallel_tensor(attn.q_b_proj.weight, dim=0)
    tensors[f"{prefix}.kv_a_proj_with_mqa.weight"] = rank0_tensor(attn.kv_a_proj_with_mqa.weight)
    tensors[f"{prefix}.kv_a_layernorm.weight"] = rank0_tensor(attn.kv_a_layernorm.weight)
    tensors[f"{prefix}.kv_b_proj.weight"] = gather_tensor_parallel_tensor(attn.kv_b_proj.weight, dim=0)
    if getattr(attn, "g_proj", None) is not None:
        tensors[f"{prefix}.g_proj.weight"] = gather_tensor_parallel_tensor(attn.g_proj.weight, dim=0)
    tensors[f"{prefix}.dense.weight"] = gather_tensor_parallel_tensor(attn.dense.weight, dim=1)


BAILING_LAYER_SPEC = LayerSpec(
    prefix="model.layers.{layer}",
    replicated=LAYER_NORM_SPECS,
    load_ops=(MLP_SPEC,),
    save_ops=(MLP_SPEC,),
    load_handlers=(load_bailing_v3_attention,),
    save_handlers=(save_bailing_v3_attention,),
    prefetch_layer=True,
)
CHECKPOINT_SPEC = CheckpointSpec(top_level=TOP_LEVEL_SPEC, layer=BAILING_LAYER_SPEC)
