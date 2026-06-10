from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from areno.engine.checkpoints.common import (
    CheckpointTensorStore,
    MergedColumnSpec,
    ParallelTensorSpec,
    ReplicatedTensorSpec,
    TopLevelSpec,
    copy_source_passthrough_weights,
    copy_merged_column,
    gather_tensor_parallel_column_tensors,
    gather_tensor_parallel_split_column_tensor,
    gather_tensor_parallel_tensor,
    load_embedding_norm_head,
    rank0_tensor,
    save_embedding_norm_head,
    write_hf_safetensors_checkpoint,
)
from areno.engine.checkpoints.io import SafetensorsIndex, _CheckpointTensorTask, _copy_row, _sync_pending_cpu_copies
from areno.engine.layers.linear import _shard_range
from areno.engine.parallel.context import get_tp_context
from areno.models.minicpmv46.model import (
    _LINEAR_HEAD_DIM,
    _LINEAR_NUM_HEADS,
    MiniCPMDecoderLayer,
    MiniCPMFullAttention,
    MiniCPMGatedDeltaNet,
    MiniCPMV46ForCausalLM,
)


TOP_LEVEL_SPEC = TopLevelSpec(
    embedding_key="model.language_model.embed_tokens.weight",
    embedding_attr="embed_tokens",
    norm_key="model.language_model.norm.weight",
    norm_attr="norm.weight",
)
LAYER_NORM_SPECS = (
    ReplicatedTensorSpec("{prefix}.input_layernorm.weight", "input_layernorm.weight"),
    ReplicatedTensorSpec("{prefix}.post_attention_layernorm.weight", "post_attention_layernorm.weight"),
)
GATE_UP_WEIGHT_SPEC = MergedColumnSpec(
    dst_attr="mlp.gate_up_proj.weight",
    keys=("{prefix}.mlp.gate_proj.weight", "{prefix}.mlp.up_proj.weight"),
)
MLP_ROW_SPEC = ParallelTensorSpec("{prefix}.mlp.down_proj.weight", "mlp.down_proj.weight", 1)
FULL_ATTN_KEYS = (
    "{prefix}.self_attn.q_proj.weight",
    "{prefix}.self_attn.k_proj.weight",
    "{prefix}.self_attn.v_proj.weight",
    "{prefix}.self_attn.o_proj.weight",
    "{prefix}.self_attn.q_norm.weight",
    "{prefix}.self_attn.k_norm.weight",
)
GDN_KEYS = (
    "{prefix}.linear_attn.in_proj_qkv.weight",
    "{prefix}.linear_attn.in_proj_z.weight",
    "{prefix}.linear_attn.in_proj_b.weight",
    "{prefix}.linear_attn.in_proj_a.weight",
    "{prefix}.linear_attn.conv1d.weight",
    "{prefix}.linear_attn.dt_bias",
    "{prefix}.linear_attn.A_log",
    "{prefix}.linear_attn.norm.weight",
    "{prefix}.linear_attn.out_proj.weight",
)


@torch.no_grad()
def load_minicpmv46_weights(model: MiniCPMV46ForCausalLM, model_path: str | Path) -> None:
    """Load MiniCPM-V-4.6 text weights from the HF safetensors checkpoint.

    The HF checkpoint wraps the language model under `model.language_model`.
    Full-attention q projection stores q and output-gate rows in one tensor,
    while GDN layers store q/k/v, z, b/a, and conv tensors separately. These
    layouts do not fit the common QKV spec directly, so the sharding rules live
    here instead of in the model definition.
    """

    ctx = get_tp_context()
    index = SafetensorsIndex(model_path)
    try:
        prefix = model.config.checkpoint_prefix
        index.set_progress_total(1 + len(model.layers), unit="stage", manual=True)
        _load_embedding_norm_head(model, index, ctx.rank, ctx.world_size)
        index.advance_progress()
        for layer_idx, layer in enumerate(model.layers):
            _load_layer(index, layer, f"{prefix}.layers.{layer_idx}", ctx.rank, ctx.world_size)
            index.advance_progress()
    finally:
        index.close()


@torch.no_grad()
def save_minicpmv46_weights(model: MiniCPMV46ForCausalLM, output_path: str | Path, source_path: str | Path | None) -> str | None:
    """Save MiniCPM-V-4.6 text weights back to HF safetensors layout."""

    tensors = CheckpointTensorStore()
    _save_embedding_norm_head(tensors, model)
    prefix = model.config.checkpoint_prefix
    for layer_idx, layer in enumerate(model.layers):
        _save_layer(tensors, layer, f"{prefix}.layers.{layer_idx}")
    saved_path = write_hf_safetensors_checkpoint(tensors, output_path, source_path)
    if saved_path is not None and source_path is not None:
        copy_source_passthrough_weights(source_path, saved_path, protected_prefix=f"{prefix}.")
    return saved_path


def _load_embedding_norm_head(model: MiniCPMV46ForCausalLM, index: SafetensorsIndex, rank: int, world_size: int) -> None:
    load_embedding_norm_head(model, index, TOP_LEVEL_SPEC, rank, world_size)
    model.norm.weight.add_(1.0)


def _save_embedding_norm_head(tensors: dict[str, torch.Tensor | None], model: MiniCPMV46ForCausalLM) -> None:
    save_embedding_norm_head(tensors, model, TOP_LEVEL_SPEC)
    tensors[TOP_LEVEL_SPEC.norm_key] = rank0_tensor(model.norm.weight - 1.0)


def _load_layer(index: SafetensorsIndex, layer: MiniCPMDecoderLayer, prefix: str, rank: int, world_size: int) -> None:
    keys = [
        *[spec.key.format(prefix=prefix) for spec in LAYER_NORM_SPECS],
        *[key.format(prefix=prefix) for key in GATE_UP_WEIGHT_SPEC.keys],
        MLP_ROW_SPEC.key.format(prefix=prefix),
    ]
    keys.extend(_attention_keys(layer.attention, prefix))
    index.prefetch([key for key in keys if key in index.weight_map])

    for spec in LAYER_NORM_SPECS:
        dst = _attr_path(layer, spec.attr)
        dst.copy_((index.get_tensor(spec.key.format(prefix=prefix)) + 1.0).to(dtype=dst.dtype))
    copy_merged_column(
        _attr_path(layer, GATE_UP_WEIGHT_SPEC.dst_attr),
        [index.get_tensor(key.format(prefix=prefix)) for key in GATE_UP_WEIGHT_SPEC.keys],
        rank,
        world_size,
    )
    _copy_row(_attr_path(layer, MLP_ROW_SPEC.attr), index.get_tensor(MLP_ROW_SPEC.key.format(prefix=prefix)), rank, world_size)

    if isinstance(layer.attention, MiniCPMFullAttention):
        _load_full_attention(index, layer.attention, prefix, rank, world_size)
        return
    if isinstance(layer.attention, MiniCPMGatedDeltaNet):
        _load_gated_delta_net(index, layer.attention, prefix, rank, world_size)
        return
    raise TypeError(f"unsupported MiniCPM attention module {type(layer.attention)!r}")


def _save_layer(tensors: dict[str, torch.Tensor | None], layer: MiniCPMDecoderLayer, prefix: str) -> None:
    for spec in LAYER_NORM_SPECS:
        tensors[spec.key.format(prefix=prefix)] = rank0_tensor(_attr_path(layer, spec.attr) - 1.0)
    gate_weight, up_weight = gather_tensor_parallel_column_tensors(list(_attr_path(layer, GATE_UP_WEIGHT_SPEC.dst_attr).split(layer.mlp.gate_up_proj.local_out_features, dim=0)))
    gate_key, up_key = [key.format(prefix=prefix) for key in GATE_UP_WEIGHT_SPEC.keys]
    tensors[gate_key] = gate_weight
    tensors[up_key] = up_weight
    tensors[MLP_ROW_SPEC.key.format(prefix=prefix)] = gather_tensor_parallel_tensor(_attr_path(layer, MLP_ROW_SPEC.attr), dim=1)

    if isinstance(layer.attention, MiniCPMFullAttention):
        _save_full_attention(tensors, layer.attention, prefix)
        return
    if isinstance(layer.attention, MiniCPMGatedDeltaNet):
        _save_gated_delta_net(tensors, layer.attention, prefix)
        return
    raise TypeError(f"unsupported MiniCPM attention module {type(layer.attention)!r}")


def _attention_keys(attn: nn.Module, prefix: str) -> list[str]:
    if isinstance(attn, MiniCPMFullAttention):
        return [key.format(prefix=prefix) for key in FULL_ATTN_KEYS]
    return [key.format(prefix=prefix) for key in GDN_KEYS]


def _load_full_attention(index: SafetensorsIndex, attn: MiniCPMFullAttention, prefix: str, rank: int, world_size: int) -> None:
    q_proj = index.get_tensor(f"{prefix}.self_attn.q_proj.weight")
    q_weight, gate_weight = _split_q_gate_by_head(q_proj, attn.num_heads, attn.head_dim)
    copy_merged_column(
        attn.qkv_proj.weight,
        [
            q_weight,
            gate_weight,
            index.get_tensor(f"{prefix}.self_attn.k_proj.weight"),
            index.get_tensor(f"{prefix}.self_attn.v_proj.weight"),
        ],
        rank,
        world_size,
    )
    _copy_row(attn.o_proj.weight, index.get_tensor(f"{prefix}.self_attn.o_proj.weight"), rank, world_size)
    attn.q_norm.weight.copy_((index.get_tensor(f"{prefix}.self_attn.q_norm.weight") + 1.0).to(dtype=attn.q_norm.weight.dtype))
    attn.k_norm.weight.copy_((index.get_tensor(f"{prefix}.self_attn.k_norm.weight") + 1.0).to(dtype=attn.k_norm.weight.dtype))


def _load_gated_delta_net(index: SafetensorsIndex, attn: MiniCPMGatedDeltaNet, prefix: str, rank: int, world_size: int) -> None:
    qkv = index.get_tensor(f"{prefix}.linear_attn.in_proj_qkv.weight")
    q, k, v = qkv.split((_LINEAR_NUM_HEADS * _LINEAR_HEAD_DIM, _LINEAR_NUM_HEADS * _LINEAR_HEAD_DIM, _LINEAR_NUM_HEADS * _LINEAR_HEAD_DIM), dim=0)
    copy_merged_column(attn.in_proj_qkvz.weight, [q, k, v, index.get_tensor(f"{prefix}.linear_attn.in_proj_z.weight")], rank, world_size)
    copy_merged_column(attn.in_proj_ba.weight, [index.get_tensor(f"{prefix}.linear_attn.in_proj_b.weight"), index.get_tensor(f"{prefix}.linear_attn.in_proj_a.weight")], rank, world_size)
    conv_qkv = index.get_tensor(f"{prefix}.linear_attn.conv1d.weight")
    conv_parts = conv_qkv.split((_LINEAR_NUM_HEADS * _LINEAR_HEAD_DIM, _LINEAR_NUM_HEADS * _LINEAR_HEAD_DIM, _LINEAR_NUM_HEADS * _LINEAR_HEAD_DIM), dim=0)
    copy_merged_column(attn.conv1d_weight, list(conv_parts), rank, world_size)
    start, end = _shard_range(_LINEAR_NUM_HEADS, rank, world_size)
    attn.dt_bias.copy_(index.get_tensor(f"{prefix}.linear_attn.dt_bias")[start:end].to(dtype=attn.dt_bias.dtype))
    attn.A_log.copy_(index.get_tensor(f"{prefix}.linear_attn.A_log")[start:end].to(dtype=attn.A_log.dtype))
    attn.norm_weight.copy_(index.get_tensor(f"{prefix}.linear_attn.norm.weight").to(dtype=attn.norm_weight.dtype))
    _copy_row(attn.out_proj.weight, index.get_tensor(f"{prefix}.linear_attn.out_proj.weight"), rank, world_size)


def _save_full_attention(tensors: dict[str, torch.Tensor | None], attn: MiniCPMFullAttention, prefix: str) -> None:
    q_size, gate_size, k_size, v_size = attn.qkv_proj.local_out_features
    q = attn.qkv_proj.weight[:q_size]
    gate = attn.qkv_proj.weight[q_size : q_size + gate_size]
    k = attn.qkv_proj.weight[q_size + gate_size : q_size + gate_size + k_size]
    v = attn.qkv_proj.weight[q_size + gate_size + k_size : q_size + gate_size + k_size + v_size]
    q_weight, gate_weight = gather_tensor_parallel_column_tensors([q, gate])
    tensors[f"{prefix}.self_attn.q_proj.weight"] = _merge_q_gate_by_head(q_weight, gate_weight, attn.num_heads, attn.head_dim)
    k_weight, v_weight = gather_tensor_parallel_column_tensors([k, v])
    tensors[f"{prefix}.self_attn.k_proj.weight"] = k_weight
    tensors[f"{prefix}.self_attn.v_proj.weight"] = v_weight
    tensors[f"{prefix}.self_attn.o_proj.weight"] = gather_tensor_parallel_tensor(attn.o_proj.weight, dim=1)
    tensors[f"{prefix}.self_attn.q_norm.weight"] = rank0_tensor(attn.q_norm.weight - 1.0)
    tensors[f"{prefix}.self_attn.k_norm.weight"] = rank0_tensor(attn.k_norm.weight - 1.0)


def _save_gated_delta_net(tensors: dict[str, torch.Tensor | None], attn: MiniCPMGatedDeltaNet, prefix: str) -> None:
    q_size, k_size, v_size, z_size = attn.in_proj_qkvz.local_out_features
    qkv = attn.in_proj_qkvz.weight[: q_size + k_size + v_size]
    z = attn.in_proj_qkvz.weight[q_size + k_size + v_size : q_size + k_size + v_size + z_size]
    tensors[f"{prefix}.linear_attn.in_proj_qkv.weight"] = gather_tensor_parallel_split_column_tensor(qkv, [q_size, k_size, v_size])
    (z_weight,) = gather_tensor_parallel_column_tensors([z])
    tensors[f"{prefix}.linear_attn.in_proj_z.weight"] = z_weight

    b_size, a_size = attn.in_proj_ba.local_out_features
    b = attn.in_proj_ba.weight[:b_size]
    a = attn.in_proj_ba.weight[b_size : b_size + a_size]
    b_weight, a_weight = gather_tensor_parallel_column_tensors([b, a])
    tensors[f"{prefix}.linear_attn.in_proj_b.weight"] = b_weight
    tensors[f"{prefix}.linear_attn.in_proj_a.weight"] = a_weight

    local_conv_qkv = attn.conv1d_weight
    conv_q = attn.local_key_dim
    conv_k = attn.local_key_dim
    conv_v = attn.local_value_dim
    tensors[f"{prefix}.linear_attn.conv1d.weight"] = gather_tensor_parallel_split_column_tensor(local_conv_qkv, [conv_q, conv_k, conv_v])
    tensors[f"{prefix}.linear_attn.dt_bias"] = gather_tensor_parallel_tensor(attn.dt_bias, dim=0)
    tensors[f"{prefix}.linear_attn.A_log"] = gather_tensor_parallel_tensor(attn.A_log, dim=0)
    tensors[f"{prefix}.linear_attn.norm.weight"] = rank0_tensor(attn.norm_weight)
    tensors[f"{prefix}.linear_attn.out_proj.weight"] = gather_tensor_parallel_tensor(attn.out_proj.weight, dim=1)


def _attr_path(obj: object, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _split_q_gate_by_head(q_proj: torch.Tensor, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_gate = q_proj.reshape(num_heads, 2, head_dim, q_proj.shape[1])
    return q_gate[:, 0].reshape(num_heads * head_dim, q_proj.shape[1]), q_gate[:, 1].reshape(num_heads * head_dim, q_proj.shape[1])


class _MergedQGateByHeadTask(_CheckpointTensorTask):
    def __init__(self, q: _CheckpointTensorTask, gate: _CheckpointTensorTask, num_heads: int, head_dim: int):
        self.q = q
        self.gate = gate
        self.num_heads = num_heads
        self.head_dim = head_dim

    def materialize(self, key: str) -> torch.Tensor | None:
        q = self.q.materialize(key)
        gate = self.gate.materialize(key)
        if q is None or gate is None:
            return None
        _sync_pending_cpu_copies()
        return _merge_q_gate_tensor_by_head(q, gate, self.num_heads, self.head_dim)


def _merge_q_gate_by_head(
    q: torch.Tensor | _CheckpointTensorTask | None,
    gate: torch.Tensor | _CheckpointTensorTask | None,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor | _CheckpointTensorTask | None:
    if q is None or gate is None:
        return None
    if isinstance(q, _CheckpointTensorTask) and isinstance(gate, _CheckpointTensorTask):
        return _MergedQGateByHeadTask(q, gate, num_heads, head_dim)
    if isinstance(q, _CheckpointTensorTask) or isinstance(gate, _CheckpointTensorTask):
        raise TypeError("q and gate checkpoint gather values must both be lazy tasks or both be tensors")
    return _merge_q_gate_tensor_by_head(q, gate, num_heads, head_dim)


def _merge_q_gate_tensor_by_head(q: torch.Tensor, gate: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    hidden = q.shape[1]
    q_by_head = q.reshape(num_heads, head_dim, hidden)
    gate_by_head = gate.reshape(num_heads, head_dim, hidden)
    return torch.stack((q_by_head, gate_by_head), dim=1).reshape(num_heads * 2 * head_dim, hidden)
