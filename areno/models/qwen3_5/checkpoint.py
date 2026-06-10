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
    copy_dense_mlp,
    copy_merged_column,
    copy_source_passthrough_weights,
    gather_moe_expert_weights,
    gather_tensor_parallel_column_tensors,
    gather_tensor_parallel_split_column_tensor,
    gather_tensor_parallel_tensor,
    load_embedding_norm_head,
    rank0_tensor,
    save_dense_mlp,
    save_embedding_norm_head,
    write_hf_safetensors_checkpoint,
)
from areno.engine.checkpoints.io import SafetensorsIndex, _all_gather_tensor_parallel, _copy_row, _owns_checkpoint_tensor, _tensor_to_cpu
from areno.engine.layers.linear import _shard_range
from areno.engine.parallel.context import get_tp_context
from areno.models.qwen3_5.model import Qwen35DecoderLayer, Qwen35ForCausalLM, Qwen35FullAttention, Qwen35GatedDeltaNet


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
def load_qwen35_weights(model: Qwen35ForCausalLM, model_path: str | Path) -> None:
    ctx = get_tp_context()
    index = SafetensorsIndex(model_path)
    try:
        prefix = _resolve_checkpoint_prefix(index)
        model.config.checkpoint_prefix = prefix
        model.config.checkpoint_lm_head_key = _resolve_lm_head_key(index, prefix, model.config.tie_word_embeddings)
        index.set_progress_total(1 + len(model.layers), unit="stage", manual=True)
        _load_embedding_norm_head(model, index, prefix, ctx.rank, ctx.world_size)
        index.advance_progress()
        for layer_idx, layer in enumerate(model.layers):
            _load_layer(index, layer, f"{prefix}.layers.{layer_idx}", ctx.rank, ctx.world_size)
            index.advance_progress()
    finally:
        index.close()


@torch.no_grad()
def save_qwen35_weights(model: Qwen35ForCausalLM, output_path: str | Path, source_path: str | Path | None) -> str | None:
    tensors = CheckpointTensorStore()
    prefix = model.config.checkpoint_prefix
    _save_embedding_norm_head(tensors, model, prefix)
    for layer_idx, layer in enumerate(model.layers):
        _save_layer(tensors, layer, f"{prefix}.layers.{layer_idx}")
    saved_path = write_hf_safetensors_checkpoint(tensors, output_path, source_path)
    if saved_path is not None and source_path is not None:
        copy_source_passthrough_weights(source_path, saved_path, protected_prefix=f"{prefix}.")
    return saved_path


def _top_level_spec(prefix: str, lm_head_key: str = "lm_head.weight") -> TopLevelSpec:
    return TopLevelSpec(
        embedding_key=f"{prefix}.embed_tokens.weight",
        embedding_attr="embed_tokens",
        norm_key=f"{prefix}.norm.weight",
        norm_attr="norm.weight",
        lm_head_key=lm_head_key,
    )


def _resolve_checkpoint_prefix(index: SafetensorsIndex) -> str:
    if "model.embed_tokens.weight" in index.weight_map:
        return "model"
    if "model.language_model.embed_tokens.weight" in index.weight_map:
        return "model.language_model"
    raise KeyError("could not find Qwen3.5 text embedding under model or model.language_model")


def _resolve_lm_head_key(index: SafetensorsIndex, prefix: str, tie_word_embeddings: bool) -> str:
    candidates = (f"{prefix}.lm_head.weight", "lm_head.weight")
    if tie_word_embeddings:
        return next((key for key in candidates if key in index.weight_map), "lm_head.weight")
    for key in candidates:
        if key in index.weight_map:
            return key
    raise KeyError(f"could not find Qwen3.5 LM head; checked: {', '.join(candidates)}")


def _load_embedding_norm_head(model: Qwen35ForCausalLM, index: SafetensorsIndex, prefix: str, rank: int, world_size: int) -> None:
    spec = _top_level_spec(prefix, model.config.checkpoint_lm_head_key)
    load_embedding_norm_head(model, index, spec, rank, world_size)
    model.norm.weight.add_(1.0)


def _save_embedding_norm_head(tensors: dict[str, torch.Tensor | None], model: Qwen35ForCausalLM, prefix: str) -> None:
    spec = _top_level_spec(prefix, model.config.checkpoint_lm_head_key)
    _require_lm_head_shape(model, spec.lm_head_key)
    save_embedding_norm_head(tensors, model, spec)
    tensors[spec.norm_key] = rank0_tensor(model.norm.weight - 1.0)


def _require_lm_head_shape(model: Qwen35ForCausalLM, lm_head_key: str) -> None:
    if model.config.tie_word_embeddings:
        return
    weight = model.lm_head.weight
    if weight.ndim != 2 or weight.shape[1] != model.config.hidden_size:
        raise ValueError(f"{lm_head_key} has invalid local shape {tuple(weight.shape)}")


def _load_layer(index: SafetensorsIndex, layer: Qwen35DecoderLayer, prefix: str, rank: int, world_size: int) -> None:
    is_moe = _is_moe_mlp(layer.mlp)
    keys = [
        *[spec.key.format(prefix=prefix) for spec in LAYER_NORM_SPECS],
    ]
    if not is_moe:
        keys.extend([*[key.format(prefix=prefix) for key in GATE_UP_WEIGHT_SPEC.keys], MLP_ROW_SPEC.key.format(prefix=prefix)])
    keys.extend(_attention_keys(layer.attention, prefix))
    index.prefetch([key for key in keys if key in index.weight_map])

    for spec in LAYER_NORM_SPECS:
        dst = _attr_path(layer, spec.attr)
        dst.copy_((index.get_tensor(spec.key.format(prefix=prefix)) + 1.0).to(dtype=dst.dtype))
    if is_moe:
        _load_moe_mlp(index, layer.mlp, f"{prefix}.mlp", rank, world_size)
    else:
        _load_dense_mlp(index, layer.mlp, f"{prefix}.mlp", rank, world_size)

    if isinstance(layer.attention, Qwen35FullAttention):
        _load_full_attention(index, layer.attention, prefix, rank, world_size)
        return
    if isinstance(layer.attention, Qwen35GatedDeltaNet):
        _load_gated_delta_net(index, layer.attention, prefix, rank, world_size)
        return
    raise TypeError(f"unsupported Qwen3.5 attention module {type(layer.attention)!r}")


def _save_layer(tensors: dict[str, torch.Tensor | None], layer: Qwen35DecoderLayer, prefix: str) -> None:
    for spec in LAYER_NORM_SPECS:
        tensors[spec.key.format(prefix=prefix)] = rank0_tensor(_attr_path(layer, spec.attr) - 1.0)
    _save_mlp(tensors, layer, prefix)

    if isinstance(layer.attention, Qwen35FullAttention):
        _save_full_attention(tensors, layer.attention, prefix)
        return
    if isinstance(layer.attention, Qwen35GatedDeltaNet):
        _save_gated_delta_net(tensors, layer.attention, prefix)
        return
    raise TypeError(f"unsupported Qwen3.5 attention module {type(layer.attention)!r}")


def _save_mlp(tensors: dict[str, torch.Tensor | None], layer: Qwen35DecoderLayer, prefix: str) -> None:
    if _is_moe_mlp(layer.mlp):
        _save_moe_mlp(tensors, layer.mlp, f"{prefix}.mlp")
        return
    _save_dense_mlp(tensors, layer.mlp, f"{prefix}.mlp")


def _save_dense_mlp(tensors: dict[str, torch.Tensor | None], mlp: nn.Module, prefix: str) -> None:
    gate_local, up_local = mlp.gate_up_proj.weight.detach().split(mlp.gate_up_proj.local_out_features, dim=0)
    gate_weight, up_weight = gather_tensor_parallel_column_tensors([gate_local, up_local])
    tensors[f"{prefix}.gate_proj.weight"] = gate_weight
    tensors[f"{prefix}.up_proj.weight"] = up_weight
    tensors[f"{prefix}.down_proj.weight"] = gather_tensor_parallel_tensor(mlp.down_proj.weight, dim=1)


def _save_moe_mlp(tensors: dict[str, torch.Tensor | None], mlp: nn.Module, prefix: str) -> None:
    tensors[f"{prefix}.gate.weight"] = rank0_tensor(getattr(mlp, "gate"))
    _save_packed_routed_experts(tensors, mlp, prefix)
    shared_expert = getattr(mlp, "shared_expert", None)
    if shared_expert is not None:
        save_dense_mlp(tensors, f"{prefix}.shared_expert", shared_expert)
    _save_shared_expert_gate(tensors, mlp, prefix)


def _save_packed_routed_experts(tensors: dict[str, torch.Tensor | None], mlp: nn.Module, prefix: str) -> None:
    gate_up_key = f"{prefix}.experts.gate_up_proj"
    down_key = f"{prefix}.experts.down_proj"
    full_weights = gather_moe_expert_weights(getattr(mlp, "experts"))
    if full_weights is None:
        tensors[gate_up_key] = None
        tensors[down_key] = None
        return
    gate_weights, up_weights, down_weights = full_weights
    gate_up = torch.cat((gate_weights, up_weights), dim=1).contiguous()
    tensors[gate_up_key] = _tensor_to_cpu(gate_up) if _owns_checkpoint_tensor(gate_up_key) else None
    tensors[down_key] = _tensor_to_cpu(down_weights.contiguous()) if _owns_checkpoint_tensor(down_key) else None


def _save_shared_expert_gate(tensors: dict[str, torch.Tensor | None], mlp: nn.Module, prefix: str) -> None:
    gate = getattr(mlp, "shared_expert_gate", None)
    if gate is None:
        return
    tensors[f"{prefix}.shared_expert_gate.weight"] = rank0_tensor(gate.reshape(1, -1))


def _attention_keys(attn: nn.Module, prefix: str) -> list[str]:
    if isinstance(attn, Qwen35FullAttention):
        return [key.format(prefix=prefix) for key in FULL_ATTN_KEYS]
    return [key.format(prefix=prefix) for key in GDN_KEYS]


def _is_moe_mlp(mlp: nn.Module) -> bool:
    return getattr(mlp, "gate", None) is not None and getattr(mlp, "num_experts", None) is not None


def _load_dense_mlp(index: SafetensorsIndex, mlp: nn.Module, prefix: str, rank: int, world_size: int) -> None:
    copy_merged_column(
        mlp.gate_up_proj.weight,
        [index.get_tensor(f"{prefix}.gate_proj.weight"), index.get_tensor(f"{prefix}.up_proj.weight")],
        rank,
        world_size,
    )
    _copy_row(mlp.down_proj.weight, index.get_tensor(f"{prefix}.down_proj.weight"), rank, world_size)


def _load_moe_mlp(index: SafetensorsIndex, mlp: nn.Module, prefix: str, rank: int, world_size: int) -> None:
    _load_router_gate(index, getattr(mlp, "gate"), prefix)
    _load_routed_experts(index, mlp, prefix, rank, world_size)
    _load_shared_expert(index, mlp, prefix, rank, world_size)


def _load_router_gate(index: SafetensorsIndex, gate: torch.Tensor, prefix: str) -> None:
    candidates = (
        f"{prefix}.gate.weight",
        f"{prefix}.gate",
        f"{prefix}.router.weight",
        f"{prefix}.router",
    )
    seen_shapes: list[str] = []
    for candidate in candidates:
        if candidate not in index.weight_map:
            continue
        tensor = index.get_tensor(candidate)
        seen_shapes.append(f"{candidate}: {tuple(tensor.shape)}")
        if tuple(tensor.shape) == tuple(gate.shape):
            gate.copy_(tensor.to(dtype=gate.dtype))
            return
        if tensor.ndim == 2 and tuple(tensor.t().shape) == tuple(gate.shape):
            gate.copy_(tensor.t().contiguous().to(dtype=gate.dtype))
            return
        if tensor.ndim == 2 and tensor.shape[0] == gate.shape[1] and tensor.shape[1] >= gate.shape[0]:
            gate.copy_(tensor[:, : gate.shape[0]].t().contiguous().to(dtype=gate.dtype))
            return
    extra = _gate_key_shape_candidates(index, prefix)
    raise KeyError(f"missing Qwen3.5-MoE router gate with shape {tuple(gate.shape)} under {prefix}; checked: {seen_shapes + extra}")


def _load_routed_experts(index: SafetensorsIndex, mlp: nn.Module, prefix: str, rank: int, world_size: int) -> None:
    experts = getattr(mlp, "experts")
    gate_up_key, down_key = _first_existing_pair(
        index,
        (
            (f"{prefix}.experts.gate_up_proj.weight", f"{prefix}.experts.down_proj.weight"),
            (f"{prefix}.experts.gate_up_proj", f"{prefix}.experts.down_proj"),
        ),
    )
    if gate_up_key is not None and down_key is not None:
        _copy_packed_gate_up_down_experts(
            experts,
            index.get_tensor(gate_up_key),
            index.get_tensor(down_key),
            _common_expert_gate(index, prefix, getattr(experts, "gate_up_weight")),
        )
        return
    gate_up_key, down_key = _first_existing_pair(
        index,
        (
            (f"{prefix}.experts.w13_weight", f"{prefix}.experts.w2_weight"),
            (f"{prefix}.experts.w13", f"{prefix}.experts.w2"),
        ),
    )
    if gate_up_key is not None and down_key is not None:
        _copy_packed_gate_up_down_experts(
            experts,
            index.get_tensor(gate_up_key),
            index.get_tensor(down_key),
        )
        return
    gate_key, up_key, down_key = _first_existing_triple(
        index,
        (
            (f"{prefix}.experts.gate_proj.weight", f"{prefix}.experts.up_proj.weight", f"{prefix}.experts.down_proj.weight"),
            (f"{prefix}.experts.gate_proj", f"{prefix}.experts.up_proj", f"{prefix}.experts.down_proj"),
        ),
    )
    if gate_key is not None and up_key is not None and down_key is not None:
        _copy_packed_split_experts(
            experts,
            index.get_tensor(gate_key),
            index.get_tensor(up_key),
            index.get_tensor(down_key),
        )
        return
    if f"{prefix}.experts.0.gate_proj.weight" in index.weight_map:
        for expert_id in range(int(getattr(mlp, "num_experts"))):
            experts.copy_expert(
                expert_id,
                index.get_tensor(f"{prefix}.experts.{expert_id}.gate_proj.weight"),
                index.get_tensor(f"{prefix}.experts.{expert_id}.up_proj.weight"),
                index.get_tensor(f"{prefix}.experts.{expert_id}.down_proj.weight"),
                rank,
                world_size,
            )
        return
    candidates = sorted(key for key in index.weight_map if key.startswith(f"{prefix}.experts."))[:8]
    raise KeyError(f"missing Qwen3.5-MoE expert weights under {prefix}.experts; found candidates: {candidates}")


def _copy_packed_gate_up_down_experts(experts: nn.Module, gate_up: torch.Tensor, down: torch.Tensor, common_gate: torch.Tensor | None = None) -> None:
    start = int(getattr(experts, "local_expert_start"))
    end = int(getattr(experts, "local_expert_end"))
    gate_up_weight = getattr(experts, "gate_up_weight")
    down_weight = getattr(experts, "down_weight")
    gate_up_weight.copy_(_local_gate_up_tensor(gate_up, gate_up_weight, start, end, common_gate=common_gate).to(dtype=gate_up_weight.dtype))
    down_weight.copy_(_local_expert_tensor(down, down_weight, start, end, "down").to(dtype=down_weight.dtype))


def _copy_packed_split_experts(experts: nn.Module, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
    start = int(getattr(experts, "local_expert_start"))
    end = int(getattr(experts, "local_expert_end"))
    gate_up_weight = getattr(experts, "gate_up_weight")
    down_weight = getattr(experts, "down_weight")
    gate_shape = (gate_up_weight.shape[0], gate_up_weight.shape[1] // 2, gate_up_weight.shape[2])
    local_gate = _local_expert_tensor(gate, gate_shape, start, end, "gate")
    local_up = _local_expert_tensor(up, gate_shape, start, end, "up")
    gate_up_weight.copy_(torch.cat((local_gate, local_up), dim=1).to(dtype=gate_up_weight.dtype))
    down_weight.copy_(_local_expert_tensor(down, down_weight, start, end, "down").to(dtype=down_weight.dtype))


def _local_gate_up_tensor(source: torch.Tensor, target: torch.Tensor, start: int, end: int, common_gate: torch.Tensor | None) -> torch.Tensor:
    try:
        return _local_expert_tensor(source, target, start, end, "gate_up")
    except ValueError:
        if common_gate is None:
            raise
    gate_shape = (target.shape[0], target.shape[1] // 2, target.shape[2])
    local_up = _local_expert_tensor(source, gate_shape, start, end, "up")
    local_gate = common_gate.unsqueeze(0).expand(local_up.shape[0], -1, -1)
    return torch.cat((local_gate, local_up), dim=1).contiguous()


def _common_expert_gate(index: SafetensorsIndex, prefix: str, target: torch.Tensor) -> torch.Tensor | None:
    key = f"{prefix}.gate.weight"
    if key not in index.weight_map:
        return None
    tensor = index.get_tensor(key)
    gate_shape = (target.shape[1] // 2, target.shape[2])
    if tuple(tensor.shape) == gate_shape:
        return tensor
    if tensor.ndim == 2 and tuple(tensor.t().shape) == gate_shape:
        return tensor.t().contiguous()
    return None


def _local_expert_tensor(source: torch.Tensor, target: torch.Tensor | tuple[int, int, int], start: int, end: int, name: str) -> torch.Tensor:
    target_shape = tuple(target.shape if isinstance(target, torch.Tensor) else target)
    if source.ndim != 3:
        raise ValueError(f"Qwen3.5-MoE {name} expert tensor must be 3D, got shape {tuple(source.shape)}")
    if tuple(source.shape) == target_shape:
        return source
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        candidate = source.permute(perm)
        if tuple(candidate.shape[1:]) == target_shape[1:] and candidate.shape[0] >= end:
            return candidate[start:end].contiguous()
    raise ValueError(f"cannot map Qwen3.5-MoE {name} expert tensor shape {tuple(source.shape)} to local shape {target_shape}")


def _load_shared_expert(index: SafetensorsIndex, mlp: nn.Module, prefix: str, rank: int, world_size: int) -> None:
    shared_expert = getattr(mlp, "shared_expert", None)
    shared_gate = getattr(mlp, "shared_expert_gate", None)
    if shared_gate is not None and f"{prefix}.shared_expert_gate.weight" in index.weight_map:
        tensor = index.get_tensor(f"{prefix}.shared_expert_gate.weight")
        shared_gate.copy_(tensor.reshape_as(shared_gate).to(dtype=shared_gate.dtype))
    if shared_expert is None:
        return
    for shared_name in ("shared_expert", "shared_experts"):
        shared_prefix = f"{prefix}.{shared_name}"
        if f"{shared_prefix}.gate_proj.weight" in index.weight_map:
            copy_dense_mlp(shared_expert, index, shared_prefix, rank, world_size)
            return
    raise KeyError(f"missing Qwen3.5-MoE shared expert weights under {prefix}.shared_expert or {prefix}.shared_experts")


def _first_existing_pair(index: SafetensorsIndex, pairs: tuple[tuple[str, str], ...]) -> tuple[str | None, str | None]:
    for first, second in pairs:
        if first in index.weight_map and second in index.weight_map:
            return first, second
    return None, None


def _first_existing_triple(index: SafetensorsIndex, triples: tuple[tuple[str, str, str], ...]) -> tuple[str | None, str | None, str | None]:
    for first, second, third in triples:
        if first in index.weight_map and second in index.weight_map and third in index.weight_map:
            return first, second, third
    return None, None, None


def _gate_key_shape_candidates(index: SafetensorsIndex, prefix: str) -> list[str]:
    out = []
    for candidate in sorted(key for key in index.weight_map if key.startswith(prefix) and ("gate" in key or "router" in key)):
        if len(out) >= 8:
            break
        out.append(candidate)
    return out


def _moe_mlp_keys(index: SafetensorsIndex, mlp: nn.Module, prefix: str) -> set[str]:
    keys = set()
    for candidate in (f"{prefix}.gate.weight", f"{prefix}.gate", f"{prefix}.router.weight", f"{prefix}.router"):
        if candidate in index.weight_map:
            keys.add(candidate)
            break
    gate_up_key, down_key = _first_existing_pair(
        index,
        (
            (f"{prefix}.experts.gate_up_proj.weight", f"{prefix}.experts.down_proj.weight"),
            (f"{prefix}.experts.gate_up_proj", f"{prefix}.experts.down_proj"),
        ),
    )
    if gate_up_key is not None and down_key is not None:
        keys.update((gate_up_key, down_key))
    else:
        gate_up_key, down_key = _first_existing_pair(
            index,
            (
                (f"{prefix}.experts.w13_weight", f"{prefix}.experts.w2_weight"),
                (f"{prefix}.experts.w13", f"{prefix}.experts.w2"),
            ),
        )
        if gate_up_key is not None and down_key is not None:
            keys.update((gate_up_key, down_key))
    gate_key, up_key, down_key = _first_existing_triple(
        index,
        (
            (f"{prefix}.experts.gate_proj.weight", f"{prefix}.experts.up_proj.weight", f"{prefix}.experts.down_proj.weight"),
            (f"{prefix}.experts.gate_proj", f"{prefix}.experts.up_proj", f"{prefix}.experts.down_proj"),
        )
    )
    if gate_key is not None and up_key is not None and down_key is not None:
        keys.update((gate_key, up_key, down_key))
    elif not any(key.startswith(f"{prefix}.experts.") for key in keys) and f"{prefix}.experts.0.gate_proj.weight" in index.weight_map:
        for expert_id in range(int(getattr(mlp, "num_experts"))):
            keys.add(f"{prefix}.experts.{expert_id}.gate_proj.weight")
            keys.add(f"{prefix}.experts.{expert_id}.up_proj.weight")
            keys.add(f"{prefix}.experts.{expert_id}.down_proj.weight")
    if getattr(mlp, "shared_expert", None) is not None:
        if f"{prefix}.shared_expert_gate.weight" in index.weight_map:
            keys.add(f"{prefix}.shared_expert_gate.weight")
        for shared_name in ("shared_expert", "shared_experts"):
            shared_prefix = f"{prefix}.{shared_name}"
            if f"{shared_prefix}.gate_proj.weight" in index.weight_map:
                keys.add(f"{shared_prefix}.gate_proj.weight")
                keys.add(f"{shared_prefix}.up_proj.weight")
                keys.add(f"{shared_prefix}.down_proj.weight")
                break
    return keys


def _load_full_attention(index: SafetensorsIndex, attn: Qwen35FullAttention, prefix: str, rank: int, world_size: int) -> None:
    q_weight = index.get_tensor(f"{prefix}.self_attn.q_proj.weight")
    k_weight = index.get_tensor(f"{prefix}.self_attn.k_proj.weight")
    v_weight = index.get_tensor(f"{prefix}.self_attn.v_proj.weight")
    if hasattr(attn.qkv_proj, "shard_ranges"):
        q_range, k_range, v_range = attn.qkv_proj.shard_ranges
        _require_weight_rows(q_weight, q_range[1], f"{prefix}.self_attn.q_proj.weight")
        _require_weight_rows(k_weight, k_range[1], f"{prefix}.self_attn.k_proj.weight")
        _require_weight_rows(v_weight, v_range[1], f"{prefix}.self_attn.v_proj.weight")
        q = q_weight[q_range[0] : q_range[1]]
        k = k_weight[k_range[0] : k_range[1]]
        v = v_weight[v_range[0] : v_range[1]]
        attn.qkv_proj.weight.copy_(torch.cat((q, k, v), dim=0).to(dtype=attn.qkv_proj.weight.dtype))
    else:
        copy_merged_column(attn.qkv_proj.weight, [q_weight, k_weight, v_weight], rank, world_size)
    _copy_row(attn.o_proj.weight, index.get_tensor(f"{prefix}.self_attn.o_proj.weight"), rank, world_size)
    attn.q_norm.weight.copy_((index.get_tensor(f"{prefix}.self_attn.q_norm.weight") + 1.0).to(dtype=attn.q_norm.weight.dtype))
    attn.k_norm.weight.copy_((index.get_tensor(f"{prefix}.self_attn.k_norm.weight") + 1.0).to(dtype=attn.k_norm.weight.dtype))


def _load_gated_delta_net(index: SafetensorsIndex, attn: Qwen35GatedDeltaNet, prefix: str, rank: int, world_size: int) -> None:
    qkv = index.get_tensor(f"{prefix}.linear_attn.in_proj_qkv.weight")
    q, k, v = qkv.split((attn.key_dim, attn.key_dim, attn.value_dim), dim=0)
    copy_merged_column(attn.in_proj_qkvz.weight, [q, k, v, index.get_tensor(f"{prefix}.linear_attn.in_proj_z.weight")], rank, world_size)
    copy_merged_column(attn.in_proj_ba.weight, [index.get_tensor(f"{prefix}.linear_attn.in_proj_b.weight"), index.get_tensor(f"{prefix}.linear_attn.in_proj_a.weight")], rank, world_size)
    conv_qkv = index.get_tensor(f"{prefix}.linear_attn.conv1d.weight")
    conv_parts = conv_qkv.split((attn.key_dim, attn.key_dim, attn.value_dim), dim=0)
    copy_merged_column(attn.conv1d_weight, list(conv_parts), rank, world_size)
    start, end = _shard_range(attn.num_value_heads, rank, world_size)
    attn.dt_bias.copy_(index.get_tensor(f"{prefix}.linear_attn.dt_bias")[start:end].to(dtype=attn.dt_bias.dtype))
    attn.A_log.copy_(index.get_tensor(f"{prefix}.linear_attn.A_log")[start:end].to(dtype=attn.A_log.dtype))
    attn.norm_weight.copy_(index.get_tensor(f"{prefix}.linear_attn.norm.weight").to(dtype=attn.norm_weight.dtype))
    _copy_row(attn.out_proj.weight, index.get_tensor(f"{prefix}.linear_attn.out_proj.weight"), rank, world_size)


def _save_full_attention(tensors: dict[str, torch.Tensor | None], attn: Qwen35FullAttention, prefix: str) -> None:
    if attn.num_kv_heads < get_tp_context().world_size:
        _save_full_attention_replicated_kv(tensors, attn, prefix)
        return
    q_size, k_size, v_size = attn.qkv_proj.local_out_features
    q_gate = attn.qkv_proj.weight[:q_size]
    k = attn.qkv_proj.weight[q_size : q_size + k_size]
    v = attn.qkv_proj.weight[q_size + k_size : q_size + k_size + v_size]
    q_weight, k_weight, v_weight = gather_tensor_parallel_column_tensors([q_gate, k, v])
    tensors[f"{prefix}.self_attn.q_proj.weight"] = q_weight
    tensors[f"{prefix}.self_attn.k_proj.weight"] = k_weight
    tensors[f"{prefix}.self_attn.v_proj.weight"] = v_weight
    tensors[f"{prefix}.self_attn.o_proj.weight"] = gather_tensor_parallel_tensor(attn.o_proj.weight, dim=1)
    tensors[f"{prefix}.self_attn.q_norm.weight"] = rank0_tensor(attn.q_norm.weight - 1.0)
    tensors[f"{prefix}.self_attn.k_norm.weight"] = rank0_tensor(attn.k_norm.weight - 1.0)


def _save_full_attention_replicated_kv(tensors: dict[str, torch.Tensor | None], attn: Qwen35FullAttention, prefix: str) -> None:
    q_size, k_size, v_size = attn.qkv_proj.local_out_features
    q_gate = attn.qkv_proj.weight[:q_size]
    k = attn.qkv_proj.weight[q_size : q_size + k_size]
    v = attn.qkv_proj.weight[q_size + k_size : q_size + k_size + v_size]
    q_weight = gather_tensor_parallel_tensor(q_gate, dim=0)
    tensors[f"{prefix}.self_attn.q_proj.weight"] = q_weight
    k_key = f"{prefix}.self_attn.k_proj.weight"
    v_key = f"{prefix}.self_attn.v_proj.weight"
    k_full = _gather_replicated_kv_heads(k, attn.num_kv_heads)
    v_full = _gather_replicated_kv_heads(v, attn.num_kv_heads)
    tensors[k_key] = _tensor_to_cpu(k_full) if _owns_checkpoint_tensor(k_key) else None
    tensors[v_key] = _tensor_to_cpu(v_full) if _owns_checkpoint_tensor(v_key) else None
    tensors[f"{prefix}.self_attn.o_proj.weight"] = gather_tensor_parallel_tensor(attn.o_proj.weight, dim=1)
    tensors[f"{prefix}.self_attn.q_norm.weight"] = rank0_tensor(attn.q_norm.weight - 1.0)
    tensors[f"{prefix}.self_attn.k_norm.weight"] = rank0_tensor(attn.k_norm.weight - 1.0)


def _gather_replicated_kv_heads(local: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    ctx = get_tp_context()
    local = local.detach().contiguous()
    if ctx.dp_rank != 0 or ctx.world_size == 1:
        return local
    gathered = _all_gather_tensor_parallel(local)
    replication = ctx.world_size // num_kv_heads
    heads = [gathered[head * replication] for head in range(num_kv_heads)]
    return torch.cat(heads, dim=0).contiguous()


def _require_weight_rows(weight: torch.Tensor, min_rows: int, key: str) -> None:
    if weight.shape[0] >= min_rows:
        return
    raise ValueError(
        f"{key} has only {weight.shape[0]} rows, but this TP shard needs rows up to {min_rows}; "
        "the checkpoint likely came from an older Qwen3.5 replicated-KV save path and is missing KV heads. "
        "Re-save from the original HF checkpoint with the fixed saver."
    )


def _save_gated_delta_net(tensors: dict[str, torch.Tensor | None], attn: Qwen35GatedDeltaNet, prefix: str) -> None:
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

    conv_q = attn.local_key_dim
    conv_k = attn.local_key_dim
    conv_v = attn.local_value_dim
    tensors[f"{prefix}.linear_attn.conv1d.weight"] = gather_tensor_parallel_split_column_tensor(attn.conv1d_weight, [conv_q, conv_k, conv_v])
    tensors[f"{prefix}.linear_attn.dt_bias"] = gather_tensor_parallel_tensor(attn.dt_bias, dim=0)
    tensors[f"{prefix}.linear_attn.A_log"] = gather_tensor_parallel_tensor(attn.A_log, dim=0)
    tensors[f"{prefix}.linear_attn.norm.weight"] = rank0_tensor(attn.norm_weight)
    tensors[f"{prefix}.linear_attn.out_proj.weight"] = gather_tensor_parallel_tensor(attn.out_proj.weight, dim=1)


def _attr_path(obj: object, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj
