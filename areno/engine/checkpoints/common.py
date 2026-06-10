"""Shared checkpoint layout helpers.

New model checkpoint support should be split into two small pieces:

1. Declare specs in `checkpoints/<model>.py` for regular tensors:
   `TopLevelSpec` for embeddings/norm/lm_head, `ReplicatedTensorSpec`
   for non-sharded layer norms, `ParallelTensorSpec` for one tensor-parallel
   shard, `MergedColumnSpec`/`SplitColumnSpec` for fused local modules
   such as QKV or gate/up projections, and `MoeSpec` for routed experts.
2. Compose `LayerSpec.load_ops` and `LayerSpec.save_ops` from built-in ops.
   Standard models should not define Python load/save handlers in their
   checkpoint files; if a new layout is needed, add a generic op here first.

This keeps checkpoint format knowledge out of model definition files while
still making the common TP shard copy/gather rules explicit and reusable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from areno.engine.checkpoints.io import (
    CheckpointTensorStore,
    SafetensorsIndex,
    _copy_column,
    _copy_row,
    _owns_checkpoint_tensor,
    _tensor_to_cpu,
    gather_tensor_parallel_column_tensors,
    gather_tensor_parallel_split_column_tensor,
    gather_tensor_parallel_tensor,
    rank0_tensor,
    write_hf_safetensors_checkpoint,
)
from areno.engine.parallel.context import get_tp_context
from areno.engine.layers.linear import _shard_range


@dataclass(frozen=True, slots=True)
class TopLevelSpec:
    """Checkpoint mapping for embeddings, final norm, and LM head."""

    embedding_key: str
    embedding_attr: str
    norm_key: str = "model.norm.weight"
    norm_attr: str = "norm.weight"
    lm_head_key: str = "lm_head.weight"
    lm_head_attr: str = "lm_head.weight"
    optional_vocab: tuple["ReplicatedTensorSpec", ...] = ()
    optional_replicated: tuple["ReplicatedTensorSpec", ...] = ()


@dataclass(frozen=True, slots=True)
class ReplicatedTensorSpec:
    """A full tensor copied identically onto every tensor-parallel rank."""

    key: str
    attr: str


@dataclass(frozen=True, slots=True)
class OptionalReplicatedTensorSpec:
    """A replicated tensor that is skipped if the module or key is absent."""

    key: str
    attr: str


@dataclass(frozen=True, slots=True)
class ParallelTensorSpec:
    """A tensor sharded along `dim` across tensor-parallel ranks."""

    key: str
    attr: str
    dim: int


@dataclass(frozen=True, slots=True)
class MergedColumnSpec:
    """Multiple HF column tensors loaded into one fused local module weight."""

    dst_attr: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KSharedQKVColumnSpec:
    """QKV load spec for checkpoints where later layers may share K/V."""

    dst_attr: str
    q_key: str
    k_key: str
    v_key: str


@dataclass(frozen=True, slots=True)
class OptionalMergedColumnSpec:
    """Optional version of `MergedColumnSpec` for bias or variant tensors."""

    dst_attr: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitColumnSpec:
    """One fused local column tensor saved back as multiple HF tensors."""

    src_attr: str
    size_attr: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RangedSplitColumnSpec:
    """Split and gather a fused column tensor using explicit per-rank ranges."""

    src_attr: str
    size_attr: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptionalSplitColumnSpec:
    """Optional version of `SplitColumnSpec` for absent bias tensors."""

    src_attr: str
    size_attr: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptionalRangedSplitColumnSpec:
    """Optional version of `RangedSplitColumnSpec` for absent bias tensors."""

    src_attr: str
    size_attr: str
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MoeSpec:
    """Checkpoint mapping for routed MoE gate and expert weights."""

    gate_weight_key: str
    gate_weight_attr: str
    expert_bias_key: str | None
    expert_bias_attr: str | None
    local_expert_bias_attr: str | None
    experts_attr: str
    num_experts_attr: str
    expert_gate_key: str
    expert_up_key: str
    expert_down_key: str
    shared_experts_attr: str | None
    shared_experts_prefix: str | None


@dataclass(frozen=True, slots=True)
class DenseOrMoeSpec:
    """Layer op that dispatches to dense MLP or MoE save/load behavior."""

    attr: str
    moe: MoeSpec


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """Generic attention checkpoint mapping used by model variants."""

    attr: str
    checkpoint_names: tuple[str, ...]
    qk_norms: tuple[ReplicatedTensorSpec, ...]
    dense_rows: tuple[ParallelTensorSpec, ...]
    dense_bias: ReplicatedTensorSpec
    linear_dense_rows: tuple[ParallelTensorSpec, ...]
    linear_dense_bias: ReplicatedTensorSpec


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """Per-layer checkpoint spec assembled from generic load/save ops."""

    prefix: str
    replicated: tuple[ReplicatedTensorSpec, ...]
    load_ops: tuple[object, ...] = ()
    save_ops: tuple[object, ...] = ()
    load_handlers: tuple[Callable[[nn.Module, SafetensorsIndex, str, int, int], None], ...] = ()
    save_handlers: tuple[Callable[[dict[str, torch.Tensor | None], str, nn.Module, object | None], None], ...] = ()


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """Full model checkpoint spec used by registry adapters."""

    top_level: TopLevelSpec
    layer: LayerSpec


def attr_path(obj: object, path: str):
    """Resolve a dotted attribute path from an object."""

    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def optional_attr_path(obj: object, path: str):
    """Resolve a dotted path, returning None if an intermediate is None."""

    for part in path.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part)
    return obj


def key(template: str, prefix: str) -> str:
    """Format a template like ``"{prefix}.weight"`` with the layer prefix."""

    return template.format(prefix=prefix)


def keys(templates: tuple[str, ...], prefix: str) -> list[str]:
    """Format a tuple of templates into concrete HF checkpoint keys."""

    return [key(template, prefix) for template in templates]


def expert_key(template: str, prefix: str, expert_id: int) -> str:
    """Format an MoE expert-indexed template with both prefix and expert id."""

    return template.format(prefix=prefix, expert=expert_id)


def copy_merged_column(dst: torch.Tensor, srcs: list[torch.Tensor], rank: int, world_size: int) -> None:
    """Pack TP shards of several HF column tensors into one fused destination.

    Each source contributes its own ``_shard_range`` slice along dim 0; the
    slices are concatenated in source order into ``dst``. Used to load fused
    QKV or fused gate/up local weights from their separate HF tensors.
    """

    offset = 0
    for src in srcs:
        start, end = _shard_range(src.shape[0], rank, world_size)
        size = end - start
        # Copy this rank's row slice of src into the next chunk of dst.
        dst[offset : offset + size].copy_(src[start:end].to(dtype=dst.dtype))
        offset += size


def copy_merged_column_from_index(dst: torch.Tensor, index: SafetensorsIndex, tensor_keys: list[str], rank: int, world_size: int) -> None:
    """Pack this rank's TP column shards without materializing full tensors."""

    offset = 0
    for tensor_key in tensor_keys:
        shard = _read_tp_shard(index, tensor_key, dim=0, rank=rank, world_size=world_size)
        size = shard.shape[0]
        dst[offset : offset + size].copy_(shard.to(dtype=dst.dtype))
        offset += size


def copy_ranged_merged_column(dst: torch.Tensor, srcs: list[torch.Tensor], ranges: tuple[tuple[int, int], ...]) -> None:
    """Variant of `copy_merged_column` with explicit per-source row ranges.

    Used when the fused local module reports its own shard ranges (for example
    QKV with non-uniform head splits across TP ranks).
    """

    offset = 0
    for src, (start, end) in zip(srcs, ranges, strict=True):
        size = end - start
        dst[offset : offset + size].copy_(src[start:end].to(dtype=dst.dtype))
        offset += size


def copy_ranged_merged_column_from_index(
    dst: torch.Tensor,
    index: SafetensorsIndex,
    tensor_keys: list[str],
    ranges: tuple[tuple[int, int], ...],
) -> None:
    """Pack explicit column ranges from safetensors without TP-even sharding."""

    offset = 0
    for tensor_key, (start, end) in zip(tensor_keys, ranges, strict=True):
        shard = _read_explicit_shard(index, tensor_key, dim=0, start=start, end=end)
        size = shard.shape[0]
        dst[offset : offset + size].copy_(shard.to(dtype=dst.dtype))
        offset += size


def copy_merged_optional_bias(
    dst: torch.Tensor | None,
    index: SafetensorsIndex,
    keys: list[str],
    rank: int,
    world_size: int,
) -> None:
    """Load a merged-column bias only when both module and all HF keys exist."""

    if dst is None or any(key not in index.weight_map for key in keys):
        return
    copy_merged_column_from_index(dst, index, keys, rank, world_size)


def split_local_tensors(tensor: torch.Tensor, sizes: list[int]) -> tuple[torch.Tensor, ...]:
    """Split a fused local tensor back into its component tensors along dim 0."""

    return tuple(tensor.split(tuple(sizes), dim=0))


def load_embedding_norm_head(model: nn.Module, index: SafetensorsIndex, spec: TopLevelSpec, rank: int, world_size: int) -> None:
    """Load token embedding, final norm, LM head, and any optional top-level tensors."""

    # Only prefetch small replicated tensors. Vocab-sharded tensors are read
    # through safetensors slices below so TP ranks do not materialize full rows.
    keys = [spec.norm_key]
    for optional in spec.optional_replicated:
        if optional_attr_path(model, optional.attr) is not None and optional.key in index.weight_map:
            keys.append(optional.key)
    index.prefetch(keys)
    embedding = attr_path(model, spec.embedding_attr)
    # Vocab embeddings are sharded across TP ranks along the vocab dimension.
    _copy_vocab_shard_from_index(embedding.weight, index, spec.embedding_key, rank, world_size)
    lm_head_weight = attr_path(model, spec.lm_head_attr)
    if model.config.tie_word_embeddings:
        # Reuse the embedding weight when LM head is tied; no separate copy needed.
        lm_head_weight.copy_(embedding.weight)
    else:
        _copy_vocab_shard_from_index(lm_head_weight, index, spec.lm_head_key, rank, world_size)
    attr_path(model, spec.norm_attr).copy_(index.get_tensor(spec.norm_key))
    # Optional vocab-sharded tensors (per-token biases etc.) follow the same vocab split.
    for optional in spec.optional_vocab:
        dst = optional_attr_path(model, optional.attr)
        if dst is not None and optional.key in index.weight_map:
            _copy_vocab_shard_from_index(dst, index, optional.key, rank, world_size)
    # Optional replicated tensors are copied identically on every rank.
    for optional in spec.optional_replicated:
        dst = optional_attr_path(model, optional.attr)
        if dst is not None and optional.key in index.weight_map:
            dst.copy_(index.get_tensor(optional.key).to(dtype=dst.dtype))


def save_embedding_norm_head(tensors: dict[str, torch.Tensor | None], model: nn.Module, spec: TopLevelSpec) -> None:
    """Stage embedding/norm/LM head tensors for the distributed safetensors writer."""

    embedding = attr_path(model, spec.embedding_attr)
    # Embedding is vocab-sharded; gather along dim 0 back to the full vocab.
    tensors[spec.embedding_key] = gather_tensor_parallel_tensor(embedding.weight, dim=0)
    tensors[spec.norm_key] = rank0_tensor(attr_path(model, spec.norm_attr))
    if not model.config.tie_word_embeddings:
        tensors[spec.lm_head_key] = gather_tensor_parallel_tensor(attr_path(model, spec.lm_head_attr), dim=0)
    for optional in spec.optional_vocab:
        value = optional_attr_path(model, optional.attr)
        if value is not None:
            tensors[optional.key] = gather_tensor_parallel_tensor(value, dim=0)
    for optional in spec.optional_replicated:
        value = optional_attr_path(model, optional.attr)
        if value is not None:
            tensors[optional.key] = rank0_tensor(value)


@torch.no_grad()
def load_checkpoint_weights(model: nn.Module, model_path: str, spec: CheckpointSpec) -> None:
    """Load a HF safetensors checkpoint into a tensor-parallel model."""

    ctx = get_tp_context()
    index = SafetensorsIndex(model_path)
    index.set_progress_total(1 + len(model.layers), unit="stage", manual=True)
    try:
        load_embedding_norm_head(model, index, spec.top_level, ctx.rank, ctx.world_size)
        index.advance_progress()
        load_layer_specs(model, index, spec.layer, ctx.rank, ctx.world_size)
    finally:
        index.close()


@torch.no_grad()
def save_checkpoint_weights(model: nn.Module, output_path: str, source_path: str | None, spec: CheckpointSpec) -> str | None:
    """Save a tensor-parallel model as a HF sharded safetensors checkpoint."""

    tensors = CheckpointTensorStore()
    save_embedding_norm_head(tensors, model, spec.top_level)
    # Some column-parallel tensors are batched until all layers are visited so
    # that a single fused all-gather can be reused across shapes.
    delayed_column_tensors: list[tuple[str, torch.Tensor]] = []
    save_layer_specs(tensors, model, spec.layer, context=delayed_column_tensors)
    if delayed_column_tensors:
        save_column_tensors(tensors, delayed_column_tensors)
    saved_path = write_hf_safetensors_checkpoint(tensors, output_path, source_path)
    if saved_path is not None and source_path is not None:
        copy_source_passthrough_weights(source_path, saved_path, protected_prefix=_protected_prefix_from_top_level(spec.top_level))
    return saved_path


def copy_source_passthrough_weights(source_path: str | Path, output_path: str | Path, protected_prefix: str) -> None:
    """Copy source checkpoint tensors outside the runtime trunk into output.

    Multimodal HF checkpoints often store text weights under
    `model.language_model.*` and vision/projector weights elsewhere. areno
    rewrites the text trunk only, so this preserves untouched non-text tensors
    and keeps the saved checkpoint structurally compatible with the source.
    """

    source = Path(source_path)
    output = Path(output_path)
    if not source.exists() or source.resolve() == output.resolve():
        return
    output_index_path = output / "model.safetensors.index.json"
    if not output_index_path.exists():
        return
    source_weight_map = _read_weight_map(source)
    output_index = json.loads(output_index_path.read_text())
    output_weight_map = dict(output_index["weight_map"])
    passthrough_keys = [key for key in source_weight_map if not key.startswith(protected_prefix) and key not in output_weight_map]
    if not passthrough_keys:
        return

    total_size = int(output_index.get("metadata", {}).get("total_size", 0))
    keys_by_file: dict[str, list[str]] = {}
    for tensor_key in passthrough_keys:
        keys_by_file.setdefault(source_weight_map[tensor_key], []).append(tensor_key)
    for file_idx, (filename, tensor_keys) in enumerate(sorted(keys_by_file.items())):
        passthrough_tensors: dict[str, torch.Tensor] = {}
        with safe_open(source / filename, framework="pt", device="cpu") as handle:
            for tensor_key in tensor_keys:
                tensor = handle.get_tensor(tensor_key)
                passthrough_tensors[tensor_key] = tensor
                total_size += tensor.numel() * tensor.element_size()
        passthrough_name = f"model-passthrough-{file_idx + 1:05d}.safetensors"
        save_file(passthrough_tensors, output / passthrough_name, metadata={"format": "pt"})
        for tensor_key in passthrough_tensors:
            output_weight_map[tensor_key] = passthrough_name

    output_index["metadata"]["total_size"] = total_size
    output_index["weight_map"] = output_weight_map
    output_index_path.write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n")


def _protected_prefix_from_top_level(spec: TopLevelSpec) -> str:
    return spec.embedding_key.rsplit(".", 2)[0] + "."


def _read_weight_map(path: Path) -> dict[str, str]:
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        return dict(json.loads(index_path.read_text())["weight_map"])
    weight_map: dict[str, str] = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as handle:
            for tensor_key in handle.keys():
                weight_map[tensor_key] = file.name
    return weight_map


def load_layer_specs(model: nn.Module, index: SafetensorsIndex, spec: LayerSpec, rank: int, world_size: int) -> None:
    """Load all transformer layers using the configured layer spec."""

    for layer_idx, layer in enumerate(model.layers):
        prefix = spec.prefix.format(layer=layer_idx)
        # Prefetch only small replicated keys. TP-sharded matrices are loaded
        # with safetensors slicing in `load_layer_op`.
        layer_keys = index.keys_for_prefix(f"{prefix}.")
        replicated_keys = [key(rep.key, prefix) for rep in spec.replicated]
        index.prefetch(replicated_keys)
        load_replicated_tensors(layer, index, prefix, spec.replicated)
        for op in spec.load_ops:
            load_layer_op(layer, index, prefix, op, rank, world_size)
        for handler in spec.load_handlers:
            handler(layer, index, prefix, rank, world_size)
        index.advance_progress()
        index.drop(layer_keys)


def save_layer_specs(tensors: dict[str, torch.Tensor | None], model: nn.Module, spec: LayerSpec, context: object | None = None) -> None:
    """Stage all per-layer tensors via the spec's `save_ops` and handlers."""

    for layer_idx, layer in enumerate(model.layers):
        prefix = spec.prefix.format(layer=layer_idx)
        save_replicated_tensors(tensors, layer, prefix, spec.replicated)
        for op in spec.save_ops:
            save_layer_op(tensors, layer, prefix, op, context)
        for handler in spec.save_handlers:
            handler(tensors, prefix, layer, context)


def load_replicated_tensors(module: nn.Module, index: SafetensorsIndex, prefix: str, specs: tuple[ReplicatedTensorSpec, ...]) -> None:
    """Copy fully replicated tensors directly into module attributes."""

    for spec in specs:
        dst = attr_path(module, spec.attr)
        dst.copy_(index.get_tensor(key(spec.key, prefix)).to(dtype=dst.dtype))


def save_replicated_tensors(
    tensors: dict[str, torch.Tensor | None],
    module: nn.Module,
    prefix: str,
    specs: tuple[ReplicatedTensorSpec, ...],
) -> None:
    """Stage replicated tensors via `rank0_tensor` so only rank 0 emits them."""

    for spec in specs:
        tensors[key(spec.key, prefix)] = rank0_tensor(attr_path(module, spec.attr))


def load_layer_op(module: nn.Module, index: SafetensorsIndex, prefix: str, op: object, rank: int, world_size: int) -> None:
    """Dispatch a single layer load op based on its declarative type."""

    if isinstance(op, ReplicatedTensorSpec):
        load_replicated_tensors(module, index, prefix, (op,))
        return
    if isinstance(op, OptionalReplicatedTensorSpec):
        # Skip silently when either the module attribute or the HF key is absent.
        dst = optional_attr_path(module, op.attr)
        tensor_key = key(op.key, prefix)
        if dst is not None and tensor_key in index.weight_map:
            dst.copy_(index.get_tensor(tensor_key).to(dtype=dst.dtype))
        return
    if isinstance(op, ParallelTensorSpec):
        _copy_parallel_tensor_from_index(attr_path(module, op.attr), index, key(op.key, prefix), op.dim, rank, world_size)
        return
    if isinstance(op, MergedColumnSpec):
        load_merged_column_spec(module, index, prefix, op, rank, world_size)
        return
    if isinstance(op, KSharedQKVColumnSpec):
        load_k_shared_qkv_column_spec(module, index, prefix, op, rank, world_size)
        return
    if isinstance(op, OptionalMergedColumnSpec):
        # Optional fused load only fires when destination and all sources exist.
        dst = optional_attr_path(module, op.dst_attr)
        tensor_keys = keys(op.keys, prefix)
        if dst is not None and all(tensor_key in index.weight_map for tensor_key in tensor_keys):
            merged_module = attr_path(module, op.dst_attr.rsplit(".", 1)[0])
            shard_ranges = getattr(merged_module, "shard_ranges", None)
            if shard_ranges is not None and len(shard_ranges) == len(tensor_keys):
                copy_ranged_merged_column_from_index(dst, index, tensor_keys, shard_ranges)
                return
            copy_merged_column_from_index(dst, index, tensor_keys, rank, world_size)
        return
    if isinstance(op, AttentionSpec):
        load_attention_spec(module, index, prefix, op, rank, world_size)
        return
    if isinstance(op, DenseOrMoeSpec):
        load_dense_or_moe_spec(module, index, prefix, op, rank, world_size)
        return
    raise TypeError(f"unsupported checkpoint load op {type(op)!r}")


def save_layer_op(tensors: dict[str, torch.Tensor | None], module: nn.Module, prefix: str, op: object, context: object | None) -> None:
    """Dispatch a single layer save op based on its declarative type."""

    if isinstance(op, ReplicatedTensorSpec):
        save_replicated_tensors(tensors, module, prefix, (op,))
        return
    if isinstance(op, OptionalReplicatedTensorSpec):
        value = optional_attr_path(module, op.attr)
        if value is not None:
            tensors[key(op.key, prefix)] = rank0_tensor(value)
        return
    if isinstance(op, ParallelTensorSpec):
        save_parallel_tensors(tensors, module, prefix, (op,))
        return
    if isinstance(op, SplitColumnSpec):
        save_split_column_spec(tensors, module, prefix, op)
        return
    if isinstance(op, RangedSplitColumnSpec):
        save_ranged_split_column_spec(tensors, module, prefix, op)
        return
    if isinstance(op, OptionalSplitColumnSpec):
        if optional_attr_path(module, op.src_attr) is not None:
            save_split_column_spec(tensors, module, prefix, SplitColumnSpec(op.src_attr, op.size_attr, op.keys))
        return
    if isinstance(op, OptionalRangedSplitColumnSpec):
        if optional_attr_path(module, op.src_attr) is not None:
            save_ranged_split_column_spec(tensors, module, prefix, RangedSplitColumnSpec(op.src_attr, op.size_attr, op.keys))
        return
    if isinstance(op, AttentionSpec):
        save_attention_spec(tensors, module, prefix, op, context)
        return
    if isinstance(op, DenseOrMoeSpec):
        save_dense_or_moe_spec(tensors, module, prefix, op)
        return
    raise TypeError(f"unsupported checkpoint save op {type(op)!r}")


def save_parallel_tensors(
    tensors: dict[str, torch.Tensor | None],
    module: nn.Module,
    prefix: str,
    specs: tuple[ParallelTensorSpec, ...],
) -> None:
    """Schedule TP-gather tasks for sharded tensors via the lazy writer."""

    for spec in specs:
        tensors[key(spec.key, prefix)] = gather_tensor_parallel_tensor(attr_path(module, spec.attr), dim=spec.dim)


def _copy_parallel_tensor(dst: torch.Tensor, src: torch.Tensor, dim: int, rank: int, world_size: int) -> None:
    """Copy a single TP shard of src into dst along dim (0=column, 1=row)."""

    if dim == 0:
        _copy_column(dst, src, rank, world_size)
        return
    if dim == 1:
        _copy_row(dst, src, rank, world_size)
        return
    raise ValueError(f"unsupported tensor-parallel dim {dim}")


def _copy_parallel_tensor_from_index(dst: torch.Tensor, index: SafetensorsIndex, tensor_key: str, dim: int, rank: int, world_size: int) -> None:
    """Copy one TP shard from safetensors without loading the full tensor."""

    shard = _read_tp_shard(index, tensor_key, dim=dim, rank=rank, world_size=world_size)
    dst.copy_(shard.to(dtype=dst.dtype))


def _copy_vocab_shard_from_index(dst: torch.Tensor, index: SafetensorsIndex, tensor_key: str, rank: int, world_size: int) -> None:
    """Copy this rank's vocab shard directly from safetensors."""

    shard = _read_tp_shard(index, tensor_key, dim=0, rank=rank, world_size=world_size)
    dst.copy_(shard.to(dtype=dst.dtype))


def _read_tp_shard(index: SafetensorsIndex, tensor_key: str, dim: int, rank: int, world_size: int) -> torch.Tensor:
    """Read only this rank's TP slice from one safetensors tensor."""

    if world_size == 1:
        return index.get_tensor(tensor_key)
    filename = index.weight_map.get(tensor_key)
    if filename is None:
        raise KeyError(f"missing HF weight {tensor_key}")
    with safe_open(index.model_path / filename, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_key)
        shape = tensor_slice.get_shape()
        start, end = _shard_range(shape[dim], rank, world_size)
        slices = [slice(None)] * len(shape)
        slices[dim] = slice(start, end)
        return tensor_slice[tuple(slices)]


def _read_explicit_shard(index: SafetensorsIndex, tensor_key: str, dim: int, start: int, end: int) -> torch.Tensor:
    """Read an explicit slice from one safetensors tensor."""

    filename = index.weight_map.get(tensor_key)
    if filename is None:
        raise KeyError(f"missing HF weight {tensor_key}")
    with safe_open(index.model_path / filename, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_key)
        slices = [slice(None)] * len(tensor_slice.get_shape())
        slices[dim] = slice(start, end)
        return tensor_slice[tuple(slices)]


def load_merged_column_spec(module: nn.Module, index: SafetensorsIndex, prefix: str, spec: MergedColumnSpec, rank: int, world_size: int) -> None:
    """Load multiple HF column tensors into one fused local column module."""

    dst = attr_path(module, spec.dst_attr)
    merged_module = attr_path(module, spec.dst_attr.rsplit(".", 1)[0])
    shard_ranges = getattr(merged_module, "shard_ranges", None)
    tensor_keys = keys(spec.keys, prefix)
    if shard_ranges is not None and len(shard_ranges) == len(tensor_keys):
        copy_ranged_merged_column_from_index(dst, index, tensor_keys, shard_ranges)
        return
    copy_merged_column_from_index(dst, index, tensor_keys, rank, world_size)


def load_k_shared_qkv_column_spec(module: nn.Module, index: SafetensorsIndex, prefix: str, spec: KSharedQKVColumnSpec, rank: int, world_size: int) -> None:
    """Load QKV where late layers may share K/V or omit them entirely.

    When K is present, V is reused from K if missing and the three tensors are
    packed via the module's shard ranges. When K is absent, only Q is loaded
    into the leading rows of the fused destination and the rest is zeroed.
    """

    q_key = key(spec.q_key, prefix)
    k_key = key(spec.k_key, prefix)
    v_key = key(spec.v_key, prefix)
    dst = attr_path(module, spec.dst_attr)
    # The parent module exposes shard_ranges for non-uniform head splits.
    qkv_module = attr_path(module, spec.dst_attr.rsplit(".", 1)[0])
    shard_ranges = getattr(qkv_module, "shard_ranges", None)
    srcs = [index.get_tensor(q_key)]
    if k_key in index.weight_map:
        k = index.get_tensor(k_key)
        v = index.get_tensor(v_key) if v_key in index.weight_map else k
        srcs.extend((k, v))
        if shard_ranges is not None:
            copy_ranged_merged_column(dst, srcs, shard_ranges)
            return
        copy_merged_column(dst, srcs, rank, world_size)
        return
    # No K shared: keep dst zero and only fill the Q portion.
    dst.zero_()
    q = srcs[0]
    start, end = shard_ranges[0] if shard_ranges is not None else _shard_range(q.shape[0], rank, world_size)
    dst[: end - start].copy_(q[start:end].to(dtype=dst.dtype))


def load_merged_optional_bias_spec(
    module: nn.Module,
    index: SafetensorsIndex,
    prefix: str,
    spec: MergedColumnSpec,
    rank: int,
    world_size: int,
) -> None:
    """Convenience wrapper that loads merged biases when both ends are present."""

    dst = attr_path(module, spec.dst_attr)
    copy_merged_optional_bias(dst, index, keys(spec.keys, prefix), rank, world_size)


def save_split_column_spec(tensors: dict[str, torch.Tensor | None], module: nn.Module, prefix: str, spec: SplitColumnSpec) -> None:
    """Split a fused local column into its component HF tensors and gather them."""

    parts = split_local_tensors(attr_path(module, spec.src_attr), attr_path(module, spec.size_attr))
    gathered = gather_tensor_parallel_column_tensors(list(parts))
    for template, tensor in zip(spec.keys, gathered, strict=True):
        tensors[key(template, prefix)] = tensor


def save_ranged_split_column_spec(tensors: dict[str, torch.Tensor | None], module: nn.Module, prefix: str, spec: RangedSplitColumnSpec) -> None:
    """Save split QKV tensors whose local K/V ranges may be replicated."""

    parts = split_local_tensors(attr_path(module, spec.src_attr), attr_path(module, spec.size_attr))
    qkv_module = attr_path(module, spec.src_attr.rsplit(".", 1)[0])
    ranges = getattr(qkv_module, "shard_ranges")
    global_sizes = getattr(qkv_module, "out_features")
    gathered = _gather_ranged_column_tensors(list(parts), ranges, global_sizes)
    for template, tensor in zip(spec.keys, gathered, strict=True):
        tensors[key(template, prefix)] = tensor


def _gather_ranged_column_tensors(
    tensors: list[torch.Tensor],
    ranges: tuple[tuple[int, int], ...],
    global_sizes: tuple[int, ...],
) -> list[torch.Tensor | None]:
    """Gather column tensors when each rank reports an explicit row range."""

    ctx = get_tp_context()
    if ctx.dp_rank != 0:
        return [None for _ in tensors]
    outputs: list[torch.Tensor | None] = []
    for tensor, (start, end), global_size in zip(tensors, ranges, global_sizes, strict=True):
        local = tensor.detach().contiguous()
        if ctx.world_size == 1:
            outputs.append(_tensor_to_cpu(local))
            continue
        gathered = torch.empty((ctx.world_size, *local.shape), dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(gathered, local, group=ctx.group)
        all_ranges: list[tuple[int, int] | None] = [None for _ in range(ctx.world_size)]
        dist.all_gather_object(all_ranges, (start, end), group=ctx.group)
        global_tensor = torch.empty((global_size, *local.shape[1:]), dtype=local.dtype, device=local.device)
        for rank in range(ctx.world_size):
            rank_range = all_ranges[rank]
            if rank_range is None:
                raise RuntimeError(f"missing TP shard range for rank {rank}")
            range_start, range_end = rank_range
            global_tensor[range_start:range_end].copy_(gathered[rank, : range_end - range_start])
        outputs.append(_tensor_to_cpu(global_tensor.contiguous()))
    return outputs


def load_row_parallel_tensors(
    module: nn.Module,
    index: SafetensorsIndex,
    prefix: str,
    specs: tuple[ParallelTensorSpec, ...],
    rank: int,
    world_size: int,
) -> None:
    """Copy row-sharded tensors (dim 1) from HF into the live module."""

    for spec in specs:
        _copy_parallel_tensor_from_index(attr_path(module, spec.attr), index, key(spec.key, prefix), dim=1, rank=rank, world_size=world_size)


def save_column_linear(tensors: dict[str, torch.Tensor | None], prefix: str, module: nn.Module) -> None:
    """Stage a column-parallel linear's weight and optional bias."""

    tensors[f"{prefix}.weight"] = gather_tensor_parallel_tensor(module.weight, dim=0)
    if module.bias is not None:
        tensors[f"{prefix}.bias"] = gather_tensor_parallel_tensor(module.bias, dim=0)


def save_column_tensors(tensors: dict[str, torch.Tensor | None], entries: list[tuple[str, torch.Tensor]]) -> None:
    """Gather a batch of column tensors, grouping like-shaped entries for one collective."""

    # Group by trailing-dim + dtype + device so each compatible group can be
    # fused into a single all-gather payload via `_ColumnGatherTask`.
    groups: dict[tuple[torch.Size, torch.dtype, torch.device], list[tuple[str, torch.Tensor]]] = {}
    for entry in entries:
        _, tensor = entry
        groups.setdefault((tensor.shape[1:], tensor.dtype, tensor.device), []).append(entry)
    for group_entries in groups.values():
        gathered = gather_tensor_parallel_column_tensors([tensor for _, tensor in group_entries])
        for (key, _), tensor in zip(group_entries, gathered, strict=True):
            tensors[key] = tensor


def save_row_linear(tensors: dict[str, torch.Tensor | None], prefix: str, module: nn.Module) -> None:
    """Stage a row-parallel linear: gather weight on dim 1, replicate the bias."""

    tensors[f"{prefix}.weight"] = gather_tensor_parallel_tensor(module.weight, dim=1)
    if module.bias is not None:
        tensors[f"{prefix}.bias"] = rank0_tensor(module.bias)


def save_dense_mlp(tensors: dict[str, torch.Tensor | None], prefix: str, mlp: nn.Module) -> None:
    """Gather a SwiGLU dense MLP (gate/up column-parallel, down row-parallel)."""

    # Gate and up share the same shape so they can be fused into one gather.
    gate_weight, up_weight = gather_tensor_parallel_column_tensors([mlp.gate_proj.weight, mlp.up_proj.weight])
    tensors[f"{prefix}.gate_proj.weight"] = gate_weight
    tensors[f"{prefix}.up_proj.weight"] = up_weight
    if mlp.gate_proj.bias is not None and mlp.up_proj.bias is not None:
        gate_bias, up_bias = gather_tensor_parallel_column_tensors([mlp.gate_proj.bias, mlp.up_proj.bias])
        tensors[f"{prefix}.gate_proj.bias"] = gate_bias
        tensors[f"{prefix}.up_proj.bias"] = up_bias
    save_row_linear(tensors, f"{prefix}.down_proj", mlp.down_proj)


def copy_dense_mlp(mlp: nn.Module, index: SafetensorsIndex, prefix: str, rank: int, world_size: int) -> None:
    """Copy gate/up/down weights of a dense MLP from HF safetensors into TP shards."""

    _copy_column(mlp.gate_proj.weight, index.get_tensor(f"{prefix}.gate_proj.weight"), rank, world_size)
    _copy_column(mlp.up_proj.weight, index.get_tensor(f"{prefix}.up_proj.weight"), rank, world_size)
    _copy_row(mlp.down_proj.weight, index.get_tensor(f"{prefix}.down_proj.weight"), rank, world_size)


def load_dense_or_moe_spec(module: nn.Module, index: SafetensorsIndex, prefix: str, spec: DenseOrMoeSpec, rank: int, world_size: int) -> None:
    """Dispatch MLP load to dense or MoE based on the live module's structure."""

    mlp = attr_path(module, spec.attr)
    mlp_prefix = f"{prefix}.mlp"
    # MoE layers expose both `gate` and `num_experts`; dense layers do not.
    if getattr(mlp, "gate", None) is not None and getattr(mlp, "num_experts", None) is not None:
        load_moe_spec(mlp, index, mlp_prefix, spec.moe, rank, world_size)
        return
    copy_dense_mlp(mlp, index, mlp_prefix, rank, world_size)


def save_dense_or_moe_spec(tensors: dict[str, torch.Tensor | None], module: nn.Module, prefix: str, spec: DenseOrMoeSpec) -> None:
    """Mirror of `load_dense_or_moe_spec` for checkpoint saving."""

    mlp = attr_path(module, spec.attr)
    mlp_prefix = f"{prefix}.mlp"
    if getattr(mlp, "gate", None) is not None and getattr(mlp, "num_experts", None) is not None:
        save_moe_spec(tensors, mlp_prefix, mlp, spec.moe)
        return
    save_dense_mlp(tensors, mlp_prefix, mlp)


def load_attention_spec(module: nn.Module, index: SafetensorsIndex, prefix: str, spec: AttentionSpec, rank: int, world_size: int) -> None:
    """Load attention block; handles MLA, fused QKV, split QKV, gated-attn variants."""

    attn = attr_path(module, spec.attr)
    # Different HF families name the attention submodule differently; pick the
    # first match that actually has tensors under it.
    attn_prefix = existing_module_prefix(index, prefix, spec.checkpoint_names)
    if getattr(attn, "kv_lora_rank", None) is not None:
        # Multi-head latent attention has its own Q + KV-projection layout.
        copy_mla_attention(attn, index, attn_prefix, rank, world_size)
        return
    qkv_prefix = optional_existing_prefix(index, attn_prefix, ("query_key_value", "qkv"))
    dense_prefix = existing_prefix(index, attn_prefix, ("dense", "o_proj"))
    if qkv_prefix is not None:
        # Pre-fused QKV in HF -> just reshard along TP heads.
        copy_attention_qkv(attn.query_key_value.weight, index.get_tensor(f"{qkv_prefix}.weight"), attn, rank, world_size)
        if attn.query_key_value.bias is not None and f"{qkv_prefix}.bias" in index.weight_map:
            copy_attention_qkv(attn.query_key_value.bias, index.get_tensor(f"{qkv_prefix}.bias"), attn, rank, world_size)
    else:
        # Separate Q/K/V tensors in HF -> assemble into the fused local layout.
        copy_split_qkv(attn, index, attn_prefix, rank, world_size)
    _copy_row(attn.dense.weight, index.get_tensor(f"{dense_prefix}.weight"), rank, world_size)
    if attn.dense.bias is not None and f"{dense_prefix}.bias" in index.weight_map:
        attn.dense.bias.copy_(index.get_tensor(f"{dense_prefix}.bias").to(dtype=attn.dense.bias.dtype))
    if getattr(attn, "query_layernorm", None) is not None:
        load_replicated_tensors(attn, index, attn_prefix, spec.qk_norms)
    if getattr(attn, "g_proj", None) is not None:
        # Gated linear attention variants carry an extra g_proj + g_norm pair.
        if f"{attn_prefix}.g_proj.weight" in index.weight_map:
            _copy_column(attn.g_proj.weight, index.get_tensor(f"{attn_prefix}.g_proj.weight"), rank, world_size)
        if f"{attn_prefix}.g_norm.weight" in index.weight_map:
            copy_g_norm_weight(attn.g_norm.weight, index.get_tensor(f"{attn_prefix}.g_norm.weight"), rank, world_size)


def save_attention_spec(
    tensors: dict[str, torch.Tensor | None],
    module: nn.Module,
    prefix: str,
    spec: AttentionSpec,
    context: object | None,
) -> None:
    """Save attention block; mirror of `load_attention_spec` with variant dispatch."""

    attn = attr_path(module, spec.attr)
    attn_prefix = f"{prefix}.attention"
    if getattr(attn, "g_proj", None) is not None:
        # Gated linear attention variant: fused QKV + extra g_proj/g_norm.
        save_fused_qkv_tensor(tensors, f"{attn_prefix}.query_key_value", attn)
        save_column_tensors(
            tensors,
            [
                (f"{attn_prefix}.g_proj.weight", attn.g_proj.weight),
            ],
        )
        if attn.g_proj.bias is not None:
            tensors[f"{attn_prefix}.g_proj.bias"] = gather_tensor_parallel_tensor(attn.g_proj.bias, dim=0)
        if attn.query_layernorm is not None:
            save_replicated_tensors(tensors, attn, attn_prefix, spec.qk_norms)
        # g_norm is a column-style tensor; defer its gather so it can share a
        # collective with sibling tensors of the same shape later.
        delayed_column_tensors(context).append((f"{attn_prefix}.g_norm.weight", attn.g_norm.weight))
        save_parallel_tensors(tensors, attn, attn_prefix, spec.linear_dense_rows)
        if attn.dense.bias is not None:
            save_replicated_tensors(tensors, attn, attn_prefix, (spec.linear_dense_bias,))
        return

    if getattr(attn, "kv_lora_rank", None) is not None:
        # MLA save: q_proj + KV down-projection + KV layer norm + kv_b_proj.
        column_entries = [
            (f"{attn_prefix}.q_proj.weight", attn.q_proj.weight),
            (f"{attn_prefix}.kv_b_proj.weight", attn.kv_b_proj.weight),
        ]
        if attn.q_proj.bias is not None:
            column_entries.append((f"{attn_prefix}.q_proj.bias", attn.q_proj.bias))
        if attn.kv_b_proj.bias is not None:
            column_entries.append((f"{attn_prefix}.kv_b_proj.bias", attn.kv_b_proj.bias))
        save_column_tensors(tensors, column_entries)
        tensors[f"{attn_prefix}.kv_a_proj_with_mqa.weight"] = rank0_tensor(attn.kv_a_proj_with_mqa.weight)
        tensors[f"{attn_prefix}.kv_a_layernorm.weight"] = rank0_tensor(attn.kv_a_layernorm.weight)
    else:
        # Standard attention: fused QKV with q/kv/kv splits.
        save_fused_qkv_tensor(tensors, f"{attn_prefix}.query_key_value", attn)
        if attn.query_layernorm is not None:
            save_replicated_tensors(tensors, attn, attn_prefix, spec.qk_norms)
    save_parallel_tensors(tensors, attn, attn_prefix, spec.dense_rows)
    if attn.dense.bias is not None:
        save_replicated_tensors(tensors, attn, attn_prefix, (spec.dense_bias,))


def save_fused_qkv_tensor(tensors: dict[str, torch.Tensor | None], prefix: str, attn: nn.Module) -> None:
    """Gather a fused QKV weight + bias with explicit q/kv/kv row splits."""

    # Compute per-head row counts so the gather task knows how to slice back
    # into Q, K, V along dim 0 after the all-gather.
    q_rows = attn.local_heads * attn.head_dim
    kv_rows = getattr(attn, "local_kv_heads", attn.local_heads) * attn.head_dim
    split_sizes = [q_rows, kv_rows, kv_rows]
    tensors[f"{prefix}.weight"] = gather_tensor_parallel_split_column_tensor(attn.query_key_value.weight, split_sizes)
    if attn.query_key_value.bias is not None:
        tensors[f"{prefix}.bias"] = gather_tensor_parallel_split_column_tensor(attn.query_key_value.bias, split_sizes)


def copy_mla_attention(attn: nn.Module, index: SafetensorsIndex, prefix: str, rank: int, world_size: int) -> None:
    """Load Multi-head Latent Attention blocks (q_proj + KV LoRA + kv_b_proj + dense)."""

    _copy_column(attn.q_proj.weight, index.get_tensor(f"{prefix}.q_proj.weight"), rank, world_size)
    # kv_a_proj_with_mqa and kv_a_layernorm are replicated across TP ranks.
    attn.kv_a_proj_with_mqa.weight.copy_(index.get_tensor(f"{prefix}.kv_a_proj_with_mqa.weight").to(dtype=attn.kv_a_proj_with_mqa.weight.dtype))
    attn.kv_a_layernorm.weight.copy_(index.get_tensor(f"{prefix}.kv_a_layernorm.weight").to(dtype=attn.kv_a_layernorm.weight.dtype))
    _copy_column(attn.kv_b_proj.weight, index.get_tensor(f"{prefix}.kv_b_proj.weight"), rank, world_size)
    dense_prefix = existing_prefix(index, prefix, ("dense", "o_proj"))
    _copy_row(attn.dense.weight, index.get_tensor(f"{dense_prefix}.weight"), rank, world_size)


def copy_attention_qkv(dst: torch.Tensor, src: torch.Tensor, attn: nn.Module, rank: int, world_size: int) -> None:
    """Shard a pre-fused HF QKV tensor across TP ranks honoring q/kv head counts."""

    head_dim = attn.head_dim
    q_rows = attn.num_heads * head_dim
    kv_heads = getattr(attn, "num_kv_heads", attn.num_heads)
    kv_rows = kv_heads * head_dim
    # Split first into the Q/K/V regions so each can be sharded independently
    # (Q and KV may have different head counts under GQA/MQA).
    q_src, k_src, v_src = src.split([q_rows, kv_rows, kv_rows], dim=0)
    q_start, q_end = _shard_range(q_rows, rank, world_size)
    k_start, k_end = _shard_range(kv_rows, rank, world_size)
    dst.copy_(torch.cat([q_src[q_start:q_end], k_src[k_start:k_end], v_src[k_start:k_end]], dim=0).to(dtype=dst.dtype))


def copy_split_qkv(attn: nn.Module, index: SafetensorsIndex, prefix: str, rank: int, world_size: int) -> None:
    """Assemble separate HF q/k/v tensors into the fused local QKV weight."""

    q_key = existing_prefix(index, prefix, ("q_proj", "query"))
    k_key = existing_prefix(index, prefix, ("k_proj", "key"))
    v_key = existing_prefix(index, prefix, ("v_proj", "value"))
    q = index.get_tensor(f"{q_key}.weight")
    k = index.get_tensor(f"{k_key}.weight")
    v = index.get_tensor(f"{v_key}.weight")
    q_start, q_end = _shard_range(q.shape[0], rank, world_size)
    k_start, k_end = _shard_range(k.shape[0], rank, world_size)
    attn.query_key_value.weight.copy_(torch.cat([q[q_start:q_end], k[k_start:k_end], v[k_start:k_end]], dim=0).to(dtype=attn.query_key_value.weight.dtype))


def copy_column_vector(dst: torch.Tensor, src: torch.Tensor, rank: int, world_size: int) -> None:
    """Generic column-vector shard copy along dim 0."""

    start, end = _shard_range(src.shape[0], rank, world_size)
    dst.copy_(src[start:end].to(dtype=dst.dtype))


def copy_g_norm_weight(dst: torch.Tensor, src: torch.Tensor, rank: int, world_size: int) -> None:
    """Copy a gated-attention g_norm, reshaping 1D HF tensors to 2D when needed."""

    if dst.ndim == 2 and src.ndim == 1:
        # HF stores g_norm flat but the live module expects [heads, head_dim].
        start, end = _shard_range(src.shape[0], rank, world_size)
        dst.copy_(src[start:end].view_as(dst).to(dtype=dst.dtype))
        return
    if dst.ndim == 2 and src.ndim == 2:
        start, end = _shard_range(src.shape[0], rank, world_size)
        dst.copy_(src[start:end].to(dtype=dst.dtype))
        return
    copy_column_vector(dst, src, rank, world_size)


def existing_prefix(index: SafetensorsIndex, base: str, names: tuple[str, ...]) -> str:
    """Return the first `{base}.{name}` with a weight in the index, or raise."""

    prefix = optional_existing_prefix(index, base, names)
    if prefix is not None:
        return prefix
    keys = ", ".join(f"{base}.{name}.weight" for name in names)
    raise KeyError(f"missing any of: {keys}")


def optional_existing_prefix(index: SafetensorsIndex, base: str, names: tuple[str, ...]) -> str | None:
    """Return the first `{base}.{name}` with a `.weight` in the index, or None."""

    for name in names:
        prefix = f"{base}.{name}"
        if f"{prefix}.weight" in index.weight_map:
            return prefix
    return None


def existing_module_prefix(index: SafetensorsIndex, base: str, names: tuple[str, ...]) -> str:
    """Return the first `{base}.{name}` that has any keys under it in the index."""

    for name in names:
        prefix = f"{base}.{name}"
        if index.keys_for_prefix(f"{prefix}."):
            return prefix
    keys = ", ".join(f"{base}.{name}.*" for name in names)
    raise KeyError(f"missing any of: {keys}")


def delayed_column_tensors(context: object | None) -> list[tuple[str, torch.Tensor]]:
    """Typed accessor for the per-save list of delayed column entries."""

    if not isinstance(context, list):
        raise TypeError("checkpoint save context must be a list of delayed column tensors")
    return context


def load_moe_spec(module: nn.Module, index: SafetensorsIndex, prefix: str, spec: MoeSpec, rank: int, world_size: int) -> None:
    """Load MoE gate, optional expert bias, all expert weights, and shared experts."""

    gate_weight = attr_path(module, spec.gate_weight_attr)
    gate_weight.copy_(index.get_tensor(key(spec.gate_weight_key, prefix)).to(dtype=gate_weight.dtype))
    if spec.expert_bias_key is not None and spec.expert_bias_attr is not None:
        bias_key = key(spec.expert_bias_key, prefix)
        if bias_key in index.weight_map:
            expert_bias = attr_path(module, spec.expert_bias_attr)
            expert_bias.copy_(index.get_tensor(bias_key).to(dtype=expert_bias.dtype))
            # Some models keep a per-rank cached copy of the expert bias.
            if spec.local_expert_bias_attr is not None:
                attr_path(module, spec.local_expert_bias_attr).copy_(expert_bias)

    experts = attr_path(module, spec.experts_attr)
    # Iterate every expert; the expert module decides how to apply EP sharding.
    for expert_id in range(int(attr_path(module, spec.num_experts_attr))):
        experts.copy_expert(
            expert_id,
            index.get_tensor(expert_key(spec.expert_gate_key, prefix, expert_id)),
            index.get_tensor(expert_key(spec.expert_up_key, prefix, expert_id)),
            index.get_tensor(expert_key(spec.expert_down_key, prefix, expert_id)),
            rank,
            world_size,
        )

    if spec.shared_experts_attr is not None and spec.shared_experts_prefix is not None:
        shared_experts = attr_path(module, spec.shared_experts_attr)
        if shared_experts is not None:
            copy_dense_mlp(shared_experts, index, key(spec.shared_experts_prefix, prefix), rank, world_size)


def save_moe_spec(tensors: dict[str, torch.Tensor | None], prefix: str, module: nn.Module, spec: MoeSpec) -> None:
    """Save MoE gate, expert bias, all routed experts, and any shared experts."""

    tensors[key(spec.gate_weight_key, prefix)] = rank0_tensor(attr_path(module, spec.gate_weight_attr))
    if spec.expert_bias_key is not None and spec.expert_bias_attr is not None:
        tensors[key(spec.expert_bias_key, prefix)] = rank0_tensor(attr_path(module, spec.expert_bias_attr))

    # One TP/EP all-gather collects every expert's gate/up/down at once.
    full_weights = gather_moe_expert_weights(attr_path(module, spec.experts_attr))
    if full_weights is None:
        return
    gate_weights, up_weights, down_weights = full_weights
    # Sharded ownership: each tensor is owned by one rank (via CRC32 of the
    # key), so non-owners stage `None` to skip the D2H copy entirely.
    for expert_id in range(gate_weights.shape[0]):
        gate_key = expert_key(spec.expert_gate_key, prefix, expert_id)
        up_key = expert_key(spec.expert_up_key, prefix, expert_id)
        down_key = expert_key(spec.expert_down_key, prefix, expert_id)
        tensors[gate_key] = _tensor_to_cpu(gate_weights[expert_id].contiguous()) if _owns_checkpoint_tensor(gate_key) else None
        tensors[up_key] = _tensor_to_cpu(up_weights[expert_id].contiguous()) if _owns_checkpoint_tensor(up_key) else None
        tensors[down_key] = _tensor_to_cpu(down_weights[expert_id].contiguous()) if _owns_checkpoint_tensor(down_key) else None

    if spec.shared_experts_attr is not None and spec.shared_experts_prefix is not None:
        shared_experts = attr_path(module, spec.shared_experts_attr)
        if shared_experts is not None:
            save_dense_mlp(tensors, key(spec.shared_experts_prefix, prefix), shared_experts)


@torch.no_grad()
def gather_moe_expert_weights(experts: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Gather local expert shards with one TP collective.

    `experts.full_expert_weights()` gathers gate/up/down in
    three separate collectives. Checkpoint save touches this for every MoE
    layer, so packing the three local tensors into one flat payload avoids two
    extra synchronization points per layer while preserving the same output
    layout on TP rank0.
    """
    ctx = get_tp_context()
    if ctx.dp_rank != 0:
        return None
    gate_weights, up_weights, down_weights = experts.expert_weights()
    gate = torch.stack(gate_weights, dim=0).detach().contiguous()
    up = torch.stack(up_weights, dim=0).detach().contiguous()
    down = torch.stack(down_weights, dim=0).detach().contiguous()
    if ctx.world_size == 1:
        return gate, up, down

    local_numel = (gate.numel(), up.numel(), down.numel())
    flat = torch.cat([gate.reshape(-1), up.reshape(-1), down.reshape(-1)]).contiguous()
    gathered = torch.empty((ctx.world_size, flat.numel()), dtype=flat.dtype, device=flat.device)
    dist.all_gather_into_tensor(gathered, flat, group=ctx.group)
    gate_flat, up_flat, down_flat = gathered.split(local_numel, dim=1)
    gate_full = gate_flat.reshape((ctx.world_size * gate.shape[0], *gate.shape[1:])).contiguous()
    up_full = up_flat.reshape((ctx.world_size * up.shape[0], *up.shape[1:])).contiguous()
    down_full = down_flat.reshape((ctx.world_size * down.shape[0], *down.shape[1:])).contiguous()
    return gate_full, up_full, down_full
