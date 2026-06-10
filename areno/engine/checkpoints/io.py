"""Low-level checkpoint IO primitives.

This module intentionally has no model-specific knowledge. It handles:
- resolving HF/local model paths,
- lazy safetensors reads with optional NCCL broadcast,
- tensor-parallel gather tasks for saving,
- distributed HF safetensors shard writing.

Model layouts live in `areno.engine.checkpoints.<model>` and are executed by
`areno.engine.checkpoints.common`.
"""

from __future__ import annotations

import json
import os
import shutil
import zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from areno.engine.parallel.context import get_tp_context
from areno.engine.layers.linear import _shard_range

_ASYNC_CPU_COPY_MAX_BYTES = 2 * 1024**3
_PENDING_CPU_COPY_BUCKETS: list[tuple[torch.cuda.Stream, int, torch.Tensor]] = []

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class _CheckpointTensorTask:
    """Lazy tensor materialization task used by checkpoint saving."""

    def materialize(self, key: str) -> torch.Tensor | None:
        raise NotImplementedError


class _ReplicatedTensorTask(_CheckpointTensorTask):
    """Wraps a fully-replicated tensor; only DP/TP rank 0 actually materializes."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.detach().contiguous()

    def materialize(self, key: str) -> torch.Tensor | None:
        del key
        ctx = get_tp_context()
        # Replicated tensors are identical across ranks, so only the global
        # rank 0 writes them to disk.
        return _tensor_to_cpu(self.tensor) if ctx.dp_rank == 0 and ctx.rank == 0 else None


class _TensorParallelGatherTask(_CheckpointTensorTask):
    """Lazy single-tensor TP all-gather, cached for reuse across keys."""

    def __init__(self, local: torch.Tensor, dim: int):
        self.local = local.detach().contiguous()
        self.dim = dim
        self._gathered: torch.Tensor | None = None

    def materialize(self, key: str) -> torch.Tensor | None:
        ctx = get_tp_context()
        if ctx.world_size == 1:
            # Single TP rank: no collective needed; copy directly to CPU.
            return _tensor_to_cpu(self.local) if _owns_checkpoint_tensor(key) else None
        if self._gathered is None:
            self._gathered = _all_gather_tensor_parallel(self.local)
        return _gathered_tensor_to_cpu(self._gathered, dim=self.dim) if _owns_checkpoint_tensor(key) else None


class _ColumnGatherTask:
    """Fused all-gather for a batch of column tensors that share shape suffix.

    The local list is concatenated along dim 0, gathered once, then split back
    so that a single NCCL collective covers all of them. The result for each
    original tensor is exposed via `_ColumnGatherResult.materialize`.
    """

    def __init__(self, locals_: list[torch.Tensor]):
        self.locals = [tensor.detach().contiguous() for tensor in locals_]
        self.row_sizes = [tensor.shape[0] for tensor in self.locals]
        self._gathered: torch.Tensor | None = None

    def result(self, index: int) -> _CheckpointTensorTask:
        return _ColumnGatherResult(self, index)

    def materialize(self, key: str, index: int) -> torch.Tensor | None:
        ctx = get_tp_context()
        if ctx.world_size == 1:
            return _tensor_to_cpu(self.locals[index]) if _owns_checkpoint_tensor(key) else None
        if self._gathered is None:
            # Concatenate once, gather once; reused for every result index.
            self._gathered = _all_gather_tensor_parallel(torch.cat(self.locals, dim=0).contiguous())
        if not _owns_checkpoint_tensor(key):
            return None
        # Slice out the (world_size, row_size, *trailing) block for this index.
        offset = sum(self.row_sizes[:index])
        row_size = self.row_sizes[index]
        shard = self._gathered[:, offset : offset + row_size]
        full_shape = (ctx.world_size * row_size, *shard.shape[2:])
        return _tensor_to_cpu(shard.reshape(full_shape).contiguous())


class _ColumnGatherResult(_CheckpointTensorTask):
    """Per-tensor view onto a shared `_ColumnGatherTask`."""

    def __init__(self, task: _ColumnGatherTask, index: int):
        self.task = task
        self.index = index

    def materialize(self, key: str) -> torch.Tensor | None:
        return self.task.materialize(key, self.index)


class _SplitColumnGatherTask(_CheckpointTensorTask):
    """All-gather a fused tensor and re-split into the original sub-tensors.

    Used for fused QKV: a single weight is stored as one packed tensor that we
    must gather across TP and then return as one HF tensor (re-concatenated
    after each sub-tensor is re-shaped to the full TP size).
    """

    def __init__(self, parts: list[torch.Tensor]):
        self.parts = [part.detach().contiguous() for part in parts]
        self.row_sizes = [part.shape[0] for part in self.parts]
        self._gathered: torch.Tensor | None = None

    def materialize(self, key: str) -> torch.Tensor | None:
        ctx = get_tp_context()
        if ctx.world_size == 1:
            return _tensor_to_cpu(torch.cat(self.parts, dim=0).contiguous()) if _owns_checkpoint_tensor(key) else None
        if self._gathered is None:
            self._gathered = _all_gather_tensor_parallel(torch.cat(self.parts, dim=0).contiguous())
        if not _owns_checkpoint_tensor(key):
            return None
        # Re-split into Q/K/V (or other named parts) before concatenating into
        # the global HF layout. Each sub-tensor is gathered across TP rows.
        outputs = []
        offset = 0
        for row_size in self.row_sizes:
            shard = self._gathered[:, offset : offset + row_size]
            full_shape = (ctx.world_size * row_size, *shard.shape[2:])
            outputs.append(shard.reshape(full_shape).contiguous())
            offset += row_size
        return _tensor_to_cpu(torch.cat(outputs, dim=0).contiguous())


class CheckpointTensorStore(dict[str, torch.Tensor | _CheckpointTensorTask | None]):
    """Dict-like staging area for distributed HF safetensors writing.

    Values may be eager tensors or lazy gather tasks. Materialization is delayed
    until shard writing so tensor-parallel all-gathers and D2H copies happen as
    late as possible and only on ranks that own a target checkpoint shard.
    """

    def __init__(self):
        super().__init__()

    def __setitem__(self, key: str, value: torch.Tensor | _CheckpointTensorTask | None) -> None:
        # Lazy tasks are materialized at assignment time so that any TP
        # all-gathers happen during save_layer_specs while peer ranks are
        # synchronized.
        if isinstance(value, _CheckpointTensorTask):
            value = value.materialize(key)
        super().__setitem__(key, value)

    def close_progress(self) -> None:
        return


def _checkpoint_tensor_owner(key: str) -> int:
    """Deterministically assign one TP rank to own each checkpoint tensor."""

    # CRC32 hashing balances ownership across TP ranks without explicit
    # coordination, so every rank computes the same owner for a given key.
    return zlib.crc32(key.encode("utf-8")) % get_tp_context().world_size


def _owns_checkpoint_tensor(key: str) -> bool:
    """True when this process should write the file shard for `key`."""

    ctx = get_tp_context()
    return ctx.dp_rank == 0 and ctx.rank == _checkpoint_tensor_owner(key)


def resolve_model_path(model: str | None) -> str | None:
    """Return a local checkpoint directory, downloading via HF Hub if needed."""

    if model is None:
        return None
    path = Path(model)
    if path.exists():
        return str(path)
    # Fall back to huggingface_hub for non-existent paths (assumed to be repo ids).
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"{model!r} is not a local path and huggingface_hub is unavailable"
        ) from exc
    return snapshot_download(model)


class SafetensorsIndex:
    """Lazy safetensors reader with optional layer-wise cache.

    Loaders prefetch one layer, copy TP shards into the live model, then drop
    that layer's tensors so loading large checkpoints does not keep the whole
    model in CPU memory.
    """

    def __init__(self, model_path: str | Path, max_workers: int = 8, progress: bool | None = None):
        self.model_path = Path(model_path)
        self.max_workers = max_workers
        self.progress = progress if progress is not None else os.environ.get("ARENO_CKPT_PROGRESS", "1") == "1"
        self._cache: dict[str, torch.Tensor] = {}
        self._progress_bar = None
        self._progress_total: int | None = None
        self._progress_unit = "tensor"
        self._manual_progress = False
        self._loaded_keys: set[str] = set()
        index_path = self.model_path / "model.safetensors.index.json"
        self._checkpoint_writer = None
        if index_path.exists():
            # Preferred path: a single index file maps every key to its shard.
            with index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.weight_map = dict(data["weight_map"])
            metadata = data.get("metadata", {})
            self._checkpoint_writer = metadata.get("areno_checkpoint_writer")
        else:
            # Fallback: enumerate every safetensors file and stitch a weight_map.
            files = sorted(self.model_path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"no safetensors files found in {self.model_path}")
            self.weight_map = {}
            for file in files:
                with safe_open(file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        self.weight_map[key] = file.name
        # Checkpoints produced by this writer already have one shard per TP
        # rank and would deadlock NCCL broadcast (different ranks read
        # different files), so disable NCCL transfer in that case.
        self._disable_nccl_transfer = self._checkpoint_writer == "distributed_tp"

    def get_tensor(self, key: str) -> torch.Tensor:
        """Read one tensor, using the prefetch cache when available."""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        filename = self.weight_map.get(key)
        if filename is None:
            raise KeyError(f"missing HF weight {key}")
        # NCCL broadcast lets the disk read happen only on rank 0 and avoids
        # filesystem contention across the cluster.
        tensor = self._get_tensor_nccl(key, filename) if self._use_nccl_transfer() else self._read_tensor(key, filename)
        self._advance_progress([key])
        return tensor

    def prefetch(self, keys: list[str] | tuple[str, ...] | set[str], *, optional: bool = False) -> None:
        """Load a set of keys, grouped by safetensors file for fewer opens."""
        # Deduplicate while preserving order; skip anything already cached.
        if isinstance(keys, set):
            keys = sorted(keys)
        keys = [key for key in dict.fromkeys(keys) if key not in self._cache]
        if not keys:
            return
        for key in keys:
            filename = self.weight_map.get(key)
            if filename is None:
                if optional:
                    continue
                raise KeyError(f"missing HF weight {key}")
        if self._use_nccl_transfer():
            # NCCL path: still go through get_tensor for the broadcast handshake.
            for key in keys:
                if key in self.weight_map:
                    self._cache[key] = self.get_tensor(key)
            return
        # Group keys by file so each safetensors file is opened only once and
        # the per-file reads can run on a thread pool.
        grouped: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            filename = self.weight_map.get(key)
            if filename is not None:
                grouped[filename].append(key)
        if not grouped:
            return
        workers = min(self.max_workers, len(grouped))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._load_file_keys, filename, file_keys) for filename, file_keys in grouped.items()]
            for future in as_completed(futures):
                loaded = future.result()
                self._cache.update(loaded)
                self._advance_progress(loaded.keys())

    def drop(self, keys: list[str] | tuple[str, ...] | set[str]) -> None:
        """Release cached tensors after their layer has been copied."""
        for key in keys:
            self._cache.pop(key, None)

    def keys_for_prefix(self, prefix: str) -> list[str]:
        """Return every key in the weight map under the given dotted prefix."""

        return [key for key in self.weight_map if key.startswith(prefix)]

    def set_progress_total(self, total: int, *, unit: str = "tensor", manual: bool = False) -> None:
        """Override the inferred progress bar total."""

        self._progress_total = total
        self._progress_unit = unit
        self._manual_progress = manual

    def advance_progress(self, count: int = 1) -> None:
        """Advance a manually-counted progress bar."""

        if not self.progress or tqdm is None or count <= 0:
            return
        if self._progress_bar is None:
            self._progress_bar = tqdm(
                total=self._progress_total,
                desc="loading checkpoint",
                unit=self._progress_unit,
                dynamic_ncols=True,
            )
        self._progress_bar.update(count)

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def close(self) -> None:
        """Drop caches and close any open progress bar."""

        self._cache.clear()
        if self._progress_bar is not None:
            # Some adapters intentionally skip checkpoint keys under the layer
            # prefix (for example non-text or variant-only tensors).  The
            # precomputed total is therefore an upper bound; make the bar close
            # at 100% for the keys that were actually visited.
            if self._progress_bar.n != self._progress_bar.total:
                self._progress_bar.total = self._progress_bar.n
                self._progress_bar.refresh()
            self._progress_bar.close()
            self._progress_bar = None

    def _load_file_keys(self, filename: str, keys: list[str]) -> dict[str, torch.Tensor]:
        """Read a batch of keys from one safetensors file inside one open handle."""

        file = self.model_path / filename
        loaded = {}
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in keys:
                loaded[key] = handle.get_tensor(key)
        return loaded

    def _read_tensor(self, key: str, filename: str) -> torch.Tensor:
        with safe_open(self.model_path / filename, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def _use_nccl_transfer(self) -> bool:
        """True when NCCL broadcast from rank 0 should replace per-rank disk reads."""

        ctx = get_tp_context()
        return (
            os.environ.get("ARENO_CKPT_NCCL_TRANSFER", "1") == "1"
            and dist.is_available()
            and dist.is_initialized()
            and ctx.world_size > 1
            and ctx.group is not None
            and ctx.device.type == "cuda"
            and not self._disable_nccl_transfer
        )

    def _get_tensor_nccl(self, key: str, filename: str) -> torch.Tensor:
        """Broadcast one tensor from rank 0 to the rest of the TP group."""

        ctx = get_tp_context()
        # In multi-DP setups, each DP replica needs its own broadcast root.
        src = ctx.dp_rank * ctx.world_size
        tensor = None
        meta = [None]
        if ctx.rank == 0:
            # Reader holds the bytes; non-readers receive shape+dtype first.
            tensor = self._read_tensor(key, filename).contiguous().to(ctx.device, non_blocking=True)
            meta[0] = (tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."))
        dist.broadcast_object_list(meta, src=src, group=ctx.group)
        shape, dtype_name = meta[0]
        dtype = _dtype_from_name(dtype_name)
        if tensor is None:
            tensor = torch.empty(shape, dtype=dtype, device=ctx.device)
        # Now ranks have matching buffers; broadcast the bytes.
        dist.broadcast(tensor, src=src, group=ctx.group)
        return tensor

    def _advance_progress(self, keys) -> None:
        """Update the tqdm progress bar once per newly observed key."""

        if self._manual_progress:
            return
        if not self.progress or tqdm is None:
            return
        new_keys = [key for key in keys if key not in self._loaded_keys]
        if not new_keys:
            return
        if self._progress_bar is None:
            self._progress_bar = tqdm(
                total=self._progress_total or len(self.weight_map),
                desc="loading checkpoint",
                unit=self._progress_unit,
                dynamic_ncols=True,
            )
        self._loaded_keys.update(new_keys)
        self._progress_bar.update(len(new_keys))


def _dtype_from_name(name: str) -> torch.dtype:
    """Look up a torch dtype by its bare name (e.g. ``"bfloat16"``)."""

    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported checkpoint tensor dtype {name!r}")
    return dtype


def _copy_column(dst: torch.Tensor, src: torch.Tensor, rank: int, world_size: int) -> None:
    """Copy this rank's column slice (dim 0) of src into dst."""

    start, end = _shard_range(src.shape[0], rank, world_size)
    dst.copy_(src[start:end].to(dtype=dst.dtype))


def _copy_row(dst: torch.Tensor, src: torch.Tensor, rank: int, world_size: int) -> None:
    """Copy this rank's row slice (dim 1) of src into dst."""

    start, end = _shard_range(src.shape[1], rank, world_size)
    dst.copy_(src[:, start:end].to(dtype=dst.dtype))


def _copy_vocab_shard(dst: torch.Tensor, src: torch.Tensor, rank: int, world_size: int) -> None:
    """Copy this rank's vocabulary slice (dim 0) into dst."""

    start, end = _shard_range(src.shape[0], rank, world_size)
    dst.copy_(src[start:end].to(dtype=dst.dtype))


def _copy_optional_column_bias(dst: torch.Tensor | None, index: SafetensorsIndex, key: str, rank: int, world_size: int) -> None:
    """Copy a column-bias slice only when both dst and the HF key are present."""

    if dst is None or key not in index.weight_map:
        return
    src = index.get_tensor(key)
    start, end = _shard_range(src.shape[0], rank, world_size)
    dst.copy_(src[start:end].to(dtype=dst.dtype))


def gather_tensor_parallel_tensor(tensor: torch.Tensor, dim: int) -> _CheckpointTensorTask | None:
    """Build a lazy task to gather one TP-sharded tensor for saving."""

    ctx = get_tp_context()
    # Non-DP-rank-0 replicas skip checkpoint saving entirely.
    if ctx.dp_rank != 0:
        return None
    local = tensor.detach().contiguous()
    if ctx.world_size == 1:
        return _ReplicatedTensorTask(local)
    return _TensorParallelGatherTask(local, dim)


def gather_tensor_parallel_column_tensors(tensors: list[torch.Tensor]) -> list[_CheckpointTensorTask | None]:
    """Build fused-gather tasks for a list of column-parallel tensors."""

    ctx = get_tp_context()
    if ctx.dp_rank != 0:
        return [None for _ in tensors]
    if not tensors:
        return []
    if len(tensors) == 1:
        # No fusing benefit for a single tensor; use the simpler path.
        return [gather_tensor_parallel_tensor(tensor, dim=0) for tensor in tensors]
    task = _ColumnGatherTask(tensors)
    return [task.result(i) for i in range(len(tensors))]


def gather_tensor_parallel_split_column_tensor(tensor: torch.Tensor, sizes: list[int]) -> _CheckpointTensorTask | None:
    """Build a lazy gather for a fused tensor that must be re-split before save."""

    ctx = get_tp_context()
    if ctx.dp_rank != 0:
        return None
    return _SplitColumnGatherTask(list(tensor.detach().split(sizes, dim=0)))


def _gathered_tensor_to_cpu(gathered: torch.Tensor, dim: int) -> torch.Tensor:
    """Reshape a [world, *local] gather tensor into the global TP shape and move to CPU."""

    if dim == 0:
        # Column-parallel: world stacks along dim 0, so multiply rows by world.
        shape = (gathered.shape[0] * gathered.shape[1], *gathered.shape[2:])
        return _tensor_to_cpu(gathered.reshape(shape))
    # Row-parallel: move the world dim to sit beside `dim`, then fuse.
    moved = gathered.movedim(0, dim)
    shape = list(moved.shape)
    shape[dim] *= shape.pop(dim + 1)
    return _tensor_to_cpu(moved.reshape(tuple(shape)).contiguous())


def _tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a CUDA tensor to pinned CPU memory using a dedicated async stream.

    Each copy gets its own CUDA stream and pinned destination buffer so D2H
    transfers overlap with subsequent compute. The pending bucket queue caps
    in-flight bytes via `_ASYNC_CPU_COPY_MAX_BYTES`; older copies are
    synchronized when the budget would be exceeded.
    """

    if not tensor.is_cuda:
        return tensor.cpu()
    source = tensor.detach()
    output = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu", pin_memory=True)
    copy_bytes = tensor.numel() * tensor.element_size()
    # Make sure this copy fits in the in-flight budget once admitted.
    _sync_pending_cpu_copies(max_pending_bytes=max(0, _ASYNC_CPU_COPY_MAX_BYTES - copy_bytes))
    stream = torch.cuda.Stream(device=tensor.device)
    # Ensure the copy stream sees writes from the default stream first.
    stream.wait_stream(torch.cuda.current_stream(tensor.device))
    with torch.cuda.stream(stream):
        output.copy_(source, non_blocking=True)
    # record_stream prevents the allocator from reusing source memory while
    # the async copy is still in flight on `stream`.
    source.record_stream(stream)
    _PENDING_CPU_COPY_BUCKETS.append((stream, copy_bytes, source))
    return output


def _sync_pending_cpu_copies(max_pending_bytes: int = 0) -> None:
    """Block on enough of the oldest async D2H copies to fit the byte budget."""

    pending_bytes = sum(nbytes for _, nbytes, _ in _PENDING_CPU_COPY_BUCKETS)
    if pending_bytes <= max_pending_bytes:
        return
    bytes_to_release = pending_bytes - max_pending_bytes
    # Walk the FIFO and pick the smallest prefix that frees enough bytes.
    num_to_sync = 0
    released = 0
    for _, nbytes, _ in _PENDING_CPU_COPY_BUCKETS:
        num_to_sync += 1
        released += nbytes
        if released >= bytes_to_release:
            break
    # Synchronize each picked stream so the source tensors can be freed.
    for stream, _, _ in _PENDING_CPU_COPY_BUCKETS[:num_to_sync]:
        stream.synchronize()
    del _PENDING_CPU_COPY_BUCKETS[:num_to_sync]


def _all_gather_tensor_parallel(local: torch.Tensor) -> torch.Tensor:
    """Run a TP all_gather_into_tensor and return the [world, *local] result."""

    ctx = get_tp_context()
    gathered = torch.empty((ctx.world_size, *local.shape), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(gathered, local, group=ctx.group)
    return gathered


def rank0_tensor(tensor: torch.Tensor) -> _CheckpointTensorTask | None:
    """Wrap a replicated tensor for saving; returns None on non-DP-rank-0."""

    ctx = get_tp_context()
    return _ReplicatedTensorTask(tensor) if ctx.dp_rank == 0 else None


def write_hf_safetensors_checkpoint(tensors: dict[str, torch.Tensor | None], output_path: str | Path, source_path: str | Path | None) -> str | None:
    """Write a sharded HF safetensors checkpoint plus its index.json.

    Each TP rank writes the tensors it owns into `model-{rank+1:05d}-of-{N:05d}.safetensors`.
    Rank 0 then collects every rank's local index and emits a unified
    `model.safetensors.index.json` and any aux assets from the source repo.
    """

    ctx = get_tp_context()
    # Only DP rank 0 participates in checkpointing.
    if ctx.dp_rank != 0:
        return None
    # Make sure every outstanding async D2H copy has landed before we serialize.
    _sync_pending_cpu_copies()
    close_progress = getattr(tensors, "close_progress", None)
    if close_progress is not None:
        close_progress()
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    if ctx.rank == 0 and source_path is not None:
        # Copy tokenizer / config / generation_config from the source repo so
        # the output directory is a self-contained HF checkpoint.
        _copy_hf_assets(Path(source_path), path)
    # Drop None entries (tensors owned by a different rank) before writing.
    final_tensors = {key: tensor for key, tensor in tensors.items() if tensor is not None}
    filename = "model.safetensors" if ctx.world_size == 1 else f"model-{ctx.rank + 1:05d}-of-{ctx.world_size:05d}.safetensors"
    if final_tensors:
        save_file(final_tensors, path / filename, metadata={"format": "pt"})

    # Each rank reports its local total bytes + key->file map for the index.
    local_metadata = {
        "total_size": sum(t.numel() * t.element_size() for t in final_tensors.values()),
        "weight_map": {key: filename for key in final_tensors},
    }
    if ctx.world_size == 1:
        all_metadata = [local_metadata]
    else:
        all_metadata = [None for _ in range(ctx.world_size)]
        dist.all_gather_object(all_metadata, local_metadata, group=ctx.group)

    if ctx.rank != 0:
        return None
    # Rank 0 merges every rank's contribution into one index file.
    weight_map = {}
    total_size = 0
    for metadata in all_metadata:
        total_size += int(metadata["total_size"])
        weight_map.update(metadata["weight_map"])
    with (path / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "total_size": total_size,
                    # Marker used by SafetensorsIndex to skip NCCL broadcast.
                    "areno_checkpoint_writer": "distributed_tp",
                },
                "weight_map": weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    return str(path)


def _copy_hf_assets(source: Path, target: Path) -> None:
    """Copy tokenizer / config / generation_config / aux files from src to dst."""

    if not source.exists() or source.resolve() == target.resolve():
        return
    # Anything that looks like a model weight file is intentionally skipped;
    # we re-emit those from our gathered tensors.
    skip_suffixes = {".safetensors", ".bin", ".pt"}
    skip_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    for item in source.iterdir():
        if item.name in skip_names or item.suffix in skip_suffixes:
            continue
        dst = target / item.name
        if item.is_dir():
            if dst.exists():
                continue
            shutil.copytree(item, dst)
        elif item.is_file():
            shutil.copy2(item, dst)
