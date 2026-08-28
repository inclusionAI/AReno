"""FP32-master AdamW with per-DP-rank shard of optimizer state.

The BF16 model weights are kept on every DP rank; the FP32 master weights and
Adam moments live in flat buckets that are sharded across DP ranks. After each
Adam update on a bucket, the updated DP shards are re-gathered and copied
back into the BF16 model parameters so forward/backward sees fresh weights.
"""

from __future__ import annotations

import mmap
import os
import tempfile
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from areno.engine.optim.master_storage import (
    BF16MasterStorage,
    decode_fp32_master_slice,
    encode_fp32_master,
)

# Per-bucket budget for the FP32 flat buffer (in elements, not bytes).
_DEFAULT_BUCKET_NUMEL = 16 * 1024 * 1024
# Per-parameter chunk size used to keep large tensors from monopolizing a bucket.
_DEFAULT_UPDATE_CHUNK_NUMEL = 4 * 1024 * 1024
# Bound disk lookahead to current/next host buffers rather than a full mmap group.
_DISK_PREFETCH_DEPTH = 2


@dataclass(slots=True)
class _ParamRef:
    """One contiguous chunk of a model parameter living inside a master bucket.

    ``param_start``/``numel`` index into the flat BF16 model parameter, while
    ``shard_start``/``shard_numel``/``shard_bucket_start`` describe this DP
    rank's slice of the chunk inside the bucket-local FP32 buffers.
    """

    model_param: torch.nn.Parameter
    param_start: int
    numel: int
    bucket_start: int
    shard_start: int
    shard_numel: int
    shard_bucket_start: int


@dataclass(slots=True)
class _MmapGroup:
    """One persistent writable raw mmap backing a group of optimizer buckets."""

    path: Path
    handle: Any
    mapping: mmap.mmap
    tensors: dict[int, dict[str, torch.Tensor]]

    def flush(self) -> None:
        self.mapping.flush()

    def close(self) -> None:
        self.tensors.clear()
        self.mapping.close()
        self.handle.close()


@dataclass(slots=True)
class _MasterBucket:
    """One flat FP32 bucket holding many parameter chunks and their Adam state."""

    numel: int
    shard_numel: int
    refs: list[_ParamRef]
    step: int = 0
    # ``master`` is a bucket-scoped FP32 work buffer, never persistent for
    # BF16 parameters.  ``master_storage`` holds only the exact low bits and a
    # packed BF16 rounding carry between optimizer steps.
    master: torch.Tensor | None = None
    master_storage: BF16MasterStorage | None = None
    exp_avg: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    grad_shard: torch.Tensor | None = None
    grad_param_ids: frozenset[int] = frozenset()
    offload_file: str | None = None
    offload_index: int | None = None
    offload_group: _MmapGroup | None = None
    offload_ready_events: tuple[torch.cuda.Event, ...] = ()


class AdamWFP32Master:
    """AdamW with sharded FP32 master parameters.

    BF16 model weights are the tensors used by forward/backward. Optimizer math
    is done on FP32 bucket shards, so each DP rank owns only a slice of the
    master weights and Adam moments. Updated shards are gathered back into the
    BF16 model parameters after each bucket update.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        bucket_numel: int = _DEFAULT_BUCKET_NUMEL,
        dp_rank: int = 0,
        dp_size: int = 1,
        dp_group: dist.ProcessGroup | None = None,
    ):
        # Only keep parameters that participate in training; freeze handling is
        # external.
        self.model_params = [param for param in params if param.requires_grad]
        self.dp_rank = dp_rank
        self.dp_size = max(dp_size, 1)
        self.dp_group = dp_group
        # Flatten and group params into bounded FP32 buckets; this also sets
        # the per-rank shard layout in each `_ParamRef`.
        self.buckets = self._build_buckets(self.model_params, max(bucket_numel, 1))
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = 1e-8
        # One input/output arena per dtype is reused by every DP collective;
        # only the compact reduced gradient shard remains attached to a bucket.
        self._collective_arenas: dict[tuple[torch.device, torch.dtype, str], torch.Tensor] = {}
        self._disk_offload_tmp: tempfile.TemporaryDirectory | None = None
        self._mmap_groups: dict[tuple[int, ...], _MmapGroup] = {}
        self._disk_prefetch_executor: ThreadPoolExecutor | None = None
        self._disk_prefetch_futures: dict[int, Future[dict[str, torch.Tensor]]] = {}
        self._disk_prefetch_in_use: dict[int, tuple[dict[str, torch.Tensor], torch.cuda.Event | None]] = {}
        self._disk_write_executor: ThreadPoolExecutor | None = None
        self._disk_write_futures: dict[tuple[int, ...], Future[None]] = {}
        self._active_offload_mode = "none"
        self._disk_offload_root: str | None = None
        self._active_offload_batch_size = 1

    @torch.no_grad()
    def step(self, closure=None):
        """Apply AdamW to every bucket that received a gradient this step."""

        if closure is not None:
            with torch.enable_grad():
                closure()
        for indices in self._bucket_groups():
            group_changed = False
            for index in indices:
                bucket = self.buckets[index]
                # A bucket is updated only if at least one of its refs has a grad;
                # this avoids materializing master state for unused parameters.
                has_grad = bucket.grad_shard is not None or any(
                    _param_grad(ref.model_param) is not None for ref in bucket.refs
                )
                if has_grad:
                    self._ensure_bucket_state(bucket)
                    if self._active_offload_mode == "disk":
                        self._schedule_disk_prefetch(index + 1)
                    self._step_bucket(bucket)
                    group_changed = True
                    if self._active_offload_mode == "disk":
                        self._stage_bucket_on_cpu(bucket)
                        self._release_disk_prefetch(index)
                elif self._active_offload_mode == "disk":
                    self._discard_disk_prefetch(index)
                    self._schedule_disk_prefetch(index + 1)
            if self._active_offload_mode == "disk" and group_changed:
                self._offload_bucket_group_to_disk(indices)
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Drop or zero out both `.grad` and the optional `.main_grad` field."""

        for param in self.model_params:
            if set_to_none:
                param.grad = None
                # Megatron-style "main_grad" buffer (FP32 accumulator) — drop too.
                if isinstance(getattr(param, "main_grad", None), torch.Tensor):
                    param.main_grad = None
            else:
                if param.grad is not None:
                    param.grad.zero_()
                if isinstance(getattr(param, "main_grad", None), torch.Tensor):
                    param.main_grad.zero_()
        for bucket in self.buckets:
            bucket.grad_shard = None
            bucket.grad_param_ids = frozenset()

    def clear_state(self) -> None:
        """Drop all master tensors and reset step counters (used before reload)."""

        for bucket in self.buckets:
            bucket.master = None
            bucket.master_storage = None
            bucket.exp_avg = None
            bucket.exp_avg_sq = None
            bucket.grad_shard = None
            bucket.grad_param_ids = frozenset()
            bucket.step = 0
            bucket.offload_file = None
            bucket.offload_index = None
            bucket.offload_group = None
            bucket.offload_ready_events = ()
        self._collective_arenas.clear()
        self._cleanup_disk_offload()
        self._active_offload_mode = "none"
        self._disk_offload_root = None
        self._active_offload_batch_size = 1

    @torch.no_grad()
    def offload_state(self, mode: str = "cpu", directory: str | None = None, batch_size: int = 1) -> None:
        """Move state to CPU or bucket-stream it into a private disk directory."""

        self.configure_state_offload(mode=mode, directory=directory, batch_size=batch_size)

        for indices in self._bucket_groups():
            if mode == "disk" and all(
                self.buckets[index].offload_file is not None for index in indices if self.buckets[index].step > 0
            ):
                continue
            for index in indices:
                bucket = self.buckets[index]
                # A BF16 bucket must not retain its transient FP32 work buffer.
                bucket.master = None
                if bucket.offload_file is not None:
                    self._load_bucket_offload(bucket, torch.device("cpu"))
                self._stage_bucket_on_cpu(bucket)
                bucket.grad_shard = None
                bucket.grad_param_ids = frozenset()
            if mode == "disk":
                self._offload_bucket_group_to_disk(indices)
        self._collective_arenas.clear()
        if mode == "cpu":
            self._cleanup_disk_offload()

    def configure_state_offload(self, mode: str, directory: str | None = None, batch_size: int = 1) -> None:
        """Set state residency policy before an optimizer step without moving tensors."""

        if mode not in {"cpu", "disk"}:
            raise ValueError("optimizer offload mode must be one of: cpu, disk")
        if mode == "disk" and not directory:
            raise ValueError("directory is required for disk optimizer offload")
        if batch_size < 1:
            raise ValueError("optimizer offload batch_size must be positive")
        self._active_offload_mode = mode
        self._disk_offload_root = directory if mode == "disk" else None
        self._active_offload_batch_size = batch_size

    def prefetch_state(self) -> None:
        """Begin bounded disk page-in before forward/backward reaches the optimizer step."""

        if self._active_offload_mode == "disk":
            self._schedule_disk_prefetch(0)

    @torch.no_grad()
    def onload_state(self, device: torch.device) -> None:
        """Move offloaded state back onto the given device."""

        for bucket in self.buckets:
            if bucket.offload_file is not None:
                self._load_bucket_offload(bucket, device)
            if bucket.master_storage is not None and bucket.master_storage.low_bits.device != device:
                bucket.master_storage = bucket.master_storage.to(device)
            if bucket.exp_avg is not None and bucket.exp_avg.device != device:
                bucket.exp_avg = bucket.exp_avg.to(device=device)
            if bucket.exp_avg_sq is not None and bucket.exp_avg_sq.device != device:
                bucket.exp_avg_sq = bucket.exp_avg_sq.to(device=device)
            bucket.offload_file = None
            bucket.offload_index = None
            bucket.offload_group = None
            bucket.offload_ready_events = ()
        self._active_offload_mode = "none"
        self._disk_offload_root = None
        self._active_offload_batch_size = 1
        self._cleanup_disk_offload()

    def state_dict(self) -> dict:
        """Return the optimizer state laid out per-bucket; each rank saves its shard."""

        payloads = [self._bucket_cpu_payload(index, bucket) for index, bucket in enumerate(self.buckets)]

        return {
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
            "eps": self.eps,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            # One flat tensor per bucket holding this rank's master shard.
            # Keep the checkpoint contract canonical: callers see one FP32
            # master shard per bucket even though runtime storage is compact.
            "master_params": [
                self._materialize_master_cpu(bucket, payload["master_storage"])
                for bucket, payload in zip(self.buckets, payloads, strict=True)
            ],
            "state": [
                {
                    "exp_avg": payload["exp_avg"],
                    "exp_avg_sq": payload["exp_avg_sq"],
                    "step": bucket.step,
                }
                for bucket, payload in zip(self.buckets, payloads, strict=True)
            ],
        }

    @torch.no_grad()
    def reduce_scatter_gradients(self) -> None:
        """Reduce one microbatch into persistent FP32 DP gradient shards."""

        for bucket in self.buckets:
            device = bucket.refs[0].model_param.device
            # Preserve the historical FP32-main-grad reduction contract. The
            # model gradient may be BF16, but casting only after the collective
            # would permanently round the cross-rank sum in BF16.
            dtype = torch.float32
            shard_size = self._max_shard_numel(bucket.numel)
            padded_numel = shard_size * self.dp_size
            send = self._arena(device, dtype, "grad_reduce_input", padded_numel)
            send.zero_()
            present: set[int] = set()
            for ref in bucket.refs:
                grad = _param_grad(ref.model_param)
                if grad is None:
                    continue
                present.add(id(ref.model_param))
                send.narrow(0, ref.bucket_start, ref.numel).copy_(
                    grad.detach().reshape(-1).narrow(0, ref.param_start, ref.numel),
                )
                if ref.param_start + ref.numel == ref.model_param.numel():
                    ref.model_param.grad = None
                    if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                        ref.model_param.main_grad = None
            if not present:
                continue
            output = torch.empty(shard_size, device=device, dtype=dtype)
            if self.dp_size == 1:
                output.copy_(send.narrow(0, 0, shard_size))
            else:
                dist.reduce_scatter_tensor(output, send, op=dist.ReduceOp.SUM, group=self.dp_group)
                output.div_(self.dp_size)
            # The last rank may own fewer real values than the equal-sized
            # collective shard; keep only its valid prefix.
            reduced = output.narrow(0, 0, bucket.shard_numel)
            if bucket.grad_shard is None:
                bucket.grad_shard = reduced.to(dtype=torch.float32)
            else:
                bucket.grad_shard.add_(reduced)
            bucket.grad_param_ids = bucket.grad_param_ids.union(present)

    def has_gradients(self) -> bool:
        """Return whether any bucket owns an accumulated gradient shard."""

        return any(bucket.grad_shard is not None for bucket in self.buckets)

    def grad_shards(self):
        """Yield (parameter, local DP-sharded gradient view) pairs."""

        for bucket in self.buckets:
            if bucket.grad_shard is None:
                continue
            for ref in bucket.refs:
                if ref.shard_numel == 0 or id(ref.model_param) not in bucket.grad_param_ids:
                    continue
                yield (
                    ref.model_param,
                    bucket.grad_shard.narrow(0, ref.shard_bucket_start, ref.shard_numel),
                )

    @torch.no_grad()
    def scale_gradients(self, coefficient: float) -> None:
        """Scale all local DP gradient shards in place."""

        for bucket in self.buckets:
            if bucket.grad_shard is not None:
                bucket.grad_shard.mul_(coefficient)

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state. Supports flat tensors and legacy per-ref lists."""

        self._cleanup_disk_offload()
        self._active_offload_mode = "none"
        self._disk_offload_root = None
        for bucket in self.buckets:
            bucket.offload_file = None
            bucket.offload_index = None
            bucket.offload_group = None
            bucket.offload_ready_events = ()
        self._load_master_params(state_dict.get("master_params", []))
        saved_states = state_dict.get("state", [])
        for saved, bucket in zip(saved_states[: len(self.buckets)], self.buckets, strict=False):
            if saved is None:
                bucket.exp_avg = None
                bucket.exp_avg_sq = None
                bucket.step = 0
                continue
            device = bucket.refs[0].model_param.device
            saved_refs = saved.get("refs") if isinstance(saved, dict) else None
            if saved_refs is not None:
                # Legacy per-ref Adam moments.
                bucket.exp_avg = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
                bucket.exp_avg_sq = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
                for saved_ref, ref in zip(saved_refs[: len(bucket.refs)], bucket.refs, strict=False):
                    if saved_ref is None or ref.shard_numel == 0:
                        continue
                    bucket.exp_avg.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(
                        saved_ref["exp_avg"].detach().to(device=device, dtype=torch.float32).view(-1)
                    )
                    bucket.exp_avg_sq.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(
                        saved_ref["exp_avg_sq"].detach().to(device=device, dtype=torch.float32).view(-1)
                    )
            else:
                # Flat-tensor format: single shard tensor for each moment.
                exp_avg = saved.get("exp_avg") if isinstance(saved, dict) else None
                exp_avg_sq = saved.get("exp_avg_sq") if isinstance(saved, dict) else None
                bucket.exp_avg = (
                    None
                    if exp_avg is None
                    else exp_avg.detach().to(device=device, dtype=torch.float32).view(-1).clone()
                )
                bucket.exp_avg_sq = (
                    None
                    if exp_avg_sq is None
                    else exp_avg_sq.detach().to(device=device, dtype=torch.float32).view(-1).clone()
                )
            bucket.step = int(saved.get("step", 0))
        # `_load_master_params` refreshes model weights while installing the
        # compact master representation.

    @torch.no_grad()
    def _step_bucket(self, bucket: _MasterBucket) -> None:
        """Update all parameter chunks that live in one flattened master bucket."""
        if bucket.refs[0].model_param.is_cuda:
            self._step_bucket_cuda(bucket)
            return
        bucket.master = self._materialize_master(bucket)
        beta1, beta2 = self.betas
        bucket.step += 1

        # Standard Adam bias-corrected step size.
        bias_correction1 = 1.0 - beta1**bucket.step
        bias_correction2 = 1.0 - beta2**bucket.step
        bias_correction2_sqrt = bias_correction2**0.5
        try:
            for ref in bucket.refs:
                grad = self._gradient_for_ref(bucket, ref)
                if grad is None:
                    continue
                effective_lr = float(getattr(ref.model_param, "_areno_lr", self.lr))
                step_size = effective_lr / bias_correction1
                self._step_param_ref(
                    bucket,
                    ref,
                    grad,
                    beta1,
                    beta2,
                    effective_lr,
                    step_size,
                    bias_correction2_sqrt,
                )
                # Once the final chunk of a parameter has consumed its grad,
                # release the autograd buffer immediately.
                if ref.param_start + ref.numel == ref.model_param.numel():
                    ref.model_param.grad = None
                    if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                        ref.model_param.main_grad = None
            self._commit_master(bucket)
            # Every rank that stepped this bucket must enter the same
            # collective, including ranks whose equal-sized DP shard contains
            # no real values. Gather the complete bucket rather than the
            # rank-local set of refs so every model replica is refreshed.
            self._all_gather_bucket(bucket)
        finally:
            # The FP32 master is a step-local bucket buffer, not persistent
            # optimizer state.
            bucket.master = None
            bucket.grad_shard = None
            bucket.grad_param_ids = frozenset()

    @torch.no_grad()
    def _step_bucket_cuda(self, bucket: _MasterBucket) -> None:
        """Run fused register-local AdamW without materializing FP32 temporaries."""

        from areno.accel.optimizer import areno_adamw_fp32_master_step

        assert bucket.master_storage is not None
        assert bucket.exp_avg is not None
        assert bucket.exp_avg_sq is not None
        beta1, beta2 = self.betas
        bucket.step += 1
        bias_correction1 = 1.0 - beta1**bucket.step
        bias_correction2_sqrt = (1.0 - beta2**bucket.step) ** 0.5
        try:
            for ref in bucket.refs:
                grad = self._gradient_for_ref(bucket, ref)
                if grad is None:
                    continue
                if ref.shard_numel > 0:
                    model_shard = (
                        ref.model_param.detach()
                        .reshape(-1)
                        .narrow(0, ref.param_start + ref.shard_start, ref.shard_numel)
                    )
                    grad_shard = (
                        grad if bucket.grad_shard is not None else grad.narrow(0, ref.shard_start, ref.shard_numel)
                    ).contiguous()
                    effective_lr = float(getattr(ref.model_param, "_areno_lr", self.lr))
                    areno_adamw_fp32_master_step(
                        model_shard,
                        bucket.master_storage.low_bits,
                        bucket.master_storage.round_up_bits,
                        grad_shard,
                        bucket.exp_avg,
                        bucket.exp_avg_sq,
                        state_offset=ref.shard_bucket_start,
                        beta1=beta1,
                        beta2=beta2,
                        effective_lr=effective_lr,
                        weight_decay=self.weight_decay,
                        eps=self.eps,
                        step_size=effective_lr / bias_correction1,
                        bias_correction2_sqrt=bias_correction2_sqrt,
                    )
                if ref.param_start + ref.numel == ref.model_param.numel():
                    ref.model_param.grad = None
                    if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                        ref.model_param.main_grad = None
            # `_step_bucket_cuda` is called only for an active bucket. Some DP
            # ranks may nevertheless own a zero-length shard, so collective
            # participation cannot depend on their local updated refs.
            self._all_gather_bucket(bucket)
        finally:
            bucket.grad_shard = None
            bucket.grad_param_ids = frozenset()

    @torch.no_grad()
    def _step_param_ref(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        beta1: float,
        beta2: float,
        effective_lr: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Apply AdamW to this rank's shard of one parameter chunk."""
        model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
        if ref.shard_numel > 0:
            # Cast just the shard's grad to FP32 to keep peak memory bounded.
            if bucket.grad_shard is not None:
                grad_shard = grad.to(dtype=torch.float32)
            else:
                grad_shard = grad.narrow(0, ref.shard_start, ref.shard_numel).to(dtype=torch.float32)
            model_shard = model_chunk.narrow(0, ref.shard_start, ref.shard_numel)
            assert bucket.master is not None
            assert bucket.exp_avg is not None
            assert bucket.exp_avg_sq is not None
            # Narrow into the per-bucket flat tensors using shard-relative offsets.
            master = bucket.master.narrow(0, ref.shard_bucket_start, ref.shard_numel)
            exp_avg = bucket.exp_avg.narrow(0, ref.shard_bucket_start, ref.shard_numel)
            exp_avg_sq = bucket.exp_avg_sq.narrow(0, ref.shard_bucket_start, ref.shard_numel)

            # Decoupled (AdamW) weight decay: shrink master before momentum.
            if self.weight_decay != 0.0:
                master.mul_(1.0 - effective_lr * self.weight_decay)
            # Standard Adam moment updates.
            exp_avg.mul_(beta1).add_(grad_shard, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_shard, grad_shard, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            master.addcdiv_(exp_avg, denom, value=-step_size)
            # Write the BF16 model shard from the FP32 master shard.
            model_shard.copy_(master)

    @torch.no_grad()
    def _all_gather_bucket(self, bucket: _MasterBucket) -> None:
        """Gather updated DP shards and copy the full bucket back to BF16 params."""
        if self.dp_size == 1:
            return
        refs = bucket.refs
        if not refs:
            return
        device = refs[0].model_param.device
        dtype = refs[0].model_param.dtype
        # Use the per-bucket max shard size so every rank contributes an
        # identically-shaped send buffer to all_gather.
        shard_size = self._max_shard_numel(bucket.numel)
        send = self._arena(device, dtype, "param_gather_input", shard_size)
        send.zero_()
        for ref in refs:
            if ref.shard_numel == 0:
                continue
            # Pack this rank's updated BF16 shard into the send buffer.
            model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
            send.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(
                model_chunk.narrow(0, ref.shard_start, ref.shard_numel)
            )
        gathered_flat = self._arena(device, dtype, "param_gather_output", shard_size * self.dp_size)
        dist.all_gather_into_tensor(gathered_flat, send, group=self.dp_group)
        for ref in refs:
            model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
            # Scatter each remote shard back into the right offsets in the
            # local BF16 chunk; bucket_start ranges may straddle ref bounds.
            for rank in range(self.dp_size):
                shard = gathered_flat.narrow(0, rank * shard_size, shard_size)
                bucket_start, bucket_numel = self._shard_range(bucket.numel, rank)
                overlap_start = max(ref.bucket_start, bucket_start)
                overlap_end = min(ref.bucket_start + ref.numel, bucket_start + bucket_numel)
                if overlap_start >= overlap_end:
                    continue
                dst_start = overlap_start - ref.bucket_start
                src_start = overlap_start - bucket_start
                numel = overlap_end - overlap_start
                model_chunk.narrow(0, dst_start, numel).copy_(shard.narrow(0, src_start, numel))

    def _gradient_for_ref(self, bucket: _MasterBucket, ref: _ParamRef) -> torch.Tensor | None:
        """Return this rank's local gradient slice for one parameter chunk."""

        if bucket.grad_shard is not None:
            if id(ref.model_param) not in bucket.grad_param_ids or ref.shard_numel == 0:
                return None
            return bucket.grad_shard.narrow(0, ref.shard_bucket_start, ref.shard_numel)
        grad = _param_grad(ref.model_param)
        if grad is None:
            return None
        return grad.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)

    def _arena(self, device: torch.device, dtype: torch.dtype, purpose: str, numel: int) -> torch.Tensor:
        """Return a reusable flat collective/cast buffer with at least ``numel`` values."""

        key = (device, dtype, purpose)
        buffer = self._collective_arenas.get(key)
        if buffer is None or buffer.numel() < numel:
            buffer = torch.empty(numel, device=device, dtype=dtype)
            self._collective_arenas[key] = buffer
        return buffer.narrow(0, 0, numel)

    @torch.no_grad()
    def _build_buckets(self, params: list[torch.nn.Parameter], bucket_numel: int) -> list[_MasterBucket]:
        """Flatten trainable parameters into bounded buckets for sharded AdamW."""
        buckets: list[_MasterBucket] = []
        current: list[_ParamRef] = []
        current_numel = 0
        current_device = None
        current_dtype = None
        for param in params:
            # Walk each parameter in update-chunk sized pieces so that a single
            # huge parameter is split across multiple buckets if needed.
            for param_start in range(0, param.numel(), _DEFAULT_UPDATE_CHUNK_NUMEL):
                numel = min(_DEFAULT_UPDATE_CHUNK_NUMEL, param.numel() - param_start)
                # Flush whenever the next chunk would overflow the bucket or
                # belongs to a different device.
                flush = current and (
                    current_device != param.device
                    or current_dtype != param.dtype
                    or (current_numel + numel > bucket_numel and current_numel > 0)
                )
                if flush:
                    buckets.append(self._make_bucket(current, current_numel))
                    current = []
                    current_numel = 0
                current.append(
                    _ParamRef(
                        model_param=param,
                        param_start=param_start,
                        numel=numel,
                        bucket_start=current_numel,
                        # shard_* fields are populated by `_make_bucket`.
                        shard_start=0,
                        shard_numel=0,
                        shard_bucket_start=0,
                    )
                )
                current_numel += numel
                current_device = param.device
                current_dtype = param.dtype
        if current:
            buckets.append(self._make_bucket(current, current_numel))
        return buckets

    @torch.no_grad()
    def _make_bucket(self, refs: list[_ParamRef], total_numel: int) -> _MasterBucket:
        """Finalize a bucket: assign each ref its slice of this rank's DP shard."""

        shard_start, shard_numel = self._shard_range(total_numel, self.dp_rank)
        for ref in refs:
            # Compute the overlap of this ref's bucket range with the shard,
            # then store offsets relative to (a) the ref's param-chunk and
            # (b) the shard's flat buffer.
            overlap_start = max(ref.bucket_start, shard_start)
            overlap_end = min(ref.bucket_start + ref.numel, shard_start + shard_numel)
            ref.shard_start = max(overlap_start - ref.bucket_start, 0)
            ref.shard_numel = max(overlap_end - overlap_start, 0)
            ref.shard_bucket_start = max(overlap_start - shard_start, 0)
        return _MasterBucket(numel=total_numel, shard_numel=shard_numel, refs=refs)

    @torch.no_grad()
    def _ensure_bucket_state(self, bucket: _MasterBucket) -> None:
        """Materialize or onload compact master metadata and FP32 moments."""
        device = bucket.refs[0].model_param.device
        if bucket.offload_file is not None:
            self._load_bucket_offload(bucket, device)
        # Lazy onload: if state was offloaded to CPU, move it back to GPU.
        if bucket.master_storage is not None and bucket.master_storage.low_bits.device != device:
            bucket.master_storage = bucket.master_storage.to(device)
        if bucket.exp_avg is not None and bucket.exp_avg.device != device:
            bucket.exp_avg = bucket.exp_avg.to(device=device)
        if bucket.exp_avg_sq is not None and bucket.exp_avg_sq.device != device:
            bucket.exp_avg_sq = bucket.exp_avg_sq.to(device=device)
        if bucket.master_storage is None:
            bucket.master_storage = BF16MasterStorage.zeros(bucket.shard_numel, device=device)
        # Adam moments start at zero.
        if bucket.exp_avg is None:
            bucket.exp_avg = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
        if bucket.exp_avg_sq is None:
            bucket.exp_avg_sq = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)

    def _disk_offload_directory(self, root: str) -> Path:
        """Create one process-private directory below the explicitly selected root."""

        if self._disk_offload_tmp is None:
            base = Path(root).expanduser()
            base.mkdir(parents=True, exist_ok=True)
            self._disk_offload_tmp = tempfile.TemporaryDirectory(
                prefix=f"areno-optimizer-dp{self.dp_rank}-p{os.getpid()}-",
                dir=base,
            )
        return Path(self._disk_offload_tmp.name)

    def _get_or_create_mmap_group(
        self,
        indices: list[int],
        specs: dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]],
    ) -> _MmapGroup:
        """Create fixed tensor views into one persistent writable raw mmap."""

        key = tuple(indices)
        existing = self._mmap_groups.get(key)
        if existing is not None:
            return existing
        if self._disk_offload_root is None:
            raise RuntimeError("disk optimizer offload is active without a usable directory")

        offsets: dict[int, dict[str, tuple[int, torch.dtype, tuple[int, ...], int]]] = {}
        cursor = 0
        for index in indices:
            offsets[index] = {}
            for name, (dtype, shape) in specs[index].items():
                cursor = _align_up(cursor, 64)
                numel = _shape_numel(shape)
                offsets[index][name] = (cursor, dtype, shape, numel)
                cursor += numel * torch.empty((), dtype=dtype).element_size()
        file_size = max(_align_up(cursor, mmap.PAGESIZE), mmap.PAGESIZE)
        directory = self._disk_offload_directory(self._disk_offload_root)
        path = directory / f"buckets-{indices[0]:06d}-{indices[-1]:06d}.mmap"
        handle = path.open("w+b")
        mapping: mmap.mmap | None = None
        try:
            handle.truncate(file_size)
            mapping = mmap.mmap(handle.fileno(), file_size, access=mmap.ACCESS_WRITE)
            tensors: dict[int, dict[str, torch.Tensor]] = {}
            for index in indices:
                tensors[index] = {}
                for name, (offset, dtype, shape, numel) in offsets[index].items():
                    if numel == 0:
                        tensor = torch.empty(shape, dtype=dtype)
                    else:
                        tensor = torch.frombuffer(mapping, dtype=dtype, count=numel, offset=offset).reshape(shape)
                    tensors[index][name] = tensor
            group = _MmapGroup(path=path, handle=handle, mapping=mapping, tensors=tensors)
        except BaseException:
            if mapping is not None:
                mapping.close()
            handle.close()
            path.unlink(missing_ok=True)
            raise
        self._mmap_groups[key] = group
        return group

    def _master_mmap_specs(self, indices: list[int]) -> dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]]:
        """Return the fixed raw-mmap layout for compact FP32-master state."""

        return {
            index: {
                "low_bits": (torch.uint16, (self.buckets[index].shard_numel,)),
                "round_up_bits": (torch.uint8, ((self.buckets[index].shard_numel + 7) // 8,)),
                "exp_avg": (torch.float32, (self.buckets[index].shard_numel,)),
                "exp_avg_sq": (torch.float32, (self.buckets[index].shard_numel,)),
            }
            for index in indices
        }

    def _offload_bucket_group_to_disk(self, indices: list[int]) -> None:
        """Persist a bounded group of staged buckets in one serialization call."""

        if self._disk_offload_root is None:
            raise RuntimeError("disk optimizer offload is active without a usable directory")
        present_indices = [index for index in indices if self.buckets[index].master_storage is not None]
        if not present_indices:
            return
        group = self._get_or_create_mmap_group(indices, self._master_mmap_specs(indices))
        payloads: dict[int, dict[str, torch.Tensor]] = {}
        ready_events: list[torch.cuda.Event] = []
        for index in present_indices:
            bucket = self.buckets[index]
            assert bucket.master_storage is not None
            assert bucket.exp_avg is not None
            assert bucket.exp_avg_sq is not None
            payloads[index] = {
                "low_bits": bucket.master_storage.low_bits,
                "round_up_bits": bucket.master_storage.round_up_bits,
                "exp_avg": bucket.exp_avg,
                "exp_avg_sq": bucket.exp_avg_sq,
            }
            ready_events.extend(bucket.offload_ready_events)
        self._submit_disk_group_write(indices, group, payloads, tuple(ready_events))
        for index in present_indices:
            bucket = self.buckets[index]
            bucket.master = None
            bucket.offload_file = str(group.path)
            bucket.offload_index = index
            bucket.offload_group = group
            bucket.master_storage = None
            bucket.exp_avg = None
            bucket.exp_avg_sq = None
            bucket.offload_ready_events = ()

    def _stage_bucket_on_cpu(self, bucket: _MasterBucket) -> None:
        """Move one bucket to CPU so only a bounded group is resident before saving."""

        bucket.master = None
        payload: dict[str, torch.Tensor] = {}
        if bucket.master_storage is not None:
            payload["low_bits"] = bucket.master_storage.low_bits
            payload["round_up_bits"] = bucket.master_storage.round_up_bits
        if bucket.exp_avg is not None:
            payload["exp_avg"] = bucket.exp_avg
        if bucket.exp_avg_sq is not None:
            payload["exp_avg_sq"] = bucket.exp_avg_sq
        staged, ready_events = self._stage_payload_on_cpu(payload)
        if bucket.master_storage is not None:
            bucket.master_storage = BF16MasterStorage(
                low_bits=staged["low_bits"],
                round_up_bits=staged["round_up_bits"],
            )
        bucket.exp_avg = staged.get("exp_avg")
        bucket.exp_avg_sq = staged.get("exp_avg_sq")
        bucket.offload_ready_events = ready_events

    def _stage_payload_on_cpu(
        self,
        payload: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.cuda.Event, ...]]:
        """Queue disk-bound CUDA tensors into pinned host buffers without synchronizing."""

        async_disk = self._active_offload_mode == "disk"
        staged: dict[str, torch.Tensor] = {}
        cuda_devices: set[torch.device] = set()
        for name, tensor in payload.items():
            if async_disk and tensor.device.type == "cuda":
                destination = torch.empty_like(tensor, device="cpu", pin_memory=True)
                destination.copy_(tensor, non_blocking=True)
                staged[name] = destination
                cuda_devices.add(tensor.device)
            else:
                staged[name] = tensor.to(device="cpu")
        ready_events: list[torch.cuda.Event] = []
        for device in cuda_devices:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            ready_events.append(event)
        return staged, tuple(ready_events)

    def _load_bucket_offload(self, bucket: _MasterBucket, device: torch.device) -> None:
        """Copy one bucket from its persistent raw mmap onto ``device``."""

        assert bucket.offload_file is not None
        assert bucket.offload_index is not None
        assert bucket.offload_group is not None
        self._wait_disk_group_write(bucket.offload_group)
        state, prefetched = self._take_disk_prefetch(
            bucket.offload_index,
            bucket.offload_group.tensors[bucket.offload_index],
        )
        bucket.master_storage = BF16MasterStorage(
            low_bits=_host_tensor_to(state["low_bits"], device, prefetched=prefetched),
            round_up_bits=_host_tensor_to(state["round_up_bits"], device, prefetched=prefetched),
        )
        bucket.exp_avg = _host_tensor_to(state["exp_avg"], device, prefetched=prefetched)
        bucket.exp_avg_sq = _host_tensor_to(state["exp_avg_sq"], device, prefetched=prefetched)
        if prefetched and device.type == "cuda":
            self._retain_disk_prefetch(bucket.offload_index, state, device)

    def _disk_mmap_group_for_index(self, index: int) -> _MmapGroup | None:
        """Return the mapped FP32-master group for an initialized bucket."""

        bucket = self.buckets[index]
        return bucket.offload_group if bucket.offload_file is not None else None

    def _schedule_disk_prefetch(self, start_index: int) -> None:
        """Keep a two-bucket mmap read window queued on one background thread."""

        if self._active_offload_mode != "disk":
            return
        next_index = start_index
        while len(self._disk_prefetch_futures) < _DISK_PREFETCH_DEPTH:
            scheduled = False
            for index in range(next_index, len(self.buckets)):
                group = self._disk_mmap_group_for_index(index)
                if group is None or index in self._disk_prefetch_futures or index in self._disk_prefetch_in_use:
                    continue
                if self._disk_prefetch_executor is None:
                    self._disk_prefetch_executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix=f"areno-optim-prefetch-dp{self.dp_rank}",
                    )
                self._disk_prefetch_futures[index] = self._disk_prefetch_executor.submit(
                    _prefetch_mmap_payload_after_write,
                    group.tensors[index],
                    torch.cuda.is_available(),
                    self._disk_write_futures.get(tuple(group.tensors)),
                )
                next_index = index + 1
                scheduled = True
                break
            if not scheduled:
                return

    def _take_disk_prefetch(
        self,
        index: int,
        fallback: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Consume a completed/in-flight lookahead payload, or use mmap directly."""

        future = self._disk_prefetch_futures.pop(index, None)
        if future is None:
            return fallback, False
        return future.result(), True

    def _discard_disk_prefetch(self, index: int) -> None:
        """Drop lookahead for an unused bucket without blocking the training thread."""

        future = self._disk_prefetch_futures.pop(index, None)
        if future is None or future.cancel():
            return
        future.add_done_callback(_discard_prefetch_result)

    def _retain_disk_prefetch(
        self,
        index: int,
        payload: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Keep pinned sources alive until their queued H2D copies complete."""

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        self._disk_prefetch_in_use[index] = (payload, event)

    def _release_disk_prefetch(self, index: int) -> None:
        """Release one prefetch buffer after the bucket update has synchronized."""

        retained = self._disk_prefetch_in_use.pop(index, None)
        if retained is not None and retained[1] is not None:
            retained[1].synchronize()

    def _shutdown_disk_prefetch(self) -> None:
        """Finish mmap readers before their mappings are closed."""

        for future in self._disk_prefetch_futures.values():
            future.cancel()
        if self._disk_prefetch_executor is not None:
            self._disk_prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._disk_prefetch_executor = None
        self._disk_prefetch_futures.clear()
        for _payload, event in self._disk_prefetch_in_use.values():
            if event is not None:
                event.synchronize()
        self._disk_prefetch_in_use.clear()

    def _submit_disk_group_write(
        self,
        indices: list[int],
        group: _MmapGroup,
        payloads: dict[int, dict[str, torch.Tensor]],
        ready_events: tuple[torch.cuda.Event, ...] = (),
    ) -> None:
        """Queue one group write while bounding retained CPU state to one group."""

        key = tuple(indices)
        previous = self._disk_write_futures.pop(key, None)
        if previous is not None:
            previous.result()
        while self._disk_write_futures:
            _oldest_key, oldest = next(iter(self._disk_write_futures.items()))
            oldest.result()
            self._disk_write_futures.pop(_oldest_key)
        if self._disk_write_executor is None:
            self._disk_write_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"areno-optim-write-dp{self.dp_rank}",
            )
        self._disk_write_futures[key] = self._disk_write_executor.submit(
            _write_mmap_payloads,
            group,
            payloads,
            ready_events,
        )

    def _wait_disk_group_write(self, group: _MmapGroup) -> None:
        """Ensure the latest async write is visible before a direct mmap read."""

        future = self._disk_write_futures.get(tuple(group.tensors))
        if future is not None:
            future.result()

    def _shutdown_disk_writes(self) -> None:
        """Finish pending mmap writes before checkpoint reads or mapping cleanup."""

        if self._disk_write_executor is not None:
            self._disk_write_executor.shutdown(wait=True, cancel_futures=False)
            self._disk_write_executor = None
        for future in self._disk_write_futures.values():
            future.result()
        self._disk_write_futures.clear()

    def _bucket_cpu_payload(self, index: int, bucket: _MasterBucket) -> dict:
        """Snapshot one bucket on CPU without changing its residency."""

        if bucket.offload_file is not None:
            assert bucket.offload_index == index
            assert bucket.offload_group is not None
            self._wait_disk_group_write(bucket.offload_group)
            payload = bucket.offload_group.tensors[index]
            storage = BF16MasterStorage(payload["low_bits"].clone(), payload["round_up_bits"].clone())
            return {
                "master_storage": storage,
                "exp_avg": payload["exp_avg"].clone(),
                "exp_avg_sq": payload["exp_avg_sq"].clone(),
            }
        storage = bucket.master_storage
        if storage is None:
            storage = BF16MasterStorage.zeros(bucket.shard_numel, device="cpu")
        elif storage.low_bits.device.type != "cpu":
            storage = storage.to("cpu")
        return {
            "master_storage": storage,
            "exp_avg": _optional_cpu_clone(bucket.exp_avg),
            "exp_avg_sq": _optional_cpu_clone(bucket.exp_avg_sq),
        }

    def _bucket_groups(self) -> list[list[int]]:
        """Return deterministic contiguous groups for batched disk I/O."""

        size = self._active_offload_batch_size if self._active_offload_mode == "disk" else 1
        return [list(range(start, min(start + size, len(self.buckets)))) for start in range(0, len(self.buckets), size)]

    @torch.no_grad()
    def _materialize_master_cpu(self, bucket: _MasterBucket, storage: BF16MasterStorage) -> torch.Tensor:
        """Reconstruct one logical master shard on CPU for serialization."""

        master = torch.empty(bucket.shard_numel, device="cpu", dtype=torch.float32)
        for ref in bucket.refs:
            if ref.shard_numel == 0:
                continue
            model_shard = (
                ref.model_param.detach()
                .reshape(-1)
                .narrow(0, ref.param_start + ref.shard_start, ref.shard_numel)
                .to(device="cpu")
            )
            destination = master.narrow(0, ref.shard_bucket_start, ref.shard_numel)
            if model_shard.dtype == torch.bfloat16:
                destination.copy_(decode_fp32_master_slice(model_shard, storage, ref.shard_bucket_start))
            elif model_shard.dtype == torch.float32:
                destination.copy_(model_shard)
            else:
                raise TypeError(f"AdamWFP32Master supports bfloat16 or float32 parameters, got {model_shard.dtype}")
        return master

    def _cleanup_disk_offload(self) -> None:
        """Remove only this optimizer's private temporary directory."""

        self._shutdown_disk_writes()
        self._shutdown_disk_prefetch()
        for group in self._mmap_groups.values():
            group.close()
        self._mmap_groups.clear()
        if self._disk_offload_tmp is not None:
            self._disk_offload_tmp.cleanup()
            self._disk_offload_tmp = None

    @torch.no_grad()
    def _copy_bucket_to_model(self, bucket: _MasterBucket) -> None:
        """Push this rank's logical FP32 master into the model + gather DP."""

        transient = bucket.master is None
        if transient:
            bucket.master = self._materialize_master(bucket)
        for ref in bucket.refs:
            model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
            if ref.shard_numel > 0:
                assert bucket.master is not None
                model_chunk.narrow(0, ref.shard_start, ref.shard_numel).copy_(
                    bucket.master.narrow(0, ref.shard_bucket_start, ref.shard_numel)
                )
        self._all_gather_bucket(bucket)
        if transient:
            bucket.master = None

    @torch.no_grad()
    def _copy_master_to_model(self) -> None:
        """Refresh every BF16 model parameter from the FP32 master state."""

        for bucket in self.buckets:
            self._copy_bucket_to_model(bucket)

    @torch.no_grad()
    def _copy_model_to_master(self) -> None:
        """Reset logical masters to the exact current model values."""

        for bucket in self.buckets:
            device = bucket.refs[0].model_param.device
            bucket.master = None
            bucket.master_storage = BF16MasterStorage.zeros(bucket.shard_numel, device=device)

    @torch.no_grad()
    def _load_master_params(self, master_params: list[torch.Tensor]) -> None:
        """Internal helper for partial master-only restore (no Adam moments)."""

        for saved, bucket in zip(master_params[: len(self.buckets)], self.buckets, strict=False):
            if saved is None:
                continue
            device = bucket.refs[0].model_param.device
            master = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
            if isinstance(saved, list):
                for saved_ref, ref in zip(saved[: len(bucket.refs)], bucket.refs, strict=False):
                    if saved_ref is not None and ref.shard_numel > 0:
                        master.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(
                            saved_ref.detach().to(device=device, dtype=torch.float32).view(-1)
                        )
            else:
                master.copy_(saved.detach().to(device=device, dtype=torch.float32).view(-1))
            bucket.master = master
            self._commit_master(bucket)
            bucket.master = None
        if master_params:
            # Every checkpoint contains only this DP rank's master shards.
            # Rebuild the replicated model weights before returning so
            # optimizer-only restore is correct even when no matching model
            # checkpoint was loaded first.
            for bucket in self.buckets:
                self._all_gather_bucket(bucket)

    @torch.no_grad()
    def _materialize_master(self, bucket: _MasterBucket) -> torch.Tensor:
        """Reconstruct one logical FP32 master shard into a bounded work buffer."""

        device = bucket.refs[0].model_param.device
        if bucket.master_storage is None:
            bucket.master_storage = BF16MasterStorage.zeros(bucket.shard_numel, device=device)
        elif bucket.master_storage.low_bits.device != device:
            bucket.master_storage = bucket.master_storage.to(device)
        master = torch.empty(bucket.shard_numel, device=device, dtype=torch.float32)
        for ref in bucket.refs:
            if ref.shard_numel == 0:
                continue
            model_shard = (
                ref.model_param.detach().reshape(-1).narrow(0, ref.param_start + ref.shard_start, ref.shard_numel)
            )
            destination = master.narrow(0, ref.shard_bucket_start, ref.shard_numel)
            if model_shard.dtype == torch.bfloat16:
                destination.copy_(decode_fp32_master_slice(model_shard, bucket.master_storage, ref.shard_bucket_start))
            elif model_shard.dtype == torch.float32:
                # FP32 model parameters are already their own master weights.
                destination.copy_(model_shard)
            else:
                raise TypeError(f"AdamWFP32Master supports bfloat16 or float32 parameters, got {model_shard.dtype}")
        return master

    @torch.no_grad()
    def _commit_master(self, bucket: _MasterBucket) -> None:
        """Persist one updated logical master into BF16 model high + low bits."""

        if bucket.master is None:
            raise RuntimeError("cannot commit an FP32 master bucket before it is materialized")
        model_dtypes = {ref.model_param.dtype for ref in bucket.refs if ref.shard_numel > 0}
        if not model_dtypes:
            return
        if len(model_dtypes) != 1:
            raise TypeError("one optimizer bucket cannot mix parameter dtypes")
        model_dtype = next(iter(model_dtypes))
        if model_dtype == torch.bfloat16:
            model_values, bucket.master_storage = encode_fp32_master(bucket.master)
        elif model_dtype == torch.float32:
            model_values = bucket.master
            bucket.master_storage = BF16MasterStorage.zeros(bucket.shard_numel, device=bucket.master.device)
        else:
            raise TypeError(f"AdamWFP32Master supports bfloat16 or float32 parameters, got {model_dtype}")
        flat_model_values = model_values.reshape(-1)
        for ref in bucket.refs:
            if ref.shard_numel == 0:
                continue
            model_shard = (
                ref.model_param.detach().reshape(-1).narrow(0, ref.param_start + ref.shard_start, ref.shard_numel)
            )
            model_shard.copy_(flat_model_values.narrow(0, ref.shard_bucket_start, ref.shard_numel))

    def _max_shard_numel(self, numel: int) -> int:
        """Ceiling-divide bucket numel by DP size to get the max shard size."""

        return (numel + self.dp_size - 1) // self.dp_size

    def _shard_range(self, numel: int, rank: int) -> tuple[int, int]:
        """Return (start, numel) of this rank's contiguous shard of a bucket."""

        shard_size = self._max_shard_numel(numel)
        # Clamp so the last rank handles whatever remainder is left.
        start = min(rank * shard_size, numel)
        end = min(start + shard_size, numel)
        return start, end - start


def _param_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    """Return Megatron's main_grad if present, else the regular `.grad`."""

    main_grad = getattr(param, "main_grad", None)
    if isinstance(main_grad, torch.Tensor):
        return main_grad
    return param.grad


def _optional_tensor_to(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    """Move an optional tensor to a device without changing its dtype."""

    return None if value is None else value.to(device=device)


def _optional_cpu_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    """Return an independent CPU snapshot of an optional tensor."""

    return None if value is None else value.detach().to(device="cpu").clone()


def _host_tensor_to(value: torch.Tensor, device: torch.device, *, prefetched: bool) -> torch.Tensor:
    """Copy host state while preserving async lifetime for prefetched sources."""

    if device.type == "cpu":
        return value.clone()
    return value.to(device=device, non_blocking=prefetched and value.is_pinned())


@torch.no_grad()
def _prefetch_mmap_payload(
    payload: dict[str, torch.Tensor],
    pin_memory: bool,
) -> dict[str, torch.Tensor]:
    """Fault one mmap bucket in off-thread and optionally stage it in pinned RAM."""

    prefetched: dict[str, torch.Tensor] = {}
    for name, tensor in payload.items():
        if not pin_memory:
            prefetched[name] = tensor.clone()
            continue
        try:
            destination = torch.empty_like(tensor, device="cpu", pin_memory=True)
        except RuntimeError:
            destination = torch.empty_like(tensor, device="cpu")
        destination.copy_(tensor)
        prefetched[name] = destination
    return prefetched


def _prefetch_mmap_payload_after_write(
    payload: dict[str, torch.Tensor],
    pin_memory: bool,
    write_future: Future[None] | None,
) -> dict[str, torch.Tensor]:
    """Wait for the producer, then fault one mmap bucket into a host buffer."""

    if write_future is not None:
        write_future.result()
    return _prefetch_mmap_payload(payload, pin_memory)


@torch.no_grad()
def _write_mmap_payloads(
    group: _MmapGroup,
    payloads: dict[int, dict[str, torch.Tensor]],
    ready_events: tuple[torch.cuda.Event, ...] = (),
) -> None:
    """Write one staged optimizer group in the background and flush once."""

    for event in ready_events:
        event.synchronize()
    for index, payload in payloads.items():
        target = group.tensors[index]
        for name, tensor in payload.items():
            target[name].copy_(tensor)
    group.flush()


def _discard_prefetch_result(future: Future[dict[str, torch.Tensor]]) -> None:
    """Retrieve a skipped prefetch result so worker exceptions are consumed."""

    if future.cancelled():
        return
    try:
        future.result()
    except Exception:
        # The bucket had no gradient and its state is not consumed this step.
        pass


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a positive byte alignment."""

    return ((value + alignment - 1) // alignment) * alignment


def _shape_numel(shape: tuple[int, ...]) -> int:
    """Return the number of elements in a tensor shape, including scalars."""

    result = 1
    for dimension in shape:
        result *= dimension
    return result
