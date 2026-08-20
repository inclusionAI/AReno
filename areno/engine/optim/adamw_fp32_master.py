"""FP32-master AdamW with per-DP-rank shard of optimizer state.

The BF16 model weights are kept on every DP rank; the FP32 master weights and
Adam moments live in flat buckets that are sharded across DP ranks. After each
Adam update on a bucket, the updated DP shards are re-gathered and copied
back into the BF16 model parameters so forward/backward sees fresh weights.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
        self._active_offload_mode = "none"
        self._disk_offload_root: str | None = None
        self._active_offload_batch_size = 32

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
                    self._step_bucket(bucket)
                    group_changed = True
                    if self._active_offload_mode == "disk":
                        self._stage_bucket_on_cpu(bucket)
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
        self._collective_arenas.clear()
        self._cleanup_disk_offload()
        self._active_offload_mode = "none"
        self._disk_offload_root = None
        self._active_offload_batch_size = 32

    @torch.no_grad()
    def offload_state(self, mode: str = "cpu", directory: str | None = None, batch_size: int = 32) -> None:
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

    def configure_state_offload(self, mode: str, directory: str | None = None, batch_size: int = 32) -> None:
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
        self._active_offload_mode = "none"
        self._disk_offload_root = None
        self._active_offload_batch_size = 32
        self._cleanup_disk_offload()

    def state_dict(self) -> dict:
        """Return the optimizer state laid out per-bucket; each rank saves its shard."""

        disk_cache: dict[str, dict[int, dict]] = {}
        payloads = [self._bucket_cpu_payload(index, bucket, disk_cache) for index, bucket in enumerate(self.buckets)]

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
            dtype = bucket.refs[0].model_param.dtype
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
        updated_refs: list[_ParamRef] = []
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
                updated_refs.append(ref)
                # Once the final chunk of a parameter has consumed its grad,
                # release the autograd buffer immediately.
                if ref.param_start + ref.numel == ref.model_param.numel():
                    ref.model_param.grad = None
                    if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                        ref.model_param.main_grad = None
            if updated_refs:
                self._commit_master(bucket)
                # Re-gather the updated BF16 shards back into every DP rank.
                self._all_gather_bucket(bucket, updated_refs)
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
        updated_refs: list[_ParamRef] = []
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
                updated_refs.append(ref)
                if ref.param_start + ref.numel == ref.model_param.numel():
                    ref.model_param.grad = None
                    if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                        ref.model_param.main_grad = None
            if updated_refs:
                self._all_gather_bucket(bucket, updated_refs)
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
    def _all_gather_bucket(self, bucket: _MasterBucket, refs: list[_ParamRef] | None = None) -> None:
        """Gather updated DP shards and copy the full bucket back to BF16 params."""
        if self.dp_size == 1:
            return
        refs = bucket.refs if refs is None else refs
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
            load_device = torch.device("cpu") if self._active_offload_mode == "disk" else device
            self._load_bucket_offload(bucket, load_device)
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

    def _write_disk_payloads(self, indices: list[int], payloads: list[dict], root: str) -> str:
        """Atomically persist a group of CPU bucket payloads and return its path."""

        directory = self._disk_offload_directory(root)
        path = directory / f"buckets-{indices[0]:06d}-{indices[-1]:06d}.pt"
        temporary = directory / f".{path.name}.tmp"
        cpu_payload = {
            "indices": indices,
            "states": [_tree_to_cpu(payload) for payload in payloads],
        }
        try:
            torch.save(cpu_payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path)

    def _offload_bucket_group_to_disk(self, indices: list[int]) -> None:
        """Persist a bounded group of staged buckets in one serialization call."""

        if self._disk_offload_root is None:
            raise RuntimeError("disk optimizer offload is active without a usable directory")
        present_indices = [index for index in indices if self.buckets[index].master_storage is not None]
        if not present_indices:
            return
        payloads = []
        for index in present_indices:
            bucket = self.buckets[index]
            assert bucket.master_storage is not None
            payloads.append(
                {
                    "low_bits": bucket.master_storage.low_bits,
                    "round_up_bits": bucket.master_storage.round_up_bits,
                    "exp_avg": bucket.exp_avg,
                    "exp_avg_sq": bucket.exp_avg_sq,
                }
            )
        path = self._write_disk_payloads(present_indices, payloads, self._disk_offload_root)
        for index in present_indices:
            bucket = self.buckets[index]
            bucket.master = None
            bucket.offload_file = path
            bucket.master_storage = None
            bucket.exp_avg = None
            bucket.exp_avg_sq = None

    def _stage_bucket_on_cpu(self, bucket: _MasterBucket) -> None:
        """Move one bucket to CPU so only a bounded group is resident before saving."""

        bucket.master = None
        if bucket.master_storage is not None:
            bucket.master_storage = bucket.master_storage.to("cpu")
        if bucket.exp_avg is not None:
            bucket.exp_avg = bucket.exp_avg.to(device="cpu")
        if bucket.exp_avg_sq is not None:
            bucket.exp_avg_sq = bucket.exp_avg_sq.to(device="cpu")

    def _load_bucket_offload(self, bucket: _MasterBucket, device: torch.device) -> None:
        """Load one disk group and invalidate its now-stale shared file."""

        assert bucket.offload_file is not None
        path = Path(bucket.offload_file)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for index, state in zip(payload["indices"], payload["states"], strict=True):
            target = self.buckets[int(index)]
            target.master_storage = BF16MasterStorage(
                low_bits=state["low_bits"].to(device=device),
                round_up_bits=state["round_up_bits"].to(device=device),
            )
            target.exp_avg = _optional_tensor_to(state.get("exp_avg"), device)
            target.exp_avg_sq = _optional_tensor_to(state.get("exp_avg_sq"), device)
            target.offload_file = None
        path.unlink(missing_ok=True)

    def _bucket_cpu_payload(self, index: int, bucket: _MasterBucket, disk_cache: dict[str, dict[int, dict]]) -> dict:
        """Snapshot one bucket on CPU without changing its residency."""

        if bucket.offload_file is not None:
            cached = disk_cache.get(bucket.offload_file)
            if cached is None:
                group = torch.load(bucket.offload_file, map_location="cpu", weights_only=True)
                cached = {
                    int(bucket_index): state
                    for bucket_index, state in zip(group["indices"], group["states"], strict=True)
                }
                disk_cache[bucket.offload_file] = cached
            payload = cached[index]
            storage = BF16MasterStorage(payload["low_bits"], payload["round_up_bits"])
            return {
                "master_storage": storage,
                "exp_avg": _optional_cpu_clone(payload.get("exp_avg")),
                "exp_avg_sq": _optional_cpu_clone(payload.get("exp_avg_sq")),
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


def _tree_to_cpu(value):
    """Copy tensors in a small optimizer-state container to CPU."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    return value
