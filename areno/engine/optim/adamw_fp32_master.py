"""FP32-master AdamW with per-DP-rank shard of optimizer state.

The BF16 model weights are kept on every DP rank; the FP32 master weights and
Adam moments live in flat buckets that are sharded across DP ranks. After each
Adam update on a bucket, the updated DP shards are re-gathered and copied
back into the BF16 model parameters so forward/backward sees fresh weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist


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
    master: torch.Tensor | None = None
    exp_avg: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None


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

    @torch.no_grad()
    def step(self, closure=None):
        """Apply AdamW to every bucket that received a gradient this step."""

        if closure is not None:
            with torch.enable_grad():
                closure()
        for bucket in self.buckets:
            # A bucket is updated only if at least one of its refs has a grad;
            # this avoids materializing master state for unused parameters.
            has_grad = False
            for ref in bucket.refs:
                has_grad = has_grad or _param_grad(ref.model_param) is not None
            if has_grad:
                self._ensure_bucket_state(bucket)
                self._step_bucket(bucket)
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

    def clear_state(self) -> None:
        """Drop all master tensors and reset step counters (used before reload)."""

        for bucket in self.buckets:
            bucket.master = None
            bucket.exp_avg = None
            bucket.exp_avg_sq = None
            bucket.step = 0

    @torch.no_grad()
    def offload_state(self) -> None:
        """Move per-bucket FP32 state to CPU to free GPU memory between steps."""

        for bucket in self.buckets:
            if bucket.master is not None:
                bucket.master = bucket.master.to(device="cpu")
            if bucket.exp_avg is not None:
                bucket.exp_avg = bucket.exp_avg.to(device="cpu")
            if bucket.exp_avg_sq is not None:
                bucket.exp_avg_sq = bucket.exp_avg_sq.to(device="cpu")

    @torch.no_grad()
    def onload_state(self, device: torch.device) -> None:
        """Move offloaded state back onto the given device."""

        for bucket in self.buckets:
            if bucket.master is not None and bucket.master.device != device:
                bucket.master = bucket.master.to(device=device)
            if bucket.exp_avg is not None and bucket.exp_avg.device != device:
                bucket.exp_avg = bucket.exp_avg.to(device=device)
            if bucket.exp_avg_sq is not None and bucket.exp_avg_sq.device != device:
                bucket.exp_avg_sq = bucket.exp_avg_sq.to(device=device)

    def state_dict(self) -> dict:
        """Return the optimizer state laid out per-bucket; each rank saves its shard."""

        return {
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
            "eps": self.eps,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            # One flat tensor per bucket holding this rank's master shard.
            "master_params": [bucket.master.detach().clone() if bucket.master is not None else None for bucket in self.buckets],
            "state": [
                {
                    "exp_avg": bucket.exp_avg.detach().clone() if bucket.exp_avg is not None else None,
                    "exp_avg_sq": bucket.exp_avg_sq.detach().clone() if bucket.exp_avg_sq is not None else None,
                    "step": bucket.step,
                }
                for bucket in self.buckets
            ],
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state. Supports flat tensors and legacy per-ref lists."""

        master_params = state_dict.get("master_params", [])
        for saved, bucket in zip(master_params[: len(self.buckets)], self.buckets, strict=False):
            if saved is None:
                continue
            device = bucket.refs[0].model_param.device
            if isinstance(saved, list):
                # Legacy format: one tensor per ref; copy them into bucket shards.
                bucket.master = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
                for saved_ref, ref in zip(saved[: len(bucket.refs)], bucket.refs, strict=False):
                    if saved_ref is not None and ref.shard_numel > 0:
                        bucket.master.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(
                            saved_ref.detach().to(device=device, dtype=torch.float32).view(-1)
                        )
            else:
                # New format: single flat shard tensor per bucket.
                bucket.master = saved.detach().to(device=device, dtype=torch.float32).view(-1).clone()
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
                bucket.exp_avg = None if exp_avg is None else exp_avg.detach().to(device=device, dtype=torch.float32).view(-1).clone()
                bucket.exp_avg_sq = None if exp_avg_sq is None else exp_avg_sq.detach().to(device=device, dtype=torch.float32).view(-1).clone()
            bucket.step = int(saved.get("step", 0))
        # The BF16 model weights must be refreshed from the loaded master copy.
        self._copy_master_to_model()

    @torch.no_grad()
    def _step_bucket(self, bucket: _MasterBucket) -> None:
        """Update all parameter chunks that live in one flattened master bucket."""
        beta1, beta2 = self.betas
        bucket.step += 1

        # Standard Adam bias-corrected step size.
        bias_correction1 = 1.0 - beta1**bucket.step
        bias_correction2 = 1.0 - beta2**bucket.step
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5
        updated_refs: list[_ParamRef] = []
        for ref in bucket.refs:
            grad = _param_grad(ref.model_param)
            if grad is None:
                continue
            # Slice out just the rows belonging to this ref's chunk.
            flat_grad = grad.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
            self._step_param_ref(bucket, ref, flat_grad, beta1, beta2, step_size, bias_correction2_sqrt)
            updated_refs.append(ref)
            # Once the final chunk of a parameter has consumed its grad, drop
            # the grad reference so the autograd graph can be freed.
            if ref.param_start + ref.numel == ref.model_param.numel():
                ref.model_param.grad = None
                if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                    ref.model_param.main_grad = None
        if updated_refs:
            # Re-gather the updated BF16 shards back into every DP rank.
            self._all_gather_bucket(bucket, updated_refs)

    @torch.no_grad()
    def _step_param_ref(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        beta1: float,
        beta2: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Apply AdamW to this rank's shard of one parameter chunk."""
        model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
        if ref.shard_numel > 0:
            # Cast just the shard's grad to FP32 to keep peak memory bounded.
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
                master.mul_(1.0 - self.lr * self.weight_decay)
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
        send = torch.empty(shard_size, device=device, dtype=dtype)
        send.zero_()
        for ref in refs:
            if ref.shard_numel == 0:
                continue
            # Pack this rank's updated BF16 shard into the send buffer.
            model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
            send.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(
                model_chunk.narrow(0, ref.shard_start, ref.shard_numel)
            )
        gathered = [torch.empty_like(send) for _ in range(self.dp_size)]
        dist.all_gather(gathered, send, group=self.dp_group)
        for ref in refs:
            model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
            # Scatter each remote shard back into the right offsets in the
            # local BF16 chunk; bucket_start ranges may straddle ref bounds.
            for rank, shard in enumerate(gathered):
                bucket_start, bucket_numel = self._shard_range(bucket.numel, rank)
                overlap_start = max(ref.bucket_start, bucket_start)
                overlap_end = min(ref.bucket_start + ref.numel, bucket_start + bucket_numel)
                if overlap_start >= overlap_end:
                    continue
                dst_start = overlap_start - ref.bucket_start
                src_start = overlap_start - bucket_start
                numel = overlap_end - overlap_start
                model_chunk.narrow(0, dst_start, numel).copy_(shard.narrow(0, src_start, numel))

    @torch.no_grad()
    def _build_buckets(self, params: list[torch.nn.Parameter], bucket_numel: int) -> list[_MasterBucket]:
        """Flatten trainable parameters into bounded buckets for sharded AdamW."""
        buckets: list[_MasterBucket] = []
        current: list[_ParamRef] = []
        current_numel = 0
        current_device = None
        for param in params:
            # Walk each parameter in update-chunk sized pieces so that a single
            # huge parameter is split across multiple buckets if needed.
            for param_start in range(0, param.numel(), _DEFAULT_UPDATE_CHUNK_NUMEL):
                numel = min(_DEFAULT_UPDATE_CHUNK_NUMEL, param.numel() - param_start)
                # Flush whenever the next chunk would overflow the bucket or
                # belongs to a different device.
                flush = (
                    current
                    and (
                        current_device != param.device
                        or (current_numel + numel > bucket_numel and current_numel > 0)
                    )
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
        """Materialize or onload the FP32 master and Adam moments for one bucket."""
        device = bucket.refs[0].model_param.device
        # Lazy onload: if state was offloaded to CPU, move it back to GPU.
        if bucket.master is not None and bucket.master.device != device:
            bucket.master = bucket.master.to(device=device)
        if bucket.exp_avg is not None and bucket.exp_avg.device != device:
            bucket.exp_avg = bucket.exp_avg.to(device=device)
        if bucket.exp_avg_sq is not None and bucket.exp_avg_sq.device != device:
            bucket.exp_avg_sq = bucket.exp_avg_sq.to(device=device)
        if bucket.master is None:
            # First touch: initialize the master shard from the current BF16
            # model weights, casting each ref's slice up to FP32.
            bucket.master = torch.empty(bucket.shard_numel, device=device, dtype=torch.float32)
            for ref in bucket.refs:
                if ref.shard_numel == 0:
                    continue
                model_shard = (
                    ref.model_param.detach()
                    .reshape(-1)
                    .narrow(0, ref.param_start + ref.shard_start, ref.shard_numel)
                )
                bucket.master.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(model_shard)
        # Adam moments start at zero.
        if bucket.exp_avg is None:
            bucket.exp_avg = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
        if bucket.exp_avg_sq is None:
            bucket.exp_avg_sq = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)

    @torch.no_grad()
    def _copy_bucket_to_model(self, bucket: _MasterBucket) -> None:
        """Push this rank's FP32 master back into BF16 model + gather across DP."""

        if bucket.master is None:
            return
        for ref in bucket.refs:
            model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
            if ref.shard_numel > 0:
                model_chunk.narrow(0, ref.shard_start, ref.shard_numel).copy_(
                    bucket.master.narrow(0, ref.shard_bucket_start, ref.shard_numel)
                )
        self._all_gather_bucket(bucket)

    @torch.no_grad()
    def _copy_master_to_model(self) -> None:
        """Refresh every BF16 model parameter from the FP32 master state."""

        for bucket in self.buckets:
            self._copy_bucket_to_model(bucket)

    @torch.no_grad()
    def _copy_model_to_master(self) -> None:
        """Re-materialize FP32 master state from current BF16 model weights."""

        for bucket in self.buckets:
            device = bucket.refs[0].model_param.device
            bucket.master = torch.empty(bucket.shard_numel, device=device, dtype=torch.float32)
            for ref in bucket.refs:
                if ref.shard_numel == 0:
                    continue
                model_shard = (
                    ref.model_param.detach().reshape(-1).narrow(0, ref.param_start + ref.shard_start, ref.shard_numel)
                )
                bucket.master.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(
                    model_shard.to(dtype=torch.float32)
                )

    @torch.no_grad()
    def _load_master_params(self, master_params: list[torch.Tensor]) -> None:
        """Internal helper for partial master-only restore (no Adam moments)."""

        for saved, bucket in zip(master_params[: len(self.buckets)], self.buckets, strict=False):
            if saved is None:
                continue
            device = bucket.refs[0].model_param.device
            if isinstance(saved, list):
                bucket.master = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
                for saved_ref, ref in zip(saved[: len(bucket.refs)], bucket.refs, strict=False):
                    if saved_ref is not None and ref.shard_numel > 0:
                        bucket.master.narrow(0, ref.shard_bucket_start, ref.shard_numel).copy_(
                            saved_ref.detach().to(device=device, dtype=torch.float32).view(-1)
                        )
            else:
                bucket.master = saved.detach().to(device=device, dtype=torch.float32).view(-1).clone()

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
