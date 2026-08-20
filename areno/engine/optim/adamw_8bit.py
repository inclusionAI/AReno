"""8-bit-state AdamW with the same DP-sharded contract as AdamWFP32Master."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from areno.engine.optim.adamw_fp32_master import (
    _DEFAULT_BUCKET_NUMEL,
    AdamWFP32Master,
    _host_tensor_to,
    _MasterBucket,
    _MmapGroup,
    _param_grad,
    _ParamRef,
)


@dataclass(slots=True)
class _Adam8bitBucketState:
    """Quantized Adam moments for one DP shard of an optimizer bucket."""

    step: int = 0
    exp_avg_q: torch.Tensor | None = None
    exp_avg_scale: torch.Tensor | None = None
    exp_avg_sq_q: torch.Tensor | None = None
    exp_avg_sq_scale: torch.Tensor | None = None
    offload_file: str | None = None
    offload_index: int | None = None
    offload_group: _MmapGroup | None = None
    offload_ready_events: tuple[torch.cuda.Event, ...] = ()


class AdamW8bit(AdamWFP32Master):
    """AdamW with uint8 Adam moments and no persistent FP32 master weights.

    The model parameters remain BF16 on every DP rank. Adam moments are stored
    for only this rank's DP shard and re-quantized after every bucket update.
    This trades optimizer precision for much lower persistent optimizer memory.
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
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            bucket_numel=bucket_numel,
            dp_rank=dp_rank,
            dp_size=dp_size,
            dp_group=dp_group,
        )
        self._states = [_Adam8bitBucketState() for _ in self.buckets]

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
                state = self._states[index]
                has_grad = bucket.grad_shard is not None or any(
                    _param_grad(ref.model_param) is not None for ref in bucket.refs
                )
                if has_grad:
                    self._ensure_bucket_state(bucket, state)
                    if self._active_offload_mode == "disk":
                        self._schedule_disk_prefetch(index + 1)
                    self._step_bucket_8bit(bucket, state)
                    group_changed = True
                    if self._active_offload_mode == "disk":
                        self._stage_8bit_state_on_cpu(state)
                        self._release_disk_prefetch(index)
                elif self._active_offload_mode == "disk":
                    self._discard_disk_prefetch(index)
                    self._schedule_disk_prefetch(index + 1)
            if self._active_offload_mode == "disk" and group_changed:
                self._offload_8bit_group_to_disk(indices)
        return None

    def clear_state(self) -> None:
        """Drop quantized moments and reset step counters."""

        for state in self._states:
            state.step = 0
            state.exp_avg_q = None
            state.exp_avg_scale = None
            state.exp_avg_sq_q = None
            state.exp_avg_sq_scale = None
            state.offload_file = None
            state.offload_index = None
            state.offload_group = None
            state.offload_ready_events = ()
        for bucket in self.buckets:
            bucket.grad_shard = None
            bucket.grad_param_ids = frozenset()
        self._collective_arenas.clear()
        self._cleanup_disk_offload()
        self._active_offload_mode = "none"
        self._disk_offload_root = None
        self._active_offload_batch_size = 1

    @torch.no_grad()
    def offload_state(self, mode: str = "cpu", directory: str | None = None, batch_size: int = 1) -> None:
        """Move quantized state to CPU or bucket-stream it to disk."""

        self.configure_state_offload(mode=mode, directory=directory, batch_size=batch_size)

        for indices in self._bucket_groups():
            if mode == "disk" and all(
                self._states[index].offload_file is not None for index in indices if self._states[index].step > 0
            ):
                continue
            for index in indices:
                state = self._states[index]
                if state.offload_file is not None:
                    self._load_state_offload(state, torch.device("cpu"))
                self._stage_8bit_state_on_cpu(state)
            if mode == "disk":
                self._offload_8bit_group_to_disk(indices)
        for bucket in self.buckets:
            bucket.grad_shard = None
            bucket.grad_param_ids = frozenset()
        self._collective_arenas.clear()
        if mode == "cpu":
            self._cleanup_disk_offload()

    @torch.no_grad()
    def onload_state(self, device: torch.device) -> None:
        """Move quantized optimizer state back to the training device."""

        for state in self._states:
            if state.offload_file is not None:
                self._load_state_offload(state, device)
            if state.exp_avg_q is not None and state.exp_avg_q.device != device:
                state.exp_avg_q = state.exp_avg_q.to(device=device)
            if state.exp_avg_scale is not None and state.exp_avg_scale.device != device:
                state.exp_avg_scale = state.exp_avg_scale.to(device=device)
            if state.exp_avg_sq_q is not None and state.exp_avg_sq_q.device != device:
                state.exp_avg_sq_q = state.exp_avg_sq_q.to(device=device)
            if state.exp_avg_sq_scale is not None and state.exp_avg_sq_scale.device != device:
                state.exp_avg_sq_scale = state.exp_avg_sq_scale.to(device=device)
            state.offload_file = None
            state.offload_index = None
            state.offload_group = None
            state.offload_ready_events = ()
        self._active_offload_mode = "none"
        self._disk_offload_root = None
        self._active_offload_batch_size = 1
        self._cleanup_disk_offload()

    def state_dict(self) -> dict:
        """Return per-rank quantized optimizer state."""

        payloads = [self._state_cpu_payload(index, state) for index, state in enumerate(self._states)]

        return {
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
            "eps": self.eps,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "adam_8bit": True,
            "state": [
                {
                    "step": state.step,
                    "exp_avg_q": payload["exp_avg_q"],
                    "exp_avg_scale": payload["exp_avg_scale"],
                    "exp_avg_sq_q": payload["exp_avg_sq_q"],
                    "exp_avg_sq_scale": payload["exp_avg_sq_scale"],
                }
                for state, payload in zip(self._states, payloads, strict=True)
            ],
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore quantized optimizer state from this rank's checkpoint."""

        self._cleanup_disk_offload()
        self._active_offload_mode = "none"
        self._disk_offload_root = None
        self._active_offload_batch_size = 1
        for state in self._states:
            state.offload_file = None
            state.offload_index = None
            state.offload_group = None
            state.offload_ready_events = ()
        saved_states = state_dict.get("state", [])
        for saved, bucket, state in zip(saved_states[: len(self.buckets)], self.buckets, self._states, strict=False):
            if saved is None:
                continue
            device = bucket.refs[0].model_param.device
            state.step = int(saved.get("step", 0))
            exp_avg_q = saved.get("exp_avg_q")
            exp_avg_scale = saved.get("exp_avg_scale")
            exp_avg_sq_q = saved.get("exp_avg_sq_q")
            exp_avg_sq_scale = saved.get("exp_avg_sq_scale")
            state.exp_avg_q = (
                None if exp_avg_q is None else exp_avg_q.detach().to(device=device, dtype=torch.uint8).view(-1).clone()
            )
            state.exp_avg_scale = (
                None
                if exp_avg_scale is None
                else exp_avg_scale.detach().to(device=device, dtype=torch.float32).view(()).clone()
            )
            state.exp_avg_sq_q = (
                None
                if exp_avg_sq_q is None
                else exp_avg_sq_q.detach().to(device=device, dtype=torch.uint8).view(-1).clone()
            )
            state.exp_avg_sq_scale = (
                None
                if exp_avg_sq_scale is None
                else exp_avg_sq_scale.detach().to(device=device, dtype=torch.float32).view(()).clone()
            )

    @torch.no_grad()
    def _ensure_bucket_state(self, bucket: _MasterBucket, state: _Adam8bitBucketState) -> None:
        """Materialize or onload quantized moments for one bucket."""

        device = bucket.refs[0].model_param.device
        if state.offload_file is not None:
            self._load_state_offload(state, device)
        if state.exp_avg_q is not None and state.exp_avg_q.device != device:
            state.exp_avg_q = state.exp_avg_q.to(device=device)
        if state.exp_avg_scale is not None and state.exp_avg_scale.device != device:
            state.exp_avg_scale = state.exp_avg_scale.to(device=device)
        if state.exp_avg_sq_q is not None and state.exp_avg_sq_q.device != device:
            state.exp_avg_sq_q = state.exp_avg_sq_q.to(device=device)
        if state.exp_avg_sq_scale is not None and state.exp_avg_sq_scale.device != device:
            state.exp_avg_sq_scale = state.exp_avg_sq_scale.to(device=device)
        if state.exp_avg_q is None:
            state.exp_avg_q = torch.full((bucket.shard_numel,), 128, device=device, dtype=torch.uint8)
            state.exp_avg_scale = torch.ones((), device=device, dtype=torch.float32)
        if state.exp_avg_sq_q is None:
            state.exp_avg_sq_q = torch.zeros(bucket.shard_numel, device=device, dtype=torch.uint8)
            state.exp_avg_sq_scale = torch.ones((), device=device, dtype=torch.float32)

    def _load_state_offload(self, state: _Adam8bitBucketState, device: torch.device) -> None:
        """Copy one quantized bucket from its persistent raw mmap."""

        assert state.offload_file is not None
        assert state.offload_index is not None
        assert state.offload_group is not None
        self._wait_disk_group_write(state.offload_group)
        saved, prefetched = self._take_disk_prefetch(
            state.offload_index,
            state.offload_group.tensors[state.offload_index],
        )
        state.exp_avg_q = _host_tensor_to(saved["exp_avg_q"], device, prefetched=prefetched)
        state.exp_avg_scale = _host_tensor_to(saved["exp_avg_scale"], device, prefetched=prefetched)
        state.exp_avg_sq_q = _host_tensor_to(saved["exp_avg_sq_q"], device, prefetched=prefetched)
        state.exp_avg_sq_scale = _host_tensor_to(saved["exp_avg_sq_scale"], device, prefetched=prefetched)
        if prefetched and device.type == "cuda":
            self._retain_disk_prefetch(state.offload_index, saved, device)

    def _disk_mmap_group_for_index(self, index: int) -> _MmapGroup | None:
        """Return the mapped Adam8bit group for an initialized bucket."""

        state = self._states[index]
        return state.offload_group if state.offload_file is not None else None

    def _state_mmap_specs(self, indices: list[int]) -> dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]]:
        """Return the fixed raw-mmap layout for quantized Adam state."""

        return {
            index: {
                "exp_avg_q": (torch.uint8, (self.buckets[index].shard_numel,)),
                "exp_avg_scale": (torch.float32, ()),
                "exp_avg_sq_q": (torch.uint8, (self.buckets[index].shard_numel,)),
                "exp_avg_sq_scale": (torch.float32, ()),
            }
            for index in indices
        }

    def _offload_8bit_group_to_disk(self, indices: list[int]) -> None:
        """Persist a bounded group of quantized states in one serialization call."""

        if self._disk_offload_root is None:
            raise RuntimeError("disk optimizer offload is active without a usable directory")
        present_indices = [index for index in indices if self._states[index].exp_avg_q is not None]
        if not present_indices:
            return
        group = self._get_or_create_mmap_group(indices, self._state_mmap_specs(indices))
        payloads: dict[int, dict[str, torch.Tensor]] = {}
        ready_events: list[torch.cuda.Event] = []
        for index in present_indices:
            state = self._states[index]
            assert state.exp_avg_q is not None
            assert state.exp_avg_scale is not None
            assert state.exp_avg_sq_q is not None
            assert state.exp_avg_sq_scale is not None
            payloads[index] = {
                "exp_avg_q": state.exp_avg_q,
                "exp_avg_scale": state.exp_avg_scale,
                "exp_avg_sq_q": state.exp_avg_sq_q,
                "exp_avg_sq_scale": state.exp_avg_sq_scale,
            }
            ready_events.extend(state.offload_ready_events)
        self._submit_disk_group_write(indices, group, payloads, tuple(ready_events))
        for index in present_indices:
            state = self._states[index]
            state.offload_file = str(group.path)
            state.offload_index = index
            state.offload_group = group
            state.exp_avg_q = None
            state.exp_avg_scale = None
            state.exp_avg_sq_q = None
            state.exp_avg_sq_scale = None
            state.offload_ready_events = ()

    def _stage_8bit_state_on_cpu(self, state: _Adam8bitBucketState) -> None:
        """Move one quantized bucket to CPU before its group is serialized."""

        payload = {
            name: tensor
            for name, tensor in {
                "exp_avg_q": state.exp_avg_q,
                "exp_avg_scale": state.exp_avg_scale,
                "exp_avg_sq_q": state.exp_avg_sq_q,
                "exp_avg_sq_scale": state.exp_avg_sq_scale,
            }.items()
            if tensor is not None
        }
        staged, state.offload_ready_events = self._stage_payload_on_cpu(payload)
        state.exp_avg_q = staged.get("exp_avg_q")
        state.exp_avg_scale = staged.get("exp_avg_scale")
        state.exp_avg_sq_q = staged.get("exp_avg_sq_q")
        state.exp_avg_sq_scale = staged.get("exp_avg_sq_scale")

    def _state_cpu_payload(
        self,
        index: int,
        state: _Adam8bitBucketState,
    ) -> dict:
        """Snapshot one quantized bucket on CPU without changing residency."""

        if state.offload_file is not None:
            assert state.offload_index == index
            assert state.offload_group is not None
            self._wait_disk_group_write(state.offload_group)
            saved = state.offload_group.tensors[index]
            return {name: tensor.clone() for name, tensor in saved.items()}
        return {
            "exp_avg_q": _cpu_clone(state.exp_avg_q),
            "exp_avg_scale": _cpu_clone(state.exp_avg_scale),
            "exp_avg_sq_q": _cpu_clone(state.exp_avg_sq_q),
            "exp_avg_sq_scale": _cpu_clone(state.exp_avg_sq_scale),
        }

    @torch.no_grad()
    def _step_bucket_8bit(self, bucket: _MasterBucket, state: _Adam8bitBucketState) -> None:
        """Update all parameter chunks in one bucket using dequantized moments."""

        assert state.exp_avg_q is not None
        assert state.exp_avg_scale is not None
        assert state.exp_avg_sq_q is not None
        assert state.exp_avg_sq_scale is not None
        beta1, beta2 = self.betas
        state.step += 1
        bias_correction1 = 1.0 - beta1**state.step
        bias_correction2 = 1.0 - beta2**state.step
        bias_correction2_sqrt = bias_correction2**0.5

        exp_avg = _dequantize_symmetric(state.exp_avg_q, state.exp_avg_scale)
        exp_avg_sq = _dequantize_positive(state.exp_avg_sq_q, state.exp_avg_sq_scale)
        updated_refs: list[_ParamRef] = []
        for ref in bucket.refs:
            grad = self._gradient_for_ref(bucket, ref)
            if grad is None:
                continue
            effective_lr = float(getattr(ref.model_param, "_areno_lr", self.lr))
            step_size = effective_lr / bias_correction1
            self._step_param_ref_8bit(
                bucket,
                ref,
                grad,
                exp_avg,
                exp_avg_sq,
                beta1,
                beta2,
                effective_lr,
                step_size,
                bias_correction2_sqrt,
            )
            updated_refs.append(ref)
            if ref.param_start + ref.numel == ref.model_param.numel():
                ref.model_param.grad = None
                if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                    ref.model_param.main_grad = None
        state.exp_avg_q, state.exp_avg_scale = _quantize_symmetric(exp_avg)
        state.exp_avg_sq_q, state.exp_avg_sq_scale = _quantize_positive(exp_avg_sq)
        if updated_refs:
            self._all_gather_bucket(bucket, updated_refs)
        bucket.grad_shard = None
        bucket.grad_param_ids = frozenset()

    @torch.no_grad()
    def _step_param_ref_8bit(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        beta1: float,
        beta2: float,
        effective_lr: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Apply one AdamW update to this rank's shard of one param chunk."""

        if ref.shard_numel == 0:
            return
        if bucket.grad_shard is not None:
            grad_shard = grad.to(dtype=torch.float32)
        else:
            grad_shard = grad.narrow(0, ref.shard_start, ref.shard_numel).to(dtype=torch.float32)
        model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
        model_shard = model_chunk.narrow(0, ref.shard_start, ref.shard_numel)
        weight = model_shard.to(dtype=torch.float32)
        if self.weight_decay != 0.0:
            weight.mul_(1.0 - effective_lr * self.weight_decay)
        moment = exp_avg.narrow(0, ref.shard_bucket_start, ref.shard_numel)
        variance = exp_avg_sq.narrow(0, ref.shard_bucket_start, ref.shard_numel)
        moment.mul_(beta1).add_(grad_shard, alpha=1.0 - beta1)
        variance.mul_(beta2).addcmul_(grad_shard, grad_shard, value=1.0 - beta2)
        denom = variance.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
        weight.addcdiv_(moment, denom, value=-step_size)
        model_shard.copy_(weight)


def _quantize_symmetric(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a signed FP32 tensor to uint8 with one bucket-level scale."""

    if tensor.numel() == 0:
        return tensor.to(dtype=torch.uint8), torch.ones((), device=tensor.device, dtype=torch.float32)
    scale = tensor.abs().amax().div(127.0).clamp_min(1.0e-30)
    quantized = torch.clamp(torch.round(tensor / scale) + 128.0, 0.0, 255.0).to(dtype=torch.uint8)
    return quantized, scale.to(dtype=torch.float32)


def _dequantize_symmetric(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize signed uint8 moments back to FP32."""

    return (quantized.to(dtype=torch.float32) - 128.0).mul_(scale)


def _quantize_positive(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a non-negative FP32 tensor to uint8 with one bucket-level scale."""

    if tensor.numel() == 0:
        return tensor.to(dtype=torch.uint8), torch.ones((), device=tensor.device, dtype=torch.float32)
    scale = tensor.amax().div(255.0).clamp_min(1.0e-30)
    quantized = torch.clamp(torch.round(tensor / scale), 0.0, 255.0).to(dtype=torch.uint8)
    return quantized, scale.to(dtype=torch.float32)


def _dequantize_positive(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize non-negative uint8 moments back to FP32."""

    return quantized.to(dtype=torch.float32).mul_(scale)


def _cpu_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    """Return an independent CPU copy of an optional quantized-state tensor."""

    return None if value is None else value.detach().to(device="cpu").clone()
