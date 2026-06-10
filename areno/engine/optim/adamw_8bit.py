"""8-bit-state AdamW with the same DP-sharded contract as AdamWFP32Master."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist

from areno.engine.optim.adamw_fp32_master import (
    _DEFAULT_BUCKET_NUMEL,
    _MasterBucket,
    _ParamRef,
    _param_grad,
    AdamWFP32Master,
)


@dataclass(slots=True)
class _Adam8bitBucketState:
    """Quantized Adam moments for one DP shard of an optimizer bucket."""

    step: int = 0
    exp_avg_q: torch.Tensor | None = None
    exp_avg_scale: torch.Tensor | None = None
    exp_avg_sq_q: torch.Tensor | None = None
    exp_avg_sq_scale: torch.Tensor | None = None


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
        for bucket, state in zip(self.buckets, self._states, strict=True):
            has_grad = False
            for ref in bucket.refs:
                has_grad = has_grad or _param_grad(ref.model_param) is not None
            if has_grad:
                self._ensure_bucket_state(bucket, state)
                self._step_bucket_8bit(bucket, state)
        return None

    def clear_state(self) -> None:
        """Drop quantized moments and reset step counters."""

        for state in self._states:
            state.step = 0
            state.exp_avg_q = None
            state.exp_avg_scale = None
            state.exp_avg_sq_q = None
            state.exp_avg_sq_scale = None

    @torch.no_grad()
    def offload_state(self) -> None:
        """Move quantized optimizer state to CPU between train phases."""

        for state in self._states:
            if state.exp_avg_q is not None:
                state.exp_avg_q = state.exp_avg_q.to(device="cpu")
            if state.exp_avg_scale is not None:
                state.exp_avg_scale = state.exp_avg_scale.to(device="cpu")
            if state.exp_avg_sq_q is not None:
                state.exp_avg_sq_q = state.exp_avg_sq_q.to(device="cpu")
            if state.exp_avg_sq_scale is not None:
                state.exp_avg_sq_scale = state.exp_avg_sq_scale.to(device="cpu")

    @torch.no_grad()
    def onload_state(self, device: torch.device) -> None:
        """Move quantized optimizer state back to the training device."""

        for state in self._states:
            if state.exp_avg_q is not None and state.exp_avg_q.device != device:
                state.exp_avg_q = state.exp_avg_q.to(device=device)
            if state.exp_avg_scale is not None and state.exp_avg_scale.device != device:
                state.exp_avg_scale = state.exp_avg_scale.to(device=device)
            if state.exp_avg_sq_q is not None and state.exp_avg_sq_q.device != device:
                state.exp_avg_sq_q = state.exp_avg_sq_q.to(device=device)
            if state.exp_avg_sq_scale is not None and state.exp_avg_sq_scale.device != device:
                state.exp_avg_sq_scale = state.exp_avg_sq_scale.to(device=device)

    def state_dict(self) -> dict:
        """Return per-rank quantized optimizer state."""

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
                    "exp_avg_q": state.exp_avg_q.detach().clone() if state.exp_avg_q is not None else None,
                    "exp_avg_scale": state.exp_avg_scale.detach().clone() if state.exp_avg_scale is not None else None,
                    "exp_avg_sq_q": state.exp_avg_sq_q.detach().clone() if state.exp_avg_sq_q is not None else None,
                    "exp_avg_sq_scale": state.exp_avg_sq_scale.detach().clone() if state.exp_avg_sq_scale is not None else None,
                }
                for state in self._states
            ],
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore quantized optimizer state from this rank's checkpoint."""

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
            state.exp_avg_q = None if exp_avg_q is None else exp_avg_q.detach().to(device=device, dtype=torch.uint8).view(-1).clone()
            state.exp_avg_scale = None if exp_avg_scale is None else exp_avg_scale.detach().to(device=device, dtype=torch.float32).view(()).clone()
            state.exp_avg_sq_q = None if exp_avg_sq_q is None else exp_avg_sq_q.detach().to(device=device, dtype=torch.uint8).view(-1).clone()
            state.exp_avg_sq_scale = None if exp_avg_sq_scale is None else exp_avg_sq_scale.detach().to(device=device, dtype=torch.float32).view(()).clone()

    @torch.no_grad()
    def _ensure_bucket_state(self, bucket: _MasterBucket, state: _Adam8bitBucketState) -> None:
        """Materialize or onload quantized moments for one bucket."""

        device = bucket.refs[0].model_param.device
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
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5

        exp_avg = _dequantize_symmetric(state.exp_avg_q, state.exp_avg_scale)
        exp_avg_sq = _dequantize_positive(state.exp_avg_sq_q, state.exp_avg_sq_scale)
        updated_refs: list[_ParamRef] = []
        for ref in bucket.refs:
            grad = _param_grad(ref.model_param)
            if grad is None:
                continue
            flat_grad = grad.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
            self._step_param_ref_8bit(bucket, ref, flat_grad, exp_avg, exp_avg_sq, beta1, beta2, step_size, bias_correction2_sqrt)
            updated_refs.append(ref)
            if ref.param_start + ref.numel == ref.model_param.numel():
                ref.model_param.grad = None
                if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                    ref.model_param.main_grad = None
        state.exp_avg_q, state.exp_avg_scale = _quantize_symmetric(exp_avg)
        state.exp_avg_sq_q, state.exp_avg_sq_scale = _quantize_positive(exp_avg_sq)
        if updated_refs:
            self._all_gather_bucket(bucket, updated_refs)

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
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Apply one AdamW update to this rank's shard of one param chunk."""

        if ref.shard_numel == 0:
            return
        grad_shard = grad.narrow(0, ref.shard_start, ref.shard_numel).to(dtype=torch.float32)
        model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
        model_shard = model_chunk.narrow(0, ref.shard_start, ref.shard_numel)
        weight = model_shard.to(dtype=torch.float32)
        if self.weight_decay != 0.0:
            weight.mul_(1.0 - self.lr * self.weight_decay)
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
