"""Packed block-wise 4-bit-state AdamW.

The first moment uses signed dynamic-exponent quantization.  The non-negative
second moment uses the zero-excluding linear map from Li et al. (NeurIPS 2023):
code ``i`` represents ``scale * (i + 1) / 16``.  Two codes are packed in each
byte and scales are stored per parameter-local block.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
import torch.distributed as dist

from areno.engine.optim.adamw_8bit import AdamW8bit
from areno.engine.optim.adamw_fp32_master import _DEFAULT_BUCKET_NUMEL, _MasterBucket, _ParamRef

_DEFAULT_QUANT_BLOCK_SIZE = 128
_STATE_FORMAT_VERSION = 1
# The signed 4-bit dynamic-exponent map used by the reference implementation
# of Li et al.  Values are normalized by each block's absolute maximum.
_SIGNED_DE_MAP = (
    -0.8875,
    -0.6625,
    -0.4375,
    -0.2125,
    -0.0775,
    -0.0325,
    -0.0055,
    0.0,
    0.0055,
    0.0325,
    0.0775,
    0.2125,
    0.4375,
    0.6625,
    0.8875,
    1.0,
)


class AdamW4bit(AdamW8bit):
    """AdamW with two packed 4-bit moments and FP32 block scales.

    Quantization blocks restart at every parameter shard.  This prevents one
    tensor's outlier from setting another tensor's scale and keeps temporary
    FP32 state bounded by ``quant_block_size`` on CPU. CUDA updates packed
    state directly with a fused block-wise kernel.
    """

    _embedding_fp32_state = False
    state_quantizer = "signed-de4/zero-excluding-linear4"

    def _precision_for_parameter(self, parameter: torch.nn.Parameter) -> str:
        del parameter
        return "8bit"

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        bucket_numel: int = _DEFAULT_BUCKET_NUMEL,
        quant_block_size: int = _DEFAULT_QUANT_BLOCK_SIZE,
        dp_rank: int = 0,
        dp_size: int = 1,
        dp_group: dist.ProcessGroup | None = None,
    ):
        if quant_block_size < 32 or quant_block_size > 1024 or quant_block_size & (quant_block_size - 1):
            raise ValueError("quant_block_size must be a power of two between 32 and 1024")
        self.quant_block_size = quant_block_size
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            bucket_numel=bucket_numel,
            dp_rank=dp_rank,
            dp_size=dp_size,
            dp_group=dp_group,
            quant_block_size=quant_block_size,
        )

    def state_dict(self) -> dict:
        """Return the versioned packed state for this DP rank."""

        payload = super().state_dict()
        payload.pop("adam_8bit", None)
        payload.pop("quantizer", None)
        payload.pop("precision_policy", None)
        payload.pop("state_memory", None)
        for state in payload["state"]:
            state.pop("precision", None)
            state.pop("quantizer", None)
            state.pop("exp_avg", None)
            state.pop("exp_avg_sq", None)
        payload["adam_4bit"] = True
        payload["state_format_version"] = _STATE_FORMAT_VERSION
        payload["quant_block_size"] = self.quant_block_size
        return payload

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore packed state, rejecting incompatible layouts explicitly."""

        version = int(state_dict.get("state_format_version", 0))
        if version != _STATE_FORMAT_VERSION:
            raise ValueError(f"unsupported AdamW4bit state format version: {version}")
        saved_block_size = int(state_dict.get("quant_block_size", 0))
        if saved_block_size != self.quant_block_size:
            raise ValueError(
                f"AdamW4bit quant_block_size mismatch: checkpoint={saved_block_size}, optimizer={self.quant_block_size}"
            )
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
            packed_numel, scale_numel = self._bucket_state_sizes(bucket)
            state.step = int(saved.get("step", 0))
            state.exp_avg_q = _load_tensor(saved, "exp_avg_q", device, torch.uint8, packed_numel)
            state.exp_avg_scale = _load_tensor(saved, "exp_avg_scale", device, torch.float32, scale_numel)
            state.exp_avg_sq_q = _load_tensor(saved, "exp_avg_sq_q", device, torch.uint8, packed_numel)
            state.exp_avg_sq_scale = _load_tensor(saved, "exp_avg_sq_scale", device, torch.float32, scale_numel)

    @torch.no_grad()
    def _ensure_bucket_state(self, bucket: _MasterBucket, state) -> None:
        """Materialize packed moments and per-block scales for one bucket."""

        device = bucket.refs[0].model_param.device
        if state.offload_file is not None:
            self._load_state_offload(state, device)
        for name in ("exp_avg_q", "exp_avg_scale", "exp_avg_sq_q", "exp_avg_sq_scale"):
            value = getattr(state, name)
            if value is not None and value.device != device:
                setattr(state, name, value.to(device=device))
        packed_numel, scale_numel = self._bucket_state_sizes(bucket)
        if state.exp_avg_q is None:
            # Signed dynamic-exponent zero has code 7, hence byte 0x77.
            state.exp_avg_q = torch.full((packed_numel,), 0x77, device=device, dtype=torch.uint8)
            state.exp_avg_scale = torch.ones(scale_numel, device=device, dtype=torch.float32)
        if state.exp_avg_sq_q is None:
            state.exp_avg_sq_q = torch.zeros(packed_numel, device=device, dtype=torch.uint8)
            # A zero scale makes the zero-excluding code initially decode to 0.
            state.exp_avg_sq_scale = torch.zeros(scale_numel, device=device, dtype=torch.float32)

    def _state_mmap_specs(self, indices: list[int]) -> dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]]:
        """Return fixed raw-mmap layouts for packed state and block scales."""

        specs: dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]] = {}
        for index in indices:
            packed_numel, scale_numel = self._bucket_state_sizes(self.buckets[index])
            specs[index] = {
                "exp_avg_q": (torch.uint8, (packed_numel,)),
                "exp_avg_scale": (torch.float32, (scale_numel,)),
                "exp_avg_sq_q": (torch.uint8, (packed_numel,)),
                "exp_avg_sq_scale": (torch.float32, (scale_numel,)),
            }
        return specs

    @torch.no_grad()
    def _step_bucket_8bit(self, bucket: _MasterBucket, state) -> None:
        """Update a bucket while materializing at most one FP32 block per moment."""

        assert state.exp_avg_q is not None
        assert state.exp_avg_scale is not None
        assert state.exp_avg_sq_q is not None
        assert state.exp_avg_sq_scale is not None
        beta1, beta2 = self.betas
        state.step += 1
        bias_correction1 = 1.0 - beta1**state.step
        bias_correction2_sqrt = (1.0 - beta2**state.step) ** 0.5
        for ref, packed_offset, scale_offset in self._iter_ref_layout(bucket):
            grad = self._gradient_for_ref(bucket, ref)
            if grad is None:
                continue
            effective_lr = float(getattr(ref.model_param, "_areno_lr", self.lr))
            self._step_param_ref_4bit(
                bucket,
                ref,
                grad,
                state,
                packed_offset,
                scale_offset,
                beta1,
                beta2,
                effective_lr,
                effective_lr / bias_correction1,
                bias_correction2_sqrt,
            )
            if ref.param_start + ref.numel == ref.model_param.numel():
                ref.model_param.grad = None
                if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                    ref.model_param.main_grad = None
        self._all_gather_bucket(bucket)
        bucket.grad_shard = None
        bucket.grad_param_ids = frozenset()

    @torch.no_grad()
    def _step_param_ref_4bit(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        state,
        packed_offset: int,
        scale_offset: int,
        beta1: float,
        beta2: float,
        effective_lr: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Apply AdamW to one parameter shard in block-sized work buffers."""

        if ref.shard_numel == 0:
            return
        grad_shard = grad if bucket.grad_shard is not None else grad.narrow(0, ref.shard_start, ref.shard_numel)
        model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
        model_shard = model_chunk.narrow(0, ref.shard_start, ref.shard_numel)
        if model_shard.is_cuda:
            from areno.accel.optimizer import areno_adamw_4bit_step

            areno_adamw_4bit_step(
                model_shard,
                grad_shard.contiguous(),
                state.exp_avg_q,
                state.exp_avg_scale,
                state.exp_avg_sq_q,
                state.exp_avg_sq_scale,
                packed_offset=packed_offset,
                scale_offset=scale_offset,
                quant_block_size=self.quant_block_size,
                beta1=beta1,
                beta2=beta2,
                effective_lr=effective_lr,
                weight_decay=self.weight_decay,
                eps=self.eps,
                step_size=step_size,
                bias_correction2_sqrt=bias_correction2_sqrt,
            )
            return
        for block_index, start in enumerate(range(0, ref.shard_numel, self.quant_block_size)):
            count = min(self.quant_block_size, ref.shard_numel - start)
            byte_start = packed_offset + start // 2
            byte_count = (count + 1) // 2
            scale_index = scale_offset + block_index
            moment = _unpack_signed_4bit(
                state.exp_avg_q.narrow(0, byte_start, byte_count),
                count,
                state.exp_avg_scale[scale_index],
            )
            variance = _unpack_positive_4bit(
                state.exp_avg_sq_q.narrow(0, byte_start, byte_count),
                count,
                state.exp_avg_sq_scale[scale_index],
            )
            grad_block = grad_shard.narrow(0, start, count).to(dtype=torch.float32)
            weight = model_shard.narrow(0, start, count).to(dtype=torch.float32)
            if not (
                torch.isfinite(grad_block).all()
                and torch.isfinite(weight).all()
                and torch.isfinite(moment).all()
                and torch.isfinite(variance).all()
            ):
                # Match the fused CUDA path: a bad block must not poison its
                # packed state or any neighboring block.
                continue
            if self.weight_decay != 0.0:
                weight.mul_(1.0 - effective_lr * self.weight_decay)
            moment.mul_(beta1).add_(grad_block, alpha=1.0 - beta1)
            variance.mul_(beta2).addcmul_(grad_block, grad_block, value=1.0 - beta2)
            denom = variance.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            weight.addcdiv_(moment, denom, value=-step_size)
            model_shard.narrow(0, start, count).copy_(weight)
            moment_q, moment_scale = _quantize_signed_4bit(moment)
            variance_q, variance_scale = _quantize_positive_4bit(variance)
            state.exp_avg_q.narrow(0, byte_start, byte_count).copy_(moment_q)
            state.exp_avg_scale[scale_index].copy_(moment_scale)
            state.exp_avg_sq_q.narrow(0, byte_start, byte_count).copy_(variance_q)
            state.exp_avg_sq_scale[scale_index].copy_(variance_scale)

    def _bucket_state_sizes(self, bucket: _MasterBucket) -> tuple[int, int]:
        packed_numel = sum((ref.shard_numel + 1) // 2 for ref in bucket.refs)
        scale_numel = sum((ref.shard_numel + self.quant_block_size - 1) // self.quant_block_size for ref in bucket.refs)
        return packed_numel, scale_numel

    def _iter_ref_layout(self, bucket: _MasterBucket) -> Iterator[tuple[_ParamRef, int, int]]:
        packed_offset = 0
        scale_offset = 0
        for ref in bucket.refs:
            yield ref, packed_offset, scale_offset
            packed_offset += (ref.shard_numel + 1) // 2
            scale_offset += (ref.shard_numel + self.quant_block_size - 1) // self.quant_block_size

    def persistent_moment_bytes(self) -> int:
        """Return resident packed-moment and scale storage in bytes."""

        total = 0
        for state in self._states:
            for value in (state.exp_avg_q, state.exp_avg_scale, state.exp_avg_sq_q, state.exp_avg_sq_scale):
                if value is not None:
                    total += value.numel() * value.element_size()
        return total


def _load_tensor(
    saved: dict,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    expected_numel: int,
) -> torch.Tensor | None:
    value = saved.get(name)
    if value is None:
        return None
    result = value.detach().to(device=device, dtype=dtype).view(-1).clone()
    if result.numel() != expected_numel:
        raise ValueError(f"AdamW4bit {name} has {result.numel()} values, expected {expected_numel}")
    return result


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 values in [0, 15], low nibble first."""

    if codes.numel() == 0:
        return codes.to(dtype=torch.uint8)
    codes = codes.to(dtype=torch.uint8).view(-1)
    if codes.numel() % 2:
        codes = torch.cat((codes, torch.zeros(1, device=codes.device, dtype=torch.uint8)))
    return codes[0::2] | (codes[1::2] << 4)


def _unpack_nibbles(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack low/high nibbles into a uint8 vector."""

    result = torch.empty(packed.numel() * 2, device=packed.device, dtype=torch.uint8)
    result[0::2] = packed & 0x0F
    result[1::2] = packed >> 4
    return result[:numel]


def _quantize_signed_4bit(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a signed block with the paper's dynamic-exponent map."""

    if tensor.numel() == 0:
        return tensor.to(dtype=torch.uint8), torch.ones((), device=tensor.device, dtype=torch.float32)
    scale = tensor.abs().amax().to(dtype=torch.float32)
    mapping = tensor.new_tensor(_SIGNED_DE_MAP, dtype=torch.float32)
    normalized = tensor / scale.clamp_min(1.0e-30)
    codes = torch.argmin((normalized.unsqueeze(-1) - mapping).abs(), dim=-1).to(dtype=torch.uint8)
    return _pack_nibbles(codes), scale.to(dtype=torch.float32)


def _unpack_signed_4bit(packed: torch.Tensor, numel: int, scale: torch.Tensor) -> torch.Tensor:
    mapping = packed.new_tensor(_SIGNED_DE_MAP, dtype=torch.float32)
    return mapping[_unpack_nibbles(packed, numel).to(dtype=torch.long)].mul_(scale)


def _quantize_positive_4bit(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with T(i)=(i+1)/16, deliberately excluding zero."""

    if tensor.numel() == 0:
        return tensor.to(dtype=torch.uint8), torch.zeros((), device=tensor.device, dtype=torch.float32)
    scale = tensor.amax().to(dtype=torch.float32)
    safe_scale = scale.clamp_min(1.0e-30)
    codes = torch.clamp(torch.round(tensor / safe_scale * 16.0 - 1.0), 0.0, 15.0).to(dtype=torch.uint8)
    return _pack_nibbles(codes), scale


def _unpack_positive_4bit(packed: torch.Tensor, numel: int, scale: torch.Tensor) -> torch.Tensor:
    return (_unpack_nibbles(packed, numel).to(dtype=torch.float32) + 1.0).mul_(scale / 16.0)


__all__ = ["AdamW4bit"]
