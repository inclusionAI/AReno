"""Packed rank-1-normalized 4-bit-state AdamW.

The first moment uses signed dynamic-exponent quantization.  The non-negative
second moment uses the zero-excluding linear map from Li et al. (NeurIPS 2023):
code ``i`` represents ``scale * (i + 1) / 16``. Matrix and higher-rank second
moments use the paper's rank-1 normalization; vectors retain B=128 block
normalization. Two codes are packed in each byte.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from math import prod

import torch
import torch.distributed as dist

from areno.engine.optim.adamw_8bit import AdamW8bit
from areno.engine.optim.adamw_fp32_master import _DEFAULT_BUCKET_NUMEL, _MasterBucket, _param_grad, _ParamRef

_DEFAULT_QUANT_BLOCK_SIZE = 128
_STATE_FORMAT_VERSION = 2
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
    """AdamW with two packed 4-bit moments and shape-aware FP32 scales.

    First-moment quantization blocks restart at every parameter shard. For a
    tensor with rank >= 2, second-moment scales are the minimum of per-axis
    maxima over the original local model-tensor shape. DP ranks combine their
    partial axis maxima with MAX. One-dimensional tensors retain parameter-
    local block scales. CPU and CUDA updates keep FP32 moment work bounded by
    ``quant_block_size``.
    """

    _embedding_fp32_state = False
    state_quantizer = "signed-de4/rank1-zero-excluding-linear4"

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
        self._rank1_metadata_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._rank1_scales: dict[int, torch.Tensor | None] = {
            id(parameter): None for parameter in self.model_params if parameter.ndim >= 2
        }

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
        payload["parameter_shapes"] = [tuple(parameter.shape) for parameter in self.model_params]
        payload["rank1_scales"] = [
            None
            if self._rank1_scales.get(id(parameter)) is None
            else self._rank1_scales[id(parameter)].detach().to(device="cpu").clone()
            for parameter in self.model_params
        ]
        payload["state_memory"] = self.state_memory_metrics()
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
        saved_shapes = [tuple(shape) for shape in state_dict.get("parameter_shapes", ())]
        current_shapes = [tuple(parameter.shape) for parameter in self.model_params]
        if saved_shapes != current_shapes:
            raise ValueError("AdamW4bit checkpoint parameter shapes do not match the optimizer")
        saved_rank1_scales = state_dict.get("rank1_scales")
        if not isinstance(saved_rank1_scales, list) or len(saved_rank1_scales) != len(self.model_params):
            raise ValueError("AdamW4bit checkpoint rank-1 scales do not match the optimizer parameters")
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
        if len(saved_states) != len(self.buckets):
            raise ValueError(
                "AdamW4bit checkpoint bucket count does not match the optimizer layout: "
                f"checkpoint={len(saved_states)}, optimizer={len(self.buckets)}"
            )
        for saved, bucket, state in zip(saved_states[: len(self.buckets)], self.buckets, self._states, strict=False):
            if saved is None:
                continue
            device = bucket.refs[0].model_param.device
            packed_numel, moment_scale_numel, variance_scale_numel = self._bucket_state_sizes(bucket)
            state.step = int(saved.get("step", 0))
            state.exp_avg_q = _load_tensor(saved, "exp_avg_q", device, torch.uint8, packed_numel)
            state.exp_avg_scale = _load_tensor(saved, "exp_avg_scale", device, torch.float32, moment_scale_numel)
            state.exp_avg_sq_q = _load_tensor(saved, "exp_avg_sq_q", device, torch.uint8, packed_numel)
            state.exp_avg_sq_scale = _load_tensor(
                saved, "exp_avg_sq_scale", device, torch.float32, variance_scale_numel
            )
        for parameter, saved_scales in zip(self.model_params, saved_rank1_scales, strict=True):
            if parameter.ndim < 2:
                if saved_scales is not None:
                    raise ValueError("AdamW4bit checkpoint has rank-1 scales for a one-dimensional parameter")
                continue
            if saved_scales is None:
                self._rank1_scales[id(parameter)] = None
                continue
            restored_scales = saved_scales.detach().to(device=parameter.device, dtype=torch.float32).view(-1).clone()
            if restored_scales.numel() != sum(parameter.shape):
                raise ValueError("AdamW4bit checkpoint rank-1 scale length does not match the parameter shape")
            self._rank1_scales[id(parameter)] = restored_scales

    def clear_state(self) -> None:
        """Drop packed moments and parameter-level rank-1 metadata."""

        super().clear_state()
        for parameter_id in self._rank1_scales:
            self._rank1_scales[parameter_id] = None
        self._rank1_metadata_cache.clear()

    @torch.no_grad()
    def offload_state(self, mode: str = "cpu", directory: str | None = None, batch_size: int = 1) -> None:
        """Offload packed buckets and keep small rank-1 metadata on CPU."""

        super().offload_state(mode=mode, directory=directory, batch_size=batch_size)
        for parameter_id, scales in self._rank1_scales.items():
            if scales is not None and scales.device.type != "cpu":
                self._rank1_scales[parameter_id] = scales.to(device="cpu")
        self._rank1_metadata_cache.clear()

    @torch.no_grad()
    def onload_state(self, device: torch.device) -> None:
        """Restore packed buckets and shared rank-1 metadata to ``device``."""

        super().onload_state(device)
        for parameter_id, scales in self._rank1_scales.items():
            if scales is not None and scales.device != device:
                self._rank1_scales[parameter_id] = scales.to(device=device)
        self._rank1_metadata_cache.clear()

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
        packed_numel, moment_scale_numel, variance_scale_numel = self._bucket_state_sizes(bucket)
        if state.exp_avg_q is None:
            # Signed dynamic-exponent zero has code 7, hence byte 0x77.
            state.exp_avg_q = torch.full((packed_numel,), 0x77, device=device, dtype=torch.uint8)
            state.exp_avg_scale = torch.ones(moment_scale_numel, device=device, dtype=torch.float32)
        if state.exp_avg_sq_q is None:
            state.exp_avg_sq_q = torch.zeros(packed_numel, device=device, dtype=torch.uint8)
            # A zero scale makes the zero-excluding code initially decode to 0.
            state.exp_avg_sq_scale = torch.zeros(variance_scale_numel, device=device, dtype=torch.float32)

    def _state_mmap_specs(self, indices: list[int]) -> dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]]:
        """Return fixed raw-mmap layouts for packed state and block scales."""

        specs: dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]] = {}
        for index in indices:
            packed_numel, moment_scale_numel, variance_scale_numel = self._bucket_state_sizes(self.buckets[index])
            specs[index] = {
                "exp_avg_q": (torch.uint8, (packed_numel,)),
                "exp_avg_scale": (torch.float32, (moment_scale_numel,)),
                "exp_avg_sq_q": (torch.uint8, (packed_numel,)),
                "exp_avg_sq_scale": (torch.float32, (variance_scale_numel,)),
            }
        return specs

    @torch.no_grad()
    def step(self, closure=None):
        """Apply a streaming statistics pass before rank-1 parameter updates."""

        if closure is not None:
            with torch.enable_grad():
                closure()
        active_rank1_parameters = {
            id(ref.model_param)
            for bucket in self.buckets
            for ref in bucket.refs
            if _uses_rank1_normalization(ref) and self._ref_has_gradient(bucket, ref)
        }
        if not active_rank1_parameters:
            return super().step()

        rank1_work: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for parameter in self.model_params:
            if id(parameter) in active_rank1_parameters:
                rank1_work[id(parameter)] = (
                    torch.zeros(_rank1_scale_numel_for_parameter(parameter), device=parameter.device),
                    torch.zeros((), device=parameter.device, dtype=torch.int32),
                )
                self._ensure_rank1_scales(parameter)

        # First streaming pass: each chunk contributes to one parameter-level
        # set of axis maxima. Disk-offloaded packed state is returned to disk
        # after each group, so this pass does not pin all buckets on the GPU.
        for indices in self._bucket_groups():
            group_changed = False
            for index in indices:
                bucket = self.buckets[index]
                state = self._states[index]
                if not any(
                    _uses_rank1_normalization(ref) and self._ref_has_gradient(bucket, ref) for ref in bucket.refs
                ):
                    continue
                self._ensure_bucket_state(bucket, state)
                for ref, packed_offset, _moment_scale_offset, _variance_scale_offset in self._iter_ref_layout(bucket):
                    if not _uses_rank1_normalization(ref) or not self._ref_has_gradient(bucket, ref):
                        continue
                    updated_scales, invalid = rank1_work[id(ref.model_param)]
                    self._rank1_variance_statistics(
                        bucket,
                        ref,
                        self._gradient_for_ref(bucket, ref),
                        state,
                        packed_offset,
                        self._ensure_rank1_scales(ref.model_param),
                        updated_scales,
                        invalid,
                        self.betas[1],
                    )
                group_changed = True
                if self._active_offload_mode == "disk":
                    self._stage_8bit_state_on_cpu(state)
            if self._active_offload_mode == "disk" and group_changed:
                self._offload_8bit_group_to_disk(indices)

        for parameter in self.model_params:
            work = rank1_work.get(id(parameter))
            if work is None:
                continue
            updated_scales, invalid = work
            if self.dp_size > 1:
                if self.dp_group is None:
                    raise RuntimeError("AdamW4bit DP rank-1 normalization requires a DP process group")
                dist.all_reduce(updated_scales, op=dist.ReduceOp.MAX, group=self.dp_group)
                dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self.dp_group)

        # Second streaming pass: recompute each bounded block, update weights,
        # and write packed moments using the now-global axis scales.
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
                    self._step_bucket_8bit(bucket, state, rank1_work)
                    group_changed = True
                    if self._active_offload_mode == "disk":
                        self._stage_8bit_state_on_cpu(state)
                elif self._active_offload_mode == "disk":
                    self._discard_disk_prefetch(index)
            if self._active_offload_mode == "disk" and group_changed:
                self._offload_8bit_group_to_disk(indices)

        for parameter in self.model_params:
            work = rank1_work.get(id(parameter))
            if work is None:
                continue
            updated_scales, invalid = work
            scale_storage = self._ensure_rank1_scales(parameter)
            if invalid.is_cuda:
                scale_storage.copy_(torch.where(invalid == 0, updated_scales, scale_storage))
            elif int(invalid.item()) == 0:
                scale_storage.copy_(updated_scales)
            if self._active_offload_mode == "disk" and scale_storage.device.type != "cpu":
                self._rank1_scales[id(parameter)] = scale_storage.to(device="cpu")
        return None

    @torch.no_grad()
    def _step_bucket_8bit(
        self,
        bucket: _MasterBucket,
        state,
        rank1_work: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        """Update a bucket while materializing at most one FP32 block per moment."""

        assert state.exp_avg_q is not None
        assert state.exp_avg_scale is not None
        assert state.exp_avg_sq_q is not None
        assert state.exp_avg_sq_scale is not None
        beta1, beta2 = self.betas
        state.step += 1
        bias_correction1 = 1.0 - beta1**state.step
        bias_correction2_sqrt = (1.0 - beta2**state.step) ** 0.5
        for ref, packed_offset, moment_scale_offset, variance_scale_offset in self._iter_ref_layout(bucket):
            has_parameter_grad = (
                id(ref.model_param) in bucket.grad_param_ids
                if bucket.grad_shard is not None
                else ref.model_param.grad is not None
                or isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor)
            )
            if not has_parameter_grad:
                continue
            grad = self._gradient_for_ref(bucket, ref)
            effective_lr = float(getattr(ref.model_param, "_areno_lr", self.lr))
            if _uses_rank1_normalization(ref):
                if rank1_work is None:
                    raise RuntimeError("AdamW4bit rank-1 update is missing its parameter statistics pass")
                updated_scales, invalid = rank1_work[id(ref.model_param)]
                if grad is not None and (invalid.is_cuda or int(invalid.item()) == 0):
                    self._step_param_ref_rank1(
                        bucket,
                        ref,
                        grad,
                        state,
                        packed_offset,
                        moment_scale_offset,
                        self._ensure_rank1_scales(ref.model_param),
                        updated_scales,
                        invalid,
                        beta1,
                        beta2,
                        effective_lr,
                        effective_lr / bias_correction1,
                        bias_correction2_sqrt,
                    )
            elif grad is not None:
                self._step_param_ref_4bit(
                    bucket,
                    ref,
                    grad,
                    state,
                    packed_offset,
                    moment_scale_offset,
                    variance_scale_offset,
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

    @staticmethod
    def _ref_has_gradient(bucket: _MasterBucket, ref: _ParamRef) -> bool:
        if bucket.grad_shard is not None:
            return id(ref.model_param) in bucket.grad_param_ids
        return _param_grad(ref.model_param) is not None

    @torch.no_grad()
    def _step_param_ref_4bit(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        state,
        packed_offset: int,
        moment_scale_offset: int,
        variance_scale_offset: int,
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
                moment_scale_offset=moment_scale_offset,
                variance_scale_offset=variance_scale_offset,
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
            moment_scale_index = moment_scale_offset + block_index
            variance_scale_index = variance_scale_offset + block_index
            moment = _unpack_signed_4bit(
                state.exp_avg_q.narrow(0, byte_start, byte_count),
                count,
                state.exp_avg_scale[moment_scale_index],
            )
            variance = _unpack_positive_4bit(
                state.exp_avg_sq_q.narrow(0, byte_start, byte_count),
                count,
                state.exp_avg_sq_scale[variance_scale_index],
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
            state.exp_avg_scale[moment_scale_index].copy_(moment_scale)
            state.exp_avg_sq_q.narrow(0, byte_start, byte_count).copy_(variance_q)
            state.exp_avg_sq_scale[variance_scale_index].copy_(variance_scale)

    @torch.no_grad()
    def _rank1_variance_statistics(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor | None,
        state,
        packed_offset: int,
        previous_scales: torch.Tensor,
        updated_scales: torch.Tensor,
        invalid: torch.Tensor,
        beta2: float,
    ) -> None:
        """Compute updated per-axis maxima without a full FP32 moment."""

        assert state.exp_avg_sq_q is not None
        if grad is None or ref.shard_numel == 0:
            return
        grad_shard = grad if bucket.grad_shard is not None else grad.narrow(0, ref.shard_start, ref.shard_numel)
        parameter_shard_start = ref.param_start + ref.shard_start
        if grad_shard.is_cuda:
            from areno.accel.optimizer import areno_adamw_4bit_rank1_stats

            shape, strides = self._rank1_metadata(ref)
            areno_adamw_4bit_rank1_stats(
                grad_shard.contiguous(),
                state.exp_avg_sq_q,
                previous_scales,
                updated_scales,
                invalid,
                shape,
                strides,
                packed_offset=packed_offset,
                parameter_shard_start=parameter_shard_start,
                quant_block_size=self.quant_block_size,
                beta2=beta2,
            )
            return

        shape_tuple = tuple(ref.model_param.shape)
        for start in range(0, ref.shard_numel, self.quant_block_size):
            count = min(self.quant_block_size, ref.shard_numel - start)
            byte_start = packed_offset + start // 2
            byte_count = (count + 1) // 2
            flat_start = parameter_shard_start + start
            element_scales = _rank1_element_scales(previous_scales, shape_tuple, flat_start, count)
            variance = _unpack_positive_4bit_elementwise(
                state.exp_avg_sq_q.narrow(0, byte_start, byte_count), count, element_scales
            )
            gradient = grad_shard.narrow(0, start, count).to(dtype=torch.float32)
            variance.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            if not bool(torch.isfinite(gradient).all() & torch.isfinite(variance).all()):
                invalid.fill_(1)
                return
            _accumulate_rank1_maxima(updated_scales, variance, shape_tuple, flat_start)

    @torch.no_grad()
    def _step_param_ref_rank1(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        state,
        packed_offset: int,
        moment_scale_offset: int,
        previous_scales: torch.Tensor,
        updated_scales: torch.Tensor,
        invalid: torch.Tensor,
        beta1: float,
        beta2: float,
        effective_lr: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Recompute bounded Adam blocks and requantize with rank-1 scales."""

        assert state.exp_avg_q is not None
        assert state.exp_avg_scale is not None
        assert state.exp_avg_sq_q is not None
        if ref.shard_numel == 0:
            return
        grad_shard = grad if bucket.grad_shard is not None else grad.narrow(0, ref.shard_start, ref.shard_numel)
        parameter_shard_start = ref.param_start + ref.shard_start
        model_shard = ref.model_param.detach().reshape(-1).narrow(0, parameter_shard_start, ref.shard_numel)
        if model_shard.is_cuda:
            from areno.accel.optimizer import areno_adamw_4bit_rank1_step

            shape, strides = self._rank1_metadata(ref)
            areno_adamw_4bit_rank1_step(
                model_shard,
                grad_shard.contiguous(),
                state.exp_avg_q,
                state.exp_avg_scale,
                state.exp_avg_sq_q,
                previous_scales,
                updated_scales,
                invalid,
                shape,
                strides,
                packed_offset=packed_offset,
                moment_scale_offset=moment_scale_offset,
                parameter_shard_start=parameter_shard_start,
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

        shape_tuple = tuple(ref.model_param.shape)
        for block_index, start in enumerate(range(0, ref.shard_numel, self.quant_block_size)):
            count = min(self.quant_block_size, ref.shard_numel - start)
            byte_start = packed_offset + start // 2
            byte_count = (count + 1) // 2
            moment_scale_index = moment_scale_offset + block_index
            moment = _unpack_signed_4bit(
                state.exp_avg_q.narrow(0, byte_start, byte_count),
                count,
                state.exp_avg_scale[moment_scale_index],
            )
            flat_start = parameter_shard_start + start
            old_element_scales = _rank1_element_scales(previous_scales, shape_tuple, flat_start, count)
            variance = _unpack_positive_4bit_elementwise(
                state.exp_avg_sq_q.narrow(0, byte_start, byte_count), count, old_element_scales
            )
            gradient = grad_shard.narrow(0, start, count).to(dtype=torch.float32)
            weight = model_shard.narrow(0, start, count).to(dtype=torch.float32).clone()
            if self.weight_decay != 0.0:
                weight.mul_(1.0 - effective_lr * self.weight_decay)
            moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            variance.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            denom = variance.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            weight.addcdiv_(moment, denom, value=-step_size)
            if not bool(
                torch.isfinite(gradient).all()
                & torch.isfinite(moment).all()
                & torch.isfinite(variance).all()
                & torch.isfinite(weight).all()
            ):
                return
            new_element_scales = _rank1_element_scales(updated_scales, shape_tuple, flat_start, count)
            moment_q, moment_scale = _quantize_signed_4bit(moment)
            variance_q = _quantize_positive_4bit_elementwise(variance, new_element_scales)
            model_shard.narrow(0, start, count).copy_(weight)
            state.exp_avg_q.narrow(0, byte_start, byte_count).copy_(moment_q)
            state.exp_avg_scale[moment_scale_index].copy_(moment_scale)
            state.exp_avg_sq_q.narrow(0, byte_start, byte_count).copy_(variance_q)

    def _rank1_metadata(self, ref: _ParamRef) -> tuple[torch.Tensor, torch.Tensor]:
        key = (id(ref.model_param), ref.model_param.device)
        cached = self._rank1_metadata_cache.get(key)
        if cached is not None:
            return cached
        shape_tuple = tuple(int(dimension) for dimension in ref.model_param.shape)
        strides_tuple = _contiguous_strides(shape_tuple)
        cached = (
            torch.tensor(shape_tuple, device=ref.model_param.device, dtype=torch.int64),
            torch.tensor(strides_tuple, device=ref.model_param.device, dtype=torch.int64),
        )
        self._rank1_metadata_cache[key] = cached
        return cached

    def _ensure_rank1_scales(self, parameter: torch.nn.Parameter) -> torch.Tensor:
        """Materialize one shared axis-scale tensor for an original parameter."""

        key = id(parameter)
        scales = self._rank1_scales.get(key)
        if scales is None:
            scales = torch.zeros(
                _rank1_scale_numel_for_parameter(parameter),
                device=parameter.device,
                dtype=torch.float32,
            )
            self._rank1_scales[key] = scales
        elif scales.device != parameter.device:
            scales = scales.to(device=parameter.device)
            self._rank1_scales[key] = scales
        return scales

    def _bucket_state_sizes(self, bucket: _MasterBucket) -> tuple[int, int, int]:
        packed_numel = sum((ref.shard_numel + 1) // 2 for ref in bucket.refs)
        moment_scale_numel = sum(
            (ref.shard_numel + self.quant_block_size - 1) // self.quant_block_size for ref in bucket.refs
        )
        variance_scale_numel = sum(self._variance_scale_count(ref) for ref in bucket.refs)
        return packed_numel, moment_scale_numel, variance_scale_numel

    def _iter_ref_layout(self, bucket: _MasterBucket) -> Iterator[tuple[_ParamRef, int, int, int]]:
        packed_offset = 0
        moment_scale_offset = 0
        variance_scale_offset = 0
        for ref in bucket.refs:
            yield ref, packed_offset, moment_scale_offset, variance_scale_offset
            packed_offset += (ref.shard_numel + 1) // 2
            moment_scale_offset += (ref.shard_numel + self.quant_block_size - 1) // self.quant_block_size
            variance_scale_offset += self._variance_scale_count(ref)

    def _variance_scale_count(self, ref: _ParamRef) -> int:
        if _uses_rank1_normalization(ref):
            return 0
        return (ref.shard_numel + self.quant_block_size - 1) // self.quant_block_size

    def persistent_moment_bytes(self) -> int:
        """Return resident packed-moment and scale storage in bytes."""

        total = 0
        for state in self._states:
            for value in (state.exp_avg_q, state.exp_avg_scale, state.exp_avg_sq_q, state.exp_avg_sq_scale):
                if value is not None:
                    total += value.numel() * value.element_size()
        for value in self._rank1_scales.values():
            if value is not None:
                total += value.numel() * value.element_size()
        return total

    def state_memory_metrics(self) -> dict[str, int]:
        """Report actual packed moments and shape/block metadata bytes."""

        quantized_state_bytes = 0
        scale_metadata_bytes = 0
        for state in self._states:
            if state.step == 0:
                continue
            for value in (state.exp_avg_q, state.exp_avg_sq_q):
                if value is not None:
                    quantized_state_bytes += value.numel() * value.element_size()
            for value in (state.exp_avg_scale, state.exp_avg_sq_scale):
                if value is not None:
                    scale_metadata_bytes += value.numel() * value.element_size()
        for value in self._rank1_scales.values():
            if value is not None:
                scale_metadata_bytes += value.numel() * value.element_size()
        return {
            "quantized_state_bytes": quantized_state_bytes,
            "scale_metadata_bytes": scale_metadata_bytes,
            "total_bytes": quantized_state_bytes + scale_metadata_bytes,
        }


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


def _uses_rank1_normalization(ref: _ParamRef) -> bool:
    return ref.model_param.ndim >= 2


def _rank1_scale_numel(ref: _ParamRef) -> int:
    return _rank1_scale_numel_for_parameter(ref.model_param)


def _rank1_scale_numel_for_parameter(parameter: torch.nn.Parameter) -> int:
    return sum(int(dimension) for dimension in parameter.shape)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    for axis in range(len(shape)):
        strides.append(prod(shape[axis + 1 :]))
    return tuple(strides)


def _rank1_element_scales(
    axis_scales: torch.Tensor,
    shape: tuple[int, ...],
    flat_start: int,
    count: int,
) -> torch.Tensor:
    """Expand paper rank-1 statistics for only one bounded flat slice."""

    if count == 0:
        return axis_scales.new_empty((0,))
    flat_indices = torch.arange(flat_start, flat_start + count, device=axis_scales.device)
    result = torch.full((count,), torch.inf, device=axis_scales.device, dtype=torch.float32)
    axis_offset = 0
    for dimension, stride in zip(shape, _contiguous_strides(shape), strict=True):
        coordinates = torch.div(flat_indices, stride, rounding_mode="floor").remainder_(dimension)
        torch.minimum(
            result,
            axis_scales.narrow(0, axis_offset, dimension)[coordinates],
            out=result,
        )
        axis_offset += dimension
    return result


def _accumulate_rank1_maxima(
    axis_maxima: torch.Tensor,
    values: torch.Tensor,
    shape: tuple[int, ...],
    flat_start: int,
) -> None:
    """Accumulate per-axis maxima for a bounded flat slice."""

    flat_indices = torch.arange(flat_start, flat_start + values.numel(), device=values.device)
    axis_offset = 0
    for dimension, stride in zip(shape, _contiguous_strides(shape), strict=True):
        coordinates = torch.div(flat_indices, stride, rounding_mode="floor").remainder_(dimension)
        axis_maxima.narrow(0, axis_offset, dimension).scatter_reduce_(
            0, coordinates, values, reduce="amax", include_self=True
        )
        axis_offset += dimension


def _unpack_positive_4bit_elementwise(
    packed: torch.Tensor,
    numel: int,
    scales: torch.Tensor,
) -> torch.Tensor:
    codes = _unpack_nibbles(packed, numel).to(dtype=torch.float32)
    return (codes + 1.0).mul_(scales / 16.0)


def _quantize_positive_4bit_elementwise(tensor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    safe_scales = scales.clamp_min(1.0e-30)
    codes = torch.clamp(torch.round(tensor / safe_scales * 16.0 - 1.0), 0.0, 15.0).to(dtype=torch.uint8)
    return _pack_nibbles(codes)


__all__ = ["AdamW4bit"]
