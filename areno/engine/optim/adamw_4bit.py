"""Packed 4-bit first-moment AdamW with factored second moments.

The first moment uses signed dynamic-exponent quantization.  The non-negative
second moment uses the zero-excluding linear map from Li et al. (NeurIPS 2023):
code ``i`` represents ``scale * (i + 1) / 16``. Vectors retain that B=128
representation. Matrix and higher-rank tensors instead use Adafactor-style
row/column second-moment statistics, eliminating their elementwise variance
state. Two first-moment codes are packed in each byte.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
import torch.distributed as dist

from areno.engine.optim.adamw_8bit import AdamW8bit
from areno.engine.optim.adamw_fp32_master import _DEFAULT_BUCKET_NUMEL, _MasterBucket, _param_grad, _ParamRef

_DEFAULT_QUANT_BLOCK_SIZE = 128
_STATE_FORMAT_VERSION = 3
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
    """AdamW with packed 4-bit momentum and factored matrix variance.

    First-moment quantization blocks restart at every parameter shard. For a
    tensor with rank >= 2, the second moment is represented by row and column
    means over the original TP-local tensor, flattened as ``[shape[0], -1]``.
    DP ranks combine partial sums before applying the exponential update.
    One-dimensional tensors retain parameter-local packed B=128 variance.
    CPU and CUDA updates keep elementwise FP32 work bounded by
    ``quant_block_size``.
    """

    _embedding_fp32_state = False
    gradient_shard_dtype = torch.bfloat16
    stream_gradient_shards = True
    state_quantizer = "signed-de4/factored-second-moment-v1"

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
        self._factored_second_moments: dict[int, torch.Tensor | None] = {
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
        payload["factored_second_moments"] = [
            None
            if self._factored_second_moments.get(id(parameter)) is None
            else self._factored_second_moments[id(parameter)].detach().to(device="cpu").clone()
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
        saved_factors = state_dict.get("factored_second_moments")
        if not isinstance(saved_factors, list) or len(saved_factors) != len(self.model_params):
            raise ValueError("AdamW4bit checkpoint factored moments do not match the optimizer parameters")
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
            moment_packed_numel, moment_scale_numel, variance_packed_numel, variance_scale_numel = (
                self._bucket_state_sizes(bucket)
            )
            state.step = int(saved.get("step", 0))
            state.exp_avg_q = _load_tensor(saved, "exp_avg_q", device, torch.uint8, moment_packed_numel)
            state.exp_avg_scale = _load_tensor(saved, "exp_avg_scale", device, torch.float32, moment_scale_numel)
            state.exp_avg_sq_q = _load_tensor(saved, "exp_avg_sq_q", device, torch.uint8, variance_packed_numel)
            state.exp_avg_sq_scale = _load_tensor(
                saved, "exp_avg_sq_scale", device, torch.float32, variance_scale_numel
            )
        for parameter, saved_factors_for_parameter in zip(self.model_params, saved_factors, strict=True):
            if parameter.ndim < 2:
                if saved_factors_for_parameter is not None:
                    raise ValueError("AdamW4bit checkpoint has factored state for a one-dimensional parameter")
                continue
            if saved_factors_for_parameter is None:
                self._factored_second_moments[id(parameter)] = None
                continue
            restored_factors = (
                saved_factors_for_parameter.detach().to(device=parameter.device, dtype=torch.float32).view(-1).clone()
            )
            if restored_factors.numel() != _factored_state_numel_for_parameter(parameter):
                raise ValueError("AdamW4bit checkpoint factored state length does not match the parameter shape")
            self._factored_second_moments[id(parameter)] = restored_factors

    def clear_state(self) -> None:
        """Drop packed moments and parameter-level factored state."""

        super().clear_state()
        for parameter_id in self._factored_second_moments:
            self._factored_second_moments[parameter_id] = None

    @torch.no_grad()
    def offload_state(self, mode: str = "cpu", directory: str | None = None, batch_size: int = 1) -> None:
        """Offload packed buckets and keep small factored state on CPU."""

        super().offload_state(mode=mode, directory=directory, batch_size=batch_size)
        for parameter_id, factors in self._factored_second_moments.items():
            if factors is not None and factors.device.type != "cpu":
                self._factored_second_moments[parameter_id] = factors.to(device="cpu")

    @torch.no_grad()
    def onload_state(self, device: torch.device) -> None:
        """Restore packed buckets and shared factored state to ``device``."""

        super().onload_state(device)
        for parameter_id, factors in self._factored_second_moments.items():
            if factors is not None and factors.device != device:
                self._factored_second_moments[parameter_id] = factors.to(device=device)

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
        moment_packed_numel, moment_scale_numel, variance_packed_numel, variance_scale_numel = self._bucket_state_sizes(
            bucket
        )
        if state.exp_avg_q is None:
            # Signed dynamic-exponent zero has code 7, hence byte 0x77.
            state.exp_avg_q = torch.full((moment_packed_numel,), 0x77, device=device, dtype=torch.uint8)
            state.exp_avg_scale = torch.ones(moment_scale_numel, device=device, dtype=torch.float32)
        if state.exp_avg_sq_q is None:
            state.exp_avg_sq_q = torch.zeros(variance_packed_numel, device=device, dtype=torch.uint8)
            # A zero scale makes the zero-excluding code initially decode to 0.
            state.exp_avg_sq_scale = torch.zeros(variance_scale_numel, device=device, dtype=torch.float32)

    def _state_mmap_specs(self, indices: list[int]) -> dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]]:
        """Return fixed raw-mmap layouts for packed state and block scales."""

        specs: dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]] = {}
        for index in indices:
            moment_packed_numel, moment_scale_numel, variance_packed_numel, variance_scale_numel = (
                self._bucket_state_sizes(self.buckets[index])
            )
            specs[index] = {
                "exp_avg_q": (torch.uint8, (moment_packed_numel,)),
                "exp_avg_scale": (torch.float32, (moment_scale_numel,)),
                "exp_avg_sq_q": (torch.uint8, (variance_packed_numel,)),
                "exp_avg_sq_scale": (torch.float32, (variance_scale_numel,)),
            }
        return specs

    @torch.no_grad()
    def step(self, closure=None):
        """Update one complete parameter at a time and release ready buckets."""

        if closure is not None:
            with torch.enable_grad():
                closure()
        layouts: dict[int, list[tuple[int, _ParamRef, int, int, int, int]]] = {}
        remaining_by_bucket: dict[int, int] = {}
        for index, bucket in enumerate(self.buckets):
            for ref, moment_packed, moment_scale, variance_packed, variance_scale in self._iter_ref_layout(bucket):
                if not self._ref_has_gradient(bucket, ref):
                    continue
                layouts.setdefault(id(ref.model_param), []).append(
                    (index, ref, moment_packed, moment_scale, variance_packed, variance_scale)
                )
                remaining_by_bucket[index] = remaining_by_bucket.get(index, 0) + 1
        if not layouts:
            return None

        beta1, beta2 = self.betas
        started_buckets: set[int] = set()
        completed_buckets: set[int] = set()

        for parameter in self.model_params:
            parameter_layouts = layouts.get(id(parameter))
            if not parameter_layouts:
                continue
            factored_work: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
            if parameter.ndim >= 2:
                previous_factors = self._ensure_factored_second_moment(parameter)
                factor_sums = torch.zeros_like(previous_factors)
                invalid = torch.zeros((), device=parameter.device, dtype=torch.int32)
                for index, ref, _moment_packed, _moment_scale, _variance_packed, _variance_scale in parameter_layouts:
                    bucket = self.buckets[index]
                    self._factored_variance_statistics(
                        bucket,
                        ref,
                        self._gradient_for_ref(bucket, ref),
                        factor_sums,
                        invalid,
                    )
                if self.dp_size > 1:
                    if self.dp_group is None:
                        raise RuntimeError("AdamW4bit factored second moments require a DP process group")
                    dist.all_reduce(factor_sums, op=dist.ReduceOp.SUM, group=self.dp_group)
                    dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self.dp_group)
                updated_factors = self._finalize_factored_statistics(parameter, previous_factors, factor_sums, beta2)
                row_mean = updated_factors[: parameter.shape[0]].mean()
                factored_work = updated_factors, row_mean, invalid

            for index, ref, moment_packed, moment_scale, variance_packed, variance_scale in parameter_layouts:
                bucket = self.buckets[index]
                state = self._states[index]
                if index not in started_buckets:
                    self._ensure_bucket_state(bucket, state)
                    state.step += 1
                    started_buckets.add(index)
                grad = self._gradient_for_ref(bucket, ref)
                if grad is not None:
                    effective_lr = float(getattr(parameter, "_areno_lr", self.lr))
                    bias_correction1 = 1.0 - beta1**state.step
                    bias_correction2_sqrt = (1.0 - beta2**state.step) ** 0.5
                    if factored_work is None:
                        self._step_param_ref_4bit(
                            bucket,
                            ref,
                            grad,
                            state,
                            moment_packed,
                            moment_scale,
                            variance_packed,
                            variance_scale,
                            beta1,
                            beta2,
                            effective_lr,
                            effective_lr / bias_correction1,
                            bias_correction2_sqrt,
                        )
                    else:
                        updated_factors, row_mean, invalid = factored_work
                        self._step_param_ref_factored(
                            bucket,
                            ref,
                            grad,
                            state,
                            moment_packed,
                            moment_scale,
                            updated_factors,
                            row_mean,
                            invalid,
                            beta1,
                            effective_lr,
                            effective_lr / bias_correction1,
                            bias_correction2_sqrt,
                        )
                remaining_by_bucket[index] -= 1
                if remaining_by_bucket[index] == 0:
                    self._all_gather_bucket(bucket)
                    bucket.grad_shard = None
                    bucket.grad_param_ids = frozenset()
                    completed_buckets.add(index)

            if factored_work is not None:
                updated_factors, _row_mean, invalid = factored_work
                factor_storage = self._ensure_factored_second_moment(parameter)
                if invalid.is_cuda:
                    factor_storage.copy_(torch.where(invalid == 0, updated_factors, factor_storage))
                elif int(invalid.item()) == 0:
                    factor_storage.copy_(updated_factors)
            parameter.grad = None
            if isinstance(getattr(parameter, "main_grad", None), torch.Tensor):
                parameter.main_grad = None

        if self._active_offload_mode == "disk":
            for indices in self._bucket_groups():
                changed = [index for index in indices if index in completed_buckets]
                for index in changed:
                    self._stage_8bit_state_on_cpu(self._states[index])
                if changed:
                    self._offload_8bit_group_to_disk(indices)
            for parameter_id, factors in self._factored_second_moments.items():
                if factors is not None and factors.device.type != "cpu":
                    self._factored_second_moments[parameter_id] = factors.to(device="cpu")
        return None

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
        moment_packed_offset: int,
        moment_scale_offset: int,
        variance_packed_offset: int,
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
                moment_packed_offset=moment_packed_offset,
                moment_scale_offset=moment_scale_offset,
                variance_packed_offset=variance_packed_offset,
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
            moment_byte_start = moment_packed_offset + start // 2
            variance_byte_start = variance_packed_offset + start // 2
            byte_count = (count + 1) // 2
            moment_scale_index = moment_scale_offset + block_index
            variance_scale_index = variance_scale_offset + block_index
            moment = _unpack_signed_4bit(
                state.exp_avg_q.narrow(0, moment_byte_start, byte_count),
                count,
                state.exp_avg_scale[moment_scale_index],
            )
            variance = _unpack_positive_4bit(
                state.exp_avg_sq_q.narrow(0, variance_byte_start, byte_count),
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
            state.exp_avg_q.narrow(0, moment_byte_start, byte_count).copy_(moment_q)
            state.exp_avg_scale[moment_scale_index].copy_(moment_scale)
            state.exp_avg_sq_q.narrow(0, variance_byte_start, byte_count).copy_(variance_q)
            state.exp_avg_sq_scale[variance_scale_index].copy_(variance_scale)

    @torch.no_grad()
    def _factored_variance_statistics(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor | None,
        factor_sums: torch.Tensor,
        invalid: torch.Tensor,
    ) -> None:
        """Accumulate row/column gradient-square sums for one DP shard."""

        if grad is None or ref.shard_numel == 0:
            return
        grad_shard = grad if bucket.grad_shard is not None else grad.narrow(0, ref.shard_start, ref.shard_numel)
        parameter_shard_start = ref.param_start + ref.shard_start
        if grad_shard.is_cuda:
            from areno.accel.optimizer import areno_adamw_4bit_factored_stats

            areno_adamw_4bit_factored_stats(
                grad_shard.contiguous(),
                factor_sums,
                invalid,
                parameter_shard_start=parameter_shard_start,
                rows=ref.model_param.shape[0],
                columns=ref.model_param.numel() // ref.model_param.shape[0],
            )
            return

        rows = ref.model_param.shape[0]
        columns = ref.model_param.numel() // rows
        row_sums = factor_sums[:rows]
        column_sums = factor_sums[rows:]
        for start in range(0, ref.shard_numel, self.quant_block_size):
            count = min(self.quant_block_size, ref.shard_numel - start)
            flat_start = parameter_shard_start + start
            gradient = grad_shard.narrow(0, start, count).to(dtype=torch.float32)
            squared = gradient.square()
            if not bool(torch.isfinite(squared).all()):
                invalid.fill_(1)
                return
            flat_indices = torch.arange(flat_start, flat_start + count, device=gradient.device)
            row_sums.index_add_(0, torch.div(flat_indices, columns, rounding_mode="floor"), squared)
            column_sums.index_add_(0, flat_indices.remainder(columns), squared)

    @staticmethod
    def _finalize_factored_statistics(
        parameter: torch.nn.Parameter,
        previous_factors: torch.Tensor,
        factor_sums: torch.Tensor,
        beta2: float,
    ) -> torch.Tensor:
        """Convert global sums to EMA row/column means in-place."""

        rows = parameter.shape[0]
        columns = parameter.numel() // rows
        updated = factor_sums
        updated[:rows].div_(columns).mul_(1.0 - beta2).add_(previous_factors[:rows], alpha=beta2)
        updated[rows:].div_(rows).mul_(1.0 - beta2).add_(previous_factors[rows:], alpha=beta2)
        return updated

    @torch.no_grad()
    def _step_param_ref_factored(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        state,
        moment_packed_offset: int,
        moment_scale_offset: int,
        updated_factors: torch.Tensor,
        row_mean: torch.Tensor,
        invalid: torch.Tensor,
        beta1: float,
        effective_lr: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Update one shard from packed momentum and factored variance."""

        assert state.exp_avg_q is not None
        assert state.exp_avg_scale is not None
        if ref.shard_numel == 0:
            return
        if not invalid.is_cuda and int(invalid.item()) != 0:
            return
        grad_shard = grad if bucket.grad_shard is not None else grad.narrow(0, ref.shard_start, ref.shard_numel)
        parameter_shard_start = ref.param_start + ref.shard_start
        model_shard = ref.model_param.detach().reshape(-1).narrow(0, parameter_shard_start, ref.shard_numel)
        if model_shard.is_cuda:
            from areno.accel.optimizer import areno_adamw_4bit_factored_step

            areno_adamw_4bit_factored_step(
                model_shard,
                grad_shard.contiguous(),
                state.exp_avg_q,
                state.exp_avg_scale,
                updated_factors,
                row_mean,
                invalid,
                moment_packed_offset=moment_packed_offset,
                moment_scale_offset=moment_scale_offset,
                parameter_shard_start=parameter_shard_start,
                quant_block_size=self.quant_block_size,
                rows=ref.model_param.shape[0],
                columns=ref.model_param.numel() // ref.model_param.shape[0],
                beta1=beta1,
                effective_lr=effective_lr,
                weight_decay=self.weight_decay,
                eps=self.eps,
                step_size=step_size,
                bias_correction2_sqrt=bias_correction2_sqrt,
            )
            return

        rows = ref.model_param.shape[0]
        columns = ref.model_param.numel() // rows
        row_factors = updated_factors[:rows]
        column_factors = updated_factors[rows:]
        for block_index, start in enumerate(range(0, ref.shard_numel, self.quant_block_size)):
            count = min(self.quant_block_size, ref.shard_numel - start)
            byte_start = moment_packed_offset + start // 2
            byte_count = (count + 1) // 2
            moment_scale_index = moment_scale_offset + block_index
            moment = _unpack_signed_4bit(
                state.exp_avg_q.narrow(0, byte_start, byte_count),
                count,
                state.exp_avg_scale[moment_scale_index],
            )
            flat_start = parameter_shard_start + start
            gradient = grad_shard.narrow(0, start, count).to(dtype=torch.float32)
            flat_indices = torch.arange(flat_start, flat_start + count, device=gradient.device)
            variance = row_factors[torch.div(flat_indices, columns, rounding_mode="floor")]
            variance = variance * column_factors[flat_indices.remainder(columns)] / row_mean.clamp_min(1.0e-30)
            weight = model_shard.narrow(0, start, count).to(dtype=torch.float32).clone()
            if self.weight_decay != 0.0:
                weight.mul_(1.0 - effective_lr * self.weight_decay)
            moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            denom = variance.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            weight.addcdiv_(moment, denom, value=-step_size)
            if not bool(
                torch.isfinite(gradient).all()
                & torch.isfinite(moment).all()
                & torch.isfinite(variance).all()
                & torch.isfinite(weight).all()
            ):
                return
            moment_q, moment_scale = _quantize_signed_4bit(moment)
            model_shard.narrow(0, start, count).copy_(weight)
            state.exp_avg_q.narrow(0, byte_start, byte_count).copy_(moment_q)
            state.exp_avg_scale[moment_scale_index].copy_(moment_scale)

    def _ensure_factored_second_moment(self, parameter: torch.nn.Parameter) -> torch.Tensor:
        """Materialize row/column second-moment statistics for one parameter."""

        key = id(parameter)
        factors = self._factored_second_moments.get(key)
        if factors is None:
            factors = torch.zeros(
                _factored_state_numel_for_parameter(parameter),
                device=parameter.device,
                dtype=torch.float32,
            )
            self._factored_second_moments[key] = factors
        elif factors.device != parameter.device:
            factors = factors.to(device=parameter.device)
            self._factored_second_moments[key] = factors
        return factors

    def _bucket_state_sizes(self, bucket: _MasterBucket) -> tuple[int, int, int, int]:
        moment_packed_numel = sum((ref.shard_numel + 1) // 2 for ref in bucket.refs)
        moment_scale_numel = sum(
            (ref.shard_numel + self.quant_block_size - 1) // self.quant_block_size for ref in bucket.refs
        )
        variance_packed_numel = sum((ref.shard_numel + 1) // 2 for ref in bucket.refs if ref.model_param.ndim < 2)
        variance_scale_numel = sum(self._variance_scale_count(ref) for ref in bucket.refs)
        return moment_packed_numel, moment_scale_numel, variance_packed_numel, variance_scale_numel

    def _iter_ref_layout(self, bucket: _MasterBucket) -> Iterator[tuple[_ParamRef, int, int, int, int]]:
        moment_packed_offset = 0
        moment_scale_offset = 0
        variance_packed_offset = 0
        variance_scale_offset = 0
        for ref in bucket.refs:
            yield ref, moment_packed_offset, moment_scale_offset, variance_packed_offset, variance_scale_offset
            moment_packed_offset += (ref.shard_numel + 1) // 2
            moment_scale_offset += (ref.shard_numel + self.quant_block_size - 1) // self.quant_block_size
            if ref.model_param.ndim < 2:
                variance_packed_offset += (ref.shard_numel + 1) // 2
            variance_scale_offset += self._variance_scale_count(ref)

    def _variance_scale_count(self, ref: _ParamRef) -> int:
        if ref.model_param.ndim >= 2:
            return 0
        return (ref.shard_numel + self.quant_block_size - 1) // self.quant_block_size

    def persistent_moment_bytes(self) -> int:
        """Return resident packed-moment and scale storage in bytes."""

        total = 0
        for state in self._states:
            for value in (state.exp_avg_q, state.exp_avg_scale, state.exp_avg_sq_q, state.exp_avg_sq_scale):
                if value is not None:
                    total += value.numel() * value.element_size()
        for value in self._factored_second_moments.values():
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
        for value in self._factored_second_moments.values():
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


def _factored_state_numel_for_parameter(parameter: torch.nn.Parameter) -> int:
    rows = int(parameter.shape[0])
    return rows + parameter.numel() // rows


__all__ = ["AdamW4bit"]
