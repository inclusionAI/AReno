"""8-bit-state AdamW with the same DP-sharded contract as AdamWFP32Master."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

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
from areno.engine.optim.dynamic_quant import (
    SIGNED_DYNAMIC_MAP,
    SIGNED_DYNAMIC_ZERO,
    UNSIGNED_DYNAMIC_MAP,
    UNSIGNED_DYNAMIC_ZERO,
)

_DEFAULT_QUANT_BLOCK_SIZE = 128
_MAX_FUSED_QUANT_BLOCK_SIZE = 4096
_DYNAMIC_QUANTIZER = "dynamic-tree-v1"
_VALID_STATE_PRECISIONS = frozenset({"8bit", "fp32"})
_CODEBOOK_CACHE: dict[tuple[torch.device, bool], torch.Tensor] = {}


@dataclass(slots=True)
class _Adam8bitBucketState:
    """Quantized Adam moments for one DP shard of an optimizer bucket."""

    step: int = 0
    exp_avg_q: torch.Tensor | None = None
    exp_avg_scale: torch.Tensor | None = None
    exp_avg_sq_q: torch.Tensor | None = None
    exp_avg_sq_scale: torch.Tensor | None = None
    exp_avg: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    precision: str = "8bit"
    quantizer: str = _DYNAMIC_QUANTIZER
    offload_file: str | None = None
    offload_index: int | None = None
    offload_group: _MmapGroup | None = None
    offload_ready_events: tuple[torch.cuda.Event, ...] = ()


class AdamW8bit(AdamWFP32Master):
    """AdamW with uint8 Adam moments and no persistent FP32 master weights.

    The model parameters remain BF16 on every DP rank. Adam moments are stored
    for only this rank's DP shard and re-quantized after every bucket update.
    Explicit token-embedding parameters keep FP32 moments for stability, while
    other parameters use the paper-compatible dynamic 8-bit codebooks.
    """

    _embedding_fp32_state = True
    state_quantizer = _DYNAMIC_QUANTIZER

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        bucket_numel: int = _DEFAULT_BUCKET_NUMEL,
        dp_rank: int = 0,
        dp_size: int = 1,
        dp_group: dist.ProcessGroup | None = None,
        quant_block_size: int = _DEFAULT_QUANT_BLOCK_SIZE,
    ):
        if quant_block_size < 1 or quant_block_size > _MAX_FUSED_QUANT_BLOCK_SIZE:
            raise ValueError(
                f"quant_block_size must be between 1 and {_MAX_FUSED_QUANT_BLOCK_SIZE}, got {quant_block_size}"
            )
        normalized_params, precision_by_id, role_by_id = _normalize_parameter_policies(params)
        self._parameter_state_precision = precision_by_id
        self._parameter_roles = role_by_id
        self._bucket_numel = max(bucket_numel, 1)
        super().__init__(
            normalized_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            bucket_numel=bucket_numel,
            dp_rank=dp_rank,
            dp_size=dp_size,
            dp_group=dp_group,
        )
        self.quant_block_size = quant_block_size
        self._states = [
            _Adam8bitBucketState(
                precision=self._precision_for_parameter(bucket.refs[0].model_param),
                quantizer=self.state_quantizer,
            )
            for bucket in self.buckets
        ]

    @torch.no_grad()
    def _build_buckets(self, params: list[torch.nn.Parameter], bucket_numel: int) -> list[_MasterBucket]:
        """Keep FP32-exempt and quantized parameters in separate buckets."""

        buckets: list[_MasterBucket] = []
        pending: list[torch.nn.Parameter] = []
        pending_precision: str | None = None
        for parameter in params:
            precision = self._precision_for_parameter(parameter)
            if pending and precision != pending_precision:
                buckets.extend(AdamWFP32Master._build_buckets(self, pending, bucket_numel))
                pending = []
            pending.append(parameter)
            pending_precision = precision
        if pending:
            buckets.extend(AdamWFP32Master._build_buckets(self, pending, bucket_numel))
        return buckets

    def _precision_for_parameter(self, parameter: torch.nn.Parameter) -> str:
        explicit = self._parameter_state_precision.get(id(parameter))
        if explicit is not None:
            return explicit
        role = self._parameter_roles.get(id(parameter), getattr(parameter, "_areno_optimizer_role", None))
        if self._embedding_fp32_state and role == "token_embedding":
            return "fp32"
        return "8bit"

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
            state.exp_avg = None
            state.exp_avg_sq = None
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
            if state.exp_avg is not None and state.exp_avg.device != device:
                state.exp_avg = state.exp_avg.to(device=device)
            if state.exp_avg_sq is not None and state.exp_avg_sq.device != device:
                state.exp_avg_sq = state.exp_avg_sq.to(device=device)
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
            "quantizer": self.state_quantizer,
            "quant_block_size": self.quant_block_size,
            "precision_policy": [
                {
                    "parameter_index": index,
                    "role": self._parameter_roles.get(id(parameter)),
                    "precision": self._precision_for_parameter(parameter),
                    "numel": parameter.numel(),
                }
                for index, parameter in enumerate(self.model_params)
            ],
            "state_memory": self.state_memory_metrics(),
            "state": [
                {
                    "step": state.step,
                    "precision": state.precision,
                    "quantizer": state.quantizer,
                    "exp_avg_q": payload["exp_avg_q"],
                    "exp_avg_scale": payload["exp_avg_scale"],
                    "exp_avg_sq_q": payload["exp_avg_sq_q"],
                    "exp_avg_sq_scale": payload["exp_avg_sq_scale"],
                    "exp_avg": payload["exp_avg"],
                    "exp_avg_sq": payload["exp_avg_sq"],
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
        saved_states = state_dict.get("state", [])
        saved_quantizer = str(state_dict.get("quantizer", ""))
        if saved_quantizer != _DYNAMIC_QUANTIZER:
            raise ValueError(f"unsupported AdamW8bit quantizer: {saved_quantizer}")
        if state_dict.get("precision_policy") is None:
            raise ValueError("AdamW8bit checkpoint is missing its precision policy")
        self._restore_precision_policy(state_dict["precision_policy"])
        if len(saved_states) != len(self.buckets):
            raise ValueError(
                "AdamW8bit checkpoint bucket count does not match the current optimizer layout: "
                f"checkpoint={len(saved_states)}, optimizer={len(self.buckets)}"
            )
        for state in self._states:
            state.step = 0
            state.exp_avg_q = None
            state.exp_avg_scale = None
            state.exp_avg_sq_q = None
            state.exp_avg_sq_scale = None
            state.exp_avg = None
            state.exp_avg_sq = None
            state.quantizer = self.state_quantizer
            state.offload_file = None
            state.offload_index = None
            state.offload_group = None
            state.offload_ready_events = ()
        if "quant_block_size" in state_dict:
            saved_block_size = int(state_dict["quant_block_size"])
            if saved_block_size < 1 or saved_block_size > _MAX_FUSED_QUANT_BLOCK_SIZE:
                raise ValueError(f"invalid saved AdamW8bit quant_block_size: {saved_block_size}")
            self.quant_block_size = saved_block_size
        for saved, bucket, state in zip(saved_states[: len(self.buckets)], self.buckets, self._states, strict=False):
            if saved is None:
                continue
            device = bucket.refs[0].model_param.device
            state.step = int(saved.get("step", 0))
            saved_precision = str(saved.get("precision", "8bit"))
            bucket_quantizer = str(saved.get("quantizer", saved_quantizer))
            if bucket_quantizer != _DYNAMIC_QUANTIZER:
                raise ValueError(f"unsupported AdamW8bit bucket quantizer: {bucket_quantizer}")
            if saved_precision != state.precision:
                raise ValueError(
                    "AdamW8bit state precision policy mismatch: "
                    f"checkpoint={saved_precision}, optimizer={state.precision}"
                )
            state.quantizer = bucket_quantizer
            if state.precision == "fp32":
                state.exp_avg = _load_optional_state_tensor(saved.get("exp_avg"), bucket, device)
                state.exp_avg_sq = _load_optional_state_tensor(saved.get("exp_avg_sq"), bucket, device)
                state.exp_avg_q = None
                state.exp_avg_scale = None
                state.exp_avg_sq_q = None
                state.exp_avg_sq_scale = None
                continue
            exp_avg_q = saved.get("exp_avg_q")
            exp_avg_scale = saved.get("exp_avg_scale")
            exp_avg_sq_q = saved.get("exp_avg_sq_q")
            exp_avg_sq_scale = saved.get("exp_avg_sq_scale")
            state.exp_avg_q = _load_optional_quantized_tensor(exp_avg_q, bucket, device)
            state.exp_avg_scale = None if exp_avg_scale is None else self._restore_scales(exp_avg_scale, bucket, device)
            state.exp_avg_sq_q = _load_optional_quantized_tensor(exp_avg_sq_q, bucket, device)
            state.exp_avg_sq_scale = (
                None if exp_avg_sq_scale is None else self._restore_scales(exp_avg_sq_scale, bucket, device)
            )

    def _restore_precision_policy(self, saved_policy: Any) -> None:
        """Rebuild buckets from the checkpoint's identity-ordered policy."""

        if not isinstance(saved_policy, list) or len(saved_policy) != len(self.model_params):
            raise ValueError(
                "AdamW8bit checkpoint precision policy does not match the current parameter count: "
                f"checkpoint={len(saved_policy) if isinstance(saved_policy, list) else 'invalid'}, "
                f"optimizer={len(self.model_params)}"
            )
        restored_precision: dict[int, str] = {}
        restored_roles: dict[int, str] = {}
        for expected_index, (entry, parameter) in enumerate(zip(saved_policy, self.model_params, strict=True)):
            if not isinstance(entry, Mapping) or int(entry.get("parameter_index", -1)) != expected_index:
                raise ValueError(
                    f"AdamW8bit checkpoint has an invalid precision policy entry at index {expected_index}"
                )
            saved_numel = int(entry.get("numel", -1))
            if saved_numel != parameter.numel():
                raise ValueError(
                    "AdamW8bit checkpoint precision policy parameter size mismatch: "
                    f"index={expected_index}, checkpoint={saved_numel}, optimizer={parameter.numel()}"
                )
            restored_precision[id(parameter)] = _normalize_state_precision(str(entry.get("precision", "8bit")))
            role = entry.get("role")
            if role is not None:
                restored_roles[id(parameter)] = str(role)
        self._parameter_state_precision = restored_precision
        self._parameter_roles = restored_roles
        self.buckets = self._build_buckets(self.model_params, self._bucket_numel)
        self._states = [
            _Adam8bitBucketState(
                precision=self._precision_for_parameter(bucket.refs[0].model_param),
                quantizer=self.state_quantizer,
            )
            for bucket in self.buckets
        ]

    def state_memory_metrics(self) -> dict[str, int]:
        """Report logical persistent moment storage for initialized buckets."""

        quantized_state_bytes = 0
        fp32_exempt_bytes = 0
        block_metadata_bytes = 0
        for bucket, state in zip(self.buckets, self._states, strict=True):
            if state.step == 0:
                continue
            if state.precision == "fp32":
                fp32_exempt_bytes += 2 * bucket.shard_numel * 4
            else:
                quantized_state_bytes += 2 * bucket.shard_numel
                block_metadata_bytes += 2 * self._bucket_scale_count(bucket) * 4
        return {
            "quantized_state_bytes": quantized_state_bytes,
            "fp32_exempt_bytes": fp32_exempt_bytes,
            "block_metadata_bytes": block_metadata_bytes,
            "total_bytes": quantized_state_bytes + fp32_exempt_bytes + block_metadata_bytes,
        }

    def _restore_scales(
        self,
        saved: torch.Tensor,
        bucket: _MasterBucket,
        device: torch.device,
    ) -> torch.Tensor:
        """Restore block scales for one quantized bucket."""

        expected = self._bucket_scale_count(bucket)
        scales = saved.detach().to(device=device, dtype=torch.float32).view(-1)
        if scales.numel() != expected:
            raise ValueError(f"AdamW8bit checkpoint has {scales.numel()} scales for a bucket requiring {expected}")
        return scales.clone()

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
        if state.exp_avg is not None and state.exp_avg.device != device:
            state.exp_avg = state.exp_avg.to(device=device)
        if state.exp_avg_sq is not None and state.exp_avg_sq.device != device:
            state.exp_avg_sq = state.exp_avg_sq.to(device=device)
        if state.precision == "fp32":
            if state.exp_avg is None:
                state.exp_avg = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
            if state.exp_avg_sq is None:
                state.exp_avg_sq = torch.zeros(bucket.shard_numel, device=device, dtype=torch.float32)
            return
        if state.exp_avg_q is None:
            state.exp_avg_q = torch.full((bucket.shard_numel,), SIGNED_DYNAMIC_ZERO, device=device, dtype=torch.uint8)
            state.exp_avg_scale = torch.zeros(self._bucket_scale_count(bucket), device=device, dtype=torch.float32)
        if state.exp_avg_sq_q is None:
            state.exp_avg_sq_q = torch.full(
                (bucket.shard_numel,), UNSIGNED_DYNAMIC_ZERO, device=device, dtype=torch.uint8
            )
            state.exp_avg_sq_scale = torch.zeros(self._bucket_scale_count(bucket), device=device, dtype=torch.float32)

    def _bucket_scale_count(self, bucket: _MasterBucket) -> int:
        """Return the number of independently scaled blocks in one DP shard."""

        return sum(_ceil_div(ref.shard_numel, self.quant_block_size) for ref in bucket.refs)

    def _ref_scale_layout(self, bucket: _MasterBucket) -> list[tuple[_ParamRef, int, int]]:
        """Map each parameter ref to its contiguous range in the scale tensors."""

        layout: list[tuple[_ParamRef, int, int]] = []
        scale_offset = 0
        for ref in bucket.refs:
            block_count = _ceil_div(ref.shard_numel, self.quant_block_size)
            layout.append((ref, scale_offset, block_count))
            scale_offset += block_count
        return layout

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
        for name in ("exp_avg_q", "exp_avg_scale", "exp_avg_sq_q", "exp_avg_sq_scale", "exp_avg", "exp_avg_sq"):
            value = saved.get(name)
            setattr(state, name, None if value is None else _host_tensor_to(value, device, prefetched=prefetched))
        if prefetched and device.type == "cuda":
            self._retain_disk_prefetch(state.offload_index, saved, device)

    def _disk_mmap_group_for_index(self, index: int) -> _MmapGroup | None:
        """Return the mapped Adam8bit group for an initialized bucket."""

        state = self._states[index]
        return state.offload_group if state.offload_file is not None else None

    def _state_mmap_specs(self, indices: list[int]) -> dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]]:
        """Return the fixed raw-mmap layout for quantized Adam state."""

        specs: dict[int, dict[str, tuple[torch.dtype, tuple[int, ...]]]] = {}
        for index in indices:
            bucket = self.buckets[index]
            if self._states[index].precision == "fp32":
                specs[index] = {
                    "exp_avg": (torch.float32, (bucket.shard_numel,)),
                    "exp_avg_sq": (torch.float32, (bucket.shard_numel,)),
                }
            else:
                specs[index] = {
                    "exp_avg_q": (torch.uint8, (bucket.shard_numel,)),
                    "exp_avg_scale": (torch.float32, (self._bucket_scale_count(bucket),)),
                    "exp_avg_sq_q": (torch.uint8, (bucket.shard_numel,)),
                    "exp_avg_sq_scale": (torch.float32, (self._bucket_scale_count(bucket),)),
                }
        return specs

    def _offload_8bit_group_to_disk(self, indices: list[int]) -> None:
        """Persist a bounded group of quantized states in one serialization call."""

        if self._disk_offload_root is None:
            raise RuntimeError("disk optimizer offload is active without a usable directory")
        present_indices = [
            index
            for index in indices
            if self._states[index].exp_avg_q is not None or self._states[index].exp_avg is not None
        ]
        if not present_indices:
            return
        group = self._get_or_create_mmap_group(indices, self._state_mmap_specs(indices))
        payloads: dict[int, dict[str, torch.Tensor]] = {}
        ready_events: list[torch.cuda.Event] = []
        for index in present_indices:
            state = self._states[index]
            payloads[index] = {
                name: value
                for name in (
                    "exp_avg_q",
                    "exp_avg_scale",
                    "exp_avg_sq_q",
                    "exp_avg_sq_scale",
                    "exp_avg",
                    "exp_avg_sq",
                )
                if (value := getattr(state, name)) is not None
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
            state.exp_avg = None
            state.exp_avg_sq = None
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
                "exp_avg": state.exp_avg,
                "exp_avg_sq": state.exp_avg_sq,
            }.items()
            if tensor is not None
        }
        staged, state.offload_ready_events = self._stage_payload_on_cpu(payload)
        state.exp_avg_q = staged.get("exp_avg_q")
        state.exp_avg_scale = staged.get("exp_avg_scale")
        state.exp_avg_sq_q = staged.get("exp_avg_sq_q")
        state.exp_avg_sq_scale = staged.get("exp_avg_sq_scale")
        state.exp_avg = staged.get("exp_avg")
        state.exp_avg_sq = staged.get("exp_avg_sq")

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
            return {
                name: None if saved.get(name) is None else saved[name].clone()
                for name in (
                    "exp_avg_q",
                    "exp_avg_scale",
                    "exp_avg_sq_q",
                    "exp_avg_sq_scale",
                    "exp_avg",
                    "exp_avg_sq",
                )
            }
        return {
            "exp_avg_q": _cpu_clone(state.exp_avg_q),
            "exp_avg_scale": _cpu_clone(state.exp_avg_scale),
            "exp_avg_sq_q": _cpu_clone(state.exp_avg_sq_q),
            "exp_avg_sq_scale": _cpu_clone(state.exp_avg_sq_scale),
            "exp_avg": _cpu_clone(state.exp_avg),
            "exp_avg_sq": _cpu_clone(state.exp_avg_sq),
        }

    @torch.no_grad()
    def _step_bucket_8bit(self, bucket: _MasterBucket, state: _Adam8bitBucketState) -> None:
        """Update one bucket without materializing full FP32 moment tensors."""

        beta1, beta2 = self.betas
        state.step += 1
        bias_correction1 = 1.0 - beta1**state.step
        bias_correction2 = 1.0 - beta2**state.step
        bias_correction2_sqrt = bias_correction2**0.5

        for ref, scale_offset, block_count in self._ref_scale_layout(bucket):
            grad = self._gradient_for_ref(bucket, ref)
            if grad is None:
                continue
            effective_lr = float(getattr(ref.model_param, "_areno_lr", self.lr))
            step_size = effective_lr / bias_correction1
            if state.precision == "fp32":
                self._step_param_ref_fp32(
                    bucket,
                    ref,
                    grad,
                    state,
                    beta1,
                    beta2,
                    effective_lr,
                    step_size,
                    bias_correction2_sqrt,
                )
            else:
                self._step_param_ref_8bit(
                    bucket,
                    ref,
                    grad,
                    state,
                    scale_offset,
                    block_count,
                    beta1,
                    beta2,
                    effective_lr,
                    step_size,
                    bias_correction2_sqrt,
                )
            if ref.param_start + ref.numel == ref.model_param.numel():
                ref.model_param.grad = None
                if isinstance(getattr(ref.model_param, "main_grad", None), torch.Tensor):
                    ref.model_param.main_grad = None
        # Collective order is bucket-global, not rank-local. A rank can own
        # no values from a small DP bucket and must still join the gather that
        # refreshes every replicated model parameter.
        self._all_gather_bucket(bucket)
        bucket.grad_shard = None
        bucket.grad_param_ids = frozenset()
        if state.precision == "8bit":
            state.quantizer = self.state_quantizer

    @torch.no_grad()
    def _step_param_ref_fp32(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        state: _Adam8bitBucketState,
        beta1: float,
        beta2: float,
        effective_lr: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Update one FP32-exempt parameter shard without a master-weight copy."""

        if ref.shard_numel == 0:
            return
        assert state.exp_avg is not None
        assert state.exp_avg_sq is not None
        grad_shard = grad if bucket.grad_shard is not None else grad.narrow(0, ref.shard_start, ref.shard_numel)
        model_shard = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start + ref.shard_start, ref.shard_numel)
        exp_avg = state.exp_avg.narrow(0, ref.shard_bucket_start, ref.shard_numel)
        exp_avg_sq = state.exp_avg_sq.narrow(0, ref.shard_bucket_start, ref.shard_numel)
        if model_shard.is_cuda:
            from areno.accel.optimizer import areno_adamw_fp32_state_step

            areno_adamw_fp32_state_step(
                model_shard,
                grad_shard.contiguous(),
                exp_avg,
                exp_avg_sq,
                beta1=beta1,
                beta2=beta2,
                effective_lr=effective_lr,
                weight_decay=self.weight_decay,
                eps=self.eps,
                step_size=step_size,
                bias_correction2_sqrt=bias_correction2_sqrt,
            )
            return
        weight = model_shard.float()
        gradient = grad_shard.float()
        if self.weight_decay != 0.0:
            weight.mul_(1.0 - effective_lr * self.weight_decay)
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
        weight.addcdiv_(exp_avg, denom, value=-step_size)
        model_shard.copy_(weight)

    @torch.no_grad()
    def _step_param_ref_8bit(
        self,
        bucket: _MasterBucket,
        ref: _ParamRef,
        grad: torch.Tensor,
        state: _Adam8bitBucketState,
        scale_offset: int,
        block_count: int,
        beta1: float,
        beta2: float,
        effective_lr: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        """Apply one AdamW update to this rank's shard of one param chunk."""

        if ref.shard_numel == 0:
            return
        assert state.exp_avg_q is not None
        assert state.exp_avg_scale is not None
        assert state.exp_avg_sq_q is not None
        assert state.exp_avg_sq_scale is not None
        if bucket.grad_shard is not None:
            grad_shard = grad
        else:
            grad_shard = grad.narrow(0, ref.shard_start, ref.shard_numel)
        model_chunk = ref.model_param.detach().reshape(-1).narrow(0, ref.param_start, ref.numel)
        model_shard = model_chunk.narrow(0, ref.shard_start, ref.shard_numel)
        moment_q = state.exp_avg_q.narrow(0, ref.shard_bucket_start, ref.shard_numel)
        variance_q = state.exp_avg_sq_q.narrow(0, ref.shard_bucket_start, ref.shard_numel)
        moment_scales = state.exp_avg_scale.narrow(0, scale_offset, block_count)
        variance_scales = state.exp_avg_sq_scale.narrow(0, scale_offset, block_count)

        if model_shard.is_cuda:
            from areno.accel.optimizer import areno_adamw_8bit_step

            signed_codebook = _dynamic_codebook(model_shard.device, signed=True)
            unsigned_codebook = _dynamic_codebook(model_shard.device, signed=False)

            areno_adamw_8bit_step(
                model_shard,
                grad_shard.contiguous(),
                moment_q,
                moment_scales,
                variance_q,
                variance_scales,
                signed_codebook,
                unsigned_codebook,
                block_size=self.quant_block_size,
                beta1=beta1,
                beta2=beta2,
                effective_lr=effective_lr,
                weight_decay=self.weight_decay,
                eps=self.eps,
                step_size=step_size,
                bias_correction2_sqrt=bias_correction2_sqrt,
            )
            return

        for block_index in range(block_count):
            start = block_index * self.quant_block_size
            numel = min(self.quant_block_size, ref.shard_numel - start)
            # ``Tensor.to(float32)`` aliases an already-FP32 model shard. Keep
            # this speculative update private until the whole block passes the
            # finite check, matching the fused CUDA kernel's two-pass commit.
            weight = model_shard.narrow(0, start, numel).to(dtype=torch.float32).clone()
            block_grad = grad_shard.narrow(0, start, numel).to(dtype=torch.float32)
            block_moment_q = moment_q.narrow(0, start, numel)
            block_variance_q = variance_q.narrow(0, start, numel)
            moment = _dequantize_dynamic(block_moment_q, moment_scales[block_index], signed=True)
            variance = _dequantize_dynamic(block_variance_q, variance_scales[block_index], signed=False)
            if self.weight_decay != 0.0:
                weight.mul_(1.0 - effective_lr * self.weight_decay)
            moment.mul_(beta1).add_(block_grad, alpha=1.0 - beta1)
            variance.mul_(beta2).addcmul_(block_grad, block_grad, value=1.0 - beta2)
            denom = variance.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            weight.addcdiv_(moment, denom, value=-step_size)
            if not bool(
                torch.isfinite(block_grad).all()
                & torch.isfinite(moment).all()
                & torch.isfinite(variance).all()
                & torch.isfinite(weight).all()
            ):
                continue
            model_shard.narrow(0, start, numel).copy_(weight)
            quantized_moment, moment_scale = _quantize_dynamic(moment, signed=True)
            quantized_variance, variance_scale = _quantize_dynamic(variance, signed=False)
            block_moment_q.copy_(quantized_moment)
            block_variance_q.copy_(quantized_variance)
            moment_scales[block_index].copy_(moment_scale)
            variance_scales[block_index].copy_(variance_scale)


def set_optimizer_state_precision(
    parameter: torch.nn.Parameter,
    precision: str,
    *,
    role: str | None = None,
) -> torch.nn.Parameter:
    """Attach an explicit AdamW8bit state policy to a parameter.

    This is intentionally parameter metadata rather than name matching, so it
    survives bucket construction and works for non-standard model layouts.
    """

    normalized = _normalize_state_precision(precision)
    parameter._areno_optimizer_state_precision = normalized
    if role is not None:
        parameter._areno_optimizer_role = str(role)
    return parameter


def _normalize_parameter_policies(
    params: Iterable[torch.nn.Parameter] | Iterable[Mapping[str, Any]],
) -> tuple[list[torch.nn.Parameter], dict[int, str], dict[int, str]]:
    values = list(params)
    flattened: list[torch.nn.Parameter] = []
    precision_by_id: dict[int, str] = {}
    role_by_id: dict[int, str] = {}
    seen: set[int] = set()
    for value in values:
        if isinstance(value, Mapping):
            group_params = list(value.get("params", ()))
            group_precision = value.get("state_precision")
            group_role = value.get("role")
        else:
            group_params = [value]
            group_precision = None
            group_role = None
        for parameter in group_params:
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError("AdamW8bit params must contain torch.nn.Parameter values")
            identity = id(parameter)
            explicit = getattr(parameter, "_areno_optimizer_state_precision", group_precision)
            if explicit is not None:
                precision_by_id[identity] = _normalize_state_precision(str(explicit))
            role = getattr(parameter, "_areno_optimizer_role", group_role)
            if role is not None:
                role_by_id[identity] = str(role)
            if identity not in seen:
                flattened.append(parameter)
                seen.add(identity)
    return flattened, precision_by_id, role_by_id


def _normalize_state_precision(precision: str) -> str:
    aliases = {"uint8": "8bit", "int8": "8bit", "float32": "fp32"}
    normalized = aliases.get(precision.lower(), precision.lower())
    if normalized not in _VALID_STATE_PRECISIONS:
        raise ValueError(f"state_precision must be one of {sorted(_VALID_STATE_PRECISIONS)}, got {precision!r}")
    return normalized


def _dynamic_codebook(device: torch.device, *, signed: bool) -> torch.Tensor:
    key = (device, signed)
    codebook = _CODEBOOK_CACHE.get(key)
    if codebook is None:
        values = SIGNED_DYNAMIC_MAP if signed else UNSIGNED_DYNAMIC_MAP
        codebook = torch.tensor(values, device=device, dtype=torch.float32)
        _CODEBOOK_CACHE[key] = codebook
    return codebook


def _quantize_dynamic(tensor: torch.Tensor, *, signed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one FP32 block with the paper's dynamic-tree codebook."""

    if tensor.numel() == 0:
        return tensor.to(dtype=torch.uint8), torch.zeros((), device=tensor.device, dtype=torch.float32)
    scale = (tensor.abs().amax() if signed else tensor.clamp_min(0).amax()).to(dtype=torch.float32)
    normalized = tensor.float().div(scale.clamp_min(torch.finfo(torch.float32).tiny))
    if not signed:
        normalized.clamp_min_(0.0)
    codebook = _dynamic_codebook(tensor.device, signed=signed)
    boundaries = (codebook[:-1] + codebook[1:]) * 0.5
    codes = torch.bucketize(normalized, boundaries).to(dtype=torch.uint8)
    return codes, scale


def _dequantize_dynamic(quantized: torch.Tensor, scale: torch.Tensor, *, signed: bool) -> torch.Tensor:
    codebook = _dynamic_codebook(quantized.device, signed=signed)
    return codebook[quantized.long()].mul_(scale)


# Private convenience aliases used by focused quantization tests.
def _quantize_symmetric(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _quantize_dynamic(tensor, signed=True)


def _dequantize_symmetric(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _dequantize_dynamic(quantized, scale, signed=True)


def _quantize_positive(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _quantize_dynamic(tensor, signed=False)


def _dequantize_positive(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _dequantize_dynamic(quantized, scale, signed=False)


def _load_optional_state_tensor(
    saved: Any,
    bucket: _MasterBucket,
    device: torch.device,
) -> torch.Tensor | None:
    if saved is None:
        return None
    value = saved.detach().to(device=device, dtype=torch.float32).view(-1)
    if value.numel() != bucket.shard_numel:
        raise ValueError(
            f"AdamW8bit checkpoint has {value.numel()} FP32 state values for a bucket requiring {bucket.shard_numel}"
        )
    return value.clone()


def _load_optional_quantized_tensor(
    saved: Any,
    bucket: _MasterBucket,
    device: torch.device,
) -> torch.Tensor | None:
    if saved is None:
        return None
    value = saved.detach().to(device=device, dtype=torch.uint8).view(-1)
    if value.numel() != bucket.shard_numel:
        raise ValueError(
            f"AdamW8bit checkpoint has {value.numel()} quantized state values "
            f"for a bucket requiring {bucket.shard_numel}"
        )
    return value.clone()


def _cpu_clone(value: torch.Tensor | None) -> torch.Tensor | None:
    """Return an independent CPU copy of an optional quantized-state tensor."""

    return None if value is None else value.detach().to(device="cpu").clone()


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
