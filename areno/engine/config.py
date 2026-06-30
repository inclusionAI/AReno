"""Engine configuration dataclasses.

`OptimizerConfig`, `RuntimeConfig`, and `ModelConfig` are small dataclasses
that describe one engine's training schedule, decode-time allocation, and
model architecture. `EngineConfig` ties them together and decides how the
configured devices map onto the tensor-parallel (TP) and data-parallel (DP)
groups consumed by the worker cluster.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

# AReno's flash path uses flash-attn features beyond the Turing-compatible
# forward kernels, including paged KV/cache and training paths, so require
# Ampere+ even though flash-attn 2.x has partial sm75 forward support.
FLASH_ATTENTION_MIN_CUDA_CAPABILITY = (8, 0)
FLASH_ATTENTION_MAX_QK_HEAD_DIM = 256


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer and learning-rate schedule config for worker training."""

    lr: float = 1e-4
    min_lr: float = 0.0
    lr_decay_steps: int = 0
    lr_warmup_steps: int = 0
    lr_decay_style: Literal["constant", "linear", "cosine"] = "constant"
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    adam_8bit: bool = False
    fp32_master_bucket_numel: int = 16 * 1024 * 1024


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime allocation config for rollout decode and CUDA graphs."""

    kv_block_size: int = 256
    attn_backend: Literal["flash", "native"] = "flash"
    activation_checkpointing: bool = True
    keep_rollout_state: bool = True
    eager_decode: bool = False
    decode_graph_buckets: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192, 256]
    )

    def __post_init__(self) -> None:
        if self.attn_backend not in {"flash", "native"}:
            raise ValueError("runtime.attn_backend must be one of: flash, native")

    def resolve_attn_backend(self, *, model: ModelConfig, devices: list[int]) -> None:
        """Switch flash-attn unsupported hardware or model shapes to native attention."""

        if self.attn_backend != "flash":
            return
        reasons = [
            reason
            for reason in (
                flash_attention_unsupported_gpu_reason(devices),
                flash_attention_unsupported_model_reason(model),
            )
            if reason is not None
        ]
        if not reasons:
            return
        reason = "; ".join(reasons)
        warnings.warn(
            f"flash-attn does not support the detected runtime configuration ({reason}); "
            "falling back to attn_backend='native'. Native attention is a compatibility path "
            "and may be slower than flash-attn on supported GPUs.",
            RuntimeWarning,
            stacklevel=2,
        )
        self.attn_backend = "native"


@dataclass(slots=True)
class ModelConfig:
    """Normalized model architecture config derived from a HF checkpoint."""

    model_type: str = "qwen3"
    checkpoint_prefix: str = "model"
    checkpoint_lm_head_key: str = "lm_head.weight"
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = False
    qkv_bias: bool = False
    qk_norm: bool = True
    v_norm: bool = False
    dtype: torch.dtype = torch.bfloat16
    hidden_act: str = "silu"
    layer_types: tuple[str, ...] | None = None
    sliding_window: int | None = None
    swa_head_dim: int | None = None
    swa_num_key_value_heads: int | None = None
    rope_parameters: dict[str, dict[str, Any]] | None = None
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int | None = None
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = False
    use_bias: bool = False
    layer_group_size: int = 1
    partial_rotary_factor: float = 1.0
    num_experts: int | None = None
    num_experts_per_tok: int = 1
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_intermediate_size: int = 0
    num_shared_experts: int | None = None
    shared_expert_intermediate_size: int = 0
    moe_router_enable_expert_bias: bool = True
    norm_topk_prob: bool = True
    moe_router_dtype: torch.dtype = torch.float32
    score_function: str = "sigmoid"
    topk_method: str = "noaux_tc"
    group_norm_size: int = 128
    num_nextn_predict_layers: int = 0
    mtp_loss_scaling_factor: float = 0.0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    kv_lora_rank: int | None = None
    linear_backend: str = "minimax"
    linear_scale: bool = True
    linear_silu: bool = False
    moe_backend: str = "grouped"
    sequence_parallel: bool = True
    moe_router_bias_update_rate: float = 0.0
    attention_softmax_scale: float | None = None
    final_logit_softcapping: float | None = None
    attn_output_gate: bool = False
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    attn_backend: Literal["flash", "native"] = "flash"

    def validate_tp(self, tp_size: int) -> None:
        """Validate tensor-parallel divisibility required by local kernels."""

        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if self.num_attention_heads % tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by tp_size")
        if self.num_key_value_heads % tp_size != 0:
            allow_replicated_kv = (
                self.model_type in {"gemma4", "qwen3_moe", "qwen3_5_moe"} and tp_size % self.num_key_value_heads == 0
            )
            if not allow_replicated_kv:
                raise ValueError("num_key_value_heads must be divisible by tp_size")
        if self.swa_num_key_value_heads is not None:
            if self.swa_num_key_value_heads % tp_size != 0:
                allow_replicated_swa_kv = self.model_type == "gemma4" and tp_size % self.swa_num_key_value_heads == 0
                if not allow_replicated_swa_kv:
                    raise ValueError("swa_num_key_value_heads must be divisible by tp_size")
        if self.intermediate_size % tp_size != 0:
            raise ValueError("intermediate_size must be divisible by tp_size")
        if self.vocab_size % tp_size != 0:
            raise ValueError("vocab_size must be divisible by tp_size")
        if self.hidden_size_per_layer_input > 0:
            ple_vocab_size = self.vocab_size_per_layer_input or self.vocab_size
            if ple_vocab_size % tp_size != 0:
                raise ValueError("vocab_size_per_layer_input must be divisible by tp_size")
        if self.layer_types and any(layer_type == "linear_attention" for layer_type in self.layer_types):
            if self.linear_num_key_heads % tp_size != 0:
                raise ValueError("linear_num_key_heads must be divisible by tp_size")
            if self.linear_num_value_heads % tp_size != 0:
                raise ValueError("linear_num_value_heads must be divisible by tp_size")
            if (self.linear_key_head_dim * self.linear_num_key_heads) % tp_size != 0:
                raise ValueError("linear key projection dim must be divisible by tp_size")
            if (self.linear_value_head_dim * self.linear_num_value_heads) % tp_size != 0:
                raise ValueError("linear value projection dim must be divisible by tp_size")


@dataclass(slots=True)
class EngineConfig:
    """Complete engine config shared by the coordinator and rank workers."""

    model: ModelConfig
    model_path: str | None = None
    train_loss_fn: Callable[[Any, torch.Tensor], torch.Tensor | tuple[torch.Tensor, dict[str, Any]]] | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tp_size: int = 1
    dp_size: int | None = None
    devices: list[int] | None = None
    dummy_load: bool = False

    def __post_init__(self) -> None:
        """Infer DP/devices and validate the distributed layout."""

        self.model.validate_tp(self.tp_size)
        if self.devices is None:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count < 1:
                    raise ValueError("CUDA is available but torch.cuda.device_count() is 0")
                self.devices = list(range(device_count))
            else:
                self.devices = list(range(self.tp_size if self.dp_size is None else self.tp_size * self.dp_size))
        if len(self.devices) < 1:
            raise ValueError("devices must be non-empty")
        if len(self.devices) % self.tp_size != 0:
            raise ValueError("len(devices) must be divisible by tp_size")
        inferred_dp_size = len(self.devices) // self.tp_size
        if self.dp_size is None:
            self.dp_size = inferred_dp_size
        elif self.dp_size != inferred_dp_size:
            raise ValueError("dp_size must equal len(devices) // tp_size")
        if self.dp_size < 1:
            raise ValueError("dp_size must be >= 1")
        if self.runtime.kv_block_size < 1:
            raise ValueError("runtime.kv_block_size must be >= 1")
        if self.runtime.kv_block_size % 256 != 0:
            raise ValueError("runtime.kv_block_size must be a multiple of 256 for FlashAttention paged KV")
        self.runtime.resolve_attn_backend(model=self.model, devices=self.devices)
        self.model.attn_backend = self.runtime.attn_backend


def flash_attention_unsupported_model_reason(model: ModelConfig) -> str | None:
    """Return a user-facing reason when a model shape cannot run flash-attn."""

    dims = [("qk head dim", model.head_dim)]
    if model.swa_head_dim is not None:
        dims.append(("swa qk head dim", model.swa_head_dim))
    if model.qk_nope_head_dim or model.qk_rope_head_dim:
        dims.append(("qk head dim", model.qk_nope_head_dim + model.qk_rope_head_dim))
    unsupported = list(
        dict.fromkeys(
            f"{name} {dim}" for name, dim in dims if dim is not None and int(dim) > FLASH_ATTENTION_MAX_QK_HEAD_DIM
        )
    )
    if not unsupported:
        return None
    return ", ".join(unsupported)


def flash_attention_unsupported_gpu_reason(devices: list[int] | None = None) -> str | None:
    """Return a user-facing reason when visible GPUs cannot run flash-attn."""

    if not torch.cuda.is_available():
        return None
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return None
    selected_devices = devices if devices is not None else list(range(device_count))
    unsupported: list[str] = []
    for device in selected_devices:
        if device < 0 or device >= device_count:
            continue
        major, minor = torch.cuda.get_device_capability(device)
        capability = (int(major), int(minor))
        if capability >= FLASH_ATTENTION_MIN_CUDA_CAPABILITY:
            continue
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = f"cuda:{device}"
        unsupported.append(f"{name} cc {major}.{minor}")
    if not unsupported:
        return None
    return ", ".join(unsupported)


def _parse_dtype(value: str | None) -> torch.dtype:
    """Parse HF dtype strings into torch dtypes."""

    if value in (None, "bfloat16", "torch.bfloat16", "bf16"):
        return torch.bfloat16
    if value in ("float16", "torch.float16", "fp16", "half"):
        return torch.float16
    if value in ("float32", "torch.float32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"unsupported torch dtype in HF config: {value}")
