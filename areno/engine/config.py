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

from areno.adapters.config import LoraConfig

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
    unfreeze_multimodal_tower: bool = False
    unfreeze_multimodal_projector: bool = False
    multimodal_tower_lr: float | None = None
    multimodal_tower_min_lr: float | None = None
    multimodal_tower_lr_decay_steps: int | None = None
    multimodal_tower_lr_decay_style: Literal["constant", "linear", "cosine"] | None = None
    multimodal_projector_lr: float | None = None
    multimodal_projector_min_lr: float | None = None
    multimodal_projector_lr_decay_steps: int | None = None
    multimodal_projector_lr_decay_style: Literal["constant", "linear", "cosine"] | None = None


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime allocation config for rollout decode and CUDA graphs."""

    kv_block_size: int = 256
    attn_backend: Literal["flash", "native"] = "flash"
    compile_model: bool = True
    activation_checkpointing: bool = True
    keep_rollout_state: bool = True
    optimizer_state_offload: Literal["none", "cpu", "disk"] | bool = "none"
    optimizer_state_offload_dir: str | None = None
    optimizer_state_offload_batch_size: int = 1
    eager_decode: bool = False
    rollout_routing_replay: bool = False
    decode_graph_buckets: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192, 256]
    )

    def __post_init__(self) -> None:
        if self.attn_backend not in {"flash", "native"}:
            raise ValueError("runtime.attn_backend must be one of: flash, native")
        if isinstance(self.optimizer_state_offload, bool):
            self.optimizer_state_offload = "cpu" if self.optimizer_state_offload else "none"
        if self.optimizer_state_offload not in {"none", "cpu", "disk"}:
            raise ValueError("runtime.optimizer_state_offload must be one of: none, cpu, disk")
        if self.optimizer_state_offload == "disk" and not self.optimizer_state_offload_dir:
            raise ValueError("runtime.optimizer_state_offload_dir is required for disk offload")
        if self.optimizer_state_offload_batch_size < 1:
            raise ValueError("runtime.optimizer_state_offload_batch_size must be positive")

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

    def resolve_compile_model(self, *, model: ModelConfig, devices: list[int]) -> None:
        """Disable torch.compile when the selected hardware cannot compile the model dtype."""

        if not self.compile_model:
            return
        reason = torch_compile_unsupported_gpu_reason(model, devices)
        if reason is None:
            return
        warnings.warn(
            f"torch.compile does not support the detected runtime configuration ({reason}); "
            "falling back to eager model execution.",
            RuntimeWarning,
            stacklevel=2,
        )
        self.compile_model = False

    def resolve_eager_decode(self, *, model: ModelConfig, lora: LoraConfig | None) -> None:
        """Use eager decode when routed-expert adapters lack a fused rollout path."""

        if self.eager_decode or lora is None:
            return
        if model.model_type == "qwen3_moe" and {
            "gate_proj",
            "up_proj",
            "down_proj",
        } & set(lora.target_modules):
            warnings.warn(
                "routed-expert LoRA uses grouped execution during rollout; falling back to eager decode.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.eager_decode = True


@dataclass(slots=True)
class ModelConfig:
    """Normalized model architecture config derived from a HF checkpoint."""

    model_type: str = "qwen3"
    checkpoint_prefix: str = "model"
    checkpoint_lm_head_key: str = "lm_head.weight"
    vocab_size: int = 151936
    pad_token_id: int = 0
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
    mrope_section: tuple[int, int, int] | None = None
    mrope_interleaved: bool = False
    num_experts: int | None = None
    num_experts_per_tok: int = 1
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_intermediate_size: int = 0
    num_shared_experts: int | None = None
    shared_expert_intermediate_size: int = 0
    vision_config: dict[str, Any] | None = None
    audio_config: dict[str, Any] | None = None
    hf_text_config: dict[str, Any] | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    audio_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
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
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    kda_safe_gate: bool = False
    kda_lower_bound: float | None = None
    no_kda_lora: bool = False
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
                self.model_type in {"gemma4", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "qwen3_5_vl", "qwen3_5_vl_moe"}
                and tp_size % self.num_key_value_heads == 0
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
    sequence_parallel: bool | None = None
    dp_size: int | None = None
    devices: list[int] | None = None
    dummy_load: bool = False
    role: Literal["train", "rollout"] = "train"
    policy_sync_bucket_mb: int = 64
    lora: LoraConfig | None = None
    lora_seed: int = 0
    reference_mode: Literal["independent", "reuse_actor_base"] = "independent"
    base_model_name_or_path: str | None = None

    def __post_init__(self) -> None:
        """Infer DP/devices and validate the distributed layout."""

        if self.sequence_parallel is not None:
            self.model.sequence_parallel = bool(self.sequence_parallel)
        self.model.validate_tp(self.tp_size)
        if self.reference_mode not in {"independent", "reuse_actor_base"}:
            raise ValueError("reference_mode must be one of: independent, reuse_actor_base")
        if self.reference_mode == "reuse_actor_base" and self.lora is None:
            raise ValueError("reference_mode='reuse_actor_base' requires native LoRA")
        if (
            self.lora is not None
            and self.model.model_type == "bailing_moe_v3"
            and self.model.moe_router_bias_update_rate != 0.0
        ):
            raise ValueError("native LoRA requires moe_router_bias_update_rate=0 to keep the base policy frozen")
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
        if any(device < 0 for device in self.devices):
            raise ValueError("devices must contain non-negative CUDA indices")
        if len(self.devices) != len(set(self.devices)):
            raise ValueError("devices must not contain duplicate CUDA indices")
        if torch.cuda.is_available():
            invalid = [device for device in self.devices if device >= torch.cuda.device_count()]
            if invalid:
                raise ValueError(f"devices are outside CUDA_VISIBLE_DEVICES: {invalid}")
        if len(self.devices) % self.tp_size != 0:
            raise ValueError("len(devices) must be divisible by tp_size")
        inferred_dp_size = len(self.devices) // self.tp_size
        if self.dp_size is None:
            self.dp_size = inferred_dp_size
        elif self.dp_size != inferred_dp_size:
            raise ValueError("dp_size must equal len(devices) // tp_size")
        if self.dp_size < 1:
            raise ValueError("dp_size must be >= 1")
        if self.policy_sync_bucket_mb < 1:
            raise ValueError("policy_sync_bucket_mb must be >= 1")
        if self.runtime.kv_block_size < 1:
            raise ValueError("runtime.kv_block_size must be >= 1")
        if self.runtime.kv_block_size % 256 != 0:
            raise ValueError("runtime.kv_block_size must be a multiple of 256 for FlashAttention paged KV")
        # CUDA rollout trainers request R3 by default. Dense checkpoints do
        # not have router decisions to capture and retain the original path.
        if self.runtime.rollout_routing_replay and self.model.num_experts is None:
            self.runtime.rollout_routing_replay = False
        self.runtime.resolve_attn_backend(model=self.model, devices=self.devices)
        self.runtime.resolve_compile_model(model=self.model, devices=self.devices)
        self.runtime.resolve_eager_decode(model=self.model, lora=self.lora)
        self.model.attn_backend = self.runtime.attn_backend

    @property
    def effective_sequence_parallel(self) -> bool:
        """Return whether training should shard activations across TP ranks."""

        return bool(self.tp_size > 1 and self.model.sequence_parallel)


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


def torch_compile_unsupported_gpu_reason(model: ModelConfig, devices: list[int] | None = None) -> str | None:
    """Return a reason when torch.compile cannot compile the model dtype on visible GPUs."""

    if model.dtype is not torch.bfloat16:
        return None
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
        if (int(major), int(minor)) >= (8, 0):
            continue
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = f"cuda:{device}"
        unsupported.append(f"{name} cc {major}.{minor} lacks native BF16 support")
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
