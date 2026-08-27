"""Native Phi-4-Multimodal language, vision, and audio adapter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from areno.accel.utils import is_cuda_graph_capturing
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention import CausalSelfAttention
from areno.engine.layers.linear import MergedColumnParallelLinear, RowParallelLinear, mark_tensor_parallel_parameter
from areno.engine.layers.mlp import GatedMLP
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.collectives import (
    all_reduce,
    copy_to_tensor_parallel_region,
    gather_from_sequence_parallel_region,
    is_sequence_parallel_active,
    scatter_to_sequence_parallel_region,
    sequence_parallel_region,
)
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.recompute import checkpoint_layer
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.models.phi4mm.audio import Phi4MMAudioConfig, Phi4MMAudioEmbedding
from areno.models.phi4mm.vision import Phi4MMExtendedEmbedding, Phi4MMVisionConfig

_IMAGE_SPECIAL_TOKEN_ID = 200010
_AUDIO_SPECIAL_TOKEN_ID = 200011


def _phi4mm_vision_config(hf_config: dict[str, Any]) -> dict[str, Any] | None:
    embedding = hf_config.get("embd_layer")
    if not isinstance(embedding, dict):
        return None
    image = embedding.get("image_embd_layer")
    if not isinstance(image, dict):
        return None
    required = {
        "embedding_cls": "tune_image",
        "image_token_compression_cls": "avg_pool_2d",
        "projection_cls": "mlp",
        "use_hd_transform": True,
        "with_learnable_separator": True,
        "hd_transform_order": "sub_glb",
    }
    for key, expected in required.items():
        actual = image.get(key)
        if actual != expected:
            raise ValueError(f"Phi4MM vision requires embd_layer.image_embd_layer.{key}={expected!r}, got {actual!r}")
    config = {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_channels": 3,
        "image_size": 448,
        "patch_size": 14,
        "layer_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "hidden_act": "gelu_pytorch_tanh",
        "feature_layer": -2,
        "crop_size": int(image.get("crop_size", 448)),
        "hd_transform_order": str(image["hd_transform_order"]),
    }
    override = hf_config.get("vision_config")
    if isinstance(override, dict):
        config.update(override)
    return config


def _phi4mm_audio_config(hf_config: dict[str, Any]) -> dict[str, Any] | None:
    embedding = hf_config.get("embd_layer")
    audio_embedding = embedding.get("audio_embd_layer") if isinstance(embedding, dict) else None
    processor = hf_config.get("audio_processor")
    if not isinstance(audio_embedding, dict) or not isinstance(processor, dict):
        return None
    required_embedding = {
        "embedding_cls": "audio",
        "projection_cls": "mlp",
        "compression_rate": 8,
        "downsample_rate": 1,
        "use_qformer": False,
        "use_conv_downsample": False,
    }
    for key, expected in required_embedding.items():
        actual = audio_embedding.get(key)
        if actual != expected:
            raise ValueError(f"Phi4MM audio requires embd_layer.audio_embd_layer.{key}={expected!r}, got {actual!r}")
    if processor.get("name") != "cascades" or not isinstance(processor.get("config"), dict):
        raise ValueError("Phi4MM audio requires the cascades processor configuration")
    values = dict(processor["config"])
    required = {
        "input_layer": "nemo_conv",
        "input_size": 80,
        "attention_dim": 1024,
        "attention_heads": 16,
        "num_blocks": 24,
        "time_reduction": 8,
        "causal": True,
        "activation": "swish",
        "conv_activation": "swish",
        "conv_glu_type": "swish",
        "batch_norm": False,
    }
    for key, expected in required.items():
        if values.get(key) != expected:
            raise ValueError(
                f"Phi4MM audio requires audio_processor.config.{key}={expected!r}, got {values.get(key)!r}"
            )
    return values


def _features_by_row(features: dict[str, Any] | list[dict[str, Any] | None], batch: int) -> list[dict[str, Any] | None]:
    if isinstance(features, list):
        if len(features) != batch:
            raise ValueError(f"Phi4MM multimodal features batch mismatch: got {len(features)} rows for batch {batch}")
        return features
    if not isinstance(features, dict):
        raise TypeError("Phi4MM multimodal features must be a dict or batch-aligned list")
    if batch == 1:
        return [features]
    rows = []
    for row_idx in range(batch):
        row = {}
        for key, value in features.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch:
                row[key] = value[row_idx]
            elif isinstance(value, list) and len(value) == batch:
                row[key] = value[row_idx]
            else:
                row[key] = value
        rows.append(row)
    return rows


def _feature_tensor(
    features: dict[str, Any], key: str, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor | None:
    value = features.get(key)
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() == 0:
        return None
    return tensor.to(device=device, dtype=dtype)


def _lora_config(config: ModelConfig, adapter: str) -> tuple[int, float, float] | None:
    values = (config.hf_text_config or {}).get(f"{adapter}_lora")
    modality_config = config.vision_config if adapter == "vision" else config.audio_config
    if modality_config is None:
        return None
    if not isinstance(values, dict):
        raise ValueError(f"Phi4MM {adapter} support requires a {adapter}_lora config")
    rank = int(values["r"])
    alpha = float(values["lora_alpha"])
    dropout = float(values.get("dp", 0.0))
    if rank <= 0 or alpha <= 0 or not 0.0 <= dropout < 1.0:
        raise ValueError(f"Phi4MM {adapter}_lora requires positive r/alpha and dp in [0, 1)")
    return rank, alpha / rank, dropout


class _Phi4MMColumnLoRA(MergedColumnParallelLinear):
    def __init__(self, in_features: int, out_features: tuple[int, ...], config: ModelConfig):
        super().__init__(in_features, out_features, bias=False)
        self.lora_scales: dict[str, float] = {}
        self.lora_dropouts: dict[str, float] = {}
        self.vision_lora_mask: torch.Tensor | None = None
        self.speech_lora_mask: torch.Tensor | None = None
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        for adapter in ("vision", "speech"):
            lora = _lora_config(config, adapter)
            if lora is None:
                continue
            rank, self.lora_scales[adapter], self.lora_dropouts[adapter] = lora
            self.lora_A[adapter] = nn.Linear(in_features, rank, bias=False)
            self.lora_B[adapter] = nn.Linear(rank, sum(self.local_out_features), bias=False)
            mark_tensor_parallel_parameter(
                self.lora_A[adapter].weight, False, sequence_parallel=False, tp_grad_allreduce=True
            )
            mark_tensor_parallel_parameter(self.lora_B[adapter].weight, True, sequence_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        if all(getattr(self, f"{adapter}_lora_mask") is None for adapter in self.lora_A):
            return output
        full_input = (
            gather_from_sequence_parallel_region(x)
            if is_sequence_parallel_active()
            else copy_to_tensor_parallel_region(x)
        )
        for adapter in self.lora_A:
            mask = getattr(self, f"{adapter}_lora_mask")
            if mask is None:
                continue
            dropped = F.dropout(full_input, p=self.lora_dropouts[adapter], training=self.training)
            delta = self.lora_B[adapter](self.lora_A[adapter](dropped)) * self.lora_scales[adapter]
            output = output + delta * mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        return output


class _Phi4MMRowLoRA(RowParallelLinear):
    def __init__(self, in_features: int, out_features: int, config: ModelConfig):
        super().__init__(in_features, out_features, bias=False)
        self.lora_scales: dict[str, float] = {}
        self.lora_dropouts: dict[str, float] = {}
        self.vision_lora_mask: torch.Tensor | None = None
        self.speech_lora_mask: torch.Tensor | None = None
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        for adapter in ("vision", "speech"):
            lora = _lora_config(config, adapter)
            if lora is None:
                continue
            rank, self.lora_scales[adapter], self.lora_dropouts[adapter] = lora
            self.lora_A[adapter] = nn.Linear(self.local_in_features, rank, bias=False)
            self.lora_B[adapter] = nn.Linear(rank, out_features, bias=False)
            mark_tensor_parallel_parameter(self.lora_A[adapter].weight, True, sequence_parallel=True)
            mark_tensor_parallel_parameter(
                self.lora_B[adapter].weight, False, sequence_parallel=False, tp_grad_allreduce=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        for adapter in self.lora_A:
            mask = getattr(self, f"{adapter}_lora_mask")
            if mask is None:
                continue
            dropped = F.dropout(x, p=self.lora_dropouts[adapter], training=self.training)
            latent = all_reduce(self.lora_A[adapter](dropped))
            delta = self.lora_B[adapter](latent) * self.lora_scales[adapter]
            delta = delta * mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
            if is_sequence_parallel_active():
                delta = scatter_to_sequence_parallel_region(delta)
            output = output + delta
        return output


def _require_bool(hf_config: dict[str, Any], key: str, expected: bool) -> None:
    value = bool(hf_config.get(key, expected))
    if value is not expected:
        raise ValueError(f"Phi4MM requires {key}={expected}, got {value}")


def _validated_longrope(hf_config: dict[str, Any], rotary_dim: int) -> dict[str, Any]:
    rope = hf_config.get("rope_scaling")
    if not isinstance(rope, dict):
        raise ValueError("Phi4MM requires a rope_scaling mapping")
    if set(rope) != {"type", "short_factor", "long_factor"}:
        raise ValueError("Phi4MM rope_scaling must contain exactly: type, short_factor, long_factor")
    if rope["type"] != "longrope":
        raise ValueError(f"Phi4MM only supports rope_scaling.type='longrope', got {rope['type']!r}")

    expected_factors = rotary_dim // 2
    normalized = {"type": "longrope"}
    for key in ("short_factor", "long_factor"):
        factors = rope[key]
        if not isinstance(factors, list) or len(factors) != expected_factors:
            raise ValueError(f"Phi4MM {key} must contain {expected_factors} values")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in factors):
            raise ValueError(f"Phi4MM {key} values must be positive numbers")
        normalized[key] = tuple(float(value) for value in factors)
    return normalized


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Phi4MMLongRoPEScaledRotaryEmbedding(nn.Module):
    """Official Phi-4 partial LongRoPE math without per-layer position caches."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hf_text_config is None:
            raise ValueError("Phi4MM requires the validated HF text config")
        self.dim = int(config.head_dim * config.partial_rotary_factor)
        if self.dim <= 0 or self.dim % 2:
            raise ValueError("Phi4MM rotary dimension must be a positive even number")
        rope_scaling = config.hf_text_config["rope_scaling"]
        expected_factors = self.dim // 2
        short_factor = rope_scaling["short_factor"]
        long_factor = rope_scaling["long_factor"]
        if len(short_factor) != expected_factors or len(long_factor) != expected_factors:
            raise ValueError(f"Phi4MM short_factor and long_factor must contain {expected_factors} values")

        self.max_position_embeddings = int(config.max_position_embeddings)
        self.original_max_position_embeddings = int(config.hf_text_config["original_max_position_embeddings"])
        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim
        base_freq = config.rope_theta**inv_freq_shape
        self.register_buffer(
            "short_inv_freq", 1.0 / (torch.tensor(short_factor, dtype=torch.float32) * base_freq), persistent=False
        )
        self.register_buffer(
            "long_inv_freq", 1.0 / (torch.tensor(long_factor, dtype=torch.float32) * base_freq), persistent=False
        )
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = (
            1.0 if scale <= 1.0 else math.sqrt(1.0 + math.log(scale) / math.log(self.original_max_position_embeddings))
        )

    def _apply(self, fn):
        super()._apply(fn)
        # Long-context phases must remain FP32 even when model weights are cast.
        self.short_inv_freq = self.short_inv_freq.float()
        self.long_inv_freq = self.long_inv_freq.float()
        return self

    @torch.no_grad()
    def cos_sin(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_length is None:
            sequence_length = int(torch.max(position_ids).item()) + 1
        inv_freq = (
            self.long_inv_freq if sequence_length > self.original_max_position_embeddings else self.short_inv_freq
        )
        expanded_inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        expanded_positions = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (expanded_inv_freq @ expanded_positions).transpose(1, 2)
            embedding = torch.cat((freqs, freqs), dim=-1)
            cos = embedding.cos() * self.scaling_factor
            sin = embedding.sin() * self.scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(q, position_ids, sequence_length)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_rot, q_pass = q[..., : self.dim], q[..., self.dim :]
        k_rot, k_pass = k[..., : self.dim], k[..., self.dim :]
        q_embed = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q_pass), dim=-1)
        k_embed = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k_pass), dim=-1)
        return q_embed, k_embed


def _phi4mm_longrope_sequence_length(
    position_ids: torch.Tensor,
    train_meta: TrainMeta | None,
    infer_meta: InferMeta | None,
    original_max_position_embeddings: int,
) -> int:
    if infer_meta is not None and infer_meta.mode == "decode":
        if infer_meta.cache_seqlens is None:
            raise ValueError("Phi4MM decode requires cache_seqlens for LongRoPE selection")
        sequence_length = int(infer_meta.cache_seqlens.max().item()) + 1
        if sequence_length > original_max_position_embeddings:
            raise ValueError(
                "Phi4MM cached decode cannot cross the LongRoPE boundary because cached keys may use short factors; "
                "run a full long-context prefill"
            )
        return sequence_length

    if infer_meta is not None:
        sequence_length = int(position_ids.max().item()) + 1
        if sequence_length > original_max_position_embeddings:
            if infer_meta.cu_seqlens is None:
                raise ValueError("Phi4MM prefill requires cu_seqlens for LongRoPE boundary validation")
            starts = infer_meta.cu_seqlens[:-1].to(dtype=torch.long)
            flat_positions = position_ids.reshape(-1)
            if bool(torch.any(flat_positions[starts] != 0)):
                raise ValueError(
                    "Phi4MM chunked prefill cannot cross the LongRoPE boundary because cached keys use short factors; "
                    "increase the prefill token budget and run a full prefill"
                )
        return sequence_length

    if train_meta is not None and train_meta.max_seqlen is not None:
        return int(train_meta.max_seqlen)
    return int(position_ids.shape[-1])


class Phi4MMAttention(CausalSelfAttention):
    """AReno GQA attention with a Phi-owned rotary implementation."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        if config.qk_norm:
            raise ValueError("Phi4MMAttention requires qk_norm=False")
        super().__init__(config, layer_idx, rotary_embedding=Phi4MMLongRoPEScaledRotaryEmbedding(config))
        self.qkv_proj = _Phi4MMColumnLoRA(
            config.hidden_size,
            (
                config.num_attention_heads * config.head_dim,
                config.num_key_value_heads * config.head_dim,
                config.num_key_value_heads * config.head_dim,
            ),
            config,
        )
        self.o_proj = _Phi4MMRowLoRA(config.num_attention_heads * config.head_dim, config.hidden_size, config)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if infer_meta is not None and infer_meta.mode == "decode" and is_cuda_graph_capturing(q):
            # DecodeGraph validates its dynamic cache lengths before replay.
            # Capture itself always records the supported short-factor path.
            sequence_length = self.rope.original_max_position_embeddings
        else:
            sequence_length = _phi4mm_longrope_sequence_length(
                position_ids,
                train_meta,
                infer_meta,
                self.rope.original_max_position_embeddings,
            )
        return self.rope(q, k, position_ids, sequence_length)


class Phi4MMDecoderLayer(nn.Module):
    """Phi-4 pre-norm decoder block composed from AReno shared layers."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Phi4MMAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)
        if config.vision_config is not None or config.audio_config is not None:
            self.mlp.gate_up_proj = _Phi4MMColumnLoRA(
                config.hidden_size, (config.intermediate_size, config.intermediate_size), config
            )
            self.mlp.down_proj = _Phi4MMRowLoRA(config.intermediate_size, config.hidden_size, config)

    def set_lora_masks(self, vision: torch.Tensor | None, speech: torch.Tensor | None) -> None:
        for module in (self.self_attn.qkv_proj, self.self_attn.o_proj, self.mlp.gate_up_proj, self.mlp.down_proj):
            if hasattr(module, "vision_lora_mask"):
                module.vision_lora_mask = vision
                module.speech_lora_mask = speech

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, position_ids, train_meta, infer_meta)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class Phi4MMModel(nn.Module):
    """Phi-4 transformer body with optional native vision and audio paths."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.embed_tokens_extend = None
        if config.vision_config is not None or config.audio_config is not None:
            vision_config = (
                Phi4MMVisionConfig.from_dict(config.vision_config) if config.vision_config is not None else None
            )
            self.embed_tokens_extend = Phi4MMExtendedEmbedding(vision_config, config.hidden_size, config.dtype)
            if config.audio_config is not None:
                self.embed_tokens_extend.audio_embed = Phi4MMAudioEmbedding(
                    Phi4MMAudioConfig.from_dict(config.audio_config), config.hidden_size, config.dtype
                )
        if self.embed_tokens_extend is not None:
            for parameter in self.embed_tokens_extend.parameters():
                mark_tensor_parallel_parameter(parameter, False, sequence_parallel=False, tp_grad_allreduce=True)
        self.layers = nn.ModuleList([Phi4MMDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.register_buffer("vision_lora_slots", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("speech_lora_slots", torch.empty(0, dtype=torch.bool), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
        features: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        vision_lora_mask, speech_lora_mask = self._lora_masks(input_ids, features, train_meta, infer_meta)
        for layer in self.layers:
            layer.set_lora_masks(vision_lora_mask, speech_lora_mask)
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self._apply_multimodal_features(hidden_states, input_ids, features)
        use_sequence_parallel = bool(train_meta is not None and train_meta.sequence_parallel)
        if use_sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        with sequence_parallel_region(use_sequence_parallel):
            for layer in self.layers:
                hidden_states = checkpoint_layer(
                    layer,
                    hidden_states,
                    position_ids,
                    train_meta,
                    infer_meta,
                    train_meta=train_meta,
                    infer_meta=infer_meta,
                )
            return self.norm(hidden_states)

    def _lora_masks(
        self,
        input_ids: torch.Tensor,
        features: dict[str, Any] | list[dict[str, Any] | None] | None,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.embed_tokens_extend is None:
            return None, None
        if infer_meta is not None and infer_meta.mode == "decode":
            if infer_meta.recurrent_slots is None:
                raise ValueError("Phi4MM multimodal decode requires recurrent modality slots")
            vision = (
                self.vision_lora_slots.index_select(0, infer_meta.recurrent_slots).view_as(input_ids)
                if self.vision_lora_slots.numel()
                else None
            )
            speech = (
                self.speech_lora_slots.index_select(0, infer_meta.recurrent_slots).view_as(input_ids)
                if self.speech_lora_slots.numel()
                else None
            )
            return vision, speech
        image_mask = self._image_token_mask(input_ids, features)
        audio_mask = self._audio_token_mask(input_ids, features)
        explicit_image_modes = None
        explicit_audio_modes = None
        if isinstance(features, dict) and features.get("image_sequence_mask") is not None:
            explicit_image_modes = torch.as_tensor(
                features["image_sequence_mask"], device=input_ids.device, dtype=torch.bool
            ).reshape(-1)
        if isinstance(features, dict) and features.get("audio_sequence_mask") is not None:
            explicit_audio_modes = torch.as_tensor(
                features["audio_sequence_mask"], device=input_ids.device, dtype=torch.bool
            ).reshape(-1)
        sequence_offsets = None
        if infer_meta is not None and infer_meta.cu_seqlens is not None:
            sequence_offsets = infer_meta.cu_seqlens
        elif train_meta is not None and train_meta.cu_seqlens is not None:
            sequence_offsets = train_meta.cu_seqlens
        if sequence_offsets is None:
            image_modes = explicit_image_modes if explicit_image_modes is not None else image_mask.any(dim=1)
            audio_modes = explicit_audio_modes if explicit_audio_modes is not None else audio_mask.any(dim=1)
            if int(image_modes.numel()) != int(input_ids.shape[0]) or int(audio_modes.numel()) != int(
                input_ids.shape[0]
            ):
                raise ValueError("Phi4MM modality sequence masks must contain one value per input row")
            # Official VISION_SPEECH mode uses the vision adapter and vision audio projector.
            speech_modes = audio_modes & ~image_modes
            return image_modes[:, None].expand_as(input_ids), speech_modes[:, None].expand_as(input_ids)
        else:
            flat_image = image_mask.reshape(-1)
            flat_audio = audio_mask.reshape(-1)
            vision_mask = torch.zeros_like(flat_image)
            speech_mask = torch.zeros_like(flat_audio)
            vision_modes = []
            speech_modes = []
            offsets = sequence_offsets.detach().to(device="cpu", dtype=torch.long).tolist()
            sequence_count = len(offsets) - 1
            if explicit_image_modes is not None and int(explicit_image_modes.numel()) != sequence_count:
                raise ValueError("Phi4MM image_sequence_mask must contain one value per packed sequence")
            if explicit_audio_modes is not None and int(explicit_audio_modes.numel()) != sequence_count:
                raise ValueError("Phi4MM audio_sequence_mask must contain one value per packed sequence")
            for sequence_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
                image_mode = (
                    bool(explicit_image_modes[sequence_idx])
                    if explicit_image_modes is not None
                    else bool(flat_image[start:end].any())
                )
                audio_mode = (
                    bool(explicit_audio_modes[sequence_idx])
                    if explicit_audio_modes is not None
                    else bool(flat_audio[start:end].any())
                )
                speech_mode = audio_mode and not image_mode
                vision_modes.append(image_mode)
                speech_modes.append(speech_mode)
                vision_mask[start:end] = image_mode
                speech_mask[start:end] = speech_mode
            if infer_meta is not None and infer_meta.recurrent_slots is not None and self.vision_lora_slots.numel() > 0:
                self.vision_lora_slots.index_copy_(
                    0,
                    infer_meta.recurrent_slots,
                    torch.tensor(vision_modes, device=self.vision_lora_slots.device, dtype=torch.bool),
                )
            if infer_meta is not None and infer_meta.recurrent_slots is not None and self.speech_lora_slots.numel() > 0:
                self.speech_lora_slots.index_copy_(
                    0,
                    infer_meta.recurrent_slots,
                    torch.tensor(speech_modes, device=self.speech_lora_slots.device, dtype=torch.bool),
                )
            return vision_mask.view_as(input_ids), speech_mask.view_as(input_ids)

    def _image_token_mask(
        self,
        input_ids: torch.Tensor,
        features: dict[str, Any] | list[dict[str, Any] | None] | None,
    ) -> torch.Tensor:
        if isinstance(features, dict) and features.get("image_token_mask") is not None:
            return torch.as_tensor(features["image_token_mask"], device=input_ids.device, dtype=torch.bool).view_as(
                input_ids
            )
        return input_ids == int(self.config.image_token_id or _IMAGE_SPECIAL_TOKEN_ID)

    def _vision_lora_mask(
        self,
        input_ids: torch.Tensor,
        features: dict[str, Any] | list[dict[str, Any] | None] | None,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor | None:
        """Backward-compatible view of the vision half of modality LoRA state."""

        return self._lora_masks(input_ids, features, train_meta, infer_meta)[0]

    def _audio_token_mask(
        self,
        input_ids: torch.Tensor,
        features: dict[str, Any] | list[dict[str, Any] | None] | None,
    ) -> torch.Tensor:
        if isinstance(features, dict) and features.get("audio_token_mask") is not None:
            return torch.as_tensor(features["audio_token_mask"], device=input_ids.device, dtype=torch.bool).view_as(
                input_ids
            )
        return input_ids == int(self.config.audio_token_id or _AUDIO_SPECIAL_TOKEN_ID)

    @torch._dynamo.disable
    def _apply_multimodal_features(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        features: dict[str, Any] | list[dict[str, Any] | None] | None,
    ) -> torch.Tensor:
        if features is None:
            return hidden_states
        if self.embed_tokens_extend is None:
            raise ValueError("Phi4MM multimodal features require a configured modality tower")
        rows = _features_by_row(features, int(input_ids.shape[0]))
        output = hidden_states.clone()
        for row_idx, row in enumerate(rows):
            if row is None:
                continue
            for modality, embeds in (
                ("image", self._project_image_feature_rows(row, hidden_states.device)),
                ("audio", self._project_audio_feature_rows(row, hidden_states.device)),
            ):
                if embeds is None:
                    continue
                mask = None if row.get(f"{modality}_feature_rows") is not None else row.get(f"{modality}_token_mask")
                if mask is None:
                    default_id = self.config.image_token_id if modality == "image" else self.config.audio_token_id
                    fallback_id = _IMAGE_SPECIAL_TOKEN_ID if modality == "image" else _AUDIO_SPECIAL_TOKEN_ID
                    token_id = int(row.get(f"{modality}_token_id", default_id or fallback_id))
                    mask = input_ids[row_idx] == token_id
                else:
                    mask = torch.as_tensor(mask, device=input_ids.device, dtype=torch.bool).reshape(-1)
                if mask.shape != input_ids[row_idx].shape:
                    raise ValueError(f"Phi4MM {modality}_token_mask must match the input token row")
                if int(mask.sum().item()) != int(embeds.shape[0]):
                    raise ValueError(
                        f"Phi4MM {modality} token count does not match projected embeddings: "
                        f"tokens={int(mask.sum().item())} embeds={int(embeds.shape[0])}"
                    )
                output[row_idx, mask] = embeds.to(device=output.device, dtype=output.dtype)
        return output

    def _project_image_feature_rows(self, features: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        rows = features.get("image_feature_rows")
        if rows is not None:
            pieces = [self._project_image_feature(dict(row), device) for row in rows if row is not None]
            pieces = [piece for piece in pieces if piece is not None]
            return torch.cat(pieces, dim=0) if pieces else None
        return self._project_image_feature(features, device)

    def _project_image_feature(self, features: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        existing = _feature_tensor(features, "image_embeds", device, self.config.dtype)
        if existing is not None:
            return existing
        pixels = _feature_tensor(features, "input_image_embeds", device, self.config.dtype)
        if pixels is None:
            return None
        sizes = _feature_tensor(features, "image_sizes", device, torch.long)
        mask = _feature_tensor(features, "image_attention_mask", device, torch.bool)
        if sizes is None or mask is None:
            raise ValueError("Phi4MM processor output requires image_sizes and image_attention_mask")
        image_embed = getattr(self.embed_tokens_extend, "image_embed", None)
        if image_embed is None:
            raise ValueError("Phi4MM image features require a configured vision tower")
        image_embeds = image_embed(pixels, sizes, mask)
        offset = int(features.get("image_token_offset", 0) or 0)
        count = features.get("image_token_count")
        if count is not None:
            return image_embeds[offset : offset + int(count)]
        return image_embeds[offset:]

    def _project_audio_feature_rows(self, features: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        rows = features.get("audio_feature_rows")
        if rows is not None:
            pieces = [self._project_audio_feature(dict(row), device) for row in rows if row is not None]
            pieces = [piece for piece in pieces if piece is not None]
            return torch.cat(pieces, dim=0) if pieces else None
        return self._project_audio_feature(features, device)

    def _project_audio_feature(self, features: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        existing = _feature_tensor(features, "audio_embeds", device, self.config.dtype)
        if existing is not None:
            return existing
        inputs = _feature_tensor(features, "input_audio_embeds", device, self.config.dtype)
        if inputs is None:
            return None
        audio_embed = getattr(self.embed_tokens_extend, "audio_embed", None)
        if audio_embed is None:
            raise ValueError("Phi4MM audio features require a configured audio tower")
        attention_mask = _feature_tensor(features, "audio_attention_mask", device, torch.bool)
        sizes = _feature_tensor(features, "audio_embed_sizes", device, torch.long)
        if sizes is None:
            raise ValueError("Phi4MM processor output requires audio_embed_sizes")
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        input_mode = features.get("input_mode", 2)
        if isinstance(input_mode, torch.Tensor):
            input_mode = int(input_mode.reshape(-1)[0].item())
        projection_mode = "vision" if int(input_mode) in (1, 3) else "speech"
        projected = audio_embed(inputs, attention_mask, projection_mode)
        sizes_list = sizes.reshape(-1).detach().to(device="cpu", dtype=torch.long).tolist()
        if len(sizes_list) != int(projected.shape[0]):
            raise ValueError("Phi4MM audio_embed_sizes must contain one entry per audio segment")
        merged = torch.cat([projected[index, : int(size)] for index, size in enumerate(sizes_list)], dim=0)
        offset = int(features.get("audio_token_offset", 0) or 0)
        count = features.get("audio_token_count")
        if count is not None:
            return merged[offset : offset + int(count)]
        return merged[offset:]


class Phi4MMForCausalLM(nn.Module):
    """Phi-4-Multimodal causal LM with a truly tied vocab-parallel head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not config.tie_word_embeddings:
            raise ValueError("Phi4MMForCausalLM requires tied word embeddings")
        self.config = config
        self.decode_cache_length_limit = int(config.hf_text_config["original_max_position_embeddings"])
        self.model = Phi4MMModel(config)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)
        self._tie_word_embeddings()

    def _tie_word_embeddings(self) -> None:
        embedding = self.model.embed_tokens
        if (self.lm_head.vocab_start, self.lm_head.vocab_end) != (embedding.vocab_start, embedding.vocab_end):
            raise ValueError("Phi4MM embedding and LM head use different TP vocabulary ranges")
        if self.lm_head.weight.shape != embedding.weight.shape:
            raise ValueError("Phi4MM embedding and LM head local weight shapes differ")
        self.lm_head.weight = embedding.weight

    @property
    def layers(self) -> nn.ModuleList:
        """Expose decoder layers to the shared checkpoint machinery."""

        return self.model.layers

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
        features: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> CausalLMOutput:
        use_sequence_parallel = bool(train_meta is not None and train_meta.sequence_parallel)
        with sequence_parallel_region(use_sequence_parallel):
            hidden_states = self.model(input_ids, position_ids, train_meta, infer_meta, features)
            logits_shard = self.lm_head(hidden_states)
        return CausalLMOutput(logits_shard=logits_shard, hidden_states=hidden_states)

    def set_kv_caches(
        self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]], *, num_slots: int | None = None
    ) -> None:
        """Bind one paged KV-cache pair to each decoder layer."""
        if len(kv_caches) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer caches, got {len(kv_caches)}")
        for layer, (k_cache, v_cache) in zip(self.layers, kv_caches, strict=True):
            layer.self_attn.set_kv_cache(k_cache, v_cache)
        slot_count = int(num_slots) if num_slots is not None else (int(kv_caches[0][0].shape[0]) if kv_caches else 0)
        self.model.vision_lora_slots = torch.zeros(slot_count, device=next(self.parameters()).device, dtype=torch.bool)
        self.model.speech_lora_slots = torch.zeros(slot_count, device=next(self.parameters()).device, dtype=torch.bool)

    @torch.no_grad()
    def reset_recurrent_cache_slots(self, slots: torch.Tensor) -> None:
        if self.model.vision_lora_slots.numel() > 0:
            self.model.vision_lora_slots.index_fill_(0, slots, False)
        if self.model.speech_lora_slots.numel() > 0:
            self.model.speech_lora_slots.index_fill_(0, slots, False)

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        return None

    @torch.no_grad()
    def clear_infer_weights(self) -> None:
        return None

    @torch.no_grad()
    def offload_train_weights(self) -> None:
        return None

    @torch.no_grad()
    def onload_train_weights(self, device: torch.device) -> None:
        del device
        return None

    @torch.no_grad()
    def finalize_router_expert_bias(self, tp_group, dp_group) -> None:
        del tp_group, dp_group
        return None

    def allocate_kv_caches(
        self, num_blocks: int, block_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Allocate the standard paged GQA cache layout for every layer."""
        caches = []
        for layer in self.layers:
            attention = layer.self_attn
            shape = (num_blocks, block_size, attention.local_kv_heads, attention.head_dim)
            caches.append(
                (
                    torch.empty(shape, device=device, dtype=self.config.dtype),
                    torch.empty(shape, device=device, dtype=self.config.dtype),
                )
            )
        return caches

    def clear_kv_caches(self) -> None:
        for layer in self.layers:
            layer.self_attn.clear_kv_cache()

    @torch.no_grad()
    def reset_kv_caches(self) -> None:
        return None

    @torch.no_grad()
    def offload_kv_caches(self) -> None:
        for layer in self.layers:
            attention = layer.self_attn
            if attention.k_cache.numel() > 0:
                attention.k_cache = attention.k_cache.to(device="cpu")
            if attention.v_cache.numel() > 0:
                attention.v_cache = attention.v_cache.to(device="cpu")
            attention.infer_backend = None

    @torch.no_grad()
    def onload_kv_caches(self, device: torch.device) -> bool:
        found = False
        for layer in self.layers:
            attention = layer.self_attn
            if attention.k_cache.numel() > 0:
                found = True
                if attention.k_cache.device != device:
                    attention.k_cache = attention.k_cache.to(device=device)
            if attention.v_cache.numel() > 0 and attention.v_cache.device != device:
                attention.v_cache = attention.v_cache.to(device=device)
        return found


class Phi4MMAdapter(ModelAdapter):
    """Translate the official Phi-4-Multimodal config into AReno semantics."""

    name = "phi4mm"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        return str(hf_config.get("model_type", "")).lower() == self.name

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        hidden_size = int(hf_config["hidden_size"])
        num_attention_heads = int(hf_config["num_attention_heads"])
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Phi4MM hidden_size must be divisible by num_attention_heads")
        head_dim = hidden_size // num_attention_heads
        partial_rotary_factor = float(hf_config.get("partial_rotary_factor", 1.0))
        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError("Phi4MM partial_rotary_factor must be in (0, 1]")
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError("Phi4MM rotary dimension must be a positive even number")

        if str(hf_config.get("hidden_act", "silu")) != "silu":
            raise ValueError("Phi4MM language backbone requires hidden_act='silu'")
        _require_bool(hf_config, "attention_bias", False)
        _require_bool(hf_config, "mlp_bias", False)
        _require_bool(hf_config, "lm_head_bias", False)
        _require_bool(hf_config, "tie_word_embeddings", True)

        original_max_position_embeddings = int(hf_config.get("original_max_position_embeddings", 4096))
        max_position_embeddings = int(hf_config.get("max_position_embeddings", original_max_position_embeddings))
        if original_max_position_embeddings <= 0 or max_position_embeddings < original_max_position_embeddings:
            raise ValueError("Phi4MM max_position_embeddings must be at least original_max_position_embeddings > 0")
        rope_scaling = _validated_longrope(hf_config, rotary_dim)

        # Preserve the validated LongRoPE fields for the Phi-specific rotary implementation.
        text_config = dict(hf_config)
        text_config["rope_scaling"] = rope_scaling
        text_config["original_max_position_embeddings"] = original_max_position_embeddings
        vision_config = _phi4mm_vision_config(hf_config)
        audio_config = _phi4mm_audio_config(hf_config)

        return ModelConfig(
            model_type=self.name,
            checkpoint_prefix="model",
            vocab_size=int(hf_config["vocab_size"]),
            pad_token_id=int(hf_config.get("pad_token_id", 0) or 0),
            hidden_size=hidden_size,
            intermediate_size=int(hf_config["intermediate_size"]),
            num_hidden_layers=int(hf_config["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(hf_config.get("num_key_value_heads", num_attention_heads)),
            head_dim=head_dim,
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-5)),
            rope_theta=float(hf_config.get("rope_theta", 10_000.0)),
            max_position_embeddings=max_position_embeddings,
            tie_word_embeddings=True,
            qkv_bias=False,
            qk_norm=False,
            dtype=_parse_dtype(hf_config.get("torch_dtype") or hf_config.get("dtype")),
            hidden_act="silu",
            sliding_window=hf_config.get("sliding_window"),
            partial_rotary_factor=partial_rotary_factor,
            sequence_parallel=bool(hf_config.get("sequence_parallel", True)),
            hf_text_config=text_config,
            vision_config=vision_config,
            audio_config=audio_config,
            image_token_id=_IMAGE_SPECIAL_TOKEN_ID if vision_config is not None else None,
            audio_token_id=_AUDIO_SPECIAL_TOKEN_ID if audio_config is not None else None,
        )

    def build(self, config: ModelConfig) -> nn.Module:
        if config.model_type != self.name:
            raise ValueError(f"Phi4MMAdapter cannot build model_type={config.model_type!r}")
        return Phi4MMForCausalLM(config)

    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        from areno.models.phi4mm.checkpoint import load_phi4mm_weights

        load_phi4mm_weights(model, model_path)

    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        from areno.models.phi4mm.checkpoint import save_phi4mm_weights

        return save_phi4mm_weights(model, output_path, source_path)
