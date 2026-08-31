"""Phi-4-Multimodal language-backbone adapter.

PR1 intentionally supports the checkpoint's text path only. The vision and
audio towers and their modality-specific LoRA adapters are not runtime model
components here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention import CausalSelfAttention
from areno.engine.layers.mlp import GatedMLP
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.collectives import (
    scatter_to_sequence_parallel_region,
    sequence_parallel_region,
)
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.recompute import checkpoint_layer
from areno.models.base import CausalLMOutput, ModelAdapter


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
        use_long_factor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_long_factor is not None:
            if use_long_factor.shape != position_ids.shape:
                raise ValueError("Phi4MM LongRoPE factor mask must match position_ids")
            # Decode batches can contain both a short-cache row and a row
            # rebuilt with long factors.  Keep that selection as tensor math:
            # materialising cache_seqlens with .item() breaks torch.compile
            # and prevents the decode CUDA graph from being captured/replayed.
            positions = position_ids.float().unsqueeze(-1)
            short_freqs = positions * self.short_inv_freq.float()
            long_freqs = positions * self.long_inv_freq.float()
            freqs = torch.where(use_long_factor.unsqueeze(-1), long_freqs, short_freqs)
        else:
            if sequence_length is None:
                sequence_length = int(torch.max(position_ids).item()) + 1
            inv_freq = (
                self.long_inv_freq if sequence_length > self.original_max_position_embeddings else self.short_inv_freq
            )
            expanded_inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            expanded_positions = position_ids[:, None, :].float()
            freqs = (expanded_inv_freq @ expanded_positions).transpose(1, 2)
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
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
        use_long_factor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(q, position_ids, sequence_length, use_long_factor)
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
            raise ValueError("Phi4MM cached decode requires cache re-prefill before crossing the LongRoPE boundary")
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

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if infer_meta is not None and infer_meta.mode == "decode":
            # A row that was re-prefilled across the boundary has its next
            # position strictly above it; rows not rebuilt are still short.
            # This tensor-only selection is essential for a captured graph:
            # cache_seqlens.max().item() would force a Dynamo graph break.
            # The scheduler requests a re-prefill at the equality boundary,
            # so no short-factor KV cache is ever paired with a long query.
            return self.rope(
                q,
                k,
                position_ids,
                use_long_factor=position_ids.ge(self.rope.original_max_position_embeddings),
            )
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
    """Text-only Phi-4 transformer body."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([Phi4MMDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.embed_tokens(input_ids)
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


class Phi4MMForCausalLM(nn.Module):
    """Text-only Phi-4 causal LM with a truly tied vocab-parallel head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not config.tie_word_embeddings:
            raise ValueError("Phi4MMForCausalLM requires tied word embeddings")
        self.config = config
        self._longrope_cache_boundary = int(config.hf_text_config["original_max_position_embeddings"])
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
    ) -> CausalLMOutput:
        use_sequence_parallel = bool(train_meta is not None and train_meta.sequence_parallel)
        with sequence_parallel_region(use_sequence_parallel):
            hidden_states = self.model(input_ids, position_ids, train_meta, infer_meta)
            logits_shard = self.lm_head(hidden_states)
        return CausalLMOutput(logits_shard=logits_shard, hidden_states=hidden_states)

    def set_kv_caches(
        self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]], *, num_slots: int | None = None
    ) -> None:
        """Bind one paged KV-cache pair to each decoder layer."""
        del num_slots
        if len(kv_caches) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer caches, got {len(kv_caches)}")
        for layer, (k_cache, v_cache) in zip(self.layers, kv_caches, strict=True):
            layer.self_attn.set_kv_cache(k_cache, v_cache)

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

    def cache_reprefill_required(self, cache_seqlens: torch.Tensor) -> torch.Tensor:
        """Request a full long-factor KV rebuild immediately before the boundary decode."""

        return cache_seqlens.eq(self._longrope_cache_boundary)

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

    def build_policy_plan(self, model: nn.Module):
        from areno.models.phi4mm.checkpoint import build_phi4mm_policy_plan

        return build_phi4mm_policy_plan(model)
