"""MiniCPM-V-4.6 language-backbone causal-LM adapter.

Targets the OpenBMB MiniCPM-V-4.6 multimodal checkpoints
(``model_type == "minicpmv4_6"``). The adapter owns the NaViT vision encoder,
window merger, visual-to-LLM projector, and hybrid text decoder. Processor
features are accepted at the model boundary and image placeholder token rows
are replaced with projected visual embeddings before the decoder runs.

Notable peculiarities:
    * Hybrid layer stack — each decoder block is either a standard softmax
      "full_attention" with a sigmoid-gated output (``MiniCPMFullAttention``)
      or a Gated Delta-Net style linear-attention block
      (``MiniCPMGatedDeltaNet``), selected by ``layer_types[layer_idx]``.
    * Full-attention layers fuse Q, an output-gate Q (same shape as Q), K
      and V into a single ``MergedColumnParallelLinear`` so the checkpoint
      loader has to split the HF ``q_proj`` tensor into a (q, gate) pair
      by head.
    * Gated Delta-Net uses depthwise causal Conv1d as a short-term mixer
      (kernel size 4), a recurrent state plus per-head learnable decay
      (``A_log``) / time-step bias (``dt_bias``), and an RMSNorm-with-SiLU
      output gate. State and conv caches live alongside the layer for
      paged-attention-style inference.
    * RMSNorm scales are loaded with a +1 offset (the HF tensors store
      ``scale - 1``) — that adjustment happens in the loader, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from areno.accel.ops import log_once
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend, build_infer_attention_backend
from areno.engine.layers.attention_backend.train import build_train_attention_backend
from areno.engine.layers.linear import MergedColumnParallelLinear, RowParallelLinear, mark_tensor_parallel_parameter
from areno.engine.layers.mlp import GatedMLP
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.rotary import PartialRotaryEmbedding
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.collectives import scatter_to_sequence_parallel_region, sequence_parallel_region
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.recompute import checkpoint_layer
from areno.models._shared.dynamo_wrappers import (
    _areno_depthwise_causal_conv1d_silu_decode_no_compile,
    _areno_depthwise_causal_conv1d_silu_no_compile,
    _areno_packed_depthwise_causal_conv1d_silu_no_compile,
    _areno_rmsnorm_silu_gate_no_compile,
    _areno_sigmoid_no_compile,
    _areno_softplus_no_compile,
    _fla_causal_conv1d_no_compile,
    _fla_chunk_gated_delta_rule_no_compile,
    _fla_fused_recurrent_gated_delta_rule_no_compile,
    _require_fla_gdn,
)
from areno.models.base import CausalLMOutput, ModelAdapter

# MiniCPM-V-4.6 uses the Qwen3.5 hybrid text topology.  Keep these defaults
# for old configs which predate the explicit linear-* fields.
_LINEAR_NUM_HEADS = 16
_LINEAR_HEAD_DIM = 128
_LINEAR_CONV_KERNEL = 4


def _feature_tensor(
    features: dict[str, Any], key: str, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    value = features.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=dtype)


def _features_by_row(features: dict[str, Any] | list[dict[str, Any] | None], batch: int) -> list[dict[str, Any] | None]:
    if isinstance(features, list):
        if len(features) != batch:
            raise ValueError(f"MiniCPM multimodal features batch mismatch: got {len(features)} rows for batch {batch}")
        return features
    if not isinstance(features, dict):
        raise TypeError("MiniCPM multimodal features must be a dict or a batch-aligned list of dicts")
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


def _image_embeds_for_row(features: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    for key in ("image_embeds", "image_features", "projected_image_embeds", "inputs_embeds"):
        value = features.get(key)
        if value is not None:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            return value.to(device=device, dtype=dtype)
    return None


def _image_token_mask(features: dict[str, Any], input_ids: torch.Tensor) -> torch.Tensor:
    mask = features.get("image_token_mask")
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        mask = mask.reshape(-1).to(device=input_ids.device, dtype=torch.bool)
        if mask.shape != input_ids.shape:
            raise ValueError(
                f"MiniCPM image_token_mask shape {tuple(mask.shape)} does not match input row {tuple(input_ids.shape)}"
            )
        return mask
    image_token_id = features.get("image_token_id")
    if image_token_id is None:
        raise ValueError("MiniCPM multimodal features require image_token_mask or image_token_id")
    return input_ids == int(image_token_id)


def _recurrent_cache_slots(infer_meta: InferMeta) -> torch.Tensor:
    if infer_meta.recurrent_slots is not None:
        return infer_meta.recurrent_slots.long()
    if infer_meta.block_table is None:
        raise RuntimeError("MiniCPM GDN inference requires recurrent_slots or block_table")
    return infer_meta.block_table[:, 0].long()


class MiniCPMV46VisionEmbeddings(nn.Module):
    """NaViT-style variable-resolution patch embedding used by MiniCPM-V."""

    def __init__(self, vision_config: dict[str, Any], dtype: torch.dtype):
        super().__init__()
        self.embed_dim = int(vision_config["hidden_size"])
        self.image_size = int(vision_config.get("image_size", 224))
        self.patch_size = int(vision_config.get("patch_size", 16))
        self.patch_embedding = nn.Conv2d(
            int(vision_config.get("num_channels", 3)),
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
            dtype=dtype,
        )
        num_side = self.image_size // self.patch_size
        self.num_patches_per_side = num_side
        self.position_embedding = nn.Embedding(num_side * num_side, self.embed_dim, dtype=dtype)

    def forward(self, pixel_values: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError("MiniCPM-V pixel_values must have shape (1, C, patch_size, packed_width)")
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        boundaries = torch.arange(
            1 / self.num_patches_per_side,
            1.0,
            1 / self.num_patches_per_side,
            device=self.position_embedding.weight.device,
        )
        position_embeddings = []
        for target_size in target_sizes.detach().cpu().to(dtype=torch.long):
            height, width = (int(target_size[0]), int(target_size[1]))
            fractional_h = torch.arange(0, 1 - 1e-6, 1 / height, device=boundaries.device)
            fractional_w = torch.arange(0, 1 - 1e-6, 1 / width, device=boundaries.device)
            bucket_h = torch.bucketize(fractional_h, boundaries, right=True)
            bucket_w = torch.bucketize(fractional_w, boundaries, right=True)
            pos_ids = (bucket_h[:, None] * self.num_patches_per_side + bucket_w).flatten()
            position_embeddings.append(self.position_embedding(pos_ids))
        if position_embeddings:
            embeddings = embeddings + torch.cat(position_embeddings, dim=0).unsqueeze(0).to(embeddings.dtype)
        return embeddings


class MiniCPMV46VisionAttention(nn.Module):
    def __init__(self, vision_config: dict[str, Any], dtype: torch.dtype):
        super().__init__()
        self.dim = int(vision_config["hidden_size"])
        self.num_heads = int(vision_config["num_attention_heads"])
        self.head_dim = self.dim // self.num_heads
        if self.head_dim * self.num_heads != self.dim:
            raise ValueError("MiniCPM vision hidden_size must be divisible by num_attention_heads")
        self.q_proj = nn.Linear(self.dim, self.dim, dtype=dtype)
        self.k_proj = nn.Linear(self.dim, self.dim, dtype=dtype)
        self.v_proj = nn.Linear(self.dim, self.dim, dtype=dtype)
        self.out_proj = nn.Linear(self.dim, self.dim, dtype=dtype)

    @torch._dynamo.disable
    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
        q_parts = torch.split(q, lengths, dim=2)
        k_parts = torch.split(k, lengths, dim=2)
        v_parts = torch.split(v, lengths, dim=2)
        outputs = [
            F.scaled_dot_product_attention(qi, ki, vi, is_causal=False).transpose(1, 2)
            for qi, ki, vi in zip(q_parts, k_parts, v_parts)
        ]
        return self.out_proj(torch.cat(outputs, dim=1).reshape(1, -1, self.dim))


class MiniCPMV46VisionMLP(nn.Module):
    def __init__(self, vision_config: dict[str, Any], dtype: torch.dtype):
        super().__init__()
        hidden = int(vision_config["hidden_size"])
        intermediate = int(vision_config["intermediate_size"])
        self.fc1 = nn.Linear(hidden, intermediate, dtype=dtype)
        self.fc2 = nn.Linear(intermediate, hidden, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class MiniCPMV46VisionEncoderLayer(nn.Module):
    def __init__(self, vision_config: dict[str, Any], dtype: torch.dtype):
        super().__init__()
        hidden = int(vision_config["hidden_size"])
        eps = float(vision_config.get("layer_norm_eps", 1e-6))
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps, dtype=dtype)
        self.self_attn = MiniCPMV46VisionAttention(vision_config, dtype)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps, dtype=dtype)
        self.mlp = MiniCPMV46VisionMLP(vision_config, dtype)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), cu_seqlens)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class MiniCPMV46VisionEncoder(nn.Module):
    def __init__(self, vision_config: dict[str, Any], dtype: torch.dtype):
        super().__init__()
        self.layers = nn.ModuleList(
            [MiniCPMV46VisionEncoderLayer(vision_config, dtype) for _ in range(int(vision_config["num_hidden_layers"]))]
        )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens)
        return hidden_states


class MiniCPMV46ViTWindowAttentionMerger(nn.Module):
    def __init__(self, vision_config: dict[str, Any], dtype: torch.dtype):
        super().__init__()
        self.window_h, self.window_w = tuple(vision_config.get("window_kernel_size", (2, 2)))
        hidden = int(vision_config["hidden_size"])
        eps = float(vision_config.get("layer_norm_eps", 1e-6))
        self.self_attn = MiniCPMV46VisionAttention(vision_config, dtype)
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps, dtype=dtype)
        merged = hidden * self.window_h * self.window_w
        self.pre_norm = nn.LayerNorm(merged, eps=eps, dtype=dtype)
        self.linear_1 = nn.Linear(merged, int(vision_config.get("intermediate_size", merged)) * 4, dtype=dtype)
        self.linear_2 = nn.Linear(int(vision_config.get("intermediate_size", merged)) * 4, hidden, dtype=dtype)

    @torch._dynamo.disable
    def _window_index(self, target_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = []
        cu = [0]
        offset = 0
        for target in target_sizes.detach().cpu().to(dtype=torch.long):
            height, width = int(target[0]), int(target[1])
            if height % self.window_h or width % self.window_w:
                raise ValueError("MiniCPM vision patch grid must be divisible by window_kernel_size")
            grid = torch.arange(height * width).reshape(height, width)
            grid = grid.reshape(height // self.window_h, self.window_h, width // self.window_w, self.window_w)
            grid = grid.permute(0, 2, 1, 3).reshape(-1, self.window_h * self.window_w)
            indices.append(grid.flatten() + offset)
            cu.extend((torch.arange(1, grid.shape[0] + 1) * (self.window_h * self.window_w) + cu[-1]).tolist())
            offset += height * width
        return torch.cat(indices), torch.tensor(cu, dtype=torch.int32)

    @torch._dynamo.disable
    def forward(
        self, hidden_states: torch.Tensor, target_sizes: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        index, window_cu = self._window_index(target_sizes)
        windowed = self.layer_norm1(hidden_states)[:, index.to(hidden_states.device), :]
        windowed = self.self_attn(windowed, window_cu.to(hidden_states.device))
        hidden_states = residual + windowed[:, torch.argsort(index).to(hidden_states.device), :]
        outputs = []
        start = 0
        for target in target_sizes.detach().cpu().to(dtype=torch.long):
            height, width = int(target[0]), int(target[1])
            patch = hidden_states[:, start : start + height * width, :].squeeze(0)
            merged_h, merged_w = height // self.window_h, width // self.window_w
            patch = patch.view(merged_h, self.window_h, merged_w, self.window_w, -1).permute(0, 2, 1, 3, 4)
            value = patch.reshape(merged_h * merged_w, -1)
            residual_patch = patch.reshape(merged_h * merged_w, self.window_h * self.window_w, -1).mean(dim=1)
            value = self.linear_2(F.gelu(self.linear_1(self.pre_norm(value)), approximate="tanh"))
            outputs.append(value + residual_patch)
            start += height * width
        return torch.cat(outputs, dim=0).unsqueeze(0)


class MiniCPMV46VisionModel(nn.Module):
    def __init__(self, vision_config: dict[str, Any], dtype: torch.dtype):
        super().__init__()
        self.config = vision_config
        hidden = int(vision_config["hidden_size"])
        self.embeddings = MiniCPMV46VisionEmbeddings(vision_config, dtype)
        self.encoder = MiniCPMV46VisionEncoder(vision_config, dtype)
        self.post_layernorm = nn.LayerNorm(hidden, eps=float(vision_config.get("layer_norm_eps", 1e-6)), dtype=dtype)
        self.vit_merger = MiniCPMV46ViTWindowAttentionMerger(vision_config, dtype)

    @torch._dynamo.disable
    def forward(
        self, pixel_values: torch.Tensor, target_sizes: torch.Tensor, use_vit_merger: bool = True
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, target_sizes)
        target_sizes = target_sizes.to(device=hidden_states.device, dtype=torch.long).reshape(-1, 2)
        cu_seqlens = F.pad(torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32), (1, 0))
        insert_layer = int(self.config.get("insert_layer_id", 6)) if use_vit_merger else -1
        for layer_idx, layer in enumerate(self.encoder.layers):
            hidden_states = layer(hidden_states, cu_seqlens)
            if layer_idx == insert_layer:
                hidden_states = self.vit_merger(hidden_states, target_sizes, cu_seqlens)
                target_sizes = target_sizes // 2
                cu_seqlens = F.pad(
                    torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32), (1, 0)
                )
        return self.post_layernorm(hidden_states)


class MiniCPMV46DownsampleMLP(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dtype: torch.dtype):
        super().__init__()
        merged = hidden_size * 4
        self.pre_norm = nn.LayerNorm(merged, eps=1e-6, dtype=dtype)
        self.linear_1 = nn.Linear(merged, merged, dtype=dtype)
        self.linear_2 = nn.Linear(merged, output_size, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        return self.linear_2(F.gelu(self.linear_1(self.pre_norm(hidden_states)), approximate="none"))


class MiniCPMV46Merger(nn.Module):
    def __init__(self, vision_config: dict[str, Any], text_hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.merge_h, self.merge_w = tuple(vision_config.get("merge_kernel_size", (2, 2)))
        self.merger_times = int(vision_config.get("merger_times", 1))
        hidden = int(vision_config["hidden_size"])
        mlps = [MiniCPMV46DownsampleMLP(hidden, hidden, dtype) for _ in range(self.merger_times - 1)]
        mlps.append(MiniCPMV46DownsampleMLP(hidden, text_hidden_size, dtype))
        self.mlp = nn.ModuleList(mlps)

    @torch._dynamo.disable
    def forward(self, hidden_states: torch.Tensor, target_sizes: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        start = 0
        for target in target_sizes.detach().cpu().to(dtype=torch.long):
            height, width = int(target[0]), int(target[1])
            count = height * width
            if height % self.merge_h or width % self.merge_w:
                raise ValueError("MiniCPM vision patch grid must be divisible by merge_kernel_size")
            value = hidden_states[0, start : start + count]
            merged_h, merged_w = height // self.merge_h, width // self.merge_w
            value = value.view(merged_h, self.merge_h, merged_w, self.merge_w, -1).permute(0, 2, 1, 3, 4)
            value = self.mlp[0](value.reshape(merged_h * merged_w, -1))
            current_h, current_w = merged_h, merged_w
            for mlp in self.mlp[1:]:
                if current_h % self.merge_h or current_w % self.merge_w:
                    raise ValueError("MiniCPM vision patch grid is not divisible for repeated merger")
                current_h //= self.merge_h
                current_w //= self.merge_w
                value = value.view(current_h, self.merge_h, current_w, self.merge_w, -1).permute(0, 2, 1, 3, 4)
                value = mlp(value.reshape(current_h * current_w, -1))
            outputs.append(value)
            start += count
        return outputs


class MiniCPMFullAttention(nn.Module):
    """Standard softmax attention layer with a per-token output gate.

    QKV plus the output-gate share one ``MergedColumnParallelLinear`` so
    ranks split the four (q, gate, k, v) columns once instead of running
    four separate projections.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        ctx = get_tp_context()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.local_heads = self.num_heads // ctx.world_size
        self.local_kv_heads = self.num_kv_heads // ctx.world_size
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # (q, gate, k, v) live in one fused projection. The gate is sized like
        # q (it gates per Q-head), then sigmoid'd to multiply the attention
        # output before ``o_proj``.
        self.qkv_proj = MergedColumnParallelLinear(config.hidden_size, (q_size, q_size, kv_size, kv_size), bias=False)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.rope = PartialRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.partial_rotary_factor,
            is_neox_style=True,
        )
        self.attn_backend = config.attn_backend
        self.train_backend = build_train_attention_backend(self.attn_backend)
        self.infer_backend: FlashAttnInferBackend | None = None
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        q_size = self.local_heads * self.head_dim
        kv_size = self.local_kv_heads * self.head_dim
        # Single fused projection, split into the four parts; ``gate`` stays
        # flat (no head split) so it can multiply the attention output as a
        # per-channel scalar after the sigmoid.
        qkv = self.qkv_proj(hidden_states)
        batch, seqlen, _ = qkv.shape
        q, gate, k, v = qkv.split((q_size, q_size, kv_size, kv_size), dim=-1)
        q = q.view(batch, seqlen, self.local_heads, self.head_dim)
        gate = gate.view(batch, seqlen, self.local_heads * self.head_dim)
        k = k.view(batch, seqlen, self.local_kv_heads, self.head_dim)
        v = v.view(batch, seqlen, self.local_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, position_ids)
        if infer_meta is not None:
            out = self._forward_infer(q, k, v, infer_meta)
        else:
            out = self.train_backend(q, k, v, train_meta)
        out = out.contiguous().view(batch, seqlen, self.local_heads * self.head_dim)
        # Sigmoid-gated output before the row-parallel projection.
        out = out * _areno_sigmoid_no_compile(gate)
        return self.o_proj(out)

    def _forward_infer(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, infer_meta: InferMeta) -> torch.Tensor:
        if self.k_cache.numel() == 0 or self.v_cache.numel() == 0:
            raise RuntimeError("MiniCPM full attention inference requires KV cache")
        if self.infer_backend is None:
            # Lazy-build the flash-attn inference backend the first time we
            # serve a request.
            self.infer_backend = build_infer_attention_backend(self.attn_backend)
        return self.infer_backend(q, k, v, self.k_cache, self.v_cache, infer_meta)

    def set_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        self.k_cache = k_cache
        self.v_cache = v_cache

    def clear_kv_cache(self) -> None:
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        self.infer_backend = None

    @torch.no_grad()
    def reset_kv_cache(self) -> None:
        return None


class MiniCPMGatedDeltaNet(nn.Module):
    """Gated Delta-Net linear-attention layer.

    Each step does:
        * Two parallel projections — ``in_proj_qkvz`` packs (q, k, v, z) so a
          single matmul produces both the attention triplet and the gate;
          ``in_proj_ba`` packs (b, a) — the b "input gate" and a "decay
          gate" that together drive the recurrent state update.
        * A depthwise causal Conv1d (kernel size 4) over the concatenated
          q/k/v output as a short-term mixer; SiLU is fused into the conv
          kernel.
        * The recurrent state update follows the gated-delta-rule:
            ``g = -exp(A_log) * softplus(a + dt_bias)`` (per-head decay),
            ``beta = sigmoid(b)`` (per-head update gate).
        * RMSNorm + SiLU on z + element-wise gate on the attention output,
          fused via ``areno_rmsnorm_silu_gate``.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        ctx = get_tp_context()
        self.layer_idx = layer_idx
        self.num_key_heads = int(config.linear_num_key_heads or _LINEAR_NUM_HEADS)
        self.num_value_heads = int(config.linear_num_value_heads or self.num_key_heads)
        self.head_k_dim = int(config.linear_key_head_dim or _LINEAR_HEAD_DIM)
        self.head_v_dim = int(config.linear_value_head_dim or self.head_k_dim)
        self.local_key_heads = self.num_key_heads // ctx.world_size
        self.local_value_heads = self.num_value_heads // ctx.world_size
        self.key_dim = self.num_key_heads * self.head_k_dim
        self.value_dim = self.num_value_heads * self.head_v_dim
        self.local_key_dim = self.local_key_heads * self.head_k_dim
        self.local_value_dim = self.local_value_heads * self.head_v_dim
        self.conv_kernel_size = int(config.linear_conv_kernel_dim or _LINEAR_CONV_KERNEL)
        # Fused (q, k, v, z) projection.
        self.in_proj_qkvz = MergedColumnParallelLinear(
            config.hidden_size,
            (self.key_dim, self.key_dim, self.value_dim, self.value_dim),
            bias=False,
        )
        # Fused (b, a) projection — per-head scalars.
        self.in_proj_ba = MergedColumnParallelLinear(
            config.hidden_size, (self.num_value_heads, self.num_value_heads), bias=False
        )
        # Depthwise causal conv1d weight: one filter per channel (groups=channels).
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.local_key_dim * 2 + self.local_value_dim, 1, self.conv_kernel_size)
        )
        mark_tensor_parallel_parameter(self.conv1d_weight, True, sequence_parallel=True)
        # Per-head time-step bias and (log) decay parameter.
        self.dt_bias = nn.Parameter(torch.empty(self.local_value_heads))
        self.A_log = nn.Parameter(torch.empty(self.local_value_heads, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.dt_bias, True, sequence_parallel=True)
        mark_tensor_parallel_parameter(self.A_log, True, sequence_parallel=True)
        self.norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.out_proj = RowParallelLinear(self.value_dim, config.hidden_size, bias=False)
        self.eps = config.rms_norm_eps
        self.scale = self.head_k_dim**-0.5
        # Recurrent state and conv-history caches, sized by ``set_state_cache``.
        self.state_cache = torch.tensor([])
        self.conv_cache = torch.tensor([])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        del position_ids
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        batch, seqlen, _ = qkvz.shape
        # Mix only the (q, k, v) prefix through the causal conv; z bypasses.
        mixed_qkv = self._causal_conv(
            qkvz[..., : self.local_key_dim * 2 + self.local_value_dim], train_meta, infer_meta
        )
        query_key, value = mixed_qkv.split((self.local_key_dim * 2, self.local_value_dim), dim=-1)
        z = qkvz[..., self.local_key_dim * 2 + self.local_value_dim :]
        b_gate, a_gate = ba.split((self.local_value_heads, self.local_value_heads), dim=-1)
        query_key = query_key.view(batch, seqlen, self.local_key_heads * 2, self.head_k_dim)
        query, key = query_key.split((self.local_key_heads, self.local_key_heads), dim=2)
        value = value.view(batch, seqlen, self.local_value_heads, self.head_v_dim)
        z = z.view(batch, seqlen, self.local_value_heads, self.head_v_dim)
        if infer_meta is not None:
            out = self._forward_infer(query, key, value, a_gate, b_gate, infer_meta)
        else:
            # Training path computes the decay and update-gate in eager Python
            # (it fuses cleanly under the recurrent kernel call).
            g = -self.A_log.float().exp().view(1, 1, -1) * F.softplus(
                a_gate.float() + self.dt_bias.float().view(1, 1, -1)
            )
            beta = torch.sigmoid(b_gate)
            out = self._forward_train(query, key, value, g, beta, train_meta)
        out = self._rmsnorm_gate(out, z).reshape(batch, seqlen, self.local_value_dim)
        return self.out_proj(out.to(dtype=hidden_states.dtype))

    def _causal_conv(
        self,
        x: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        # Three modes: inference (uses conv_cache), packed training (per-doc
        # convolutions to respect doc boundaries), or dense single-sequence.
        if infer_meta is not None:
            return self._causal_conv_infer(x, infer_meta)
        if train_meta is not None and train_meta.packed and train_meta.cu_seqlens is not None:
            return self._causal_conv_train_packed(x, train_meta.cu_seqlens)
        _require_fla_gdn()
        log_once("minicpm_gdn_fla_conv", "using FLA causal-conv training kernel")
        out = _fla_causal_conv1d_no_compile(
            x,
            weight=self.conv1d_weight.squeeze(1),
            activation="silu",
        )
        return out

    @torch._dynamo.disable
    def _causal_conv_train_packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        # Packed sequences need per-document convolutions so the causal kernel
        # doesn't leak context across boundaries.
        if x.shape[0] != 1:
            raise ValueError("packed ARENO causal-conv expects flattened packed input with batch size 1")
        log_once("minicpm_gdn_areno_packed_conv", "using ARENO packed causal-conv training kernel")
        return _areno_packed_depthwise_causal_conv1d_silu_no_compile(x, self.conv1d_weight, cu_seqlens)

    def _causal_conv_dense(self, x: torch.Tensor) -> torch.Tensor:
        return _areno_depthwise_causal_conv1d_silu_no_compile(x, self.conv1d_weight)

    def _causal_conv_infer(self, x: torch.Tensor, infer_meta: InferMeta) -> torch.Tensor:
        if infer_meta.block_table is None:
            raise RuntimeError("MiniCPM GDN inference requires block_table")
        if self.conv_cache.numel() == 0:
            raise RuntimeError("MiniCPM GDN inference requires conv state cache")
        # One conv-history slot per request (first column of the block table).
        slots = _recurrent_cache_slots(infer_meta)
        if infer_meta.mode == "decode":
            # Decode: one token per request. Pull the (kernel-1)-history,
            # run the fused decode kernel, then slide the window forward.
            current = x[:, :, :].reshape(-1, x.shape[-1])
            history = self.conv_cache.index_select(0, slots).to(dtype=current.dtype)
            out = _areno_depthwise_causal_conv1d_silu_decode_no_compile(current, history, self.conv1d_weight)
            window = torch.cat((history, current.unsqueeze(-1)), dim=-1)
            # Drop the oldest column to keep the cache at (kernel-1) length.
            self.conv_cache[slots] = window[:, :, 1:].detach().to(dtype=self.conv_cache.dtype)
            return out.view(x.shape)
        if infer_meta.mode == "prefill":
            if infer_meta.cu_seqlens is None:
                raise RuntimeError("MiniCPM GDN prefill requires cu_seqlens")
            return self._causal_conv_infer_prefill(x, infer_meta.cu_seqlens, slots)
        raise ValueError(f"unsupported inference mode: {infer_meta.mode}")

    @torch._dynamo.disable
    def _causal_conv_infer_prefill(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        cu = cu_seqlens.to(device=x.device, dtype=torch.long)
        for idx, slot in enumerate(slots):
            start, end = int(cu[idx].item()), int(cu[idx + 1].item())
            segment = x[:, start:end]
            out[:, start:end] = self._causal_conv_dense(segment)
            # Seed the per-request conv cache with the (kernel-1) trailing
            # tokens so subsequent decode steps see the right history.
            tail = segment.reshape(-1, x.shape[-1])[-(self.conv_kernel_size - 1) :]
            cache = torch.zeros(x.shape[-1], self.conv_kernel_size - 1, device=x.device, dtype=self.conv_cache.dtype)
            cache[:, -tail.shape[0] :] = tail.transpose(0, 1).to(dtype=self.conv_cache.dtype)
            self.conv_cache[slot] = cache
        return out

    def _forward_train(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        train_meta: TrainMeta | None,
    ) -> torch.Tensor:
        cu = None
        if train_meta is not None and train_meta.packed and train_meta.cu_seqlens is not None:
            cu = train_meta.cu_seqlens.to(device=q.device, dtype=torch.long)
        _require_fla_gdn()
        if cu is not None and q.shape[0] != 1:
            raise ValueError("FLA chunk gated-delta expects flattened packed input with batch size 1")
        log_once("minicpm_gdn_fla_chunk", "using FLA chunk gated-delta training kernel")
        out, _ = _fla_chunk_gated_delta_rule_no_compile(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=self.scale,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )
        return out

    def _forward_infer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_meta: InferMeta,
    ) -> torch.Tensor:
        if infer_meta.block_table is None:
            raise RuntimeError("MiniCPM GDN inference requires block_table")
        if self.state_cache.numel() == 0:
            raise RuntimeError("MiniCPM GDN inference requires recurrent state cache")
        slots = _recurrent_cache_slots(infer_meta)
        cu = infer_meta.cu_seqlens
        if infer_meta.mode == "decode":
            # Decode is single-token per request, so collapse batch/seq into
            # one effective batch and run FLA recurrent scan against the
            # selected per-request states.
            decode_shape = q.shape
            q = q.reshape(1, -1, self.local_key_heads, self.head_k_dim)
            k = k.reshape(1, -1, self.local_key_heads, self.head_k_dim)
            v = v.reshape(1, -1, self.local_value_heads, self.head_v_dim)
            a = a.reshape(1, -1, self.local_value_heads)
            b = b.reshape(1, -1, self.local_value_heads)
            _require_fla_gdn()
            g = -self.A_log.float().exp().view(1, 1, -1) * _areno_softplus_no_compile(
                a.float() + self.dt_bias.float().view(1, 1, -1)
            )
            beta = _areno_sigmoid_no_compile(b)
            initial_state = self.state_cache.index_select(0, slots).to(device=q.device)
            out, state = _fla_fused_recurrent_gated_delta_rule_no_compile(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                scale=self.scale,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=torch.arange(slots.numel() + 1, device=q.device, dtype=torch.long),
            )
            self.state_cache[slots] = state.detach().to(dtype=self.state_cache.dtype)
            return out.reshape(decode_shape[:-2] + (self.local_value_heads, self.head_v_dim))
        if cu is None:
            raise RuntimeError("MiniCPM GDN inference requires cu_seqlens")
        _require_fla_gdn()
        log_once("minicpm_gdn_fla_infer", "using FLA recurrent gated-delta inference kernel")
        # Prefill: compute decay/beta in Python (the kernel takes them
        # pre-computed), seed the recurrent kernel with the saved state, and
        # write the final state back into the cache.
        g = -self.A_log.float().exp().view(1, 1, -1) * _areno_softplus_no_compile(
            a.float() + self.dt_bias.float().view(1, 1, -1)
        )
        beta = _areno_sigmoid_no_compile(b)
        initial_state = self.state_cache.index_select(0, slots).to(device=q.device)
        out, state = _fla_fused_recurrent_gated_delta_rule_no_compile(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=self.scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu.to(device=q.device, dtype=torch.long),
            use_qk_l2norm_in_kernel=True,
        )
        # Persist the final state for the subsequent decode steps.
        self.state_cache[slots] = state.detach().to(dtype=self.state_cache.dtype)
        return out

    def _rmsnorm_gate(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Fused RMSNorm + SiLU(gate) * x in a single kernel call.
        return _areno_rmsnorm_silu_gate_no_compile(x, gate, self.norm_weight, self.eps)

    def set_state_cache(self, state_cache: torch.Tensor, conv_cache: torch.Tensor) -> None:
        self.state_cache = state_cache
        self.conv_cache = conv_cache

    def clear_kv_cache(self) -> None:
        self.state_cache = torch.tensor([])
        self.conv_cache = torch.tensor([])

    @torch.no_grad()
    def reset_kv_cache(self) -> None:
        # Zero in-place so the buffer can be reused across batches.
        if self.state_cache.numel() > 0:
            self.state_cache.zero_()
        if self.conv_cache.numel() > 0:
            self.conv_cache.zero_()


class MiniCPMDecoderLayer(nn.Module):
    """One MiniCPM-V-4.6 decoder block: pre-norm attention (full or GDN) +
    pre-norm GatedMLP."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        # Layer type comes from the HF config's ``layer_types`` list.
        layer_type = (config.layer_types or ())[layer_idx]
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = (
            MiniCPMFullAttention(config, layer_idx)
            if layer_type == "full_attention"
            else MiniCPMGatedDeltaNet(config, layer_idx)
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None,
        infer_meta: InferMeta | None,
    ) -> torch.Tensor:
        # Standard pre-norm residual: norm -> sublayer -> add.
        residual = hidden_states
        hidden_states = residual + self.attention(
            self.input_layernorm(hidden_states), position_ids, train_meta, infer_meta
        )
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class MiniCPMV46ForCausalLM(nn.Module):
    """MiniCPM-V-4.6 language model with an AReno-owned vision path."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([MiniCPMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)
        if config.vision_config is not None:
            vision_config = dict(config.vision_config)
            self.vision_tower = MiniCPMV46VisionModel(vision_config, config.dtype)
            self.merger = MiniCPMV46Merger(vision_config, config.hidden_size, config.dtype)
            for parameter in (*self.vision_tower.parameters(), *self.merger.parameters()):
                mark_tensor_parallel_parameter(parameter, False, sequence_parallel=False, tp_grad_allreduce=True)
        else:
            self.vision_tower = None
            self.merger = None

    @property
    def language_model(self) -> MiniCPMV46ForCausalLM:
        """Expose the direct AReno text trunk under the HF-compatible name."""

        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
        features: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> CausalLMOutput:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.embed_tokens(input_ids)
        if features is not None:
            hidden_states = self._apply_multimodal_features(
                hidden_states,
                input_ids,
                self._project_pixel_values(features, input_ids.device, int(input_ids.shape[0])),
            )
        use_sequence_parallel = bool(train_meta is not None and train_meta.sequence_parallel)
        if use_sequence_parallel:
            # Split sequence dim across TP ranks before entering the SP region.
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
            hidden_states = self.norm(hidden_states)
            logits_shard = self.lm_head(hidden_states)
        return CausalLMOutput(logits_shard=logits_shard, hidden_states=hidden_states)

    @torch._dynamo.disable
    def _apply_multimodal_features(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        features: dict[str, Any] | list[dict[str, Any] | None] | None,
    ) -> torch.Tensor:
        if features is None:
            return hidden_states
        rows = _features_by_row(features, int(input_ids.shape[0]))
        if not any(row is not None for row in rows):
            return hidden_states
        hidden_states = hidden_states.clone()
        for row_idx, row in enumerate(rows):
            if row is None:
                continue
            image_embeds = _image_embeds_for_row(row, hidden_states.device, hidden_states.dtype)
            if image_embeds is None:
                continue
            if image_embeds.ndim != 2 or int(image_embeds.shape[-1]) != self.config.hidden_size:
                raise ValueError(f"MiniCPM image_embeds must have shape (num_image_tokens, {self.config.hidden_size})")
            mask = _image_token_mask(row, input_ids[row_idx])
            if int(mask.sum().item()) != int(image_embeds.shape[0]):
                raise ValueError(
                    "MiniCPM image token count does not match image_embeds: "
                    f"tokens={int(mask.sum().item())} embeds={int(image_embeds.shape[0])}"
                )
            hidden_states[row_idx, mask] = image_embeds
        return hidden_states

    @torch._dynamo.disable
    def _project_pixel_values(
        self,
        features: dict[str, Any] | list[dict[str, Any] | None] | None,
        device: torch.device,
        batch: int,
    ) -> dict[str, Any] | list[dict[str, Any] | None] | None:
        if features is None or self.vision_tower is None or self.merger is None:
            return features
        rows = _features_by_row(features, batch)
        projected: list[dict[str, Any] | None] = []
        changed = False
        for row in rows:
            if row is None:
                projected.append(None)
                continue
            row_features = dict(row)
            if _image_embeds_for_row(row_features, device, self.config.dtype) is None:
                image_embeds = self._project_image_feature_rows(row_features, device)
                if image_embeds is not None:
                    row_features["image_embeds"] = image_embeds
                    changed = True
            if row_features.get("image_token_id") is None and self.config.image_token_id is not None:
                row_features["image_token_id"] = self.config.image_token_id
                changed = True
            projected.append(row_features)
        if not changed:
            return features
        return projected[0] if batch == 1 else projected

    @torch._dynamo.disable
    def _project_image_feature_rows(self, features: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        rows = features.get("image_feature_rows")
        if rows is not None:
            embeds = []
            for row in rows:
                if row is None:
                    continue
                value = _image_embeds_for_row(dict(row), device, self.config.dtype)
                if value is None:
                    value = self._project_pixel_feature(dict(row), device)
                if value is not None:
                    embeds.append(value)
            return torch.cat(embeds, dim=0) if embeds else None
        return self._project_pixel_feature(features, device)

    @torch._dynamo.disable
    def _project_pixel_feature(self, features: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        if features.get("pixel_values") is None or features.get("target_sizes") is None:
            return None
        pixel_values = _feature_tensor(features, "pixel_values", device, self.config.dtype)
        target_sizes = _feature_tensor(features, "target_sizes", device, torch.long)
        if pixel_values is None or target_sizes is None:
            return None
        downsample_mode = str(features.get("downsample_mode", self.config.vision_config.get("downsample_mode", "16x")))
        insert_layer_id = int(self.config.vision_config.get("insert_layer_id", 6))
        use_vit_merger = downsample_mode != "4x" and 0 <= insert_layer_id < len(self.vision_tower.encoder.layers)
        hidden = self.vision_tower(pixel_values, target_sizes, use_vit_merger=use_vit_merger)
        if use_vit_merger:
            target_sizes = target_sizes // 2
        per_image = self.merger(hidden, target_sizes)
        image_embeds = (
            torch.cat(per_image, dim=0) if per_image else torch.empty((0, self.config.hidden_size), device=device)
        )
        offset = int(features.get("image_token_offset", 0) or 0)
        count = features.get("image_token_count")
        if count is not None:
            image_embeds = image_embeds[offset : offset + int(count)]
        elif offset:
            image_embeds = image_embeds[offset:]
        return image_embeds

    def set_kv_caches(
        self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]], *, num_slots: int | None = None
    ) -> None:
        """Bind per-full-attention KV caches and allocate GDN state caches.

        ``kv_caches`` only contains entries for ``MiniCPMFullAttention``
        layers (in order). For each GDN layer we synthesise a zeroed
        recurrent-state tensor and a zeroed conv-history tensor sized from
        the first KV cache's slot count.
        """
        idx = 0
        device = next(self.parameters()).device
        num_slots = int(num_slots) if num_slots is not None else (int(kv_caches[0][0].shape[0]) if kv_caches else 1)
        for layer in self.layers:
            attn = layer.attention
            if isinstance(attn, MiniCPMFullAttention):
                attn.set_kv_cache(*kv_caches[idx])
                idx += 1
            else:
                # GDN needs both a recurrent state ([value_heads, key_dim, value_dim])
                # and (kernel-1) columns of conv history per slot.
                state = torch.zeros(
                    num_slots,
                    attn.local_value_heads,
                    attn.head_k_dim,
                    attn.head_v_dim,
                    device=device,
                    dtype=torch.float32,
                )
                conv = torch.zeros(
                    num_slots,
                    attn.local_key_dim * 2 + attn.local_value_dim,
                    attn.conv_kernel_size - 1,
                    device=device,
                    dtype=self.config.dtype,
                )
                attn.set_state_cache(state, conv)

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        # No fused inference weights to pre-stage (no MoE/expert tiles here).
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
        # No MoE router, nothing to balance.
        del tp_group, dp_group
        return None

    def allocate_kv_caches(
        self, num_blocks: int, block_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Allocate paged KV caches for the full-attention layers only."""
        caches = []
        for layer in self.layers:
            attn = layer.attention
            if not isinstance(attn, MiniCPMFullAttention):
                continue
            k_cache = torch.empty(
                num_blocks, block_size, attn.local_kv_heads, attn.head_dim, device=device, dtype=self.config.dtype
            )
            v_cache = torch.empty(
                num_blocks, block_size, attn.local_kv_heads, attn.head_dim, device=device, dtype=self.config.dtype
            )
            caches.append((k_cache, v_cache))
        return caches

    def clear_kv_caches(self) -> None:
        for layer in self.layers:
            layer.attention.clear_kv_cache()

    @torch.no_grad()
    def reset_kv_caches(self) -> None:
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    @torch.no_grad()
    def reset_recurrent_cache_slots(self, slots: torch.Tensor) -> None:
        slots = slots.to(device=next(self.parameters()).device, dtype=torch.long)
        for layer in self.layers:
            attn = layer.attention
            if isinstance(attn, MiniCPMGatedDeltaNet):
                attn.state_cache.index_fill_(0, slots, 0)
                attn.conv_cache.index_fill_(0, slots, 0)

    @torch.no_grad()
    def offload_kv_caches(self) -> None:
        """Move all per-layer caches to CPU (used when swapping to training)."""
        for layer in self.layers:
            attn = layer.attention
            if isinstance(attn, MiniCPMFullAttention):
                if attn.k_cache.numel() > 0:
                    attn.k_cache = attn.k_cache.cpu()
                if attn.v_cache.numel() > 0:
                    attn.v_cache = attn.v_cache.cpu()
                # Drop the lazily-built backend so it gets rebuilt for the new device.
                attn.infer_backend = None
            elif isinstance(attn, MiniCPMGatedDeltaNet):
                if attn.state_cache.numel() > 0:
                    attn.state_cache = attn.state_cache.cpu()
                if attn.conv_cache.numel() > 0:
                    attn.conv_cache = attn.conv_cache.cpu()

    @torch.no_grad()
    def onload_kv_caches(self, device: torch.device) -> bool:
        """Move any previously-offloaded caches back onto ``device``.

        Returns True iff at least one cache was found (regardless of whether
        it actually had to be moved), so callers can detect a no-op onload.
        """
        found = False
        for layer in self.layers:
            attn = layer.attention
            for name in ("k_cache", "v_cache", "state_cache", "conv_cache"):
                cache = getattr(attn, name, None)
                if isinstance(cache, torch.Tensor) and cache.numel() > 0:
                    found = True
                    if cache.device != device:
                        setattr(attn, name, cache.to(device=device))
        return found


class MiniCPMV46Adapter(ModelAdapter):
    """Model adapter binding HF MiniCPM-V-4.6 checkpoints to the areno runtime."""

    name = "minicpmv46"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        return str(hf_config.get("model_type", "")).lower() == "minicpmv4_6"

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        """Translate the nested text and vision configs into AReno fields."""
        text = hf_config["text_config"]
        dtype = _parse_dtype(hf_config.get("torch_dtype") or hf_config.get("dtype") or text.get("dtype"))
        rope = text.get("rope_parameters") or {}
        vision = dict(hf_config.get("vision_config") or {})
        vision["insert_layer_id"] = int(hf_config.get("insert_layer_id", vision.get("insert_layer_id", 6)))
        vision["merge_kernel_size"] = tuple(hf_config.get("merge_kernel_size", (2, 2)))
        vision["merger_times"] = int(hf_config.get("merger_times", 1))
        vision["downsample_mode"] = str(hf_config.get("downsample_mode", "16x"))
        return ModelConfig(
            model_type=self.name,
            checkpoint_prefix="model.language_model",
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", 10_000.0)),
            max_position_embeddings=int(text.get("max_position_embeddings", 262144)),
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", True)),
            qkv_bias=False,
            qk_norm=True,
            dtype=dtype,
            hidden_act=str(text.get("hidden_act", "silu")),
            layer_types=tuple(text["layer_types"]),
            partial_rotary_factor=float(text.get("partial_rotary_factor", rope.get("partial_rotary_factor", 0.25))),
            linear_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(text.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(text.get("linear_value_head_dim", 128)),
            linear_num_key_heads=int(text.get("linear_num_key_heads", 16)),
            linear_num_value_heads=int(text.get("linear_num_value_heads", 32)),
            sequence_parallel=bool(text.get("sequence_parallel", True)),
            vision_config=vision,
            image_token_id=(int(hf_config["image_token_id"]) if hf_config.get("image_token_id") is not None else None),
        )

    def build(self, config: ModelConfig) -> nn.Module:
        if config.layer_types is None:
            raise ValueError("MiniCPM-V-4.6 requires layer_types in config")
        return MiniCPMV46ForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, MiniCPMV46ForCausalLM):
            raise TypeError(f"MiniCPMV46Adapter cannot load weights into {type(model)!r}")
        # Import lazily so the model file can be imported without the
        # checkpoint module (and vice versa, breaking a circular import).
        from areno.models.minicpmv46.checkpoint import load_minicpmv46_weights

        load_minicpmv46_weights(model, model_path)

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, MiniCPMV46ForCausalLM):
            raise TypeError(f"MiniCPMV46Adapter cannot save weights from {type(model)!r}")
        from areno.models.minicpmv46.checkpoint import save_minicpmv46_weights

        return save_minicpmv46_weights(model, output_path, source_path)

    def build_policy_plan(self, model: nn.Module):
        from areno.models.minicpmv46.checkpoint import build_minicpmv46_policy_plan

        return build_minicpmv46_policy_plan(model)


# Match the Transformers public class name while retaining AReno's existing
# causal-LM adapter type and checkpoint dispatch.
MiniCPMV46ForConditionalGeneration = MiniCPMV46ForCausalLM
