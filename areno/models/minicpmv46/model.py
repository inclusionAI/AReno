"""MiniCPM-V-4.6 language-backbone causal-LM adapter.

Targets the OpenBMB MiniCPM-V-4.6 multimodal checkpoints
(``model_type == "minicpmv4_6"``). The HF release ships a vision encoder and
a multimodal projector on top of a text decoder; only the text decoder is
realised here — image embeddings are produced upstream (by the caller's
preprocessing pipeline) and arrive as ordinary token embeddings via
``input_ids`` already containing image-placeholder ids. The adapter therefore
simply wires the language model, leaving the vision tower and the
hidden->LM projector to the data pipeline.

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

from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend, build_infer_attention_backend
from areno.engine.layers.attention_backend.train import build_train_attention_backend
from areno.engine.layers.linear import MergedColumnParallelLinear, RowParallelLinear, mark_tensor_parallel_parameter
from areno.engine.layers.mlp import GatedMLP
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.rotary import PartialRotaryEmbedding
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.accel.ops import log_once
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


# Gated Delta-Net topology constants (fixed by the MiniCPM-V-4.6 architecture,
# not exposed in the HF config). 16 heads of dim 128 -> 2048-d key/value path.
_LINEAR_NUM_HEADS = 16
_LINEAR_HEAD_DIM = 128
# Short-term Conv1d kernel size: each step sees the current token plus
# (kernel-1) immediate predecessors.
_LINEAR_CONV_KERNEL = 4


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
        self.train_backend = build_train_attention_backend()
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
        batch, seqlen, _ = hidden_states.shape
        q_size = self.local_heads * self.head_dim
        kv_size = self.local_kv_heads * self.head_dim
        # Single fused projection, split into the four parts; ``gate`` stays
        # flat (no head split) so it can multiply the attention output as a
        # per-channel scalar after the sigmoid.
        q, gate, k, v = self.qkv_proj(hidden_states).split((q_size, q_size, kv_size, kv_size), dim=-1)
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
            self.infer_backend = build_infer_attention_backend()
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
        self.num_heads = _LINEAR_NUM_HEADS
        self.head_dim = _LINEAR_HEAD_DIM
        self.local_heads = self.num_heads // ctx.world_size
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_heads * self.head_dim
        self.local_key_dim = self.key_dim // ctx.world_size
        self.local_value_dim = self.value_dim // ctx.world_size
        self.conv_kernel_size = _LINEAR_CONV_KERNEL
        # Fused (q, k, v, z) projection.
        self.in_proj_qkvz = MergedColumnParallelLinear(
            config.hidden_size,
            (self.key_dim, self.key_dim, self.value_dim, self.value_dim),
            bias=False,
        )
        # Fused (b, a) projection — per-head scalars.
        self.in_proj_ba = MergedColumnParallelLinear(config.hidden_size, (self.num_heads, self.num_heads), bias=False)
        # Depthwise causal conv1d weight: one filter per channel (groups=channels).
        self.conv1d_weight = nn.Parameter(torch.empty(self.local_key_dim * 2 + self.local_value_dim, 1, self.conv_kernel_size))
        mark_tensor_parallel_parameter(self.conv1d_weight, True, sequence_parallel=True)
        # Per-head time-step bias and (log) decay parameter.
        self.dt_bias = nn.Parameter(torch.empty(self.local_heads))
        self.A_log = nn.Parameter(torch.empty(self.local_heads, dtype=torch.float32))
        mark_tensor_parallel_parameter(self.dt_bias, True, sequence_parallel=True)
        mark_tensor_parallel_parameter(self.A_log, True, sequence_parallel=True)
        self.norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.out_proj = RowParallelLinear(self.value_dim, config.hidden_size, bias=False)
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
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
        batch, seqlen, _ = hidden_states.shape
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        # Mix only the (q, k, v) prefix through the causal conv; z bypasses.
        mixed_qkv = self._causal_conv(qkvz[..., : self.local_key_dim * 2 + self.local_value_dim], train_meta, infer_meta)
        query_key, value = mixed_qkv.split((self.local_key_dim * 2, self.local_value_dim), dim=-1)
        z = qkvz[..., self.local_key_dim * 2 + self.local_value_dim :]
        b_gate, a_gate = ba.split((self.local_heads, self.local_heads), dim=-1)
        query_key = query_key.view(batch, seqlen, self.local_heads * 2, self.head_dim)
        query, key = query_key.split((self.local_heads, self.local_heads), dim=2)
        value = value.view(batch, seqlen, self.local_heads, self.head_dim)
        z = z.view(batch, seqlen, self.local_heads, self.head_dim)
        if infer_meta is not None:
            out = self._forward_infer(query, key, value, a_gate, b_gate, infer_meta)
        else:
            # Training path computes the decay and update-gate in eager Python
            # (it fuses cleanly under the recurrent kernel call).
            g = -self.A_log.float().exp().view(1, 1, -1) * F.softplus(a_gate.float() + self.dt_bias.float().view(1, 1, -1))
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
        slots = infer_meta.block_table[:, 0].long()
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
    def _causal_conv_infer_prefill(self, x: torch.Tensor, cu_seqlens: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
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
        slots = infer_meta.block_table[:, 0].long()
        cu = infer_meta.cu_seqlens
        if infer_meta.mode == "decode":
            # Decode is single-token per request, so collapse batch/seq into
            # one effective batch and run FLA recurrent scan against the
            # selected per-request states.
            decode_shape = q.shape
            q = q.reshape(1, -1, self.local_heads, self.head_dim)
            k = k.reshape(1, -1, self.local_heads, self.head_dim)
            v = v.reshape(1, -1, self.local_heads, self.head_dim)
            a = a.reshape(1, -1, self.local_heads)
            b = b.reshape(1, -1, self.local_heads)
            _require_fla_gdn()
            g = -self.A_log.float().exp().view(1, 1, -1) * _areno_softplus_no_compile(a.float() + self.dt_bias.float().view(1, 1, -1))
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
            return out.reshape(decode_shape)
        if cu is None:
            raise RuntimeError("MiniCPM GDN inference requires cu_seqlens")
        _require_fla_gdn()
        log_once("minicpm_gdn_fla_infer", "using FLA recurrent gated-delta inference kernel")
        # Prefill: compute decay/beta in Python (the kernel takes them
        # pre-computed), seed the recurrent kernel with the saved state, and
        # write the final state back into the cache.
        g = -self.A_log.float().exp().view(1, 1, -1) * _areno_softplus_no_compile(a.float() + self.dt_bias.float().view(1, 1, -1))
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
        self.attention = MiniCPMFullAttention(config, layer_idx) if layer_type == "full_attention" else MiniCPMGatedDeltaNet(config, layer_idx)
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
        hidden_states = residual + self.attention(self.input_layernorm(hidden_states), position_ids, train_meta, infer_meta)
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class MiniCPMV46ForCausalLM(nn.Module):
    """Top-level MiniCPM-V-4.6 text-only causal LM.

    The vision encoder and multimodal projector live outside this module;
    callers preprocess image regions into the input id stream (using the
    model's image-token placeholders) and the standard
    ``VocabParallelEmbedding`` handles them like normal tokens.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([MiniCPMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> CausalLMOutput:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.embed_tokens(input_ids)
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

    def set_kv_caches(self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Bind per-full-attention KV caches and allocate GDN state caches.

        ``kv_caches`` only contains entries for ``MiniCPMFullAttention``
        layers (in order). For each GDN layer we synthesise a zeroed
        recurrent-state tensor and a zeroed conv-history tensor sized from
        the first KV cache's slot count.
        """
        idx = 0
        device = next(self.parameters()).device
        num_slots = kv_caches[0][0].shape[0] if kv_caches else 1
        for layer in self.layers:
            attn = layer.attention
            if isinstance(attn, MiniCPMFullAttention):
                attn.set_kv_cache(*kv_caches[idx])
                idx += 1
            else:
                # GDN needs both a recurrent state ([heads, head_dim, head_dim])
                # and (kernel-1) columns of conv history per slot.
                state = torch.zeros(num_slots, attn.local_heads, attn.head_dim, attn.head_dim, device=device, dtype=torch.float32)
                conv = torch.zeros(num_slots, attn.local_key_dim * 2 + attn.local_value_dim, attn.conv_kernel_size - 1, device=device, dtype=self.config.dtype)
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

    def allocate_kv_caches(self, num_blocks: int, block_size: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Allocate paged KV caches for the full-attention layers only."""
        caches = []
        for layer in self.layers:
            attn = layer.attention
            if not isinstance(attn, MiniCPMFullAttention):
                continue
            k_cache = torch.empty(num_blocks, block_size, attn.local_kv_heads, attn.head_dim, device=device, dtype=self.config.dtype)
            v_cache = torch.empty(num_blocks, block_size, attn.local_kv_heads, attn.head_dim, device=device, dtype=self.config.dtype)
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
        """Pull the text-trunk subset of the HF multimodal config.

        MiniCPM-V wraps the language backbone settings under ``text_config``
        and exposes a separate ``rope_parameters`` dict; the vision config is
        ignored here because the vision tower is handled externally.
        ``checkpoint_prefix`` tells the loader to look under
        ``model.language_model.*`` rather than the usual ``model.*``.
        """
        text = hf_config["text_config"]
        dtype = _parse_dtype(hf_config.get("torch_dtype") or hf_config.get("dtype") or text.get("dtype"))
        rope = text.get("rope_parameters") or {}
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
            rope_theta=float(rope.get("rope_theta", 10_000_000.0)),
            max_position_embeddings=int(text.get("max_position_embeddings", 262144)),
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", True)),
            qkv_bias=False,
            qk_norm=True,
            dtype=dtype,
            hidden_act=str(text.get("hidden_act", "silu")),
            layer_types=tuple(text["layer_types"]),
            partial_rotary_factor=float(text.get("partial_rotary_factor", rope.get("partial_rotary_factor", 0.25))),
            sequence_parallel=bool(text.get("sequence_parallel", True)),
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

