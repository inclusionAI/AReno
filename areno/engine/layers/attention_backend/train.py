"""Training-time FlashAttention backend.

Supports two activation layouts:

- padded (default): tensors are ``(batch, seqlen, heads, head_dim)`` and
  use `flash_attn_func` directly.
- varlen packed: when `TrainMeta.cu_seqlens` is supplied the tensors are
  flattened to ``(total_tokens, heads, head_dim)`` and routed through
  `flash_attn_varlen_func` so packed batches without padding can be
  trained efficiently.

The ``native`` backend keeps rollout prefill/decode on native kernels while
training uses PyTorch's SDPA math backend to avoid the slow native backward
diagnostic kernel.
Unsupported flash-attn shapes fail with an actionable message instead of
silently falling back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from areno.engine.layers.attention_backend.common import (
    AttnBackend,
    build_attention_call,
    expand_kv_heads,
    require_flash_attention_supported,
    use_native_attention,
)
from areno.engine.runtime.metadata import TrainMeta

_SDPA_MATH_QUERY_CHUNK = 512


class TrainAttentionBackend(nn.Module, ABC):
    """Abstract training attention backend."""

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: TrainMeta | None,
        window_size: tuple[int, int] | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class FlashAttnTrainAttentionBackend(TrainAttentionBackend):
    """FlashAttention backend shared by padded and varlen packed training."""

    def __init__(self, attn_backend: AttnBackend = "flash"):
        super().__init__()
        self.attn_backend = attn_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: TrainMeta | None,
        window_size: tuple[int, int] | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        call = build_attention_call(q, k, v, window_size, softmax_scale)
        if use_native_attention(self.attn_backend):
            out = _native_train(call.q, call.k, call.v, meta, call.window_size, call.softmax_scale)
            return call.trim_value_dim(out)
        require_flash_attention_supported(call, mode="training attention")
        if meta is not None and meta.cu_seqlens is not None:
            # Varlen packed path: tensors must be flattened to (T, H, D) so
            # cu_seqlens can carve out the per-sequence boundaries.
            if meta.max_seqlen is None:
                raise ValueError("TrainMeta.max_seqlen is required with cu_seqlens")
            batch, seqlen = q.shape[:2]
            del batch, seqlen
            q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
            k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
            v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
            # Rebuild the AttentionCall on the flat layout so V padding is
            # applied to the tensor we actually hand to the kernel.
            flat_call = build_attention_call(q_flat, k_flat, v_flat, window_size, softmax_scale)
            out = _flash_attn_varlen_train_no_compile(
                flat_call.q,
                flat_call.k,
                flat_call.v,
                cu_seqlens_q=meta.cu_seqlens,
                cu_seqlens_k=meta.cu_seqlens,
                max_seqlen_q=meta.max_seqlen,
                max_seqlen_k=meta.max_seqlen,
                causal=True,
                window_size=flat_call.window_size,
                softmax_scale=flat_call.softmax_scale,
            )
            out = flat_call.trim_value_dim(out)
            # Restore the original (B, S, H, D) layout for downstream code.
            return out.view(q.shape[0], q.shape[1], q.shape[2], flat_call.value_dim)

        # Padded path: directly call flash-attn over the 4D batch tensor.
        out = _flash_attn_train_no_compile(
            call.q,
            call.k,
            call.v,
            causal=True,
            window_size=call.window_size,
            softmax_scale=call.softmax_scale,
        )
        return call.trim_value_dim(out)


def build_train_attention_backend(attn_backend: AttnBackend = "flash") -> TrainAttentionBackend:
    """Build the default FlashAttention training backend."""

    return FlashAttnTrainAttentionBackend(attn_backend=attn_backend)


@torch._dynamo.disable
def _flash_attn_train_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper for the dense flash-attn training kernel."""

    from flash_attn import flash_attn_func

    return flash_attn_func(*args, **kwargs)


@torch._dynamo.disable
def _flash_attn_varlen_train_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper for the packed varlen flash-attn training kernel."""

    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(*args, **kwargs)


@torch._dynamo.disable
def _native_train(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: TrainMeta | None,
    window_size: tuple[int, int],
    softmax_scale: float | None,
) -> torch.Tensor:
    """Training path for attn_backend=native using PyTorch SDPA math."""

    if meta is not None and meta.sequence_parallel:
        raise RuntimeError(
            "native attention backend training does not support sequence parallelism with SDPA math yet. "
            "Disable sequence parallelism for logprob diagnostics or use --attn-backend flash."
        )
    if meta is not None and meta.cu_seqlens is not None:
        cu = meta.cu_seqlens.detach().cpu().tolist()
        q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
        k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
        v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
        outs = [
            _sdpa_math_sequence(q_flat[start:end], k_flat[start:end], v_flat[start:end], window_size, softmax_scale)
            for start, end in zip(cu[:-1], cu[1:], strict=True)
            if end > start
        ]
        return torch.cat(outs, dim=0).view(q.shape)
    k = expand_kv_heads(k, q.shape[-2])
    v = expand_kv_heads(v, q.shape[-2])
    return _sdpa_math_padded(q, k, v, window_size, softmax_scale)


@contextmanager
def _sdpa_math_kernel() -> Iterator[None]:
    with sdpa_kernel(SDPBackend.MATH):
        yield


def _sdpa_math_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int],
    softmax_scale: float | None,
) -> torch.Tensor:
    q_seq = q.transpose(1, 2)
    k_seq = k.transpose(1, 2)
    v_seq = v.transpose(1, 2)
    if q.shape[1] > _SDPA_MATH_QUERY_CHUNK:
        return _sdpa_math_chunked(q_seq, k_seq, v_seq, window_size, softmax_scale).transpose(1, 2)
    with _sdpa_math_kernel():
        if window_size == (-1, -1):
            out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True, scale=softmax_scale)
        else:
            out = F.scaled_dot_product_attention(
                q_seq,
                k_seq,
                v_seq,
                attn_mask=_causal_window_mask(q.shape[1], k.shape[1], window_size, q.device, 0),
                scale=softmax_scale,
            )
    return out.transpose(1, 2)


def _sdpa_math_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int],
    softmax_scale: float | None,
) -> torch.Tensor:
    k = expand_kv_heads(k, q.shape[-2])
    v = expand_kv_heads(v, q.shape[-2])
    q_seq = q.transpose(0, 1).unsqueeze(0)
    k_seq = k.transpose(0, 1).unsqueeze(0)
    v_seq = v.transpose(0, 1).unsqueeze(0)
    if q.shape[0] > _SDPA_MATH_QUERY_CHUNK:
        return _sdpa_math_chunked(q_seq, k_seq, v_seq, window_size, softmax_scale).squeeze(0).transpose(0, 1)
    with _sdpa_math_kernel():
        if window_size == (-1, -1):
            out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True, scale=softmax_scale)
        else:
            out = F.scaled_dot_product_attention(
                q_seq,
                k_seq,
                v_seq,
                attn_mask=_causal_window_mask(q.shape[0], k.shape[0], window_size, q.device, 0),
                scale=softmax_scale,
            )
    return out.squeeze(0).transpose(0, 1)


def _sdpa_math_chunked(
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    window_size: tuple[int, int],
    softmax_scale: float | None,
) -> torch.Tensor:
    chunks = []
    seqlen = q_seq.shape[-2]
    with _sdpa_math_kernel():
        for start in range(0, seqlen, _SDPA_MATH_QUERY_CHUNK):
            end = min(start + _SDPA_MATH_QUERY_CHUNK, seqlen)
            chunks.append(
                F.scaled_dot_product_attention(
                    q_seq[:, :, start:end],
                    k_seq,
                    v_seq,
                    attn_mask=_causal_window_mask(end - start, k_seq.shape[-2], window_size, q_seq.device, start),
                    scale=softmax_scale,
                )
            )
    return torch.cat(chunks, dim=-2)


def _causal_window_mask(
    q_len: int,
    k_len: int,
    window_size: tuple[int, int],
    device: torch.device,
    query_start: int,
) -> torch.Tensor:
    if window_size != (-1, -1) and window_size[1] != 0:
        raise ValueError("SDPA math training only supports causal right window 0")
    rows = torch.arange(query_start, query_start + q_len, device=device).view(q_len, 1)
    cols = torch.arange(k_len, device=device).view(1, k_len)
    mask = cols <= rows
    if window_size[0] >= 0:
        mask = mask & (cols >= rows - int(window_size[0]))
    return mask.view(1, 1, q_len, k_len)
