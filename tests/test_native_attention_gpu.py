from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from areno.accel.attention import areno_paged_causal_attention_decode, areno_varlen_causal_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native attention equivalence tests require CUDA")


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-5, 2e-4
    return 2e-2, 5e-2


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    atol, rtol = _tolerance(actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _attention_mask(
    query_start: int, query_len: int, key_len: int, window_left: int | None, device: torch.device
) -> torch.Tensor:
    query_positions = torch.arange(query_start, query_start + query_len, device=device).view(query_len, 1)
    key_positions = torch.arange(key_len, device=device).view(1, key_len)
    mask = key_positions <= query_positions
    if window_left is not None:
        mask = mask & (key_positions >= query_positions - window_left)
    return mask.view(1, 1, query_len, key_len)


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_start: int,
    window_left: int | None,
    softmax_scale: float,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=_attention_mask(query_start, q.shape[-2], k.shape[-2], window_left, q.device),
            scale=softmax_scale,
        )


def _expand_kv_heads(x: torch.Tensor, q_heads: int) -> torch.Tensor:
    kv_heads = x.shape[-2]
    repeat = q_heads // kv_heads
    return (
        x.unsqueeze(-2)
        .expand(*x.shape[:-2], kv_heads, repeat, x.shape[-1])
        .reshape(*x.shape[:-2], q_heads, x.shape[-1])
    )


def _varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    window_left: int | None,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    boundaries = cu_seqlens.cpu().tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
        k_seq = _expand_kv_heads(k[start:end], q.shape[1]).transpose(0, 1).unsqueeze(0)
        v_seq = _expand_kv_heads(v[start:end], q.shape[1]).transpose(0, 1).unsqueeze(0)
        outputs.append(
            _sdpa_reference(
                q_seq,
                k_seq,
                v_seq,
                query_start=0,
                window_left=window_left,
                softmax_scale=softmax_scale,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    return torch.cat(outputs, dim=0)


def _paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    window_left: int | None,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    q_heads = q.shape[1]
    block_size = k_cache.shape[1]
    for batch_idx in range(q.shape[0]):
        length = int(cache_seqlens[batch_idx].item()) + 1
        positions = torch.arange(length, device=q.device)
        block_columns = torch.div(positions, block_size, rounding_mode="floor")
        block_offsets = positions % block_size
        block_ids = block_table[batch_idx, block_columns]
        k_seq = _expand_kv_heads(k_cache[block_ids, block_offsets], q_heads).transpose(0, 1).unsqueeze(0)
        v_seq = _expand_kv_heads(v_cache[block_ids, block_offsets], q_heads).transpose(0, 1).unsqueeze(0)
        outputs.append(
            _sdpa_reference(
                q[batch_idx : batch_idx + 1].unsqueeze(2),
                k_seq,
                v_seq,
                query_start=length - 1,
                window_left=window_left,
                softmax_scale=softmax_scale,
            )
            .squeeze(2)
            .squeeze(0)
        )
    return torch.stack(outputs, dim=0)


def _make_inputs(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(81)
    q = torch.randn(9, 4, 32, device=device, dtype=dtype)
    k = torch.randn(9, 2, 32, device=device, dtype=dtype)
    v = torch.randn(9, 2, 32, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
@pytest.mark.parametrize("window_left", [None, 2], ids=["full", "window2"])
def test_varlen_native_attention_matches_sdpa_forward_and_backward(dtype: torch.dtype, window_left: int | None) -> None:
    device = torch.device("cuda")
    q, k, v = _make_inputs(dtype, device)
    cu_seqlens = torch.tensor([0, 1, 4, 9], device=device, dtype=torch.int32)
    softmax_scale = 0.17

    q_native, k_native, v_native = (value.detach().clone().requires_grad_() for value in (q, k, v))
    native = areno_varlen_causal_attention(
        q_native,
        k_native,
        v_native,
        cu_seqlens,
        window_left=window_left,
        softmax_scale=softmax_scale,
    )
    grad_output = torch.randn_like(native)
    native.backward(grad_output)

    q_reference, k_reference, v_reference = (value.detach().clone().requires_grad_() for value in (q, k, v))
    reference = _varlen_reference(
        q_reference,
        k_reference,
        v_reference,
        cu_seqlens,
        window_left=window_left,
        softmax_scale=softmax_scale,
    )
    reference.backward(grad_output)
    torch.cuda.synchronize()

    _assert_close(native, reference)
    _assert_close(q_native.grad, q_reference.grad)
    _assert_close(k_native.grad, k_reference.grad)
    _assert_close(v_native.grad, v_reference.grad)


def _make_paged_inputs(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(81)
    q = torch.randn(2, 4, 32, device=device, dtype=dtype)
    k_update = torch.randn(2, 2, 32, device=device, dtype=dtype)
    v_update = torch.randn(2, 2, 32, device=device, dtype=dtype)
    k_cache = torch.randn(5, 4, 2, 32, device=device, dtype=dtype)
    v_cache = torch.randn(5, 4, 2, 32, device=device, dtype=dtype)
    block_table = torch.tensor([[2, 0, 0], [1, 3, 4]], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([5, 8], device=device, dtype=torch.int32)
    return q, k_update, v_update, k_cache, v_cache, block_table, cache_seqlens


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
@pytest.mark.parametrize("window_left", [None, 2], ids=["full", "window2"])
def test_paged_native_attention_matches_sdpa_and_updates_cache(dtype: torch.dtype, window_left: int | None) -> None:
    device = torch.device("cuda")
    q, k_update, v_update, k_cache, v_cache, block_table, cache_seqlens = _make_paged_inputs(dtype, device)
    expected_k_cache = k_cache.clone()
    expected_v_cache = v_cache.clone()
    block_size = k_cache.shape[1]
    for batch_idx in range(q.shape[0]):
        position = int(cache_seqlens[batch_idx].item())
        block_col, block_offset = divmod(position, block_size)
        block_id = int(block_table[batch_idx, block_col].item())
        expected_k_cache[block_id, block_offset] = k_update[batch_idx]
        expected_v_cache[block_id, block_offset] = v_update[batch_idx]

    native = areno_paged_causal_attention_decode(
        q,
        k_update,
        v_update,
        k_cache,
        v_cache,
        block_table,
        cache_seqlens,
        window_left=window_left,
        num_splits=8,
        softmax_scale=0.17,
    )
    torch.cuda.synchronize()
    reference = _paged_reference(
        q,
        expected_k_cache,
        expected_v_cache,
        block_table,
        cache_seqlens,
        window_left=window_left,
        softmax_scale=0.17,
    )

    _assert_close(native, reference)
    torch.testing.assert_close(k_cache, expected_k_cache, atol=0, rtol=0)
    torch.testing.assert_close(v_cache, expected_v_cache, atol=0, rtol=0)
