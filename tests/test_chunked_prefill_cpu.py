"""Scheduler/attention contracts with an explicit CPU math substitute, not CUDA evidence."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
from areno.engine.layers.attention_backend import infer


def paged_reference(
    q, k_cache, v_cache, *, cu_seqlens_q, cu_seqlens_k, block_table, window_size, softmax_scale, **kwargs
):
    outputs = []
    for row, (start, end) in enumerate(zip(cu_seqlens_q[:-1], cu_seqlens_q[1:], strict=True)):
        total = int(cu_seqlens_k[row + 1] - cu_seqlens_k[row])
        pos = torch.arange(total)
        blocks = block_table[row, pos // k_cache.shape[1]].long()
        k, v = k_cache[blocks, pos % k_cache.shape[1]], v_cache[blocks, pos % k_cache.shape[1]]
        repeat = q.shape[-2] // k.shape[-2]
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        visible = pos[None, :] <= torch.arange(total - int(end - start), total)[:, None]
        if window_size != (-1, -1):
            visible &= pos[None, :] >= torch.arange(total - int(end - start), total)[:, None] - window_size[0]
        outputs.append(
            F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
                attn_mask=visible,
                scale=softmax_scale,
            ).transpose(0, 1)
        )
    return torch.cat(outputs)


@pytest.mark.parametrize("backend", ["flash", "native"])
def test_zero_scores_see_all_previous_chunks(monkeypatch, backend):
    def flash(q, k, v, **kwargs):
        if kwargs.get("block_table") is not None:
            return paged_reference(q, k, v, **kwargs)
        return F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True
        ).transpose(0, 1)

    def native(q, k, v, cache_k, cache_v, meta, window, scale):
        cu = torch.arange(q.shape[0] + 1, dtype=torch.int32)
        lengths = meta.cache_seqlens + 1
        return paged_reference(
            q,
            cache_k,
            cache_v,
            cu_seqlens_q=cu,
            cu_seqlens_k=torch.cat((cu[:1], lengths.cumsum(0, dtype=torch.int32))),
            block_table=meta.block_table,
            window_size=window,
            softmax_scale=scale,
        )

    monkeypatch.setattr(infer, "_flash_attn_varlen_no_compile", flash)
    monkeypatch.setattr(infer, "_native_prefill", lambda q, k, v, meta, window, scale: flash(q, k, v))
    monkeypatch.setattr(infer, "_native_decode", native)
    state = InferenceBatchState([[1, 3, 5, 7]], 1, max_prefill_tokens=2, kv_block_size=2)
    cache_k = torch.zeros(3, 2, 1, 8)
    cache_v = torch.zeros_like(cache_k)
    outputs = []
    while state.has_pending_prompts:
        payload = state.build_prefill_payload()
        meta = payload_to_infer_meta(payload, torch.device("cpu"))
        q = torch.zeros(1, 2, 1, 8)
        v = payload["input_ids"].float().view(1, 2, 1, 1).expand_as(q)
        outputs.append(infer.FlashAttnInferBackend(backend)(q, q, v, cache_k, cache_v, meta))
    torch.testing.assert_close(torch.cat(outputs, 1)[0, :, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_mixed_prefix_metadata_and_single_paged_launch(monkeypatch):
    state = InferenceBatchState([[1] * 5, [2] * 3], 1, max_prefill_tokens=3, kv_block_size=2)
    fresh = payload_to_infer_meta(state.build_prefill_payload(), torch.device("cpu"))
    assert fresh.cache_seqlens is None and fresh.cu_seqlens_k is None
    state.max_prefill_tokens = 5
    meta = payload_to_infer_meta(state.build_prefill_payload(), torch.device("cpu"))
    assert meta.cache_seqlens.tolist() == [3, 0]
    assert meta.cu_seqlens.tolist() == [0, 2, 5]
    assert meta.cu_seqlens_k.tolist() == [0, 5, 8]
    assert meta.max_seqlen_k == 5
    calls = []

    def flash(q, k, v, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(q)

    monkeypatch.setattr(infer, "_flash_attn_varlen_no_compile", flash)
    monkeypatch.setattr(infer, "_flash_attn_with_kvcache_no_compile", lambda *a, **kw: pytest.fail("per-row dispatch"))
    q = torch.zeros(1, 5, 2, 8)
    cache = torch.zeros(state.num_cache_blocks, 2, 2, 8)
    infer.FlashAttnInferBackend("flash")(q, q, q, cache, cache.clone(), meta)
    assert len(calls) == 1
    assert calls[0]["block_table"] is meta.block_table
    assert calls[0]["cu_seqlens_k"] is meta.cu_seqlens_k


@pytest.mark.parametrize("chunks", [[2, 3, 2], [1] * 7])
def test_ling_kda_short_conv_keeps_history_across_chunks(monkeypatch, chunks):
    pytest.importorskip("triton")
    from areno.models.bailing_v3 import model

    torch.manual_seed(11)
    weight = torch.randn(3, 1, 4)
    x = torch.randn(1, 7, 3)

    def conv(x, w):
        return F.silu(F.conv1d(F.pad(x.transpose(1, 2), (3, 0)), w, groups=3)).transpose(1, 2)

    monkeypatch.setattr(model, "_areno_depthwise_causal_conv1d_silu_no_compile", conv)
    attn = SimpleNamespace(conv_kernel_size=4, conv_cache=torch.zeros(2, 3, 3, 3))
    attn._causal_conv_infer_prefill = MethodType(model.BailingKDAAttention._causal_conv_infer_prefill, attn)
    start, outputs = 0, []
    for count in chunks:
        outputs.append(
            model.BailingKDAAttention._causal_conv_infer(
                attn,
                x[:, start : start + count],
                weight,
                0,
                infer.InferMeta(
                    mode="prefill",
                    cu_seqlens=torch.tensor([0, count]),
                    block_table=torch.tensor([[0]]),
                    recurrent_slots=torch.tensor([1]),
                    cache_seqlens=None if start == 0 else torch.tensor([start]),
                ),
            )
        )
        start += count
    torch.testing.assert_close(torch.cat(outputs, 1), conv(x, weight))
    torch.testing.assert_close(attn.conv_cache[1, 0], x[0, -3:].T)
    assert torch.count_nonzero(attn.conv_cache[0]) == 0


def test_npu_cached_prefix_is_rejected_before_mutating_cache(monkeypatch):
    call = SimpleNamespace(q=SimpleNamespace(device=SimpleNamespace(type="npu")))
    monkeypatch.setattr(infer, "build_attention_call", lambda *a: call)
    monkeypatch.setattr(infer, "_store_prefill_cache", lambda *a: pytest.fail("unsupported prefix mutated cache"))
    meta = infer.InferMeta(
        mode="prefill",
        cu_seqlens=torch.tensor([0, 1]),
        max_seqlen=1,
        cache_seqlens=torch.tensor([4]),
        block_table=torch.tensor([[0]]),
    )
    q = torch.zeros(1, 1, 1, 8)
    with pytest.raises(ValueError, match="not supported on NPU"):
        infer.FlashAttnInferBackend()(q, q, q, q, q, meta)


def test_ling_short_conv_mixed_fresh_and_continuing_rows(monkeypatch):
    pytest.importorskip("triton")
    from areno.models.bailing_v3 import model

    torch.manual_seed(635)
    weight = torch.randn(3, 1, 4)
    x = torch.randn(1, 4, 3)

    def conv(x, w):
        return F.silu(F.conv1d(F.pad(x.transpose(1, 2), (3, 0)), w, groups=3)).transpose(1, 2)

    monkeypatch.setattr(model, "_areno_depthwise_causal_conv1d_silu_no_compile", conv)
    # Slot 0 is fresh and must ignore its old contents even when another row
    # in the same packed batch has a cached prefix.
    attn = SimpleNamespace(conv_kernel_size=4, conv_cache=torch.randn(2, 3, 3, 3))
    history = attn.conv_cache[1, 0].T.unsqueeze(0).clone()
    actual = model.BailingKDAAttention._causal_conv_infer_prefill(
        attn, x, weight, 0, torch.tensor([0, 2, 4]), torch.tensor([0, 1]), torch.tensor([0, 5])
    )
    expected = torch.cat((conv(x[:, :2], weight), conv(torch.cat((history, x[:, 2:]), 1), weight)[:, -2:]), 1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(attn.conv_cache[0, 0, :, 0], torch.zeros(3))


def test_chunked_prefill_continues_in_last_allocated_page():
    state = InferenceBatchState([[1, 3, 5, 7]], 1, max_prefill_tokens=1, kv_block_size=256)
    for position in range(4):
        payload = state.build_prefill_payload()
        assert payload is not None
        assert payload["position_ids"].tolist() == [position]
        assert payload["sample_indices"].numel() == int(position == 3)
    assert not state.has_pending_prompts
