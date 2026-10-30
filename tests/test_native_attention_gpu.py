from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from areno.accel.attention import areno_paged_causal_attention_decode, areno_varlen_causal_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native attention equivalence tests require CUDA")


@pytest.mark.parametrize("backend", ["native", "flash"])
@pytest.mark.parametrize("window_left", [None, 3])
@torch.inference_mode()
def test_chunked_prefill_reads_cached_prefix_in_mixed_batch(backend, window_left):
    from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend
    from areno.engine.runtime.metadata import InferMeta

    if backend == "flash":
        pytest.importorskip("flash_attn")
    torch.manual_seed(37)
    device = torch.device("cuda")
    dtype = torch.float16
    block_size = 256
    q = torch.randn(1, 5, 4, 32, device=device, dtype=dtype)
    k = torch.randn(1, 5, 2, 32, device=device, dtype=dtype)
    v = torch.randn_like(k)
    k_cache = torch.randn(6, block_size, 2, 32, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor([[2, 0, 4], [1, 3, 5]], device=device, dtype=torch.int32)
    prefixes = [block_size + 1, 0]
    offsets = [0, 3, 5]
    block_ids, block_offsets = [], []
    expected_k, expected_v = k_cache.clone(), v_cache.clone()
    for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        positions = torch.arange(prefixes[row], prefixes[row] + end - start, device=device)
        ids = block_table[row, positions // block_size].long()
        block_ids.append(ids)
        block_offsets.append(positions % block_size)
        expected_k[ids, positions % block_size] = k[0, start:end]
        expected_v[ids, positions % block_size] = v[0, start:end]
    meta = InferMeta(
        mode="prefill",
        cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=device),
        max_seqlen=3,
        cache_seqlens=torch.tensor(prefixes, dtype=torch.int32, device=device),
        block_table=block_table,
        cache_block_ids=torch.cat(block_ids),
        cache_block_offsets=torch.cat(block_offsets),
    )

    actual = FlashAttnInferBackend(backend)(
        q, k, v, k_cache, v_cache, meta, window_size=None if window_left is None else (window_left, 0)
    )
    expected = []
    for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        positions = torch.arange(prefixes[row] + end - start, device=device)
        ids = block_table[row, positions // block_size].long()
        keys = _expand_kv_heads(expected_k[ids, positions % block_size], 4).transpose(0, 1).unsqueeze(0)
        values = _expand_kv_heads(expected_v[ids, positions % block_size], 4).transpose(0, 1).unsqueeze(0)
        expected.append(
            _sdpa_reference(
                q[0, start:end].transpose(0, 1).unsqueeze(0),
                keys,
                values,
                query_start=prefixes[row],
                window_left=window_left,
                softmax_scale=32**-0.5,
            )
        )
    expected = torch.cat([value.transpose(1, 2) for value in expected], dim=1)
    _assert_close(actual, expected)
    torch.testing.assert_close(k_cache, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(v_cache, expected_v, atol=0, rtol=0)


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


@pytest.mark.parametrize("backend", ["flash", "native"])
@pytest.mark.parametrize("partitions", [[2, 2], [1, 1, 1, 1]])
@torch.inference_mode()
def test_zero_score_chunk_partitions_and_padded_values(backend, partitions):
    from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
    from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend

    if backend == "flash":
        pytest.importorskip("flash_attn")
    state = InferenceBatchState([[1, 3, 5, 7]], 1, max_prefill_tokens=partitions[0], kv_block_size=256)
    cache_k = torch.zeros(2, 256, 1, 32, dtype=torch.float16, device="cuda")
    cache_v = torch.zeros_like(cache_k)
    outputs = []
    for count in partitions:
        state.max_prefill_tokens = count
        p = state.build_prefill_payload()
        meta = payload_to_infer_meta(p, torch.device("cuda"))
        q = torch.zeros(1, count, 2, 32, dtype=torch.float16, device="cuda")
        k = q[:, :, :1]
        v = p["input_ids"].cuda().half().view(1, count, 1, 1).expand(1, count, 1, 16)
        outputs.append(FlashAttnInferBackend(backend)(q, k, v, cache_k, cache_v, meta))
    actual = torch.cat(outputs, 1)
    assert actual.shape == (1, 4, 2, 16)
    torch.testing.assert_close(actual[0, :, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda").half())


@torch.inference_mode()
def test_dense_model_chunked_logits_logprobs_and_graph_decode():
    """Real CUDA Qwen3 adapter with small seeded weights, without downloading a checkpoint."""
    from areno.engine.config import EngineConfig, ModelConfig, RuntimeConfig
    from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
    from areno.engine.modeling import build_model_on_device
    from areno.engine.runtime.decode_graph import DecodeGraph
    from areno.engine.runtime.metadata import InferMeta

    pytest.importorskip("flash_attn")
    torch.manual_seed(635)
    device = torch.device("cuda", 0)
    config = ModelConfig(
        model_type="qwen3",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        dtype=torch.float16,
        max_position_embeddings=512,
    )
    model = build_model_on_device(
        EngineConfig(config, devices=[0], role="rollout", runtime=RuntimeConfig(compile_model=False)), device
    ).eval()
    model.set_kv_caches(model.allocate_kv_caches(8, 256, device))
    tokens = torch.randint(1, 128, (257,)).tolist()

    def prefill(budget):
        state = InferenceBatchState(
            [tokens], 4, max_cache_len=512, max_prefill_tokens=budget, kv_block_size=256, num_cache_blocks=7
        )
        while state.has_pending_prompts:
            p = state.build_prefill_payload()
            meta = payload_to_infer_meta(p, device)
            logits = model(
                p["input_ids"].to(device)[None], p["position_ids"].to(device)[None], infer_meta=meta
            ).logits_shard[0, -1]
        return logits.clone(), state

    def decode(state):
        table = torch.tensor([state._seq_to_blocks[0]], device=device, dtype=torch.int32)
        outputs = []
        for step, token in enumerate([11, 17, 23, 29]):
            meta = InferMeta(
                mode="decode",
                cache_seqlens=torch.tensor([len(tokens) + step], device=device, dtype=torch.int32),
                block_table=table,
                recurrent_slots=torch.tensor([state._seq_to_recurrent_slot[0]], device=device),
            )
            outputs.append(
                model(
                    torch.tensor([[token]], device=device),
                    torch.tensor([[len(tokens) + step]], device=device),
                    infer_meta=meta,
                )
                .logits_shard[0, -1]
                .clone()
            )
        return torch.stack(outputs)

    reference, reference_state = prefill(512)
    reference_decode = decode(reference_state)
    for budget in [128, 65, 1]:
        actual, state = prefill(budget)
        torch.testing.assert_close(actual, reference, atol=0.005, rtol=0.02)
        torch.testing.assert_close(
            torch.log_softmax(actual.float(), -1)[[11, 17, 23]],
            torch.log_softmax(reference.float(), -1)[[11, 17, 23]],
            atol=0.005,
            rtol=0.005,
        )
        assert int(actual.argmax()) == int(reference.argmax())
        torch.testing.assert_close(decode(state), reference_decode, atol=0.005, rtol=0.02)
    table = torch.tensor([state._seq_to_blocks[0]], device=device, dtype=torch.int32)
    graph = DecodeGraph(model, 1, 2, 7, 2, device)
    graph.warmup(iterations=2)
    graph.capture()
    fixed_token = torch.tensor([[11]], device=device)
    positions = torch.tensor([[len(tokens)]], device=device)
    lengths = torch.tensor([len(tokens)], device=device, dtype=torch.int32)
    slots = torch.tensor([0], device=device)
    prefill(65)
    meta = InferMeta(mode="decode", cache_seqlens=lengths, block_table=table, recurrent_slots=slots)
    eager = model(fixed_token, positions, infer_meta=meta).logits_shard.clone()
    prefill(65)
    captured = graph.replay_tensors(fixed_token, positions, lengths, table, slots).clone()
    torch.testing.assert_close(captured, eager, atol=0.005, rtol=0.02)


@pytest.mark.parametrize("budget", [128, 65, 1])
@torch.inference_mode()
def test_ling_kda_chunked_outputs_and_recurrent_state(budget):
    from areno.engine.config import ModelConfig
    from areno.engine.runtime.metadata import InferMeta
    from areno.models.bailing_v3.model import BailingKDAAttention

    torch.manual_seed(635)
    device = torch.device("cuda")
    config = ModelConfig(
        model_type="bailing_moe_v3",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        v_head_dim=32,
        dtype=torch.bfloat16,
        kda_safe_gate=True,
        kda_lower_bound=-5,
    )
    with torch.device(device):
        attn = BailingKDAAttention(config, 0).eval()
    for param in attn.parameters():
        torch.nn.init.normal_(param, std=0.1)
    attn.set_state_cache(torch.zeros(2, 2, 32, 32, device=device), torch.zeros(2, 3, 64, 3, device=device))
    hidden = torch.randn(1, 257, 64, device=device, dtype=torch.bfloat16)

    def run(chunk):
        attn.reset_kv_cache()
        outputs = []
        for start in range(0, 257, chunk):
            count = min(chunk, 257 - start)
            meta = InferMeta(
                mode="prefill",
                cu_seqlens=torch.tensor([0, count], device=device, dtype=torch.int32),
                max_seqlen=count,
                block_table=torch.tensor([[0]], device=device, dtype=torch.int32),
                recurrent_slots=torch.tensor([1], device=device),
                cache_seqlens=None if start == 0 else torch.tensor([start], device=device, dtype=torch.int32),
            )
            outputs.append(
                attn(
                    hidden[:, start : start + count],
                    torch.arange(start, start + count, device=device)[None],
                    None,
                    meta,
                )
            )
        return torch.cat(outputs, 1), attn.state_cache.clone(), attn.conv_cache.clone()

    expected = run(257)
    actual = run(budget)
    # BF16 FLA chunk boundaries round local reductions differently. Compare
    # attention outputs and FP32 continuation state separately from MoE logits.
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, atol=0.005, rtol=0.02)
    assert torch.count_nonzero(attn.state_cache[0]) == 0
    assert torch.count_nonzero(attn.conv_cache[0]) == 0
