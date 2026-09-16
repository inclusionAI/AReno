"""Gaudi-only tests through public accel APIs, including graph glue and autograd."""

import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from areno.accel._extension import extension
from areno.accel.attention import (
    areno_causal_attention,
    areno_paged_causal_attention_decode,
    areno_varlen_causal_attention,
)
from areno.accel.conv import (
    areno_depthwise_causal_conv1d_silu,
    areno_depthwise_causal_conv1d_silu_decode,
    areno_packed_depthwise_causal_conv1d_silu,
)
from areno.accel.embedding import areno_vocab_embedding
from areno.accel.kda import areno_kda_chunk, areno_kda_recurrent_update
from areno.accel.linear import areno_grouped_linear
from areno.accel.moe import areno_moe_topk_permute, areno_moe_unpermute
from areno.accel.ops import FusedMoeConfig, SegLaMeta, areno_fused_experts, rms_norm_gate_fwd, seg_la_fwd
from areno.accel.router import areno_grouped_topk_router
from areno.accel.routing import areno_moe_align
from areno.accel.topk import areno_topk_softmax
from tests.test_hpu_activation import DTYPES, TOLERANCES
from tests.test_hpu_activation import hpu_core as hpu_core
from tests.test_hpu_kda_source_cpu import reference as kda_reference


def compare(actual, expected, dtype, core):
    core.mark_step()
    torch.hpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.detach().to(actual.dtype), **TOLERANCES[dtype])


def test_all_cuda_extension_exports_exist(hpu_core):
    cuda = Path(__file__).resolve().parents[1] / "areno/accel/csrc/extension.cpp"
    exports = set(re.findall(r'm\.def\(\s*"(\w+)"', cuda.read_text()))
    native = extension("hpu")
    assert exports <= set(dir(native))


@pytest.mark.parametrize("dtype", DTYPES)
def test_embedding_large_indices_and_repeated_backward(hpu_core, dtype):
    ids = torch.tensor([[5, 6, 8, 5], [0, -1, 2**32 + 5, 2**40]])
    weight = torch.randn(4, 65).to(dtype)
    actual_weight = weight.to("hpu").requires_grad_()
    actual = areno_vocab_embedding(ids.to("hpu"), actual_weight, 5, 9)
    ref = weight.float().requires_grad_()
    expected = F.embedding((ids - 5).clamp(0, 3), ref) * ((ids >= 5) & (ids < 9))[..., None]
    grad = torch.randn(actual.shape).to(dtype)
    actual.backward(grad.to("hpu"))
    expected.backward(grad.float())
    compare(actual, expected, dtype, hpu_core)
    compare(actual_weight.grad, ref.grad, dtype, hpu_core)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", [False, True])
def test_attention_forward_backward(hpu_core, dtype, packed):
    shape = (7, 2, 65) if packed else (1, 2, 7, 65)
    values = [torch.randn(shape).to(dtype) * 0.2 for _ in range(3)]
    actual_inputs = [x.to("hpu").requires_grad_() for x in values]
    reference_inputs = [x.float().requires_grad_() for x in values]
    if packed:
        cu = torch.tensor([0, 0, 3, 7], dtype=torch.int32)
        actual = areno_varlen_causal_attention(*actual_inputs, cu.to("hpu"), window_left=2)
        boundaries = [(0, 3), (3, 7)]
        expected = []
        for first, last in boundaries:
            q, k, v = [x[first:last].transpose(0, 1) for x in reference_inputs]
            mask = torch.arange(last - first)[:, None] - torch.arange(last - first)[None, :]
            expected.append(
                F.scaled_dot_product_attention(q, k, v, attn_mask=(mask >= 0) & (mask <= 2)).transpose(0, 1)
            )
        expected = torch.cat(expected)
    else:
        actual = areno_causal_attention(*actual_inputs, window_left=2)
        mask = torch.arange(7)[:, None] - torch.arange(7)[None, :]
        expected = F.scaled_dot_product_attention(*reference_inputs, attn_mask=(mask >= 0) & (mask <= 2))
    grad = torch.randn(expected.shape).to(dtype)
    actual.backward(grad.to("hpu"))
    expected.backward(grad.float())
    compare(actual, expected, dtype, hpu_core)
    for a, b in zip(actual_inputs, reference_inputs):
        compare(a.grad, b.grad, dtype, hpu_core)


@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_attention_updates_only_selected_slots(hpu_core, dtype):
    q = torch.randn(2, 4, 65).to(dtype) * 0.2
    ku, vu = [torch.randn(2, 2, 65).to(dtype) * 0.2 for _ in range(2)]
    kc, vc = [torch.randn(6, 4, 2, 65).to(dtype) * 0.2 for _ in range(2)]
    table = torch.tensor([[2, 0], [4, 1]], dtype=torch.int32)
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    hkc, hvc = kc.to("hpu"), vc.to("hpu")
    pointers = (hkc.data_ptr(), hvc.data_ptr())
    actual = areno_paged_causal_attention_decode(
        q.to("hpu"), ku.to("hpu"), vu.to("hpu"), hkc, hvc, table.to("hpu"), lengths.to("hpu")
    )
    expected = []
    for b, length in enumerate(lengths.tolist()):
        block = table[b, length // 4]
        kc[block, length % 4] = ku[b]
        vc[block, length % 4] = vu[b]
        keys = torch.stack([kc[table[b, t // 4], t % 4] for t in range(length + 1)]).repeat_interleave(2, dim=1)
        vals = torch.stack([vc[table[b, t // 4], t % 4] for t in range(length + 1)]).repeat_interleave(2, dim=1)
        score = torch.einsum("hd,thd->ht", q[b].float(), keys.float()) / 65**0.5
        expected.append(torch.einsum("ht,thd->hd", score.softmax(-1), vals.float()))
    compare(actual, torch.stack(expected), dtype, hpu_core)
    torch.testing.assert_close(hkc.cpu(), kc, rtol=0, atol=0)
    torch.testing.assert_close(hvc.cpu(), vc, rtol=0, atol=0)
    assert pointers == (hkc.data_ptr(), hvc.data_ptr())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", [False, True])
def test_conv_training_and_decode(hpu_core, dtype, packed):
    x = torch.randn(1, 7, 65).to(dtype)
    w = torch.randn(65, 1, 4)
    hx, hw = x.to("hpu").requires_grad_(), w.to("hpu").requires_grad_()
    rx, rw = x.float().requires_grad_(), w.clone().requires_grad_()
    if packed:
        cu = torch.tensor([0, 0, 2, 7], dtype=torch.int32)
        actual = areno_packed_depthwise_causal_conv1d_silu(hx, hw, cu.to("hpu"))
        parts = [
            F.silu(F.conv1d(rx[:, a:b].transpose(1, 2), rw, padding=3, groups=65)[:, :, : b - a]).transpose(1, 2)
            for a, b in ((0, 2), (2, 7))
        ]
        expected = torch.cat(parts, dim=1)
    else:
        actual = areno_depthwise_causal_conv1d_silu(hx, hw)
        expected = F.silu(F.conv1d(rx.transpose(1, 2), rw, padding=3, groups=65)[:, :, :7]).transpose(1, 2)
    grad = torch.randn(x.shape).to(dtype)
    actual.backward(grad.to("hpu"))
    expected.backward(grad.float())
    compare(actual, expected, dtype, hpu_core)
    compare(hx.grad, rx.grad, dtype, hpu_core)
    compare(hw.grad, rw.grad, dtype, hpu_core)
    current = x[:, 6]
    history = x[:, 3:6].transpose(1, 2)
    out = areno_depthwise_causal_conv1d_silu_decode(current.to("hpu"), history.to("hpu"), hw.detach())
    compare(out, F.silu((history.float() * w[:, 0, :-1]).sum(-1) + current.float() * w[:, 0, -1]), dtype, hpu_core)


@pytest.mark.parametrize("dtype", DTYPES)
def test_router_moe_grouped_linear_training(hpu_core, dtype):
    logits = torch.randn(5, 4).to(dtype)
    hl = logits.to("hpu").requires_grad_()
    rl = logits.float().requires_grad_()
    ids, weights = areno_topk_softmax(hl, 2, True)
    expected_ids = rl.detach().argsort(dim=-1, descending=True)[:, :2]
    rw = rl.gather(-1, expected_ids).softmax(-1)
    torch.testing.assert_close(ids.cpu(), expected_ids)
    compare(weights, rw, torch.float32, hpu_core)
    x = torch.randn(5, 65).to(dtype)
    w = torch.randn(4, 33, 65).to(dtype) * 0.1
    hx, hw = x.to("hpu").requires_grad_(), w.to("hpu").requires_grad_()
    rx, rweight = x.float().requires_grad_(), w.float().requires_grad_()
    routed, route_weight, token_ids, counts = areno_moe_topk_permute(hx, ids, weights, 0, 4)
    projected = areno_grouped_linear(routed, hw, counts)
    actual = areno_moe_unpermute(projected.float() * route_weight[:, None], token_ids, (5, 33))
    expected = torch.stack(
        [
            sum(F.linear(rx[t], rweight[e]).to(dtype).float() * rw[t, p] for p, e in enumerate(expected_ids[t]))
            for t in range(5)
        ]
    )
    grad = torch.randn(actual.shape)
    actual.backward(grad.to("hpu"))
    expected.backward(grad)
    compare(actual, expected, dtype, hpu_core)
    for a, b in ((hx, rx), (hw, rweight), (hl, rl)):
        compare(a.grad, b.grad, dtype, hpu_core)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_fused_experts_and_grouped_gate(hpu_core, dtype, activation):
    x = torch.randn(4, 65).to(dtype) * 0.2
    w1 = torch.randn(3, 66, 65).to(dtype) * 0.1
    w2 = torch.randn(3, 65, 33).to(dtype) * 0.1
    ids = torch.tensor([[0, 2], [2, 1], [1, 0], [0, -1]])
    weights = torch.rand(4, 2)
    config = FusedMoeConfig(3, 65, 33, 2, routed_scaling_factor=1.3)
    out = areno_fused_experts(
        x.to("hpu"), w1.to("hpu"), w2.to("hpu"), weights.to("hpu"), ids.to("hpu"), config, activation=activation
    )
    expected = torch.zeros_like(x).float()
    for t in range(4):
        for p, e in enumerate(ids[t]):
            if e < 0:
                continue
            up = F.linear(x[t].float(), w1[e].float()).to(dtype).float()
            gate, linear = up.chunk(2)
            act = F.silu(gate) if activation == "silu" else F.gelu(gate, approximate="tanh")
            middle = (act * linear).to(dtype).float()
            expected[t] += (F.linear(middle, w2[e].float()) * weights[t, p]).to(dtype).float()
    compare(out, expected * 1.3, dtype, hpu_core)
    xx = torch.randn(3, 2, 65).to(dtype).transpose(0, 1)
    gg = torch.randn_like(xx)
    ww = torch.randn(3, 65).to(dtype)
    norm, inv = rms_norm_gate_fwd(xx.to("hpu"), gg.to("hpu"), ww.to("hpu"), 1e-5)
    rinv = (xx.float().square().mean(-1) + 1e-5).rsqrt()
    compare(norm, xx.float() * rinv[..., None] * ww.float() * gg.float().sigmoid(), dtype, hpu_core)
    compare(inv, rinv, torch.float32, hpu_core)


def test_align_in_place_and_grouped_router(hpu_core):
    ids = torch.tensor([[1, -1], [0, 2], [1, 2]], device="hpu", dtype=torch.int32)
    cap = ids.numel() + 4 * 3
    sorted_ids = torch.full((cap,), -9, device="hpu", dtype=torch.int32)
    experts = torch.full(((cap + 3) // 4,), -8, device="hpu", dtype=torch.int32)
    total = torch.empty(1, device="hpu", dtype=torch.int32)
    scratch = torch.empty(5, device="hpu", dtype=torch.int32)
    before = [x.data_ptr() for x in (sorted_ids, experts, total, scratch)]
    areno_moe_align(ids, 4, 4, sorted_ids, experts, total, scratch)
    hpu_core.mark_step()
    assert total.cpu().item() == 16
    assert before == [x.data_ptr() for x in (sorted_ids, experts, total, scratch)]
    torch.testing.assert_close(
        sorted_ids.cpu()[:16], torch.tensor([1, 6, 6, 6, 2, 6, 6, 6, 0, 4, 6, 6, 3, 5, 6, 6], dtype=torch.int32)
    )
    logits = torch.arange(16, dtype=torch.float32).view(1, 16) * 0.1
    selected, weights = areno_grouped_topk_router(logits.to("hpu"), torch.zeros(16, device="hpu"), 4, 4, 2)
    torch.testing.assert_close(selected.cpu(), torch.tensor([[15, 14, 13, 12]]))
    expected = logits.sigmoid()[:, [15, 14, 13, 12]]
    expected /= expected.sum(-1, keepdim=True)
    compare(weights, expected, torch.float32, hpu_core)


@pytest.mark.parametrize("dtype", DTYPES)
def test_kda_cpp_autograd_and_recurrent_state_alias(hpu_core, dtype):
    rng = torch.Generator().manual_seed(75)
    shapes = [(1, 5, 1, 3), (1, 5, 1, 3), (1, 5, 2, 4), (1, 5, 2, 3), (2,), (2, 3), (1, 5, 2), (3, 2, 4, 3)]
    cpu = [
        (torch.randn(shape, generator=rng) * 0.2).to(dtype if i in (0, 1, 2, 3, 6) else torch.float32)
        for i, shape in enumerate(shapes)
    ]
    hp = [x.to("hpu").requires_grad_() for x in cpu]
    ref = [x.float().requires_grad_() for x in cpu]
    q, k, v, g, a, bias, beta, initial = hp
    cu = torch.tensor([0, 0, 2, 5], dtype=torch.int32)
    ids = torch.tensor([0, 1, 2])
    out, final = areno_kda_chunk(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial,
        state_indices=ids.to("hpu"),
        output_final_state=True,
        scale=0.4,
        cu_seqlens=cu.to("hpu"),
        a_log=a,
        dt_bias=bias,
        lower_bound=-5.0,
    )
    rq, rk, rv, rg, ra, rbias, rbeta, rinitial = ref
    expected, expected_final = kda_reference(
        rq[0],
        rk[0],
        rv[0],
        rg[0],
        ra,
        rbias,
        rbeta[0],
        rinitial,
        cu.tolist(),
        ids.tolist(),
        dtype=dtype,
        normalize=True,
        recurrent=False,
        bound=-5.0,
        scale=0.4,
    )
    go = torch.randn(out.shape, generator=rng).to(dtype)
    gf = torch.randn(final.shape, generator=rng)
    torch.autograd.backward((out, final), (go.to("hpu"), gf.to("hpu")))
    torch.autograd.backward((expected, expected_final), (go[0].float(), gf))
    compare(out, expected[None], dtype, hpu_core)
    compare(final, expected_final, dtype, hpu_core)
    for actual, reference in zip(hp, ref):
        compare(actual.grad, reference.grad, dtype, hpu_core)
    state = cpu[-1].to("hpu")
    pointer = state.data_ptr()
    output = areno_kda_recurrent_update(
        *[x.detach() for x in (q, k, v, g.flatten(2), beta)],
        state=state,
        state_indices=ids.to("hpu"),
        scale=0.4,
        cu_seqlens=cu.to("hpu"),
        a_log=a.detach(),
        dt_bias=bias.detach(),
        lower_bound=None,
    )
    expected, expected_final = kda_reference(
        rq[0],
        rk[0],
        rv[0],
        rg[0],
        ra,
        rbias,
        rbeta[0],
        rinitial,
        cu.tolist(),
        ids.tolist(),
        dtype=dtype,
        normalize=True,
        recurrent=True,
        bound=None,
        scale=0.4,
    )
    compare(output, expected[None], dtype, hpu_core)
    compare(state, expected_final, dtype, hpu_core)
    assert state.data_ptr() == pointer


def test_segmented_mtp_preserves_state_and_cache_tails(hpu_core):
    q, k, v = [torch.randn(6, 2, 33) * 0.2 for _ in range(3)]
    state = torch.randn(4, 2, 33, 33) * 0.1
    cache = torch.full((4, 4, 2, 33, 33), -17.0)
    slots = torch.tensor([2, -1, 0])
    decay = torch.tensor([0.03, 0.2])
    meta = SegLaMeta(
        3,
        2,
        torch.tensor([0, 2, 4, 6], device="hpu"),
        slots.to("hpu"),
        torch.full((3,), 2, device="hpu"),
        torch.ones(3, device="hpu"),
    )
    hs, hc = state.to("hpu"), cache.to("hpu")
    out = seg_la_fwd(q.to("hpu"), k.to("hpu"), v.to("hpu"), hs, decay.to("hpu"), meta, hc, 0.3)
    expected = torch.zeros_like(q)
    for seq, slot in enumerate(slots):
        if slot < 0:
            continue
        current = state[slot].clone()
        for t in range(2):
            row = seq * 2 + t
            current = current * (-decay).exp()[:, None, None] + k[row][:, :, None] * v[row][:, None, :]
            expected[row] = (q[row][:, :, None] * current).sum(1) * 0.3
            cache[slot, t] = current
    compare(out, expected, torch.float32, hpu_core)
    compare(hc, cache, torch.float32, hpu_core)
    torch.testing.assert_close(hs.cpu(), state, rtol=0, atol=0)
