"""CUDA exactness controls, including realistic lift dimensions."""

import pytest
import torch

from areno.engine.optim.galore import GaLoreAdamW

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(32768, 1024), (1024, 3072)])
@pytest.mark.parametrize("tile_rows", [512, 16384])
def test_lift_tiles_cuda_exact(shape, tile_rows):
    m, n = shape
    r = 128
    gen = torch.Generator(device="cuda").manual_seed(593)
    basis = torch.randn((r, n) if m >= n else (m, r), device="cuda", generator=gen)
    update = torch.randn((m, r) if m >= n else (r, n), device="cuda", generator=gen)
    dense = update @ basis if m >= n else basis @ update
    tiled = torch.cat(
        [
            update[s : s + tile_rows] @ basis if m >= n else basis[s : s + tile_rows] @ update
            for s in range(0, m, tile_rows)
        ]
    )
    assert torch.equal(dense, tiled), f"tiled lift differs: {(dense - tiled).abs().max().item()}"


@pytest.mark.parametrize("shape", [(512, 256), (256, 512), (513,)])
@pytest.mark.parametrize("fused", [False, True])
def test_compact_resume_cuda_exact(shape, fused):
    torch.manual_seed(593)
    p = torch.nn.Parameter(torch.randn(shape, device="cuda").bfloat16())
    q = torch.nn.Parameter(p.detach().clone())
    kw = dict(lr=0.003, rank=16, update_proj_gap=3, weight_decay=0.01)
    ref = GaLoreAdamW([q], **kw)
    actual = GaLoreAdamW([p], compact_master=True, tile_rows=128, release_gradients=True, fused_writeback=fused, **kw)
    for step in range(5):
        grad = torch.randn(shape, device="cuda")
        p.main_grad = grad.clone()
        q.main_grad = grad.clone()
        actual.step()
        ref.step()
        a = actual.state_dict()["state"][0]
        b = ref.state_dict()["state"][0]
        assert torch.equal(p, q)
        for key in ("master", "exp_avg", "exp_avg_sq", "basis"):
            assert a[key] is b[key] if a[key] is None else torch.equal(a[key], b[key])
        if step == 2:
            saved = actual.state_dict()
            actual = GaLoreAdamW([p], lr=0.5)
            actual.load_state_dict(saved)


def test_cuda_extreme_finite_bit_patterns():
    from areno.engine.optim.master_storage import decode_fp32_master

    bits = torch.tensor(
        [0, -2147483648, 1, 0x007FFFFF, 0x00800000, 0x3F7F8000, 0x3F808000, 0x3F818000, 0x7F7FFFFF, -8388609],
        dtype=torch.int32,
        device="cuda",
    )
    model, packed = GaLoreAdamW._checked_encode(bits.view(torch.float32))
    assert torch.equal(decode_fp32_master(model, packed).view(torch.int32), bits)


def test_fused_writeback_edge_bits_and_nonfinite_flag():
    from areno.engine.optim.master_storage import decode_fp32_master

    bits = torch.tensor(
        [0, -2147483648, 1, 0x007FFFFF, 0x00800000, 0x3F7F8000, 0x3F808000, 0x3F818000, 0x7F7FFFFF, -8388609],
        dtype=torch.int32,
        device="cuda",
    )
    original = bits.view(torch.float32)
    model, packed = GaLoreAdamW._checked_encode(original)
    p = torch.nn.Parameter(model)
    opt = GaLoreAdamW([p], lr=0.003, compact_master=True, fused_writeback=True)
    opt._writeback_invalid = torch.zeros((), dtype=torch.int32, device="cuda")
    update = torch.zeros_like(original)
    opt._write_update(p, {"master": packed}, update, 0, 0.003)
    expected = original.clone().mul_(1.0).add_(update, alpha=-0.003)
    assert torch.equal(decode_fp32_master(p, packed).view(torch.int32), expected.view(torch.int32))
    assert opt._writeback_invalid.item() == 0
    opt._write_update(p, {"master": packed}, torch.full_like(update, float("inf")), 0, 0.003)
    assert opt._writeback_invalid.item() == 1
