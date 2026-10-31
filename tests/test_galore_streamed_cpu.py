"""Exact compact-master storage, streamed lift, and canonical checkpoint contract."""

import pytest
import torch

from areno.engine.optim.galore import GaLoreAdamW
from areno.engine.optim.master_storage import BF16MasterStorage, decode_fp32_master, decode_fp32_master_slice


@pytest.mark.parametrize(
    "bits",
    [
        [0, -2147483648, 1, 0x007FFFFF, 0x00800000, 0x3F7F8000, 0x3F808000, 0x3F818000],
        [-1082163200, -1082097664, 0x7F7FFFFF, -8388609, 0x3F807FFF, 0x3F808001, -1 + 1],
    ],
)
def test_bit_patterns_roundtrip(bits):
    original = torch.tensor(bits, dtype=torch.int32).view(torch.float32)
    model, packed = GaLoreAdamW._checked_encode(original)
    assert torch.equal(decode_fp32_master(model, packed).view(torch.int32), original.view(torch.int32))
    for offset in range(original.numel()):
        result = decode_fp32_master_slice(model[offset : offset + 1], packed, offset)
        assert torch.equal(result.view(torch.int32), original[offset : offset + 1].view(torch.int32))
    assert packed.nbytes == original.numel() * 2 + (original.numel() + 7) // 8


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_is_explicit_rejection(value):
    with pytest.raises(RuntimeError, match="nonfinite"):
        GaLoreAdamW._checked_encode(torch.tensor([value]))


@pytest.mark.parametrize("shape", [(32, 16), (16, 32), (17,), (4, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tile_rows", [0, 8])
def test_streamed_updates_resume_and_bytes(shape, dtype, tile_rows):
    torch.manual_seed(593)
    param = torch.nn.Parameter(torch.randn(shape).to(dtype))
    reference_param = torch.nn.Parameter(param.detach().clone())
    kw = dict(lr=0.003, betas=(0.8, 0.9), weight_decay=0.1, rank=2, update_proj_gap=3, scale=0.7)
    reference = GaLoreAdamW([reference_param], **kw)
    actual = GaLoreAdamW([param], compact_master=True, tile_rows=tile_rows, release_gradients=True, **kw)
    for step in range(7):
        grad = torch.randn(shape)
        param.main_grad = grad.clone()
        reference_param.main_grad = grad.clone()
        actual.step()
        reference.step()
        # Exactness contract, not widened numerical tolerance.
        assert torch.equal(param, reference_param)
        snapshot = actual.state_dict()
        ref_snapshot = reference.state_dict()
        for key in ("master", "exp_avg", "exp_avg_sq", "basis"):
            a = snapshot["state"][0][key]
            b = ref_snapshot["state"][0][key]
            assert a is b if a is None else torch.equal(a, b)
        assert param.main_grad is None
        if dtype == torch.bfloat16:
            assert isinstance(actual.state[param]["master"], BF16MasterStorage)
            assert actual.state_memory_metrics()["master_bytes"] == param.numel() * 2 + (param.numel() + 7) // 8
        else:
            assert actual.state_memory_metrics()["master_bytes"] == param.numel() * 4
        if step in (1, 3):
            restored_param = torch.nn.Parameter(torch.zeros_like(param))
            restored = GaLoreAdamW([restored_param], lr=0.2)
            restored.load_state_dict(snapshot)
            assert restored.compact_master and restored.release_gradients
            assert restored.tile_rows == actual.tile_rows
            actual, param = restored, restored_param
    actual.offload_state()
    actual.onload_state(torch.device("cpu"))
    assert torch.equal(actual.state_dict()["state"][0]["master"], reference.state_dict()["state"][0]["master"])


@pytest.mark.parametrize("shape", [(32, 16), (16, 32)])
def test_tiled_lift_matches_dense_exactly(shape):
    torch.manual_seed(593)
    m, n = shape
    r = 2
    basis = torch.randn((r, n) if m >= n else (m, r))
    update = torch.randn((m, r) if m >= n else (r, n))
    dense = update @ basis if m >= n else basis @ update
    tiled = torch.cat([update[s : s + 8] @ basis if m >= n else basis[s : s + 8] @ update for s in range(0, m, 8)])
    assert torch.equal(dense, tiled)
