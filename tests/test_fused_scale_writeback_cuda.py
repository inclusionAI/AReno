"""Exact scalar conversion and full deferred compact-writeback regression."""

import pytest
import torch

from areno.engine.optim.galore import GaLoreAdamW
from areno.engine.optim.master_storage import decode_fp32_master

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def assert_bits_equal(first, second):
    assert first.dtype == second.dtype and first.shape == second.shape
    assert torch.equal(
        first.contiguous().reshape(-1).view(torch.uint8),
        second.contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.parametrize("scale", [True, 1, 2, 4, 2**63, 2**64 - 1, 0.1, 1.3, 1e-40, 1e-300, 1e300])
@pytest.mark.parametrize("offset,count", [(0, 10), (8, 1025)])
@pytest.mark.parametrize("alpha", [-0.003, 0.0])
def test_fused_scale_matches_pytorch_scalar_conversion(scale, offset, count, alpha):
    from areno.engine.optim.compact_writeback import compact_writeback

    # Explicit positive and negative zero, tiny values and overflow-producing
    # updates; strided update exercises the existing contiguous-copy contract.
    words = torch.tensor(
        [
            0,
            -2147483648,
            1,
            -2147483647,
            8388607,
            8388608,
            1065353216,
            -1082130432,
            2139095039,
            -8388609,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    source = words.view(torch.float32).repeat((count + 9) // 10)[:count]
    update = torch.stack([source, source], dim=1)[:, 0]
    original = torch.linspace(-0.5, 0.5, offset + count, device="cuda")
    control_model, control_storage = GaLoreAdamW._checked_encode(original)
    candidate_model, candidate_storage = GaLoreAdamW._checked_encode(original)
    control_invalid = torch.zeros((), device="cuda", dtype=torch.int32)
    candidate_invalid = torch.zeros_like(control_invalid)
    separately_scaled = update.clone().mul_(scale)
    compact_writeback(
        control_model,
        control_storage,
        separately_scaled,
        offset,
        0.9999,
        alpha,
        control_invalid,
    )
    compact_writeback(
        candidate_model,
        candidate_storage,
        update,
        offset,
        0.9999,
        alpha,
        candidate_invalid,
        scale=scale,
    )
    for control, candidate in (
        (control_model, candidate_model),
        (control_storage.low_bits, candidate_storage.low_bits),
        (control_storage.round_up_bits, candidate_storage.round_up_bits),
        (control_invalid, candidate_invalid),
        (
            decode_fp32_master(control_model, control_storage),
            decode_fp32_master(candidate_model, candidate_storage),
        ),
    ):
        assert_bits_equal(control, candidate)


def assert_nested_bits(first, second):
    if isinstance(first, torch.Tensor):
        assert_bits_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_bits(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_nested_bits(a, b)
    else:
        assert first == second


def make_optimizer(monkeypatch, initial, scale, enabled, tile_rows):
    monkeypatch.setenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION", "1")
    monkeypatch.setenv("ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK", str(int(enabled)))
    parameters = [torch.nn.Parameter(x.clone()) for x in initial]
    optimizer = GaLoreAdamW(
        parameters,
        lr=1e-5,
        rank=2,
        update_proj_gap=2,
        scale=scale,
        projector_backend="recycled",
        projector_oversampling=4,
        projector_seed=17,
        compact_master=True,
        tile_rows=tile_rows,
        release_gradients=True,
        fused_writeback=True,
        transport_moments=True,
    )
    return parameters, optimizer


def update(parameters, optimizer, gradients):
    optimizer.begin_deferred_projection()
    sum((p.float() * g).sum() for p, g in zip(parameters, gradients, strict=True)).backward()
    optimizer.finish_deferred_backward()
    optimizer.clip_deferred_gradients(optimizer.deferred_grad_norms()["global"], 1)
    optimizer.step()
    assert not optimizer._deferred_pending and not optimizer._deferred_ready


@pytest.mark.parametrize("scale", [1, 1.3, 1e-40, 1e-300, 2**64 - 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tile_rows", [0, 8])
def test_fused_scalar_full_pipeline_and_cross_flag_resume(monkeypatch, scale, dtype, tile_rows):
    generator = torch.Generator().manual_seed(593)
    initial = [
        torch.randn(13, 9, generator=generator).to(device="cuda", dtype=dtype),
        torch.randn(5, generator=generator).to("cuda"),
    ]
    control_parameters, control = make_optimizer(monkeypatch, initial, scale, False, tile_rows)
    parameters, optimizer = make_optimizer(monkeypatch, initial, scale, True, tile_rows)
    resumed = None
    for index in range(4):
        gradients = [torch.randn(x.shape, generator=generator).to("cuda") for x in initial]
        update(control_parameters, control, gradients)
        update(parameters, optimizer, gradients)
        assert_nested_bits(control.state_dict(), optimizer.state_dict())
        assert_nested_bits(control_parameters, parameters)
        if resumed is not None:
            rp, ro = resumed
            update(rp, ro, gradients)
            assert_nested_bits(optimizer.state_dict(), ro.state_dict())
            assert_nested_bits(parameters, rp)
        if index == 1:
            rp, ro = make_optimizer(
                monkeypatch,
                [torch.zeros_like(p) for p in parameters],
                scale,
                False,
                tile_rows,
            )
            ro.load_state_dict(optimizer.state_dict())
            resumed = rp, ro
    assert optimizer.state[parameters[0]]["projector_refreshes"] == 2
    assert optimizer.state[parameters[0]]["transport_count"] == 1


@pytest.mark.parametrize("scale", [2**64, 2**100, torch.tensor(1.3)])
def test_non_kernel_scalars_preserve_original_fallback(monkeypatch, scale):
    initial = [torch.ones(13, 9, device="cuda", dtype=torch.bfloat16)]
    control_parameters, control = make_optimizer(monkeypatch, initial, scale, False, 8)
    parameters, optimizer = make_optimizer(monkeypatch, initial, scale, True, 8)
    assert not optimizer._fused_scale_writeback
    failures = []
    gradients = [torch.ones_like(initial[0], dtype=torch.float32)]
    for ps, opt in ((control_parameters, control), (parameters, optimizer)):
        try:
            update(ps, opt, gradients)
        except OverflowError as exc:
            failures.append((type(exc), str(exc)))
            opt.abort_deferred_projection()
        else:
            failures.append(None)
    assert failures[0] == failures[1]
    assert_nested_bits(control.state_dict(), optimizer.state_dict())
    assert_nested_bits(control_parameters, parameters)
