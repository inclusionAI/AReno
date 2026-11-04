from __future__ import annotations

import gc
import math

import pytest
import torch

from areno.accel.activation_compress_reference import (
    CompressedActivation,
    compress_activation,
    restore_activation,
)


def _assert_bit_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    if actual.numel():
        assert torch.equal(
            actual.detach().contiguous().view(-1).view(torch.uint8),
            expected.detach().contiguous().view(-1).view(torch.uint8),
        )


def _error_bound(source: torch.Tensor, payload: CompressedActivation) -> float:
    """Independent conservative bound, including arithmetic and output casting."""
    if not source.numel():
        return 0.0
    compute_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    compute_info = torch.finfo(compute_dtype)
    output_info = torch.finfo(source.dtype)
    max_source = source.detach().abs().max().item()
    max_scale = payload.scales.max().item()
    cast_allowance = 0.0
    if source.dtype in (torch.float16, torch.bfloat16):
        cast_allowance = output_info.eps * max_source + output_info.tiny
    return max_scale / 2 + 8 * compute_info.eps * max_source + cast_allowance + compute_info.tiny


def _assert_bounded_restore(source: torch.Tensor, payload: CompressedActivation) -> torch.Tensor:
    assert not payload.used_fallback
    restored = restore_activation(payload)
    assert restored.shape == source.shape
    assert restored.dtype == source.dtype
    assert restored.device == source.device
    if source.numel():
        observed = (restored.to(torch.float64) - source.to(torch.float64)).abs().max().item()
    else:
        observed = 0.0
    bound = _error_bound(source, payload)
    assert observed <= bound, f"observed error {observed} exceeds independent bound {bound}"
    assert 0.0 <= payload.max_abs_error <= bound
    return restored


@pytest.mark.parametrize("bit_width", (2, 4, 8))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((), (3,), (2, 3), (2, 3, 5), (2, 0, 3)))
def test_round_trip_preserves_layout_metadata_and_error_bound(shape, dtype, bit_width) -> None:
    generator = torch.Generator().manual_seed(591)
    random_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    source = torch.randn(shape, generator=generator, dtype=random_dtype).to(dtype)

    payload = compress_activation(source, bit_width=bit_width, block_size=7, chunk_size=11)

    _assert_bounded_restore(source, payload)
    assert payload.shape == shape
    assert payload.dtype == dtype


@pytest.mark.parametrize("shape", ((), (0,), (2, 0, 3), (4, 5)))
def test_zero_and_empty_round_trip_is_bit_exact(shape) -> None:
    source = torch.zeros(shape, dtype=torch.float32)

    payload = compress_activation(source, block_size=5, chunk_size=7)
    restored = payload.restore()

    _assert_bit_exact(restored, source)
    assert payload.max_abs_error == 0.0


def test_noncontiguous_input_restores_logical_values() -> None:
    source = torch.linspace(-2.0, 2.0, 42).view(6, 7).transpose(0, 1)
    assert not source.is_contiguous()

    payload = compress_activation(source, bit_width=4, block_size=6, chunk_size=10)

    _assert_bounded_restore(source, payload)


def test_subnormal_values_respect_scale_and_error_bound() -> None:
    subnormal = torch.finfo(torch.float32).tiny * 0.5
    source = torch.tensor([subnormal, -subnormal, 0.0], dtype=torch.float32)

    payload = compress_activation(source, bit_width=4, block_size=3)

    assert payload.scales.max().item() == torch.finfo(torch.float32).tiny
    _assert_bounded_restore(source, payload)


def test_error_budget_fallback_is_bit_exact_and_owns_source_copy() -> None:
    source = torch.tensor([-3.0, -1.0, -0.5, 0.25, 1.0, 2.0, 4.0])
    snapshot = source.clone()

    payload = compress_activation(source, bit_width=2, block_size=3, max_abs_error=0.0)
    source.fill_(99.0)

    assert payload.used_fallback
    assert payload.fallback_reason == "error_budget_exceeded"
    assert payload.packed_nbytes == payload.scale_nbytes == 0
    assert payload.raw_nbytes == payload.source_nbytes == snapshot.numel() * snapshot.element_size()
    assert payload.storage_nbytes == payload.raw_nbytes
    assert payload.reconstruction_workspace_nbytes == 0
    _assert_bit_exact(payload.restore(), snapshot)


def test_nonfinite_input_falls_back_without_losing_any_bits() -> None:
    source = torch.tensor([0.0, math.nan, math.inf, -math.inf], dtype=torch.float32)

    payload = compress_activation(source)

    assert payload.used_fallback
    assert payload.fallback_reason == "nonfinite_input"
    assert math.isnan(payload.max_abs_error)
    _assert_bit_exact(payload.restore(), source)


def test_error_budget_comparison_uses_strict_greater_than() -> None:
    source = torch.linspace(-1.0, 1.0, 11)
    baseline = compress_activation(source, bit_width=2, block_size=4)
    assert baseline.max_abs_error > 0.0

    at_budget = compress_activation(source, bit_width=2, block_size=4, max_abs_error=baseline.max_abs_error)
    below_budget = compress_activation(
        source,
        bit_width=2,
        block_size=4,
        max_abs_error=math.nextafter(baseline.max_abs_error, -math.inf),
    )

    assert not at_budget.used_fallback
    assert below_budget.used_fallback
    assert below_budget.fallback_reason == "error_budget_exceeded"


@pytest.mark.parametrize("bit_width", (2, 4, 8))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_storage_and_workspace_accounting_match_independent_formulas(dtype, bit_width) -> None:
    source = torch.linspace(-1.0, 1.0, 30, dtype=dtype).view(2, 3, 5)
    block_size = 7

    payload = compress_activation(source, bit_width=bit_width, block_size=block_size, chunk_size=11)

    numel = 30
    scale_count = (numel + block_size - 1) // block_size
    scale_element_size = 8 if dtype == torch.float64 else 4
    assert payload.numel == numel
    assert payload.scale_count == scale_count
    assert payload.packed_nbytes == (numel * bit_width + 7) // 8
    assert payload.scale_nbytes == scale_count * scale_element_size
    assert payload.raw_nbytes == 0
    assert payload.source_nbytes == numel * source.element_size()
    assert payload.storage_nbytes == payload.packed_nbytes + payload.scale_nbytes
    assert payload.reconstruction_workspace_nbytes == numel + scale_count * block_size * scale_element_size


def test_chunk_size_does_not_change_codes_scales_or_restoration() -> None:
    source = torch.linspace(-2.0, 2.0, 37)
    payloads = [
        compress_activation(source, bit_width=4, block_size=4, chunk_size=chunk_size)
        for chunk_size in (1, 3, 4, 5, 7, 8, 17, 1024)
    ]
    baseline = payloads[0]
    expected = baseline.restore()

    for payload in payloads[1:]:
        assert torch.equal(payload.packed, baseline.packed)
        assert torch.equal(payload.scales, baseline.scales)
        assert payload.scale_count == baseline.scale_count
        assert payload.max_abs_error == baseline.max_abs_error
        _assert_bit_exact(payload.restore(), expected)


@pytest.mark.parametrize("block_size", (1, 2, 3, 8, 64))
def test_block_size_changes_remain_within_their_independent_bounds(block_size) -> None:
    generator = torch.Generator().manual_seed(1591)
    source = torch.randn(5, 7, generator=generator)

    payload = compress_activation(source, bit_width=4, block_size=block_size, chunk_size=block_size * 2 + 1)

    _assert_bounded_restore(source, payload)
    assert payload.scale_count == (source.numel() + block_size - 1) // block_size


def test_checkpoint_release_and_interleaved_restore_do_not_mix_state() -> None:
    first_source = torch.linspace(-2.0, 2.0, 30).view(6, 5)
    second_source = torch.linspace(3.0, -3.0, 30).view(6, 5)
    first = compress_activation(first_source, bit_width=4, block_size=3, chunk_size=5)
    second = compress_activation(second_source, bit_width=2, block_size=4, chunk_size=9)
    expected_first = first.restore()
    expected_second = second.restore()
    assert not torch.equal(expected_first, expected_second)

    first_source.fill_(100.0)
    second_source.fill_(-100.0)
    del first_source, second_source
    gc.collect()

    for _ in range(2):
        _assert_bit_exact(restore_activation(second), expected_second)
        _assert_bit_exact(restore_activation(first), expected_first)

    restored_first = first.restore()
    restored_first.zero_()
    _assert_bit_exact(first.restore(), expected_first)
    _assert_bit_exact(second.restore(), expected_second)


@pytest.mark.parametrize(
    ("overrides", "exception"),
    (
        ({"bit_width": 3}, ValueError),
        ({"bit_width": 4.0}, TypeError),
        ({"block_size": 0}, ValueError),
        ({"chunk_size": 0}, ValueError),
        ({"max_abs_error": -1.0}, ValueError),
        ({"max_abs_error": math.nan}, ValueError),
    ),
)
def test_invalid_compression_configuration_is_rejected(overrides, exception) -> None:
    with pytest.raises(exception):
        compress_activation(torch.ones(4), **overrides)


def test_nonfloating_activation_is_rejected() -> None:
    with pytest.raises(TypeError, match="unsupported activation dtype"):
        compress_activation(torch.ones(4, dtype=torch.int64))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable in the torch 2.7.1+cpu environment; enable this check on a GPU host",
)
def test_cuda_round_trip_when_device_is_available() -> None:
    source = torch.linspace(-1.0, 1.0, 35, dtype=torch.float16, device="cuda")

    payload = compress_activation(source, bit_width=4, block_size=7, chunk_size=13)

    _assert_bounded_restore(source, payload)
    assert payload.device.type == "cuda"
