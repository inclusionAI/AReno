from __future__ import annotations

import pytest
import torch

from areno.accel.kivi_reference import (
    KiviReferenceCache,
    quantize_key_per_channel,
    quantize_value_per_token,
)


def _key_value(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(590)
    key = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
    value = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
    return key, value


def test_quantization_axes_scales_and_saved_metadata() -> None:
    key = torch.tensor([[[[1.0, 10.0], [3.0, 20.0]]]])
    value = key.clone()

    key_block = quantize_key_per_channel(key, bits=2, group_size=2)
    value_block = quantize_value_per_token(value, bits=2, group_size=2)

    assert key_block.codes.shape == (1, 1, 2, 1, 2)
    assert key_block.scales.shape == (1, 1, 2, 1, 1)
    assert value_block.codes.shape == (1, 1, 2, 1, 2)
    assert value_block.scales.shape == (1, 1, 2, 1, 1)
    assert key_block.codes.dtype == torch.uint8
    assert value_block.codes.dtype == torch.uint8
    assert key_block.scales.dtype == torch.float32
    assert value_block.minima.dtype == torch.float32
    torch.testing.assert_close(
        key_block.scales.flatten(), torch.tensor([2.0 / 3.0, 10.0 / 3.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        value_block.scales.flatten(), torch.tensor([3.0, 17.0 / 3.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(key_block.dequantize(), key, atol=1e-6, rtol=0)
    torch.testing.assert_close(value_block.dequantize(), value, atol=1e-6, rtol=0)


def test_constant_groups_reconstruct_without_division_by_zero() -> None:
    key = torch.full((1, 1, 4, 2), -3.25)
    value = torch.full((1, 1, 4, 2), 7.5)

    key_block = quantize_key_per_channel(key, group_size=4)
    value_block = quantize_value_per_token(value, group_size=2)

    assert torch.count_nonzero(key_block.scales) == 0
    assert torch.count_nonzero(value_block.scales) == 0
    assert torch.count_nonzero(key_block.codes) == 0
    assert torch.count_nonzero(value_block.codes) == 0
    torch.testing.assert_close(key_block.dequantize(), key, atol=0, rtol=0)
    torch.testing.assert_close(value_block.dequantize(), value, atol=0, rtol=0)


def test_fp32_quantization_error_is_at_most_half_the_largest_scale() -> None:
    key, value = _key_value((2, 3, 8, 8))
    key_block = quantize_key_per_channel(key, bits=2, group_size=4)
    value_block = quantize_value_per_token(value, bits=2, group_size=4)
    largest_scale = max(key_block.scales.max().item(), value_block.scales.max().item())
    atol = largest_scale / 2 + 1e-6

    key_error = (key_block.dequantize() - key).abs().max().item()
    value_error = (value_block.dequantize() - value).abs().max().item()
    assert key_error <= atol
    assert value_error <= atol
    torch.testing.assert_close(key_block.dequantize(), key, atol=atol, rtol=0)
    torch.testing.assert_close(value_block.dequantize(), value, atol=atol, rtol=0)


def test_bf16_quantized_prefix_has_explicit_error_bound_and_exact_residual() -> None:
    key, value = _key_value((1, 2, 8, 8), torch.bfloat16)
    cache = KiviReferenceCache(bits=2, group_size=4, residual_length=2)
    cache.append(key, value)

    reconstructed_key, reconstructed_value = cache.materialize()
    quantized_prefix = 4
    largest_scale = max(
        max(block.scales.max().item() for block in cache.key_blocks),
        max(block.scales.max().item() for block in cache.value_blocks),
    )
    largest_magnitude = max(
        key.abs().max().item(),
        value.abs().max().item(),
        reconstructed_key.abs().max().item(),
        reconstructed_value.abs().max().item(),
    )
    bf16_rounding = torch.finfo(torch.bfloat16).eps * largest_magnitude
    atol = largest_scale / 2 + bf16_rounding + 1e-6

    torch.testing.assert_close(
        reconstructed_key[:, :, :quantized_prefix], key[:, :, :quantized_prefix], atol=atol, rtol=0
    )
    torch.testing.assert_close(
        reconstructed_value[:, :, :quantized_prefix], value[:, :, :quantized_prefix], atol=atol, rtol=0
    )
    torch.testing.assert_close(reconstructed_key[:, :, quantized_prefix:], key[:, :, quantized_prefix:], atol=0, rtol=0)
    torch.testing.assert_close(
        reconstructed_value[:, :, quantized_prefix:], value[:, :, quantized_prefix:], atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_materialize_preserves_shape_dtype_and_token_order(dtype: torch.dtype) -> None:
    key, value = _key_value((2, 2, 6, 4), dtype)
    cache = KiviReferenceCache(bits=4, group_size=2, residual_length=2)

    cache.append(key, value)
    reconstructed_key, reconstructed_value = cache.materialize()

    assert reconstructed_key.shape == key.shape
    assert reconstructed_value.shape == value.shape
    assert reconstructed_key.dtype == dtype
    assert reconstructed_value.dtype == dtype
    assert cache.num_tokens == cache.quantized_tokens + cache.pending_tokens + cache.residual_tokens
    torch.testing.assert_close(reconstructed_key[:, :, -2:], key[:, :, -2:], atol=0, rtol=0)
    torch.testing.assert_close(reconstructed_value[:, :, -2:], value[:, :, -2:], atol=0, rtol=0)


def test_zero_residual_quantizes_every_complete_group_immediately() -> None:
    key, value = _key_value((1, 1, 3, 2))
    cache = KiviReferenceCache(group_size=1, residual_length=0)

    assert cache.residual_tokens == 0
    cache.append(key, value)

    assert cache.quantized_tokens == 3
    assert cache.pending_tokens == 0
    assert cache.residual_tokens == 0
    assert cache.residual == (None, None)
    assert len(cache.key_blocks) == 3
    assert len(cache.value_blocks) == 3


def test_one_token_residual_switches_only_when_window_overflows() -> None:
    first_key, first_value = _key_value((1, 1, 1, 2))
    second_key, second_value = _key_value((1, 1, 1, 2))
    second_key = second_key + 10
    second_value = second_value - 10
    cache = KiviReferenceCache(group_size=1, residual_length=1)

    cache.append(first_key, first_value)
    assert cache.quantized_tokens == 0
    assert cache.residual_tokens == 1
    torch.testing.assert_close(cache.materialize()[0], first_key, atol=0, rtol=0)

    cache.append(second_key, second_value)
    reconstructed_key, reconstructed_value = cache.materialize()
    assert cache.quantized_tokens == 1
    assert cache.residual_tokens == 1
    torch.testing.assert_close(reconstructed_key[:, :, -1:], second_key, atol=0, rtol=0)
    torch.testing.assert_close(reconstructed_value[:, :, -1:], second_value, atol=0, rtol=0)


def test_full_residual_window_remains_exact_until_next_token() -> None:
    key, value = _key_value((1, 1, 3, 2))
    cache = KiviReferenceCache(group_size=1, residual_length=3)

    cache.append(key, value)
    assert cache.quantized_tokens == 0
    assert cache.residual_tokens == 3
    torch.testing.assert_close(cache.materialize()[0], key, atol=0, rtol=0)
    torch.testing.assert_close(cache.materialize()[1], value, atol=0, rtol=0)

    next_key, next_value = _key_value((1, 1, 1, 2))
    cache.append(next_key, next_value)
    assert cache.quantized_tokens == 1
    assert cache.residual_tokens == 3
    torch.testing.assert_close(cache.materialize()[0][:, :, 1:3], key[:, :, 1:], atol=0, rtol=0)
    torch.testing.assert_close(cache.materialize()[0][:, :, -1:], next_key, atol=0, rtol=0)
    torch.testing.assert_close(cache.materialize()[1][:, :, -1:], next_value, atol=0, rtol=0)


def test_pending_tokens_are_distinct_from_the_residual_window() -> None:
    key, value = _key_value((1, 1, 6, 4))
    cache = KiviReferenceCache(group_size=4, residual_length=2)

    cache.append(key[:, :, :3], value[:, :, :3])
    assert cache.quantized_tokens == 0
    assert cache.pending_tokens == 1
    assert cache.residual_tokens == 2
    torch.testing.assert_close(cache.materialize()[0], key[:, :, :3], atol=0, rtol=0)

    cache.append(key[:, :, 3:], value[:, :, 3:])
    assert cache.quantized_tokens == 4
    assert cache.pending_tokens == 0
    assert cache.residual_tokens == 2
    torch.testing.assert_close(cache.materialize()[0][:, :, -2:], key[:, :, -2:], atol=0, rtol=0)


def test_quantized_blocks_are_independent_of_update_chunking() -> None:
    key, value = _key_value((1, 2, 10, 4))
    one_shot = KiviReferenceCache(group_size=4, residual_length=2)
    incremental = KiviReferenceCache(group_size=4, residual_length=2)

    one_shot.append(key, value)
    start = 0
    for length in (3, 1, 4, 2):
        stop = start + length
        incremental.append(key[:, :, start:stop], value[:, :, start:stop])
        start = stop

    assert incremental.quantized_tokens == one_shot.quantized_tokens == 8
    assert incremental.pending_tokens == one_shot.pending_tokens == 0
    assert incremental.residual_tokens == one_shot.residual_tokens == 2
    for actual_blocks, expected_blocks in (
        (incremental.key_blocks, one_shot.key_blocks),
        (incremental.value_blocks, one_shot.value_blocks),
    ):
        assert len(actual_blocks) == len(expected_blocks)
        for actual, expected in zip(actual_blocks, expected_blocks, strict=True):
            torch.testing.assert_close(actual.codes, expected.codes, atol=0, rtol=0)
            torch.testing.assert_close(actual.scales, expected.scales, atol=0, rtol=0)
            torch.testing.assert_close(actual.minima, expected.minima, atol=0, rtol=0)
    torch.testing.assert_close(incremental.materialize()[0], one_shot.materialize()[0], atol=0, rtol=0)
    torch.testing.assert_close(incremental.materialize()[1], one_shot.materialize()[1], atol=0, rtol=0)


def test_reset_clears_state_and_allows_reuse_with_new_metadata() -> None:
    key, value = _key_value((1, 2, 5, 4), torch.bfloat16)
    cache = KiviReferenceCache(group_size=2, residual_length=1)
    cache.append(key, value)
    assert cache.num_tokens > 0

    cache.reset()

    assert cache.num_tokens == 0
    assert cache.quantized_tokens == 0
    assert cache.pending_tokens == 0
    assert cache.residual_tokens == 0
    assert cache.key_blocks == ()
    assert cache.value_blocks == ()
    assert cache.residual == (None, None)
    with pytest.raises(RuntimeError, match="empty cache"):
        cache.materialize()

    new_key, new_value = _key_value((1, 1, 2, 2), torch.float32)
    cache.append(new_key, new_value)
    reconstructed_key, reconstructed_value = cache.materialize()
    assert reconstructed_key.shape == new_key.shape
    assert reconstructed_key.dtype == torch.float32
    assert cache.num_tokens == 2
    torch.testing.assert_close(reconstructed_key, new_key, atol=0, rtol=0)
    torch.testing.assert_close(reconstructed_value, new_value, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bits": 1}, "bits"),
        ({"bits": 9}, "bits"),
        ({"group_size": 0}, "group_size"),
        ({"residual_length": -1}, "residual_length"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        KiviReferenceCache(**kwargs)


def test_unsupported_attention_type_raises_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="unsupported attention type.*linear"):
        KiviReferenceCache(attention_type="linear")


@pytest.mark.parametrize(
    ("key", "value", "error", "match"),
    [
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), ValueError, "4D"),
        (torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 4), ValueError, "same shape"),
        (torch.zeros(1, 1, 0, 3), torch.zeros(1, 1, 0, 3), ValueError, "at least one token"),
        (torch.zeros(1, 1, 2, 3, dtype=torch.int32), torch.zeros(1, 1, 2, 3), TypeError, "dtype"),
        (torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3, dtype=torch.float16), TypeError, "same dtype"),
    ],
)
def test_invalid_updates_are_rejected(
    key: torch.Tensor, value: torch.Tensor, error: type[Exception], match: str
) -> None:
    cache = KiviReferenceCache(group_size=1, residual_length=1)
    with pytest.raises(error, match=match):
        cache.append(key, value)
    assert cache.num_tokens == 0


def test_nonfinite_and_incompatible_later_updates_are_rejected() -> None:
    key, value = _key_value((1, 1, 2, 4))
    cache = KiviReferenceCache(group_size=1, residual_length=1)
    cache.append(key, value)

    with pytest.raises(ValueError, match="match the existing cache"):
        cache.append(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
    with pytest.raises(TypeError, match="update dtype"):
        cache.append(key[:, :, :1].bfloat16(), value[:, :, :1].bfloat16())

    invalid = key[:, :, :1].clone()
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        cache.append(invalid, value[:, :, :1])
    assert cache.num_tokens == 2


def test_quantizers_reject_a_partial_group() -> None:
    key, value = _key_value((1, 1, 3, 3))
    with pytest.raises(ValueError, match="key tokens.*divisible"):
        quantize_key_per_channel(key, group_size=2)
    with pytest.raises(ValueError, match="value head_dim.*divisible"):
        quantize_value_per_token(value, group_size=2)
