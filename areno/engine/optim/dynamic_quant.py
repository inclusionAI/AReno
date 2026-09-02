"""Reference codebooks for Dettmers et al. block-wise dynamic quantization."""

from __future__ import annotations


def create_dynamic_map(*, signed: bool, max_exponent_bits: int = 7, total_bits: int = 8) -> tuple[float, ...]:
    """Build the paper's sorted dynamic-tree codebook without a torch dependency.

    The construction matches the reference implementation released with
    ``8-Bit Optimizers via Block-wise Quantization``. Signed maps reserve one
    sign bit; unsigned maps reclaim it for an additional fractional bit.
    """

    if total_bits != 8:
        raise ValueError("AReno dynamic optimizer states currently require total_bits=8")
    if max_exponent_bits < 1 or max_exponent_bits >= total_bits:
        raise ValueError("max_exponent_bits must be between 1 and total_bits - 1")

    values: list[float] = []
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    last_index = 0
    for index in range(max_exponent_bits):
        last_index = index
        fraction_items = 2 ** (index + non_sign_bits - max_exponent_bits + (0 if signed else 1)) + 1
        step = 0.9 / (fraction_items - 1)
        means = (0.1 + (item + 0.5) * step for item in range(fraction_items - 1))
        scale = 10 ** (-(max_exponent_bits - 1) + index)
        positive = [scale * mean for mean in means]
        values.extend(positive)
        if signed:
            values.extend(-value for value in positive)

    if additional_items > 0:
        step = 0.9 / additional_items
        means = (0.1 + (item + 0.5) * step for item in range(additional_items))
        scale = 10 ** (-(max_exponent_bits - 1) + last_index)
        positive = [scale * mean for mean in means]
        values.extend(positive)
        if signed:
            values.extend(-value for value in positive)

    values.extend((0.0, 1.0))
    if len(values) != 2**total_bits:
        raise AssertionError(f"dynamic codebook has {len(values)} entries, expected {2**total_bits}")
    return tuple(sorted(values))


SIGNED_DYNAMIC_MAP = create_dynamic_map(signed=True)
UNSIGNED_DYNAMIC_MAP = create_dynamic_map(signed=False)
SIGNED_DYNAMIC_ZERO = SIGNED_DYNAMIC_MAP.index(0.0)
UNSIGNED_DYNAMIC_ZERO = UNSIGNED_DYNAMIC_MAP.index(0.0)


__all__ = [
    "SIGNED_DYNAMIC_MAP",
    "SIGNED_DYNAMIC_ZERO",
    "UNSIGNED_DYNAMIC_MAP",
    "UNSIGNED_DYNAMIC_ZERO",
    "create_dynamic_map",
]
