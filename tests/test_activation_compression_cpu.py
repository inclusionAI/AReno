"""Saved-activation compression: codes, accounting, autograd hooks and calibration."""

import gc
import math
import weakref

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from areno.adapters.config import LoraConfig
from areno.engine.config import EngineConfig, ModelConfig, RuntimeConfig
from areno.engine.runtime.activation_compression import (
    GROUP_SIZE,
    PLAN_FILE,
    UNCOMPRESSED,
    ActivationCompressor,
    PackedActivation,
    dequantize,
    outlier_count,
    plan_bits,
    quantize,
    relative_error,
    saved_uncompressed,
    stored_bytes,
    uniform_noise,
)


def _reference_lowbias32(h: int) -> int:
    # lowbias32 (C. Wellons) in exact Python integers.
    h ^= h >> 16
    h = (h * 0x7FEB352D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846CA68B) & 0xFFFFFFFF
    return h ^ (h >> 16)


def _reference_uniform(index: int, seed: int) -> float:
    return (_reference_lowbias32(_reference_lowbias32(index & 0xFFFFFFFF) ^ seed) >> 8) / float(1 << 24)


def _reference_quantize(x: torch.Tensor, bits: int, *, stochastic: bool, seed: int):
    """Element-by-element NumPy FP32 reference for codes, minima, steps and outliers."""

    values = x.float().numpy().reshape(-1).astype(np.float32)
    channels = x.shape[-1]
    rows = values.reshape(-1, channels)
    peak = np.maximum(rows.max(axis=0), -rows.min(axis=0))
    outliers = np.argsort(-peak, kind="stable")[: outlier_count(channels)]
    kept = np.zeros(channels, dtype=bool)
    kept[outliers] = True
    levels = np.float32((1 << bits) - 1)
    groups = -(-values.size // GROUP_SIZE)
    codes = np.zeros(groups * GROUP_SIZE, dtype=np.int64)
    minima = np.zeros(groups, dtype=np.float32)
    steps = np.zeros(groups, dtype=np.float32)
    for group in range(groups):
        members = [
            i for i in range(group * GROUP_SIZE, min((group + 1) * GROUP_SIZE, values.size)) if not kept[i % channels]
        ]
        low = np.float32(min(values[i] for i in members)) if members else np.float32(0.0)
        high = np.float32(max(values[i] for i in members)) if members else np.float32(0.0)
        step = np.float32(np.float32(high - low) / levels)
        step = step if step > 0 else np.float32(1.0)
        minima[group], steps[group] = low, step
        for i in members:
            q = np.float32(np.float32(values[i] - low) / step)
            u = np.float32(_reference_uniform(i, seed)) if stochastic else np.float32(0.5)
            codes[i] = min(max(int(np.floor(np.float32(q + u))), 0), int(levels))
    return codes, minima, steps, outliers


def _unpack_codes(packed: PackedActivation) -> np.ndarray:
    per_byte = 8 // packed.bits
    data = packed.codes.numpy().astype(np.int64)
    return np.stack([(data >> (packed.bits * j)) & ((1 << packed.bits) - 1) for j in range(per_byte)], 1).reshape(-1)


def _distinct(shape, dtype, seed=0):
    # Designated outlier channels with distinct magnitudes keep the ranking free of ties.
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=generator) * 2 + 0.3
    for rank, channel in enumerate((1, 7, 50, 90)):
        if channel < shape[-1]:
            x[..., channel] = (40.0 - 5.0 * rank) * torch.sign(x[..., channel])
    return x.to(dtype)


@pytest.mark.parametrize("bits", [8, 4, 2])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize(
    "shape,dtype", [((6, 130), torch.bfloat16), ((3, 2, 96), torch.float16), ((4, 256), torch.float32)]
)
def test_quantize_matches_independent_reference_cpu(bits, stochastic, shape, dtype):
    x = _distinct(shape, dtype, seed=bits)
    packed = quantize(x, bits, stochastic=stochastic, seed=1234, use_kernel=False)
    codes, minima, steps, outliers = _reference_quantize(x, bits, stochastic=stochastic, seed=1234)

    np.testing.assert_array_equal(_unpack_codes(packed), codes)
    np.testing.assert_array_equal(packed.minimum.numpy(), minima)
    np.testing.assert_array_equal(packed.scale.numpy(), steps)
    np.testing.assert_array_equal(np.sort(packed.outlier_index.numpy()), np.sort(outliers))

    restored = dequantize(packed, use_kernel=False)
    assert restored.shape == x.shape and restored.dtype == x.dtype and restored.is_contiguous()
    group = np.arange(codes.size) // GROUP_SIZE
    expected = (codes.astype(np.float32) * steps[group]).astype(np.float32) + minima[group]
    expected = torch.from_numpy(expected[: x.numel()].astype(np.float32)).view(-1, shape[-1])
    expected[:, torch.from_numpy(outliers)] = x.float().reshape(-1, shape[-1])[:, torch.from_numpy(outliers)]
    torch.testing.assert_close(restored, expected.view(shape).to(dtype), atol=0, rtol=0)


def test_uniform_noise_matches_integer_reference_cpu():
    index = torch.tensor([0, 1, 7, 65_535, 2**31 - 1, 2**31 + 5, 2**40 + 3], dtype=torch.int64)
    for seed in (0, 1, 2**31 - 1):
        got = uniform_noise(index, seed).tolist()
        assert got == [_reference_uniform(int(i), seed) for i in index]


@pytest.mark.parametrize("bits", [8, 4, 2])
@pytest.mark.parametrize("stochastic", [False, True])
def test_reconstruction_error_is_bounded_by_the_group_step_cpu(bits, stochastic):
    x = _distinct((33, 192), torch.float32, seed=3)
    packed = quantize(x, bits, stochastic=stochastic, seed=5, use_kernel=False)
    restored = dequantize(packed, use_kernel=False)
    step = packed.scale.repeat_interleave(GROUP_SIZE)[: x.numel()].view_as(x)
    bound = step * (1.0 if stochastic else 0.5) + 1e-6 * x.abs()
    assert bool(((restored - x).abs() <= bound).all())
    outliers = packed.outlier_index
    assert torch.equal(restored[:, outliers], x[:, outliers])


def test_stochastic_rounding_is_unbiased_cpu():
    x = _distinct((4, 256), torch.float32, seed=9)
    total = torch.zeros_like(x)
    draws = 400
    for seed in range(draws):
        total += dequantize(quantize(x, 2, stochastic=True, seed=seed, use_kernel=False), use_kernel=False)
    step = quantize(x, 2, stochastic=True, seed=0, use_kernel=False).scale
    bias = (total / draws - x).reshape(-1, GROUP_SIZE)
    # Per-draw error is at most one step, so the mean of 400 draws is within ~step / 40.
    assert float((bias.abs().amax(dim=1) / step).max()) < 0.25
    assert abs(float(bias.mean())) < 2e-3 * float(step.mean())


def test_noise_is_uniform_and_independent_across_seeds_cpu():
    index = torch.arange(1024, dtype=torch.int64)
    draws = torch.stack([uniform_noise(index, seed) for seed in range(400)])
    # 409,600 draws: the mean of U[0, 1) is 0.5 +- 0.00045 and each element's
    # 400-draw mean has standard deviation 1 / sqrt(12 * 400) = 0.0144.
    assert abs(float(draws.mean()) - 0.5) < 0.003
    assert 0.012 < float(draws.mean(dim=0).std()) < 0.017
    assert float(torch.corrcoef(draws[:, :64].T).fill_diagonal_(0).abs().max()) < 0.25


def test_stored_bytes_counts_codes_scales_and_outlier_channels_cpu():
    x = _distinct((70, 1024), torch.bfloat16)
    for bits in (8, 4, 2):
        packed = quantize(x, bits, use_kernel=False)
        total = packed.packed_bytes + packed.scale_bytes + packed.outlier_bytes
        assert total == stored_bytes(x.numel(), x.element_size(), 1024, bits)
        assert packed.outliers.shape == (70, outlier_count(1024)) == (70, 16)
    assert stored_bytes(x.numel(), 2, 1024, UNCOMPRESSED) == x.numel() * 2
    assert stored_bytes(x.numel(), 2, 1024, 8) < x.numel() * 2


def test_quantize_rejects_unsupported_inputs_cpu():
    with pytest.raises(ValueError, match="bits"):
        quantize(torch.zeros(4, 64), 3)
    with pytest.raises(ValueError, match="contiguous"):
        quantize(torch.zeros(64, 4).t(), 8)
    with pytest.raises(ValueError, match="contiguous"):
        quantize(torch.zeros(64, dtype=torch.int32).view(4, 16), 8)


class _Block(nn.Module):
    def __init__(self, width: int = 256):
        super().__init__()
        self.up = nn.Linear(width, 2 * width)
        self.down = nn.Linear(2 * width, width)
        self.embed = nn.Embedding(16, width)
        self.register_buffer("offset", torch.zeros(1024, width))

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        squared = x * x  # saves x twice: one deduplicated slot
        hidden = F.silu(self.up(x))
        return squared + self.down(hidden) + self.embed(ids) + self.offset[: x.shape[0]] * x


def _block_grads(block, x, ids, compressor=None, plan="current"):
    block.zero_grad(set_to_none=True)
    if compressor is None:
        out = block(x, ids)
    else:
        with compressor.capture(plan):
            out = block(x, ids)
    out.square().mean().backward()
    return [p.grad.detach().clone() for p in block.parameters()]


def _block_inputs(rows=512, width=256, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, width, generator=generator).requires_grad_()
    return x, torch.randint(0, 16, (rows,), generator=generator)


def test_capture_records_only_eligible_activations_cpu():
    torch.manual_seed(0)
    block = _Block()
    x, ids = _block_inputs()
    compressor = ActivationCompressor(block)
    _block_grads(block, x, ids, compressor, plan=None)
    capture = compressor.last_capture
    # x (saved twice by the product and once by `up`), the `up` output (saved by SiLU)
    # and the SiLU output (saved by `down`); weight views, the buffer slice and the
    # integer ids are never slots.
    assert capture.signatures == [("torch.float32", 2, 256), ("torch.float32", 2, 512), ("torch.float32", 2, 512)]
    assert capture.stats.slots == 3 and capture.stats.compressed_slots == 0
    assert capture.stats.saved_bytes == capture.stats.kept_bytes == 512 * (256 + 512 + 512) * 4


def test_uncompressed_plan_is_bit_identical_and_compressed_plan_is_close_cpu():
    torch.manual_seed(0)
    block = _Block()
    x, ids = _block_inputs()
    compressor = ActivationCompressor(block)
    exact = _block_grads(block, x, ids)
    _block_grads(block, x, ids, compressor, plan=None)
    compressor.signatures = list(compressor.last_capture.signatures)
    kept = _block_grads(block, x, ids, compressor, plan=[UNCOMPRESSED] * 3)
    assert all(torch.equal(a, b) for a, b in zip(kept, exact))
    with pytest.raises(ValueError, match="one entry per recorded slot"):
        _block_grads(block, x, ids, compressor, plan=[8, 8])
    eight = _block_grads(block, x, ids, compressor, plan=[8, 8, 8])
    assert compressor.last_capture.stats.compressed_slots == 3
    assert 0 < relative_error(eight, exact) < 0.02
    two = _block_grads(block, x, ids, compressor, plan=[2, 2, 2])
    assert relative_error(two, exact) > relative_error(eight, exact)


def test_compressed_originals_and_packs_are_released_cpu():
    torch.manual_seed(0)
    block = _Block()
    x, ids = _block_inputs()
    compressor = ActivationCompressor(block)
    _block_grads(block, x, ids, compressor, plan=None)
    compressor.signatures = list(compressor.last_capture.signatures)
    compressor.plan = [8, 8, 8]
    originals, codes = [], []
    with compressor.capture() as state:
        pack = state.pack

        def tracking_pack(tensor):
            handle = pack(tensor)
            if isinstance(handle, PackedActivation) and tensor is not x:
                originals.append(weakref.ref(tensor))
                codes.append(weakref.ref(handle.codes))
            return handle

        def unpack(handle):
            return dequantize(handle) if isinstance(handle, PackedActivation) else handle

        with torch.autograd.graph.saved_tensors_hooks(tracking_pack, unpack):
            out = block(x, ids)
    gc.collect()
    assert len(originals) == 2 and all(ref() is None for ref in originals)
    assert all(ref() is not None for ref in codes)
    out.square().mean().backward()
    del out
    gc.collect()
    assert all(ref() is None for ref in codes)


def test_checkpoint_boundary_is_compressed_and_recomputed_cpu():
    torch.manual_seed(0)
    block = _Block()
    x, ids = _block_inputs()
    x.requires_grad_()
    compressor = ActivationCompressor(block)

    def grads(plan):
        block.zero_grad(set_to_none=True)
        x.grad = None
        with compressor.capture(plan):
            out = checkpoint(block, x * 1.0, ids, use_reentrant=False)
        out.square().mean().backward()
        return [x.grad.clone(), *(p.grad.clone() for p in block.parameters())]

    exact = grads(None)
    # Only the checkpoint input is saved outside the recomputed region.
    assert compressor.last_capture.signatures == [("torch.float32", 2, 256)]
    compressor.signatures = list(compressor.last_capture.signatures)
    compressed = grads([8])
    assert compressor.last_capture.stats.compressed_slots == 1
    assert 0 < relative_error(compressed, exact) < 0.02


def test_saved_uncompressed_context_creates_no_slots_cpu():
    torch.manual_seed(0)
    block = _Block()
    x, ids = _block_inputs()
    compressor = ActivationCompressor(block)
    with compressor.capture(None):
        with saved_uncompressed():
            out = block(x, ids)
    assert compressor.last_capture.stats.slots == 0
    out.sum().backward()


class _Aliasing(nn.Module):
    """Saves views of one storage both uncompressed (like attention) and as slots."""

    def __init__(self, width: int = 256, uncompressed_first: bool = True):
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.uncompressed_first = uncompressed_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        if self.uncompressed_first:
            with saved_uncompressed():
                a = h.sin()  # saves h
            b = F.silu(h.view_as(h))  # saves a view of the same storage
        else:
            b = F.silu(h)
            with saved_uncompressed():
                a = h.sin()
        return a + b


def test_aliases_of_uncompressed_saves_are_never_compressed_cpu():
    torch.manual_seed(0)
    x = torch.randn(512, 256, requires_grad=True)
    # An alias saved after the uncompressed save is not even a slot ...
    model = _Aliasing(uncompressed_first=True)
    compressor = ActivationCompressor(model)
    with compressor.capture(None):
        model(x).sum().backward()
    assert compressor.last_capture.signatures == [("torch.float32", 2, 256)]  # only the input x
    # ... and a slot saved before one is pinned uncompressed by calibration.
    model = _Aliasing(uncompressed_first=False)
    compressor = ActivationCompressor(model, error_budget=10.0)

    def run(plan):
        model.zero_grad(set_to_none=True)
        with compressor.capture(plan):
            model(x).sum().backward()
        return [p.grad.clone() for p in model.parameters()]

    compressor.calibrate(run, run)
    assert compressor.calibration["pinned_slots"] == [1]
    assert compressor.plan[1] == UNCOMPRESSED and compressor.plan[0] != UNCOMPRESSED


def test_plan_bits_prefers_least_error_per_byte_within_target_cpu():
    slots = [(1 << 20, 2, 1024)] * 3
    sensitivity = [{8: 1e-6, 4: 1e-4, 2: 1e-2}, {8: 5e-6, 4: 2e-5}, {8: 2e-3}]
    assert plan_bits(sensitivity, slots, 0.0) == [UNCOMPRESSED] * 3
    assert plan_bits(sensitivity, slots, 1e-5) == [8, 8, UNCOMPRESSED]
    assert plan_bits(sensitivity, slots, 1.5e-4) == [4, 4, UNCOMPRESSED]
    # Slot 1 has no 2-bit estimate (pruned), so it can never go below 4 bits.
    assert plan_bits(sensitivity, slots, 1.0) == [2, 4, 8]
    assert plan_bits(sensitivity, slots, 1.0, bits=(8,)) == [8, 8, 8]


def _calibration(block, budget, seed=0):
    x, ids = _block_inputs(seed=seed)
    held_x, held_ids = _block_inputs(seed=seed + 1)
    compressor = ActivationCompressor(block, error_budget=budget)

    def runner(inputs, ids_):
        def run(plan):
            return _block_grads(block, inputs, ids_, compressor, plan=plan)

        return run

    metrics = compressor.calibrate(runner(x, ids), runner(held_x, held_ids))
    return compressor, metrics


def test_calibration_compresses_within_budget_on_both_batches_cpu():
    torch.manual_seed(0)
    block = _Block()
    compressor, metrics = _calibration(block, budget=0.02)
    c = compressor.calibration
    assert any(bits != UNCOMPRESSED for bits in compressor.plan)
    assert c["calibration_error"] <= 0.02 and c["held_out_error"] <= 0.02
    assert metrics["actc_held_out_error"] == c["held_out_error"]
    assert metrics["actc_plan_stored_fraction"] < 1.0
    assert c["noise_floor"] < 1e-6  # CPU backward is deterministic
    assert len(c["sensitivity"]) == c["slots"] == 3


def test_calibration_bisects_after_an_over_budget_plan_cpu():
    # Synthetic joint errors: 8-bit slots behave like BF16 rounding noise (one
    # shared contribution, however many are compressed) while 4-bit slots add
    # up, so rescaling from the first plan's error overshoots.
    torch.manual_seed(0)
    block = nn.Sequential(*(nn.Linear(256, 256) for _ in range(6)))
    x = torch.randn(512, 256, requires_grad=True)
    compressor = ActivationCompressor(block, error_budget=0.01)

    def run(plan):
        block.zero_grad(set_to_none=True)
        with compressor.capture(plan):
            block(x).sum().backward()
        plan = compressor.plan if isinstance(plan, str) else plan
        squared = 0.0
        if plan is not None and any(bits != UNCOMPRESSED for bits in plan):
            squared = 2.0e-5 + sum(1.4e-5 for bits in plan if bits == 4) + sum(1.0 for bits in plan if bits == 2)
        reference = torch.ones(1000)
        return [reference + math.sqrt(squared)]

    compressor.calibrate(run, run)
    c = compressor.calibration
    errors = [max(r["calibration_error"], r["held_out_error"]) for r in c["rounds"]]
    assert any(error > 0.01 for error in errors)  # the rescale overshot ...
    assert 0.0072 <= c["calibration_error"] <= 0.01  # ... and the bracket found the window
    assert compressor.plan.count(4) == 4 and compressor.plan.count(8) == 2
    assert len(c["rounds"]) <= 6


def test_calibration_falls_back_to_uncompressed_when_nothing_fits_cpu():
    torch.manual_seed(0)
    block = _Block()
    compressor, metrics = _calibration(block, budget=1e-7)
    assert compressor.plan == [UNCOMPRESSED] * 3
    assert metrics["actc_uncompressed_fallback"] == 1.0
    assert metrics["actc_plan_stored_fraction"] == 1.0


def test_check_keeps_a_plan_within_budget_and_recalibrates_one_over_it_cpu():
    torch.manual_seed(0)
    block = _Block()
    compressor, _ = _calibration(block, budget=0.02)
    plan = list(compressor.plan)
    x, ids = _block_inputs(seed=7)
    held_x, held_ids = _block_inputs(seed=8)

    def run(plan_):
        return _block_grads(block, x, ids, compressor, plan=plan_)

    def held_out(plan_):
        return _block_grads(block, held_x, held_ids, compressor, plan=plan_)

    metrics = compressor.check(run, held_out)
    assert 0 < metrics["actc_check_error"] <= 0.02 and "actc_recalibrated" not in metrics
    assert compressor.plan == plan
    compressor.error_budget = 1e-7  # the same plan is now over budget
    metrics = compressor.check(run, held_out)
    assert metrics["actc_recalibrated"] == 1.0 and metrics["actc_uncompressed_fallback"] == 1.0
    assert compressor.plan == [UNCOMPRESSED] * 3


def test_plan_state_round_trip_and_settings_check_cpu(tmp_path):
    torch.manual_seed(0)
    block = _Block()
    compressor, _ = _calibration(block, budget=0.02)
    compressor.save(tmp_path)
    assert (tmp_path / PLAN_FILE).is_file()
    restored = ActivationCompressor(block, error_budget=0.02)
    assert restored.load(tmp_path)
    assert restored.plan == compressor.plan and restored.signatures == compressor.signatures
    assert not ActivationCompressor(block, error_budget=0.05).load(tmp_path)
    assert not ActivationCompressor(block, error_budget=0.02, bits=(8,)).load(tmp_path)
    assert not ActivationCompressor(block).load(tmp_path / "missing")


def test_plan_follows_signatures_when_other_slots_disappear_cpu():
    torch.manual_seed(0)
    block = _Block()
    x, ids = _block_inputs()
    compressor = ActivationCompressor(block)
    # Calibrated with an extra (dropped) slot of another shape at position 0;
    # the remaining slots keep their bits by (signature, occurrence).
    compressor.signatures = [("torch.float32", 2, 7), ("torch.float32", 2, 256), ("torch.float32", 2, 512)] + [
        ("torch.float32", 2, 512)
    ]
    compressor.plan = [8, UNCOMPRESSED, 4, 8]
    _block_grads(block, x, ids, compressor)
    stats = compressor.last_capture.stats
    assert stats.compressed_slots == 2 and stats.plan_misses == 0
    assert stats.packed_bytes == 512 * 512 * (4 + 8) // 8


def test_plan_miss_keeps_unexpected_slots_uncompressed_cpu():
    torch.manual_seed(0)
    block = _Block()
    x, ids = _block_inputs()
    compressor = ActivationCompressor(block)
    compressor.plan = [8]
    compressor.signatures = [("torch.float32", 2, 999)]
    _block_grads(block, x, ids, compressor)
    stats = compressor.last_capture.stats
    assert stats.compressed_slots == 0 and stats.plan_misses == 3


def test_runtime_config_validates_activation_compression_cpu():
    assert RuntimeConfig().activation_compression == "none"
    runtime = RuntimeConfig(activation_compression="adaptive", activation_compression_bits=[8, 4])
    assert runtime.activation_compression_bits == (8, 4)
    for kwargs, message in (
        ({"activation_compression": "int8"}, "activation_compression must"),
        ({"activation_compression_error_budget": 0.0}, "error_budget"),
        ({"activation_compression_bits": (8, 3)}, "bits"),
        ({"activation_compression_check_interval": -1}, "interval"),
    ):
        with pytest.raises(ValueError, match=message):
            RuntimeConfig(**kwargs)


def test_engine_config_limits_activation_compression_to_one_dense_gpu_cpu(monkeypatch):
    # Validate layouts without the visible-device check of GPU hosts.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = {"activation_compression": "adaptive"}
    dense = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32)
    EngineConfig(model=dense, tp_size=1, devices=[0], runtime=RuntimeConfig(**runtime))
    with pytest.raises(ValueError, match="one CUDA training GPU"):
        EngineConfig(model=dense, tp_size=2, devices=[0, 1], runtime=RuntimeConfig(**runtime))
    with pytest.raises(ValueError, match="one CUDA training GPU"):
        EngineConfig(model=dense, tp_size=1, devices=[0, 1], runtime=RuntimeConfig(**runtime))
    with pytest.raises(ValueError, match="full-parameter"):
        EngineConfig(model=dense, devices=[0], runtime=RuntimeConfig(**runtime), lora=LoraConfig(rank=4, alpha=8))
    moe = ModelConfig(num_attention_heads=4, num_key_value_heads=4, intermediate_size=16, vocab_size=32, num_experts=4)
    with pytest.raises(ValueError, match="dense"):
        EngineConfig(model=moe, devices=[0], runtime=RuntimeConfig(**runtime))
    # Rollout partitions share the runtime config but never compress.
    EngineConfig(model=dense, tp_size=1, devices=[0, 1], runtime=RuntimeConfig(**runtime), role="rollout")


def test_trainer_configs_forward_activation_compression_to_the_runtime_cpu():
    from areno.api.trainer_config import PolicyTrainerConfig, TrainerConfig

    options = dict(
        ckpt="model",
        dataset_path="data",
        backend="cuda",
        activation_compression="adaptive",
        activation_compression_error_budget=0.02,
        activation_compression_bits=(8, 4),
        activation_compression_check_interval=50,
    )
    # RolloutTrainerConfig overrides cuda_config, so check both builders.
    for config in (TrainerConfig(algo="sft", **options), PolicyTrainerConfig(algo="gspo", **options)):
        runtime = config.cuda_config().runtime
        assert runtime["activation_compression"] == "adaptive"
        assert runtime["activation_compression_error_budget"] == 0.02
        assert runtime["activation_compression_bits"] == (8, 4)
        assert runtime["activation_compression_check_interval"] == 50
        RuntimeConfig(**runtime)
    with pytest.raises(ValueError, match="activation_compression must"):
        TrainerConfig(algo="sft", ckpt="m", dataset_path="d", backend="cuda", activation_compression="int4")
    with pytest.raises(ValueError, match="CUDA backend"):
        TrainerConfig(algo="sft", ckpt="m", dataset_path="d", backend="mlx", activation_compression="adaptive")


def test_relative_error_handles_missing_gradients_cpu():
    ref = [torch.ones(3), None, torch.zeros(2)]
    assert relative_error(ref, ref) == 0.0
    assert math.isclose(relative_error([torch.zeros(3), None, torch.zeros(2)], ref), 1.0)
    assert math.isclose(relative_error([torch.ones(3), torch.ones(4), torch.zeros(2)], ref), 2.0 / math.sqrt(3))
