"""Ascend AdamW acceptance against CPU math and the shared checkpoint codec."""

import importlib.util
import math

import pytest
import torch

from areno.accel import optimizer
from areno.accel._extension import extension
from areno.engine.optim import AdamW4bit, AdamW8bit, AdamWFP32Master
from areno.engine.optim.adamw_4bit import _SIGNED_DE_MAP
from areno.engine.optim.dynamic_quant import SIGNED_DYNAMIC_MAP, UNSIGNED_DYNAMIC_MAP
from areno.engine.optim.master_storage import BF16MasterStorage, decode_fp32_master, encode_fp32_master


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    assert extension("npu").optimizer_implementation == "ascendc"


def hyperparameters(step):
    return dict(
        beta1=0.8,
        beta2=0.95,
        effective_lr=0.03,
        weight_decay=0.02,
        eps=1e-6,
        step_size=0.03 / (1 - 0.8**step),
        bias_correction2_sqrt=math.sqrt(1 - 0.95**step),
    )


def reference(weight, grad, moment, variance, args):
    weight = weight.float() * (1 - args["effective_lr"] * args["weight_decay"])
    moment = args["beta1"] * moment + (1 - args["beta1"]) * grad.float()
    variance = args["beta2"] * variance + (1 - args["beta2"]) * grad.float() * grad.float()
    denominator = variance.sqrt() / args["bias_correction2_sqrt"] + args["eps"]
    return weight - args["step_size"] * moment / denominator, moment, variance


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 1023, 1024, 1025, 32769])
def test_fp32_state_multiple_steps_and_storage_canaries(model_dtype, grad_dtype, size):
    generator = torch.Generator().manual_seed(341)
    initial = torch.randn(size + 2, generator=generator).to(model_dtype)
    backing = initial.to("npu")
    model = backing[1:-1]
    m_backing = torch.full((size + 2,), 0.125, device="npu")
    v_backing = torch.full((size + 2,), 0.5, device="npu")
    moment, variance = m_backing[1:-1], v_backing[1:-1]
    expected_m, expected_v = moment.cpu(), variance.cpu()
    for step in range(1, 5):
        gradient = torch.randn(size, generator=generator).to(grad_dtype)
        # Use the current stored model as input each step, so BF16 rounding
        # cannot accumulate and hide a bad FP32-state update.
        expected, expected_m, expected_v = reference(
            model.cpu(), gradient, expected_m, expected_v, hyperparameters(step)
        )
        optimizer.areno_adamw_fp32_state_step(model, gradient.to("npu"), moment, variance, **hyperparameters(step))
        tolerance = 2e-5 if model_dtype == torch.float32 else 8e-3
        torch.testing.assert_close(model.cpu(), expected.to(model_dtype), atol=2e-6, rtol=tolerance)
        torch.testing.assert_close(moment.cpu(), expected_m, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(variance.cpu(), expected_v, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(backing.cpu()[[0, -1]], initial[[0, -1]], atol=0, rtol=0)
    torch.testing.assert_close(m_backing.cpu()[[0, -1]], torch.full((2,), 0.125), atol=0, rtol=0)
    torch.testing.assert_close(v_backing.cpu()[[0, -1]], torch.full((2,), 0.5), atol=0, rtol=0)


@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("offset", [0, 1, 7, 8, 9, 1023, 1024, 1025])
@pytest.mark.parametrize("size", [0, 1, 7, 1025, 32769])
def test_compact_master_slice_preserves_other_weights_and_carry_bits(grad_dtype, offset, size):
    generator = torch.Generator().manual_seed(778)
    original = torch.randn(offset + size + 11, generator=generator)
    rounded, metadata = encode_fp32_master(original)
    full_model = rounded.to("npu")
    storage = metadata.to("npu")
    moment = torch.full_like(full_model, 0.125, dtype=torch.float32)
    variance = torch.full_like(full_model, 0.5, dtype=torch.float32)
    region = slice(offset, offset + size)
    expected = original[region].clone()
    expected_m, expected_v = moment[region].cpu(), variance[region].cpu()
    for step in range(1, 5):
        gradient = torch.randn(size, generator=generator).to(grad_dtype)
        expected, expected_m, expected_v = reference(expected, gradient, expected_m, expected_v, hyperparameters(step))
        optimizer.areno_adamw_fp32_master_step(
            full_model[region],
            storage.low_bits,
            storage.round_up_bits,
            gradient.to("npu"),
            moment,
            variance,
            state_offset=offset,
            **hyperparameters(step),
        )
        decoded = decode_fp32_master(full_model.cpu(), storage.to("cpu"))
        torch.testing.assert_close(decoded[region], expected, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(moment[region].cpu(), expected_m, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(variance[region].cpu(), expected_v, atol=3e-6, rtol=3e-6)
        for outside in (slice(None, offset), slice(offset + size, None)):
            assert torch.equal(decoded[outside].view(torch.int32), original[outside].view(torch.int32))
            torch.testing.assert_close(moment[outside].cpu(), torch.full_like(original[outside], 0.125), atol=0, rtol=0)
            torch.testing.assert_close(variance[outside].cpu(), torch.full_like(original[outside], 0.5), atol=0, rtol=0)


@pytest.mark.parametrize("offset", [1, 7, 1023])
def test_compact_master_preserves_fp32_bits_and_bf16_ties(offset):
    words = torch.tensor([0x3F808000, 0x3F818000, -1082097664, -1082032128], dtype=torch.int32).repeat(513)
    original = words.view(torch.float32)
    rounded, metadata = encode_fp32_master(original)
    full_model, storage = rounded.to("npu"), metadata.to("npu")
    args = hyperparameters(1) | dict(beta1=1.0, beta2=1.0, weight_decay=0.0, step_size=0.0)
    model = full_model[offset:-1]
    optimizer.areno_adamw_fp32_master_step(
        model,
        storage.low_bits,
        storage.round_up_bits,
        torch.zeros_like(model),
        torch.ones_like(full_model, dtype=torch.float32),
        torch.ones_like(full_model, dtype=torch.float32),
        state_offset=offset,
        **args,
    )
    decoded = decode_fp32_master(full_model.cpu(), storage.to("cpu"))
    assert torch.equal(decoded.view(torch.int32), words)
    assert torch.equal(full_model.cpu().view(torch.int16), rounded.view(torch.int16))
    assert torch.equal(storage.low_bits.cpu(), metadata.low_bits)
    assert torch.equal(storage.round_up_bits.cpu(), metadata.round_up_bits)


@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_fp32_master_model_uses_offset_and_leaves_metadata_untouched(grad_dtype):
    offset, n = 7, 2051
    values = torch.linspace(-1, 2, n)
    model = values.to("npu")
    metadata = BF16MasterStorage(
        torch.full((offset + n + 2,), 12345, dtype=torch.uint16),
        torch.full(((offset + n + 9) // 8,), 0xA5, dtype=torch.uint8),
    )
    storage = metadata.to("npu")
    moment = torch.full((offset + n + 2,), 0.125, device="npu")
    variance = torch.ones_like(moment)
    gradient = torch.linspace(-1, 1, n).to(grad_dtype)
    expected, expected_m, expected_v = reference(
        values, gradient, moment[offset : offset + n].cpu(), variance[offset : offset + n].cpu(), hyperparameters(1)
    )
    optimizer.areno_adamw_fp32_master_step(
        model,
        storage.low_bits,
        storage.round_up_bits,
        gradient.to("npu"),
        moment,
        variance,
        state_offset=offset,
        **hyperparameters(1),
    )
    torch.testing.assert_close(model.cpu(), expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(moment[offset : offset + n].cpu(), expected_m, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(variance[offset : offset + n].cpu(), expected_v, atol=3e-6, rtol=3e-6)
    assert torch.equal(storage.low_bits.cpu(), metadata.low_bits)
    assert torch.equal(storage.round_up_bits.cpu(), metadata.round_up_bits)


def test_fp32_state_zero_variance_and_nonfinite_propagation():
    values = torch.ones(1025)
    gradients = torch.zeros_like(values)
    gradients[1:4] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    model, gradient = values.to("npu"), gradients.to("npu")
    moment, variance = torch.zeros_like(model), torch.zeros_like(model)
    expected, expected_m, expected_v = reference(values, gradients, moment.cpu(), variance.cpu(), hyperparameters(1))
    optimizer.areno_adamw_fp32_state_step(model, gradient, moment, variance, **hyperparameters(1))
    for actual, target in ((model, expected), (moment, expected_m), (variance, expected_v)):
        torch.testing.assert_close(actual.cpu(), target, atol=3e-6, rtol=3e-6, equal_nan=True)


def test_optimizer_current_stream_and_tensor_device():
    device = 1 if torch.npu.device_count() >= 2 else 0
    model = torch.empty(1025, device=f"npu:{device}")
    grad, moment, variance = (torch.empty_like(model) for _ in range(3))
    stream = torch.npu.Stream(device=device)
    with torch.npu.stream(stream):
        model.fill_(1)
        grad.fill_(0.25)
        moment.zero_()
        variance.zero_()
        optimizer.areno_adamw_fp32_state_step(model, grad, moment, variance, **hyperparameters(1))
    stream.synchronize()
    expected, _, _ = reference(
        torch.ones(1025), torch.full((1025,), 0.25), torch.zeros(1025), torch.zeros(1025), hyperparameters(1)
    )
    torch.testing.assert_close(model.cpu(), expected, atol=3e-6, rtol=3e-5)


def test_native_fp32_optimizer_rejects_invalid_metadata():
    native = extension("npu")
    x = torch.ones(7, device="npu")
    args = tuple(hyperparameters(1).values())
    with pytest.raises(RuntimeError, match="moments must be FP32"):
        native.areno_adamw_fp32_state_step(x, x, x.to(torch.bfloat16), x, *args)
    with pytest.raises(RuntimeError, match="length must match"):
        native.areno_adamw_fp32_state_step(x, x, torch.ones(8, device="npu"), torch.ones(8, device="npu"), *args)
    low = torch.zeros(7, dtype=torch.uint16, device="npu")
    carries = torch.zeros(1, dtype=torch.uint8, device="npu")
    for offset in (-1, 1):
        with pytest.raises(RuntimeError, match="out of bounds"):
            native.areno_adamw_fp32_master_step(x, low, carries, x, x, x, offset, *args)
    with pytest.raises(RuntimeError, match="metadata lengths"):
        native.areno_adamw_fp32_master_step(x, low, carries[:0], x, x, x, 0, *args)
    with pytest.raises(RuntimeError, match="contiguous"):
        native.areno_adamw_fp32_state_step(x[::2], x[::2], x[:4], x[:4], *args)


def test_shared_master_optimizer_checkpoint_resume(tmp_path):
    initial = [torch.linspace(-1, 1, 3).to(torch.bfloat16), torch.linspace(-2, 2, 2053).to(torch.bfloat16)]
    params = [torch.nn.Parameter(t.to("npu")) for t in initial]
    reference_params = [torch.nn.Parameter(t.float()) for t in initial]
    kwargs = dict(lr=0.003, betas=(0.8, 0.95), weight_decay=0.02)
    actual = AdamWFP32Master(params, bucket_numel=1024, **kwargs)
    expected = torch.optim.AdamW(reference_params, **kwargs)
    for step in range(5):
        for index, (p, ref) in enumerate(zip(params, reference_params, strict=True)):
            grad = torch.sin(torch.arange(p.numel()) * 0.17 + step + index).to(torch.bfloat16)
            p.grad, ref.grad = grad.to("npu"), grad.float()
        actual.step()
        expected.step()
        masters = torch.cat(actual.state_dict()["master_params"])
        torch.testing.assert_close(masters, torch.cat([p.detach() for p in reference_params]), atol=4e-6, rtol=4e-5)
        if step == 1:
            path = tmp_path / "optimizer.pt"
            torch.save(actual.state_dict(), path)
            actual = AdamWFP32Master(params, bucket_numel=1024, **kwargs)
            actual.load_state_dict(torch.load(path, weights_only=True))


def quantized_reference(bits, state, gradient, block_size, offsets, args):
    model, mq, ms, vq, vs = (x.clone() for x in state)
    mo, mso, vo, vso = offsets
    signed = torch.tensor(_SIGNED_DE_MAP if bits == 4 else SIGNED_DYNAMIC_MAP)
    unsigned = torch.arange(1, 17) / 16 if bits == 4 else torch.tensor(UNSIGNED_DYNAMIC_MAP)
    for block, start in enumerate(range(0, model.numel(), block_size)):
        stop = min(start + block_size, model.numel())
        positions = torch.arange(start, stop)
        if bits == 4:
            mc = (mq[mo + positions // 2] >> (4 * (positions % 2))) & 15
            vc = (vq[vo + positions // 2] >> (4 * (positions % 2))) & 15
        else:
            mc, vc = mq[positions], vq[positions]
        m, v = signed[mc.long()] * ms[mso + block], unsigned[vc.long()] * vs[vso + block]
        p, m, v = reference(model[start:stop], gradient[start:stop], m, v, args)
        if not all(torch.isfinite(t).all() for t in (gradient[start:stop], p, m, v)):
            continue
        ms[mso + block], vs[vso + block] = m.abs().max(), v.max()
        mn = m / ms[mso + block].clamp_min(1e-30)
        vn = v / vs[vso + block].clamp_min(1e-30)
        mc = (mn[:, None] - signed).abs().argmin(dim=-1).byte()
        if bits == 4:
            vc = (vn * 16 - 1).round().clamp(0, 15).byte()
            if mc.numel() % 2:
                mc = torch.cat((mc, torch.tensor([7], dtype=torch.uint8)))
                vc = torch.cat((vc, torch.tensor([0], dtype=torch.uint8)))
            mq[mo + start // 2 : mo + (stop + 1) // 2] = mc[::2] | (mc[1::2] << 4)
            vq[vo + start // 2 : vo + (stop + 1) // 2] = vc[::2] | (vc[1::2] << 4)
        else:
            vc = (vn[:, None] - unsigned).abs().argmin(dim=-1).byte()
            mq[start:stop], vq[start:stop] = mc, vc
        model[start:stop] = p.to(model.dtype)
    return model, mq, ms, vq, vs


def quantized_candidate(bits, state, gradient, block_size, offsets, args):
    model, mq, ms, vq, vs = state
    if bits == 8:
        optimizer.areno_adamw_8bit_step(
            model,
            gradient,
            mq,
            ms,
            vq,
            vs,
            torch.tensor(SIGNED_DYNAMIC_MAP, device=model.device),
            torch.tensor(UNSIGNED_DYNAMIC_MAP, device=model.device),
            block_size=block_size,
            **args,
        )
    else:
        mo, mso, vo, vso = offsets
        optimizer.areno_adamw_4bit_step(
            model,
            gradient,
            mq,
            ms,
            vq,
            vs,
            moment_packed_offset=mo,
            moment_scale_offset=mso,
            variance_packed_offset=vo,
            variance_scale_offset=vso,
            quant_block_size=block_size,
            **args,
        )


@pytest.mark.parametrize("bits,block_size", [(8, 1), (8, 31), (8, 128), (8, 4096), (4, 32), (4, 128), (4, 1024)])
@pytest.mark.parametrize("size", [0, 1, 33, 1025, 8193])
@pytest.mark.parametrize(
    "model_dtype,grad_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_quantized_steps_match_codes_scales_and_packed_offsets(bits, block_size, size, model_dtype, grad_dtype):
    generator = torch.Generator().manual_seed(962)
    offsets = (3, 2, 5, 4) if bits == 4 else (0, 0, 0, 0)
    mo, mso, vo, vso = offsets
    codes, count = ((size + 1) // 2 if bits == 4 else size), (size + block_size - 1) // block_size
    tail = 3 if bits == 4 else 0
    state = (
        torch.randn(size, generator=generator).to(model_dtype),
        torch.randint(0, 256, (mo + codes + tail,), generator=generator, dtype=torch.uint8),
        torch.rand(mso + count + tail, generator=generator) + 0.01,
        torch.randint(0, 256, (vo + codes + tail,), generator=generator, dtype=torch.uint8),
        torch.rand(vso + count + tail, generator=generator) + 0.01,
    )
    # Back every device tensor with additional sentinels and a nonzero storage
    # offset, including the 8-bit entries whose visible lengths must be exact.
    backing = [
        torch.cat((torch.full((1,), 12, dtype=t.dtype), t, torch.full((1,), 12, dtype=t.dtype))).to("npu")
        for t in state
    ]
    native = tuple(t[1:-1] for t in backing)
    for step in range(1, 4):
        gradient = torch.randn(size, generator=generator).to(grad_dtype)
        before = tuple(t.cpu() for t in native)
        expected = quantized_reference(bits, before, gradient, block_size, offsets, hyperparameters(step))
        quantized_candidate(bits, native, gradient.to("npu"), block_size, offsets, hyperparameters(step))
        for index, (actual, target) in enumerate(zip(native, expected, strict=True)):
            if target.dtype == torch.uint8:
                assert torch.equal(actual.cpu(), target), f"{bits}-bit codes differ at state index {index}"
            else:
                rtol = 8e-3 if target.dtype == torch.bfloat16 else 3e-5
                torch.testing.assert_close(actual.cpu(), target, atol=3e-6, rtol=rtol)
        for parent in backing:
            torch.testing.assert_close(parent.cpu()[[0, -1]], torch.full((2,), 12, dtype=parent.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1e30])
def test_quantized_invalid_gradient_skips_only_its_block(bits, bad):
    n, block_size = 97, 32
    state = (
        torch.linspace(-1, 1, n),
        torch.full(((n + 1) // 2 if bits == 4 else n,), 0x77 if bits == 4 else 0, dtype=torch.uint8),
        torch.zeros(4),
        torch.zeros((n + 1) // 2 if bits == 4 else n, dtype=torch.uint8),
        torch.zeros(4),
    )
    gradient = torch.full((n,), 0.5)
    gradient[39] = bad
    native = tuple(t.to("npu") for t in state)
    args = hyperparameters(1)
    expected = quantized_reference(bits, state, gradient, block_size, (0, 0, 0, 0), args)
    quantized_candidate(bits, native, gradient.to("npu"), block_size, (0, 0, 0, 0), args)
    for actual, target in zip(native, expected, strict=True):
        torch.testing.assert_close(actual.cpu(), target, atol=3e-6, rtol=3e-5)
    # All persistent data in the invalid block is byte-for-byte untouched.
    for index in range(5):
        region = (
            slice(32, 64)
            if index == 0 or (bits == 8 and index in (1, 3))
            else (slice(16, 32) if index in (1, 3) else slice(1, 2))
        )
        assert torch.equal(native[index].cpu()[region], state[index][region])


@pytest.mark.parametrize("optimizer_cls", [AdamW4bit, AdamW8bit])
def test_shared_quantized_vector_optimizer_resume(tmp_path, optimizer_cls):
    initial = torch.linspace(-1, 1, 4099).to(torch.bfloat16)
    param, ref = torch.nn.Parameter(initial.to("npu")), torch.nn.Parameter(initial.clone())
    kwargs = dict(lr=0.003, betas=(0.8, 0.95), weight_decay=0.02, bucket_numel=1024, quant_block_size=128)
    actual, expected = optimizer_cls([param], **kwargs), optimizer_cls([ref], **kwargs)
    for step in range(5):
        gradient = torch.sin(torch.arange(initial.numel()) * 0.03 + step * 0.2).to(torch.bfloat16)
        param.grad, ref.grad = gradient.to("npu"), gradient.clone()
        actual.step()
        expected.step()
        torch.testing.assert_close(param.cpu(), ref, atol=2e-3, rtol=0)
        if step == 1:
            path = tmp_path / "quantized.pt"
            torch.save(actual.state_dict(), path)
            actual = optimizer_cls([param], **kwargs)
            actual.load_state_dict(torch.load(path, weights_only=True))
