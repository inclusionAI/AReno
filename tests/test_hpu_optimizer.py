"""Exercise public optimizer entry points on Gaudi, including state-buffer offsets."""

import itertools

import numpy as np
import pytest
import torch

from areno.accel import optimizer
from tests.test_hpu_activation import hpu_core as hpu_core
from tests.test_hpu_optimizer_source_cpu import floats
from tests.test_hpu_quantized_optimizer_source_cpu import initial_state, reference_step


@pytest.mark.parametrize("model_dtype,grad_dtype", list(itertools.product((0, 1), repeat=2)))
@pytest.mark.parametrize("kind", [3, 4, 8])
@pytest.mark.parametrize("count", [0, 1, 65, 257])
def test_quantized_public_api(hpu_core, model_dtype, grad_dtype, kind, count):
    block, rows, columns, start = 32, 17, 19, 7
    state = initial_state(count, block, 8 if kind == 8 else 4, model_dtype, grad_dtype)
    state[8] = np.linspace(0.1, 2, rows + columns, dtype=np.float32)
    state[9][0] = state[8][:rows].mean()
    tensors = [torch.from_numpy(x.copy()) for x in state]
    for i, bf16 in enumerate((model_dtype, grad_dtype)):
        if bf16:
            tensors[i] = tensors[i].view(torch.bfloat16)
    tensors = [x.to("hpu") for x in tensors]
    offsets = {2: 3, 3: 5, 4: 7, 5: 9}
    if kind != 8:
        for i, offset in offsets.items():
            tensors[i] = torch.cat(
                (
                    torch.full((offset,), 23, dtype=tensors[i].dtype, device="hpu"),
                    tensors[i],
                    torch.full((4,), 23, dtype=tensors[i].dtype, device="hpu"),
                )
            )
    for step in range(1, 4):
        scalars = np.array(
            [0.9, 0.99, 0.001, 0.1, 1e-8, 0.001 / (1 - 0.9**step), (1 - 0.99**step) ** 0.5], dtype=np.float32
        )
        expected = reference_step(state, kind, block, scalars, model_dtype, grad_dtype, start, rows, columns)
        kwargs = dict(
            zip(
                ("beta1", "beta2", "effective_lr", "weight_decay", "eps", "step_size", "bias_correction2_sqrt"),
                map(float, scalars),
            )
        )
        if kind == 8:
            optimizer.areno_adamw_8bit_step(*tensors[:8], block_size=block, **kwargs)
        elif kind == 4:
            optimizer.areno_adamw_4bit_step(
                *tensors[:6],
                moment_packed_offset=3,
                moment_scale_offset=5,
                variance_packed_offset=7,
                variance_scale_offset=9,
                quant_block_size=block,
                **kwargs,
            )
        else:
            kwargs.pop("beta2")
            optimizer.areno_adamw_4bit_factored_step(
                *tensors[:4],
                *tensors[8:11],
                moment_packed_offset=3,
                moment_scale_offset=5,
                parameter_shard_start=start,
                quant_block_size=block,
                rows=rows,
                columns=columns,
                **kwargs,
            )
        hpu_core.mark_step()
        torch.hpu.synchronize()
        for i, ref in zip((0, 2, 3, 4, 5), expected):
            if kind == 3 and i in (4, 5):
                continue
            actual = tensors[i].cpu()
            if i and kind != 8:
                offset = offsets[i]
                assert torch.all(actual[:offset] == 23) and torch.all(actual[offset + ref.size :] == 23)
                actual = actual[offset : offset + ref.size]
            if i == 0:
                torch.testing.assert_close(actual.float(), floats(ref, model_dtype), rtol=2e-4, atol=2e-5)
            elif i in (2, 4):
                torch.testing.assert_close(actual, torch.from_numpy(ref))
            else:
                torch.testing.assert_close(actual, torch.from_numpy(ref), rtol=2e-4, atol=2e-6)
            # Next reference step uses the actual quantized storage.
            state[i] = actual.view(torch.uint16).numpy().copy() if i == 0 and model_dtype else actual.numpy().copy()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bad", [False, True])
def test_factored_statistics_public_api(hpu_core, dtype, bad):
    rows, columns = 7, 13
    grad = torch.linspace(-2, 3, rows * columns).to(dtype)
    if bad:
        grad[17] = float("nan")
    sums = torch.zeros(rows + columns, device="hpu")
    flag = torch.zeros(1, device="hpu", dtype=torch.int32)
    for start, end in ((0, 5), (5, 32), (32, grad.numel())):
        optimizer.areno_adamw_4bit_factored_stats(
            grad[start:end].to("hpu"), sums, flag, parameter_shard_start=start, rows=rows, columns=columns
        )
    hpu_core.mark_step()
    torch.hpu.synchronize()
    squared = grad.float().square().nan_to_num(nan=0).reshape(rows, columns)
    expected = torch.cat((squared.sum(1), squared.sum(0)))
    torch.testing.assert_close(sums.cpu(), expected, rtol=2e-5, atol=2e-5)
    assert flag.cpu().item() == int(bad)


@pytest.mark.parametrize("model_dtype,grad_dtype", list(itertools.product((torch.float32, torch.bfloat16), repeat=2)))
@pytest.mark.parametrize("master", [False, True])
def test_fp32_state_and_compact_master(hpu_core, model_dtype, grad_dtype, master):
    count, offset, total = 65, 3, 73
    original = torch.linspace(-1, 1, total)
    model = original[offset : offset + count].to(model_dtype).to("hpu")
    grad = torch.linspace(-0.1, 0.2, count).to(grad_dtype)
    words = original.numpy().view(np.uint32)
    rounded = original.bfloat16().view(torch.uint16).numpy()
    low_cpu = (words & 0xFFFF).astype(np.uint16)
    carry_cpu = np.zeros((total + 7) // 8, dtype=np.uint8)
    for i in range(total):
        carry_cpu[i // 8] |= np.uint8((int(rounded[i]) != int(words[i] >> 16)) << (i % 8))
    low, carry = torch.from_numpy(low_cpu.copy()).to("hpu"), torch.from_numpy(carry_cpu.copy()).to("hpu")
    size = total if master else count
    moment, variance = torch.zeros(size, device="hpu"), torch.zeros(size, device="hpu")
    beta1, beta2 = float(np.float32(0.9)), float(np.float32(0.99))
    kwargs = dict(
        beta1=beta1,
        beta2=beta2,
        effective_lr=0.001,
        weight_decay=0.1,
        eps=1e-8,
        step_size=0.01,
        bias_correction2_sqrt=0.1,
    )
    if master:
        optimizer.areno_adamw_fp32_master_step(
            model, low, carry, grad.to("hpu"), moment, variance, state_offset=offset, **kwargs
        )
    else:
        optimizer.areno_adamw_fp32_state_step(model, grad.to("hpu"), moment, variance, **kwargs)
    hpu_core.mark_step()
    torch.hpu.synchronize()
    expected_m = (1 - beta1) * grad.float()
    expected_v = (1 - beta2) * grad.float().square()
    initial = original[offset : offset + count] if master else original[offset : offset + count].to(model_dtype).float()
    expected = initial * (1 - 0.001 * 0.1) - 0.01 * expected_m / (expected_v.sqrt() / 0.1 + 1e-8)
    torch.testing.assert_close(model.cpu(), expected.to(model_dtype), rtol=2e-4, atol=2e-5)
    if master:
        for actual, ref in ((moment, expected_m), (variance, expected_v)):
            value = actual.cpu()
            assert torch.count_nonzero(value[:offset]) == 0 and torch.count_nonzero(value[offset + count :]) == 0
            torch.testing.assert_close(value[offset : offset + count], ref, rtol=2e-5, atol=2e-7)
        low_out, carry_out = low.cpu().numpy(), carry.cpu().numpy()
        if model_dtype == torch.bfloat16:
            high = model.cpu().view(torch.uint16).numpy()
            restored = np.empty(count, dtype=np.uint32)
            for i in range(count):
                j = offset + i
                bit = (int(carry_out[j // 8]) >> (j % 8)) & 1
                restored[i] = (((int(high[i]) - bit) & 0xFFFF) << 16) | int(low_out[j])
            torch.testing.assert_close(torch.from_numpy(restored.view(np.float32)), expected, rtol=2e-5, atol=2e-7)
        for i in range(total):
            if model_dtype == torch.float32 or not offset <= i < offset + count:
                assert low_out[i] == low_cpu[i]
                assert ((int(carry_out[i // 8]) ^ int(carry_cpu[i // 8])) & (1 << (i % 8))) == 0
    else:
        torch.testing.assert_close(moment.cpu(), expected_m, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(variance.cpu(), expected_v, rtol=2e-5, atol=2e-7)
