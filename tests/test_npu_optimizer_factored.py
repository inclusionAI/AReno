"""Native factored AdamW4bit statistics and updates, including shard boundaries."""

import importlib.util

import pytest
import torch

from areno.accel import optimizer
from areno.accel._extension import extension
from areno.engine.optim import AdamW4bit
from areno.engine.optim.adamw_4bit import _SIGNED_DE_MAP


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    native = extension("npu")
    assert native.optimizer_implementation == "ascendc"
    assert callable(native.areno_adamw_4bit_factored_stats)
    assert callable(native.areno_adamw_4bit_factored_step)


def arguments():
    return dict(beta1=0.8, effective_lr=0.03, weight_decay=0.02, eps=1e-6, step_size=0.15, bias_correction2_sqrt=0.25)


def reference(
    model,
    gradient,
    codes,
    scales,
    factors,
    mean,
    invalid,
    *,
    start,
    rows,
    columns,
    code_offset=3,
    scale_offset=2,
    block_size=128,
):
    model, codes, scales = (x.clone() for x in (model, codes, scales))
    if invalid:
        return model, codes, scales
    mapping = torch.tensor(_SIGNED_DE_MAP)
    args = arguments()
    for block, begin in enumerate(range(0, model.numel(), block_size)):
        end = min(begin + block_size, model.numel())
        local = torch.arange(begin, end)
        position = start + local
        old_codes = (codes[code_offset + local // 2] >> (4 * (local % 2))) & 15
        moment = mapping[old_codes.long()] * scales[scale_offset + block]
        moment = args["beta1"] * moment + (1 - args["beta1"]) * gradient[begin:end].float()
        variance = factors[position // columns] * factors[rows + position % columns]
        variance = variance / torch.fmax(mean, torch.tensor(1e-30))
        weight = model[begin:end].float() * (1 - args["effective_lr"] * args["weight_decay"])
        weight -= args["step_size"] * moment / (variance.sqrt() / args["bias_correction2_sqrt"] + args["eps"])
        if not all(torch.isfinite(x).all() for x in (gradient[begin:end], moment, variance, weight)):
            continue
        scale = moment.abs().max()
        new_codes = ((moment / scale.clamp_min(1e-30))[:, None] - mapping).abs().argmin(dim=-1).byte()
        if new_codes.numel() % 2:
            new_codes = torch.cat((new_codes, torch.tensor([7], dtype=torch.uint8)))
        codes[code_offset + begin // 2 : code_offset + (end + 1) // 2] = new_codes[::2] | (new_codes[1::2] << 4)
        scales[scale_offset + block] = scale
        model[begin:end] = weight.to(model.dtype)
    return model, codes, scales


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("columns", [1, 3, 31, 1023, 1024, 1025, 4097])
@pytest.mark.parametrize("bad", [False, True])
def test_factored_stats_accumulate_shards_and_ignore_only_nonfinite_contributions(dtype, columns, bad):
    rows = 65
    generator = torch.Generator().manual_seed(617)
    values = torch.randn(rows * columns, generator=generator).to(dtype)
    if bad:
        values[[1, values.numel() // 2, -2]] = torch.tensor([float("nan"), float("inf"), 1e30]).to(dtype)
    grad = values.to("npu")
    parent = torch.full((rows + columns + 2,), 0.25, device="npu")
    sums = parent[1:-1]
    flag_parent = torch.tensor([9, 0, 11], device="npu", dtype=torch.int32)
    flag = flag_parent[1:2]
    cuts = sorted({0, 1, columns - 1, columns + 3, values.numel() - 2, values.numel()})
    for begin, end in zip(cuts, cuts[1:]):
        optimizer.areno_adamw_4bit_factored_stats(
            grad[begin:end],
            sums,
            flag,
            parameter_shard_start=begin,
            rows=rows,
            columns=columns,
        )
    # The CUDA contract accumulates finite entries even in a shard that also
    # contains NaN, infinity or a finite gradient whose square overflows.
    squared = values.float().square().reshape(rows, columns)
    squared = torch.where(torch.isfinite(squared), squared, 0)
    expected = torch.cat((squared.sum(1), squared.sum(0))) + 0.25
    torch.testing.assert_close(sums.cpu(), expected, atol=3e-5, rtol=3e-5)
    assert flag.cpu().item() == int(bad)
    torch.testing.assert_close(parent.cpu()[[0, -1]], torch.full((2,), 0.25), atol=0, rtol=0)
    assert flag_parent.cpu().tolist()[::2] == [9, 11]
    # An empty trailing shard cannot reset the accumulated flag or sums.
    optimizer.areno_adamw_4bit_factored_stats(
        grad[:0],
        sums,
        flag,
        parameter_shard_start=values.numel(),
        rows=rows,
        columns=columns,
    )
    torch.testing.assert_close(sums.cpu(), expected, atol=3e-5, rtol=3e-5)
    assert flag.cpu().item() == int(bad)


@pytest.mark.parametrize(
    "shape,start,size",
    [
        ((257, 1), 7, 129),
        ((70, 3), 2, 205),
        ((7, 1025), 1023, 4099),
        ((65, 67), 65, 1025),
        ((3, 7), 21, 0),
        ((1, 5000), 7, 1025),
        ((3, 4097), 4095, 33),
    ],
)
@pytest.mark.parametrize(
    "model_dtype,grad_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("block_size", [32, 128, 1024])
def test_factored_update_crosses_rows_and_preserves_packed_neighbors(
    shape, start, size, model_dtype, grad_dtype, block_size
):
    rows, columns = shape
    generator = torch.Generator().manual_seed(477)
    model = torch.randn(size, generator=generator).to(model_dtype)
    gradient = torch.randn(size, generator=generator).to(grad_dtype)
    codes = torch.randint(0, 256, (3 + (size + 1) // 2 + 5,), generator=generator, dtype=torch.uint8)
    scales = torch.rand(2 + (size + block_size - 1) // block_size + 3, generator=generator) + 0.01
    factors = torch.rand(rows + columns, generator=generator) + 0.1
    mean = factors[:rows].mean()
    expected = reference(
        model, gradient, codes, scales, factors, mean, 0, start=start, rows=rows, columns=columns, block_size=block_size
    )
    backing = [
        torch.cat((torch.full((1,), 12, dtype=x.dtype), x, torch.full((1,), 12, dtype=x.dtype))).to("npu")
        for x in (model, codes, scales)
    ]
    actual = [x[1:-1] for x in backing]
    native_factors = factors.to("npu")
    optimizer.areno_adamw_4bit_factored_step(
        actual[0],
        gradient.to("npu"),
        actual[1],
        actual[2],
        native_factors,
        mean.to("npu"),
        torch.zeros((), device="npu", dtype=torch.int32),
        moment_packed_offset=3,
        moment_scale_offset=2,
        parameter_shard_start=start,
        quant_block_size=block_size,
        rows=rows,
        columns=columns,
        **arguments(),
    )
    for result, target in zip(actual, expected, strict=True):
        if result.dtype == torch.uint8:
            assert torch.equal(result.cpu(), target)
        else:
            torch.testing.assert_close(
                result.cpu(), target, atol=3e-6, rtol=8e-3 if model_dtype == torch.bfloat16 else 3e-5
            )
    assert torch.equal(native_factors.cpu(), factors)
    for tensor in backing:
        torch.testing.assert_close(tensor.cpu()[[0, -1]], torch.full((2,), 12, dtype=tensor.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("case", ["flag", "negative_flag", "gradient", "row", "column", "zero_factors", "nan_mean"])
def test_factored_skip_and_row_mean_clamp(case):
    rows, columns, start, size, block_size = 7, 65, 33, 257, 32
    model = torch.linspace(-1, 2, size)
    gradient = torch.full((size,), 0.5)
    codes = torch.full((3 + (size + 1) // 2 + 5,), 0x77, dtype=torch.uint8)
    scales = torch.zeros(2 + (size + block_size - 1) // block_size + 3)
    factors = torch.ones(rows + columns)
    flag = 1 if case == "flag" else -1 if case == "negative_flag" else 0
    if case == "gradient":
        gradient[39] = float("nan")
    elif case == "row":
        factors[2] = float("inf")
    elif case == "column":
        factors[rows + 10] = float("nan")
    elif case in {"zero_factors", "nan_mean"}:
        factors.zero_()
    mean = torch.tensor(float("nan")) if case == "nan_mean" else factors[:rows].mean()
    expected = reference(
        model,
        gradient,
        codes,
        scales,
        factors,
        mean,
        flag,
        start=start,
        rows=rows,
        columns=columns,
        block_size=block_size,
    )
    actual = [x.to("npu") for x in (model, codes, scales)]
    optimizer.areno_adamw_4bit_factored_step(
        actual[0],
        gradient.to("npu"),
        actual[1],
        actual[2],
        factors.to("npu"),
        mean.to("npu"),
        torch.tensor(flag, device="npu", dtype=torch.int32),
        moment_packed_offset=3,
        moment_scale_offset=2,
        parameter_shard_start=start,
        quant_block_size=block_size,
        rows=rows,
        columns=columns,
        **arguments(),
    )
    for result, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.cpu(), target, atol=3e-6, rtol=3e-5)


def test_factored_stats_current_stream_and_tensor_device():
    device = 1 if torch.npu.device_count() >= 2 else 0
    gradient = torch.empty(2051, device=f"npu:{device}", dtype=torch.bfloat16)
    sums = torch.empty(8 + 1025, device=gradient.device)
    flag = torch.empty((), device=gradient.device, dtype=torch.int32)
    stream = torch.npu.Stream(device=device)
    with torch.npu.stream(stream):
        gradient.fill_(0.5)
        sums.zero_()
        flag.zero_()
        optimizer.areno_adamw_4bit_factored_stats(
            gradient, sums, flag, parameter_shard_start=1023, rows=8, columns=1025
        )
    stream.synchronize()
    padded = torch.zeros(8 * 1025)
    padded[1023 : 1023 + 2051] = 0.25
    matrix = padded.reshape(8, 1025)
    torch.testing.assert_close(sums.cpu(), torch.cat((matrix.sum(1), matrix.sum(0))), atol=0, rtol=0)
    assert flag.cpu().item() == 0


def test_factored_native_bounds_and_dtypes():
    native = extension("npu")
    grad = torch.ones(7, device="npu")
    factors = torch.zeros(6, device="npu")
    flag = torch.zeros((), device="npu", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="out of bounds"):
        native.areno_adamw_4bit_factored_stats(grad, factors, flag, 3, 3, 3)
    with pytest.raises(RuntimeError, match="factor count"):
        native.areno_adamw_4bit_factored_stats(grad, factors[:5], flag, 0, 3, 3)
    with pytest.raises(RuntimeError, match="one int32"):
        native.areno_adamw_4bit_factored_stats(grad, factors, flag.float(), 0, 3, 3)
    with pytest.raises(RuntimeError, match="shape is invalid"):
        native.areno_adamw_4bit_factored_stats(grad, factors, flag, 0, 2**62, 4)


@pytest.mark.parametrize("shape", [(65, 67), (17, 3, 11), (5, 1025)])
def test_shared_factored_optimizer_checkpoint_resume(tmp_path, shape):
    initial = torch.linspace(-1, 1, int(torch.tensor(shape).prod())).reshape(shape).to(torch.bfloat16)
    param, ref = torch.nn.Parameter(initial.to("npu")), torch.nn.Parameter(initial.clone())
    kwargs = dict(lr=0.003, betas=(0.8, 0.95), weight_decay=0.02, bucket_numel=1024, quant_block_size=128)
    actual, expected = AdamW4bit([param], **kwargs), AdamW4bit([ref], **kwargs)
    for step in range(5):
        gradient = torch.sin(torch.arange(initial.numel()) * 0.03 + step * 0.2).reshape(shape).to(torch.bfloat16)
        param.grad, ref.grad = gradient.to("npu"), gradient.clone()
        actual.step()
        expected.step()
        torch.testing.assert_close(param.cpu(), ref, atol=2e-3, rtol=0)
        native_factors = actual._factored_second_moments[id(param)].cpu()
        torch.testing.assert_close(native_factors, expected._factored_second_moments[id(ref)], atol=3e-6, rtol=3e-5)
        assert native_factors.numel() == shape[0] + initial.numel() // shape[0]
        if step == 1:
            path = tmp_path / "factored.pt"
            torch.save(actual.state_dict(), path)
            actual = AdamW4bit([param], **kwargs)
            actual.load_state_dict(torch.load(path, weights_only=True))
