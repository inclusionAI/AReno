"""Ascend vocab embedding through the shared CUDA/Torch autograd wrapper."""

import importlib.util

import pytest
import torch

from areno.accel._extension import extension
from areno.accel.embedding import areno_vocab_embedding


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    assert extension("npu").embedding_implementation == "ascendc"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [0, 1, 7, 65, 1023, 1024, 1025, 8193])
@pytest.mark.parametrize("strided", [False, True])
def test_embedding_forward_backward_masks_and_strided_weight(dtype, width, strided):
    start, end = 19, 26
    weights = torch.linspace(-1, 1, (end - start) * width * 2).reshape(end - start, width * 2).to(dtype)
    all_ids = torch.tensor([[18, 19, 20, 25, 26], [19, -1, 25, 2**40, -(2**40)]])
    ids = all_ids.to("npu")
    weight = weights.to("npu")
    if strided:
        ids, all_ids = ids.t(), all_ids.t()
        weight, weights = weight[:, ::2], weights[:, ::2]
    else:
        weight, weights = weight[:, :width].contiguous(), weights[:, :width].contiguous()
    weight.requires_grad_()
    output = areno_vocab_embedding(ids, weight, start, end)
    expected = torch.zeros(*all_ids.shape, width, dtype=dtype)
    mask = (all_ids >= start) & (all_ids < end)
    expected[mask] = weights[all_ids[mask] - start]
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    # Binary fractions make repeated-id accumulation exactly representable in
    # every storage dtype, so a wrong scatter address cannot hide in tolerance.
    full_gradient = (
        ((torch.arange(output.numel() * 2) % 5 - 2) * 0.125).reshape(*output.shape[:-1], width * 2).to(dtype)
    )
    gradient = full_gradient.to("npu")[..., ::2]
    reference_gradient = full_gradient[..., ::2]
    output.backward(gradient)
    expected_grad = torch.zeros_like(weights)
    expected_grad.index_add_(0, all_ids[mask] - start, reference_gradient[mask])
    torch.testing.assert_close(weight.grad.cpu(), expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_embedding_scalar_and_empty_ids(dtype, shape):
    ids = torch.full(shape, 7, dtype=torch.int64, device="npu")
    weight = torch.ones(3, 65, device="npu", dtype=dtype, requires_grad=True)
    output = areno_vocab_embedding(ids, weight, 6, 9)
    assert output.shape == (*shape, 65)
    output.backward(torch.ones_like(output))
    expected = torch.zeros(3, 65, dtype=dtype)
    expected[1].fill_(ids.numel())
    torch.testing.assert_close(weight.grad.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_embedding_bitwise_forward_and_nonlocal_bad_gradients(dtype):
    source = torch.tensor([[-0.0, float("nan"), float("inf"), -float("inf"), 1.25]], dtype=dtype)
    weight = source.to("npu").requires_grad_()
    ids = torch.tensor([5, 4, 6, 5], device="npu")
    output = areno_vocab_embedding(ids, weight, 5, 6)
    word_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    result = output.cpu()
    assert torch.equal(result[[0, 3]].view(word_dtype), source.expand(2, -1).contiguous().view(word_dtype))
    assert torch.count_nonzero(result[[1, 2]].view(word_dtype)) == 0
    gradient = torch.ones_like(output)
    gradient[1].fill_(float("nan"))
    gradient[2].fill_(float("inf"))
    output.backward(gradient)
    torch.testing.assert_close(weight.grad.cpu(), torch.full_like(source, 2), atol=0, rtol=0)


def test_empty_vocab_shard_and_tp_shard_sum():
    ids = torch.tensor([[0, 1, 7, 8, 12, 16, -1, 17]], device="npu")
    table = torch.arange(17 * 65).reshape(17, 65).float() / 128
    outputs, shards = [], []
    for start, end in ((0, 1), (1, 8), (8, 8), (8, 17)):
        weight = table[start:end].to("npu").requires_grad_()
        outputs.append(areno_vocab_embedding(ids, weight, start, end))
        shards.append((start, end, weight))
    total = sum(outputs)
    target = torch.zeros(*ids.shape, 65)
    cpu_ids = ids.cpu()
    mask = (cpu_ids >= 0) & (cpu_ids < 17)
    target[mask] = table[cpu_ids[mask]]
    torch.testing.assert_close(total.cpu(), target, atol=0, rtol=0)
    total.sum().backward()
    expected = torch.zeros_like(table)
    expected.index_add_(0, cpu_ids[mask], torch.ones(int(mask.sum()), 65))
    for start, end, weight in shards:
        torch.testing.assert_close(weight.grad.cpu(), expected[start:end], atol=0, rtol=0)


def test_embedding_uses_tensor_device_and_current_stream():
    device = 1 if torch.npu.device_count() >= 2 else 0
    ids = torch.empty(3, device=f"npu:{device}", dtype=torch.int64)
    weight = torch.empty(5, 1025, device=ids.device, requires_grad=True)
    stream = torch.npu.Stream(device=device)
    with torch.npu.stream(stream):
        ids.fill_(12)
        with torch.no_grad():
            weight.fill_(0.75)
        result = areno_vocab_embedding(ids, weight, 10, 15)
        result.sum().backward()
    stream.synchronize()
    assert result.device == weight.device
    torch.testing.assert_close(result.cpu(), torch.full((3, 1025), 0.75), atol=0, rtol=0)
    expected = torch.zeros(5, 1025)
    expected[2].fill_(3)
    torch.testing.assert_close(weight.grad.cpu(), expected, atol=0, rtol=0)


def test_embedding_native_input_contract():
    native = extension("npu")
    ids = torch.tensor([0, 1], device="npu")
    weight = torch.ones(3, 65, device="npu")
    with pytest.raises(RuntimeError, match="range must match"):
        native.areno_vocab_embedding_forward(ids, weight, 0, 4)
    with pytest.raises(RuntimeError, match="ids must be int64"):
        native.areno_vocab_embedding_forward(ids.int(), weight, 0, 3)
    with pytest.raises(RuntimeError, match="contiguous"):
        native.areno_vocab_embedding_forward(ids, weight[:, ::2], 0, 3)
    with pytest.raises(RuntimeError, match="gradient shape and dtype"):
        native.areno_vocab_embedding_backward(torch.ones(2, 64, device="npu"), ids, weight, 0, 3)
