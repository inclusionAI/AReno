"""Ascend FLA acceptance, independent of the AReno native extension build."""

import importlib.util

import pytest
import torch

from areno.accel.ops import SegLaMeta, chunk_lightning_attn, seg_la_fwd


@pytest.fixture(scope="module")
def npu():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend hardware and torch_npu are required")
    import torch_npu  # noqa: F401
    from fla.ops.simple_gla import chunk_simple_gla  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    return "npu:0"


def linear_reference(q, k, v, log_decay, initial, *, boundaries=None, scale=0.25):
    """CPU recurrence: H[t] = exp(log_decay) H[t-1] + outer(k[t], v[t])."""
    sequences = (
        [(0, start, end) for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)]
        if boundaries is not None
        else [(batch, 0, q.shape[1]) for batch in range(q.shape[0])]
    )
    outputs, states = [], []
    for sequence, (batch, start, end) in enumerate(sequences):
        state = initial[sequence]
        for step in range(start, end):
            state = log_decay.exp()[:, None, None] * state + torch.einsum("hk,hv->hkv", k[batch, step], v[batch, step])
            outputs.append(torch.einsum("hk,hkv->hv", q[batch, step] * scale, state))
        states.append(state)
    return torch.stack(outputs).reshape_as(v), torch.stack(states)


def inputs(shape, dtype, device, seed, *, requires_grad=False):
    # Strided inputs exercise the actual library layout/contiguity adapter.
    cpu = (torch.randn(*shape, 2, generator=torch.Generator().manual_seed(seed)) * 0.2).to(dtype)
    return (
        cpu.to(device)[..., 0].detach().requires_grad_(requires_grad),
        cpu[..., 0].double().detach().requires_grad_(requires_grad),
    )


def close(actual, expected):
    tolerance = 0.04 if actual.dtype == torch.bfloat16 else 0.006
    torch.testing.assert_close(actual.float().cpu(), expected.detach().float(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("head_first", [False, True])
def test_lightning_training_state_and_backward(npu, dtype, packed, head_first):
    shape = (1, 67, 2, 16) if packed else (2, 67, 2, 16)
    pairs = [inputs(shape, dtype, npu, seed, requires_grad=True) for seed in (1, 2, 3)]
    tensors, refs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    boundaries = [0, 3, 67] if packed else None
    state, state_ref = inputs((2, 2, 16, 16), torch.float32, npu, 4, requires_grad=True)
    # Deliberately different from the decay recomputed by upstream Lightning,
    # including a nonzero first local head, as on a nonzero tensor-parallel rank.
    decay = torch.tensor([-0.07, -0.003], dtype=torch.float32)
    args = [x.transpose(1, 2) for x in tensors] if head_first else tensors
    out, final = chunk_lightning_attn(
        *args,
        layer_idx=3,
        num_layers=12,
        g_gamma=decay.to(npu),
        scale=0.25,
        initial_state=state,
        output_final_state=True,
        head_first=head_first,
        cu_seqlens=torch.tensor(boundaries, device=npu, dtype=torch.long) if packed else None,
    )
    if head_first:
        out = out.transpose(1, 2)
    expected, expected_final = linear_reference(*refs, decay.double(), state_ref, boundaries=boundaries)
    close(out, expected)
    close(final, expected_final)
    grad, grad_ref = inputs(shape, dtype, npu, 5)
    final_grad, final_grad_ref = inputs(state.shape, torch.float32, npu, 6)
    torch.autograd.backward((out, final), (grad, final_grad))
    torch.autograd.backward((expected, expected_final), (grad_ref, final_grad_ref))
    for tensor, ref in zip((*tensors, state), (*refs, state_ref), strict=True):
        close(tensor.grad, ref.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_seg_la_prefill_then_decode_preserves_state_slots(npu, dtype):
    slots = torch.tensor([3, 1], dtype=torch.int32)
    boundaries = [0, 3, 67]
    state, state_ref = inputs((5, 2, 16, 16), torch.float32, npu, 11)
    # A state pool can be strided. Copies below must update the original tensor.
    before = state_ref.clone()
    rates = torch.tensor([0.07, 0.003])
    pairs = [inputs((67, 2, 16), dtype, npu, seed) for seed in (12, 13, 14)]
    tensors, refs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    meta = SegLaMeta(
        2,
        64,
        torch.tensor(boundaries, dtype=torch.int32, device=npu),
        slots.to(npu),
        torch.tensor([3, 64], device=npu),
        torch.tensor([0, 1], device=npu),
    )
    initial = state_ref[slots.long()].clone()
    initial[0].zero_()
    expected, final = linear_reference(
        *(x.unsqueeze(0) for x in refs),
        -rates.double(),
        initial,
        boundaries=boundaries,
    )
    out = seg_la_fwd(*tensors, state, rates.to(npu), meta, softmax_scale=0.25)
    close(out, expected.squeeze(0))
    close(state[slots.long().to(npu)], final)
    torch.testing.assert_close(state[[0, 2, 4]].cpu(), before[[0, 2, 4]].float(), atol=0, rtol=0)

    pairs = [inputs((2, 2, 16), dtype, npu, seed) for seed in (15, 16, 17)]
    tensors, refs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    meta.q_offsets = torch.arange(3, dtype=torch.int32, device=npu)
    meta.q_lengths = torch.ones(2, dtype=torch.int32, device=npu)
    meta.s_scales = torch.ones(2, dtype=torch.int32, device=npu)
    meta.max_q_length = 1
    # Use the actual persisted FP32 state for the next step's reference, so this
    # comparison isolates decode from any accumulated prefill rounding error.
    persisted = state[slots.long().to(npu)].cpu().double()
    expected, final = linear_reference(*(x.unsqueeze(1) for x in refs), -rates.double(), persisted)
    out = seg_la_fwd(*tensors, state, rates.to(npu), meta, softmax_scale=0.25)
    close(out, expected.squeeze(1))
    close(state[slots.long().to(npu)], final)
    torch.testing.assert_close(state[[0, 2, 4]].cpu(), before[[0, 2, 4]].float(), atol=0, rtol=0)
