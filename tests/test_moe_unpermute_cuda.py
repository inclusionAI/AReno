"""MoE sums and derivatives must survive arbitrary within-expert ordering."""

import pytest
import torch

from areno.accel.moe import areno_moe_unpermute


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA deterministic MoE reduction")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_moe_unpermute_fixed_expert_order_and_gradients_cuda(dtype):
    torch.manual_seed(59)
    experts, tokens, hidden = 8, 32, 257
    values = (torch.randint(-64, 64, (experts, tokens, hidden)).float() / 32).to(dtype)
    values[:3, 0, 0] = torch.tensor([512, 1, -512], dtype=dtype)
    values[3:, 0, 0] = 0
    expected = values.double().sum(0).to(dtype).cuda()
    upstream = ((torch.arange(tokens * hidden).reshape(tokens, hidden) % 13 - 6) / 16).to(dtype).cuda()
    for seed in range(3):
        torch.manual_seed(seed)
        permutations = [torch.randperm(tokens) for _ in range(experts)]
        indices = torch.cat(permutations).cuda()
        routed = torch.cat([values[i, permutation] for i, permutation in enumerate(permutations)]).cuda()
        physical = torch.empty(routed.shape[0], hidden * 2, device="cuda", dtype=dtype)
        physical[:, ::2] = routed
        states = physical[:, ::2].requires_grad_()
        output = areno_moe_unpermute(states, indices, (tokens, hidden))
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        output.backward(upstream)
        torch.testing.assert_close(states.grad, upstream[indices], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture")
def test_moe_unpermute_graph_capture_replay_cuda():
    tokens, hidden = 31, 63
    indices = torch.arange(tokens, device="cuda").repeat(3)
    states = ((torch.arange(indices.numel() * hidden, device="cuda").reshape(-1, hidden) % 17 - 8) / 8).float()
    for _ in range(3):
        areno_moe_unpermute(states, indices, (tokens, hidden))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = areno_moe_unpermute(states, indices, (tokens, hidden))
    states.mul_(2)
    indices.copy_(indices.flip(0))
    graph.replay()
    expected = torch.zeros(tokens, hidden, device="cuda").index_add_(0, indices, states)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA empty MoE reduction")
@pytest.mark.parametrize("tokens,hidden", [(0, 31), (17, 31), (17, 0)])
def test_moe_unpermute_empty_routes_backward_cuda(tokens, hidden):
    states = torch.empty(0, hidden, device="cuda", requires_grad=True)
    indices = torch.empty(0, dtype=torch.long, device="cuda")
    output = areno_moe_unpermute(states, indices, (tokens, hidden))
    torch.testing.assert_close(output, torch.zeros(tokens, hidden, device="cuda"), atol=0, rtol=0)
    output.sum().backward()
    assert states.grad is not None and states.grad.shape == states.shape
