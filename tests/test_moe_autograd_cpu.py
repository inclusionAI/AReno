"""Check shared autograd state; these tests do not execute device kernels."""

from types import SimpleNamespace

import torch

from areno.accel import moe


def test_unpermute_backward_reuses_contiguous_forward_indices(monkeypatch):
    seen = []

    def forward(x, ids, tokens, hidden):
        assert ids.is_contiguous()
        seen.append(ids)
        return x.new_zeros((tokens, hidden)).index_add_(0, ids, x)

    def gather(grad, ids):
        assert ids.is_contiguous()
        assert ids is seen[0]
        return grad[ids]

    native = SimpleNamespace(areno_moe_unpermute_forward=forward, areno_moe_gather_by_token_index=gather)
    monkeypatch.setattr(moe, "_extension", lambda device: native)
    indices = torch.tensor([[2, -1], [0, -1], [2, -1], [1, -1]])[:, 0]
    assert not indices.is_contiguous()
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    out = moe._MoeUnpermute.apply(x, indices, 3, 3)
    gradient = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    out.backward(gradient)
    torch.testing.assert_close(out.detach(), torch.stack((x[1], x[3], x[0] + x[2])))
    torch.testing.assert_close(x.grad, gradient[indices])
