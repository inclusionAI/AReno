"""CUDA projection/reference and refresh-boundary restoration checks."""

import pytest
import torch

from areno.engine.optim.galore import GaLoreAdamW
from tests.test_galore_cpu import Reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(11, 7), (7, 11), (9, 9)])
def test_cuda_matches_reference_across_refresh(shape):
    torch.manual_seed(593)
    param = torch.nn.Parameter(torch.randn(shape, device="cuda"))
    reference = Reference(param.detach())
    optim = GaLoreAdamW([param], lr=0.003, betas=(0.8, 0.9), weight_decay=0.1, rank=2, update_proj_gap=3, scale=0.7)
    for _ in range(7):
        param.grad = torch.randn_like(param)
        reference.update(param.grad)
        optim.step()
        torch.testing.assert_close(param, reference.weight, atol=1e-6, rtol=1e-4)


def test_cuda_bfloat16_resume_before_refresh():
    torch.manual_seed(593)
    param = torch.nn.Parameter(torch.randn(11, 7, device="cuda", dtype=torch.bfloat16))
    optim = GaLoreAdamW([param], lr=0.001, rank=2, update_proj_gap=3)
    for _ in range(3):
        param.main_grad = torch.randn_like(param, dtype=torch.float32)
        optim.step()
    other_param = torch.nn.Parameter(torch.zeros_like(param))
    resumed = GaLoreAdamW([other_param], lr=1, rank=1)
    resumed.load_state_dict(optim.state_dict())
    for _ in range(4):
        grad = torch.randn_like(param, dtype=torch.float32)
        param.main_grad = grad
        other_param.main_grad = grad.clone()
        optim.step()
        resumed.step()
        torch.testing.assert_close(param, other_param, atol=0, rtol=0)
        torch.testing.assert_close(optim.state[param]["master"], resumed.state[other_param]["master"], atol=0, rtol=0)
