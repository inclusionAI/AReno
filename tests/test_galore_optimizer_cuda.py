from __future__ import annotations

import pytest
import torch

from areno.engine.layers.linear import mark_linear_weight
from areno.engine.optim import GaLoreAdamW

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _param(rows, cols, *, device, dtype):
    generator = torch.Generator().manual_seed(0)
    param = torch.nn.Parameter((0.1 * torch.randn(sum(rows), cols, generator=generator)).to(device=device, dtype=dtype))
    mark_linear_weight(param, rows)
    return param


def _gradient(rows, cols, generator):
    blocks = []
    for count in rows:
        left, _ = torch.linalg.qr(torch.randn(count, count, generator=generator, dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(cols, cols, generator=generator, dtype=torch.float64))
        size = min(count, cols)
        values = 0.8 ** torch.arange(size, dtype=torch.float64)
        blocks.append((left[:, :size] * values) @ right[:, :size].T)
    return torch.cat(blocks).float()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_galore_matches_cpu_across_refreshes_with_tf32_enabled(dtype) -> None:
    rows, cols = (48, 24), 40
    kwargs = {"lr": 2.0e-3, "betas": (0.9, 0.99), "weight_decay": 0.01, "rank": 6, "update_proj_gap": 3, "scale": 2.0}
    cuda_param = _param(rows, cols, device="cuda", dtype=dtype)
    cpu_param = _param(rows, cols, device="cpu", dtype=dtype)
    cuda_optimizer = GaLoreAdamW([cuda_param], **kwargs)
    cpu_optimizer = GaLoreAdamW([cpu_param], **kwargs)
    generator = torch.Generator().manual_seed(3)
    previous = torch.get_float32_matmul_precision()
    # Workers run with TF32 matmuls; GaLore must still project in IEEE FP32.
    torch.set_float32_matmul_precision("high")
    try:
        for _ in range(8):
            gradient = _gradient(rows, cols, generator)
            cuda_param.main_grad = gradient.cuda()
            cpu_param.main_grad = gradient.clone()
            cuda_optimizer.step()
            cpu_optimizer.step()
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)

    cuda_master = cuda_optimizer._matrix_master(cuda_optimizer._matrices[0]).cpu()
    cpu_master = cpu_optimizer._matrix_master(cpu_optimizer._matrices[0])
    torch.testing.assert_close(cuda_master, cpu_master, atol=1.0e-6, rtol=1.0e-4)
    assert cuda_param.main_grad is None
