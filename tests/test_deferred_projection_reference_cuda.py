"""CUDA independent preclip/transport reference and exact serialized continuation."""

import pytest
import torch

from tests.test_deferred_projection_reference_cpu import case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(9, 9), (13, 9), (9, 13)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("stress", [False, True])
def test_deferred_independent_preclip_reference_cuda(monkeypatch, shape, dtype, stress):
    monkeypatch.setenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION", "1")
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        case(shape, dtype, stress, "cuda")
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
