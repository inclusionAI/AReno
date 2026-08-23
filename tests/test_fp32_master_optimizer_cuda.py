from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from areno.engine.optim import AdamW8bit, AdamWFP32Master

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_fused_fp32_master_adamw_matches_torch_reference() -> None:
    device = torch.device("cuda", 0)
    initial = torch.linspace(-1.0, 1.0, 4099, device=device, dtype=torch.float32).to(torch.bfloat16)
    candidate_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.float())
    kwargs = {"lr": 2.0e-4, "betas": (0.9, 0.999), "weight_decay": 0.05}
    candidate = AdamWFP32Master([candidate_param], bucket_numel=1024, **kwargs)
    reference = torch.optim.AdamW([reference_param], eps=1.0e-8, **kwargs)
    generator = torch.Generator(device=device).manual_seed(29)

    for _ in range(5):
        gradient = torch.randn(candidate_param.shape, device=device, generator=generator).to(torch.bfloat16)
        candidate_param.grad = gradient.clone()
        reference_param.grad = gradient.float()
        candidate.step()
        reference.step()

        master = torch.cat(candidate.state_dict()["master_params"])
        torch.testing.assert_close(master, reference_param.detach().cpu(), rtol=3e-6, atol=3e-7)
        assert torch.equal(candidate_param, reference_param.detach().to(torch.bfloat16))
        assert torch.isfinite(master).all()
        assert all(bucket.master is None for bucket in candidate.buckets)


def test_fused_fp32_master_handles_multiple_refs_and_partial_bytes() -> None:
    device = torch.device("cuda", 0)
    first = torch.nn.Parameter(torch.tensor([0.5, -0.25, 1.0], device=device, dtype=torch.bfloat16))
    second = torch.nn.Parameter(torch.linspace(-2.0, 2.0, 14, device=device).to(torch.bfloat16))
    optimizer = AdamWFP32Master(
        [first, second],
        lr=1.0e-3,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        bucket_numel=32,
    )
    first.grad = torch.tensor([0.25, -0.5, 1.0], device=device, dtype=torch.bfloat16)
    second.grad = torch.linspace(1.0, -1.0, 14, device=device).to(torch.bfloat16)

    optimizer.step()

    expected_first = torch.tensor([0.499, -0.249, 0.999], device=device).to(torch.bfloat16)
    torch.testing.assert_close(first, expected_first, rtol=0.0, atol=0.0)
    state = optimizer.state_dict()
    assert state["master_params"][0].numel() == 17
    assert torch.isfinite(state["master_params"][0]).all()


@pytest.mark.parametrize("optimizer_cls", [AdamWFP32Master, AdamW8bit])
def test_disk_offload_prefetch_uses_pinned_double_buffer_and_preserves_updates(tmp_path, optimizer_cls) -> None:
    from areno.engine.optim.adamw_fp32_master import _write_mmap_payloads

    device = torch.device("cuda", 0)
    initial = [torch.linspace(-1.0 + index, 1.0 + index, 1024, device=device).to(torch.bfloat16) for index in range(3)]
    candidate_params = [torch.nn.Parameter(value.clone()) for value in initial]
    reference_params = [torch.nn.Parameter(value.clone()) for value in initial]
    kwargs = {
        "lr": 4.0e-4,
        "betas": (0.9, 0.99),
        "weight_decay": 0.02,
        "bucket_numel": 1024,
    }
    candidate = optimizer_cls(candidate_params, **kwargs)
    reference = optimizer_cls(reference_params, **kwargs)
    candidate.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=2)
    staged_writes = []

    def inspect_write(group, payloads, ready_events=()) -> None:
        staged_writes.append((payloads, ready_events))
        _write_mmap_payloads(group, payloads, ready_events)

    for step in range(2):
        if step == 1:
            candidate.prefetch_state()
            assert len(candidate._disk_prefetch_futures) == 2
            prefetched = [future.result() for future in candidate._disk_prefetch_futures.values()]
            assert all(tensor.is_pinned() for payload in prefetched for tensor in payload.values())
        for index, (candidate_param, reference_param) in enumerate(
            zip(candidate_params, reference_params, strict=True)
        ):
            gradient = torch.linspace(-0.5 + index, 0.75 - index, 1024, device=device).to(torch.bfloat16)
            candidate_param.grad = gradient
            reference_param.grad = gradient.clone()
        with patch("areno.engine.optim.adamw_fp32_master._write_mmap_payloads", inspect_write):
            candidate.step()
            candidate._shutdown_disk_writes()
        reference.step()

    assert staged_writes
    assert all(ready_events for _payloads, ready_events in staged_writes)
    assert all(
        tensor.is_pinned()
        for payloads, _ready_events in staged_writes
        for payload in payloads.values()
        for tensor in payload.values()
    )
    for candidate_param, reference_param in zip(candidate_params, reference_params, strict=True):
        torch.testing.assert_close(candidate_param, reference_param, rtol=0.0, atol=0.0)
    assert not candidate._disk_prefetch_futures
    assert not candidate._disk_prefetch_in_use
    candidate.onload_state(device)
    assert not list(tmp_path.rglob("*.mmap"))
