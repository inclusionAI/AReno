"""Exercise offload lifetimes with real mmap state and mocked device transfers."""

from types import SimpleNamespace

import pytest
import torch

from areno.engine.optim import AdamW4bit, AdamW8bit, AdamWFP32Master, adamw_8bit, adamw_fp32_master


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize("optimizer_cls", [AdamWFP32Master, AdamW8bit, AdamW4bit])
def test_prefetch_uses_bucket_device_and_retains_sources_until_copy_completes(
    monkeypatch, tmp_path, device_type, optimizer_cls
):
    param = torch.nn.Parameter(torch.linspace(-1, 1, 32).to(torch.bfloat16))
    optimizer = optimizer_cls([param], lr=0.001, betas=(0.9, 0.99), weight_decay=0.01)
    param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.offload_state(mode="disk", directory=str(tmp_path))
    optimizer._shutdown_disk_writes()
    events = []
    # This device is only used by the mocked transfer boundary; optimizer math
    # and disk IO above/below really run on CPU.
    device = SimpleNamespace(type=device_type, index=1)
    stream = object()

    class Event:
        def record(self, actual_stream):
            assert actual_stream is stream
            events.append("record")

        def synchronize(self):
            events.append("completed")

    def current_stream(actual_device):
        assert actual_device is device
        return stream

    def unexpected_probe():
        pytest.fail("prefetch must use the bucket's device, not CUDA availability")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    monkeypatch.setattr(torch, device_type, SimpleNamespace(Event=Event, current_stream=current_stream), raising=False)

    def prefetch(payload, pin_memory, write_future):
        assert pin_memory
        if write_future is not None:
            write_future.result()
        return {name: tensor.clone() for name, tensor in payload.items()}

    def transfer(tensor, actual_device, *, prefetched):
        assert actual_device is device and prefetched
        events.append("copy")
        return tensor.clone()

    monkeypatch.setattr(adamw_fp32_master, "_prefetch_mmap_payload_after_write", prefetch)
    monkeypatch.setattr(adamw_fp32_master, "_host_tensor_to", transfer)
    monkeypatch.setattr(adamw_8bit, "_host_tensor_to", transfer)
    bucket = optimizer.buckets[0]
    original = bucket.refs[0].model_param
    bucket.refs[0].model_param = SimpleNamespace(device=device)
    try:
        optimizer.prefetch_state()
        prefetched = optimizer._disk_prefetch_futures[0].result()
        if optimizer_cls is AdamWFP32Master:
            optimizer._load_bucket_offload(bucket, device)
        else:
            optimizer._load_state_offload(optimizer._states[0], device)
        assert events[-1] == "record" and events.count("copy") == len(prefetched)
        assert optimizer._disk_prefetch_in_use[0][0] is prefetched
        assert "completed" not in events
        optimizer._release_disk_prefetch(0)
        assert events[-1] == "completed"
        assert not optimizer._disk_prefetch_in_use
    finally:
        bucket.refs[0].model_param = original
        # Restore real copies before the optimizer releases its mmap files.
        monkeypatch.undo()
        optimizer.onload_state(torch.device("cpu"))
    assert not list(tmp_path.rglob("*.mmap"))
