"""Ascend graph/stream/offload/checkpoint acceptance; requires real NPU hardware."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from areno.accel._extension import extension
from areno.engine.checkpoints import io
from areno.engine.optim import AdamW4bit, AdamW8bit, AdamWFP32Master


@pytest.fixture(params=[0, 1])
def device(request):
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    if torch.npu.device_count() <= request.param:
        pytest.skip("a second NPU is required for device 1 coverage")
    with torch.npu.device(request.param):
        yield torch.device("npu", request.param)


@torch.inference_mode()
def test_decode_graph_matches_eager_with_changing_inputs_and_cache(device):
    from areno.accel import areno_linear, areno_rmsnorm, areno_silu, areno_vocab_embedding
    from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend
    from areno.engine.runtime.decode_graph import DecodeGraph
    from areno.engine.runtime.metadata import InferMeta

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(71)
            self.embedding = torch.randn(16, 32, device=device, dtype=torch.bfloat16) / 4
            self.weight = torch.randn(32, 32, device=device, dtype=torch.bfloat16) / 4
            self.norm = torch.ones(32, device=device, dtype=torch.bfloat16)
            self.k = torch.randn(9, 4, 2, 16, device=device, dtype=torch.bfloat16) / 4
            self.v = torch.randn_like(self.k) / 4
            self.attention = FlashAttnInferBackend("native")

        def forward(self, input_ids, position_ids, infer_meta):
            x = areno_vocab_embedding(input_ids, self.embedding, 0, 16)
            x = x + position_ids.unsqueeze(-1).to(x.dtype) / 16
            x = areno_silu(areno_rmsnorm(areno_linear(x, self.weight), self.norm, 1e-6))
            q = x.view(1, -1, 2, 16)
            out = self.attention(q, q * 0.5, q * 0.25, self.k, self.v, infer_meta)
            return SimpleNamespace(logits_shard=areno_linear(out.flatten(-2), self.weight))

    model = Model().eval()
    before_k, before_v = model.k.clone(), model.v.clone()
    graph = DecodeGraph(model, 4, 2, 8, 4, device)
    graph.warmup()
    graph.capture()
    # Warmup and capture must only touch the scratch block.
    torch.testing.assert_close(model.k[:-1], before_k[:-1], atol=0, rtol=0)
    torch.testing.assert_close(model.v[:-1], before_v[:-1], atol=0, rtol=0)
    pointers = (model.k.data_ptr(), model.v.data_ptr(), graph.input_ids.data_ptr())
    for count, offset in ((4, 0), (2, 3), (1, 4)):
        model.k.copy_(before_k)
        model.v.copy_(before_v)
        tokens = torch.arange(count, device=device) + offset
        positions = torch.arange(count, device=device) + offset
        lengths = positions.to(torch.int32)
        table = torch.arange(count * 2, device=device, dtype=torch.int32).view(count, 2)
        slots = torch.arange(count, device=device)
        actual = graph.replay_tensors(tokens, positions, lengths, table, slots)[0, :count].clone()
        replay_k, replay_v = model.k.clone(), model.v.clone()
        model.k.copy_(before_k)
        model.v.copy_(before_v)
        meta = InferMeta(mode="decode", cache_seqlens=lengths, block_table=table, recurrent_slots=slots)
        expected = model(tokens.view(1, count), positions.view(1, count), meta).logits_shard[0]
        torch.npu.synchronize(device)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(replay_k[:-1], model.k[:-1], atol=0, rtol=0)
        torch.testing.assert_close(replay_v[:-1], model.v[:-1], atol=0, rtol=0)
        assert pointers == (model.k.data_ptr(), model.v.data_ptr(), graph.input_ids.data_ptr())


@pytest.mark.parametrize("pageable", [False, True])
def test_checkpoint_copy_waits_for_producer_and_preserves_strided_data(device, pageable, monkeypatch):
    # Checkpoint transport can be validated before compiling AReno's kernels.
    io._sync_pending_cpu_copies()
    assert not io._PENDING_CPU_COPY_BUCKETS
    monkeypatch.setattr(io, "_ASYNC_CPU_COPY_MAX_BYTES", 256)
    producer = torch.npu.Stream(device=device)
    snapshots = []
    try:
        with torch.npu.stream(producer):
            for index in range(4):
                source = torch.arange(512, dtype=torch.float32, device=device).reshape(16, 32)
                source.add_(index * 1000)
                view = source[1::2, ::3].T
                if pageable:
                    with io.pageable_checkpoint_staging():
                        snapshot = io._tensor_to_cpu(view)
                    assert not io._PENDING_CPU_COPY_BUCKETS
                else:
                    snapshot = io._tensor_to_cpu(view)
                    assert snapshot.is_pinned()
                    assert len(io._PENDING_CPU_COPY_BUCKETS) == 1
                snapshots.append(snapshot)
                del view, source
                # Reuse device allocations while copy streams may be in flight.
                torch.empty((16, 32), device=device).fill_(-999)
        io._sync_pending_cpu_copies()
        for index, snapshot in enumerate(snapshots):
            expected = (torch.arange(512).reshape(16, 32) + index * 1000)[1::2, ::3].T.float()
            torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)
    finally:
        io._sync_pending_cpu_copies()


@pytest.mark.parametrize("mode", ["cpu", "disk"])
@pytest.mark.parametrize("optimizer_cls", [AdamWFP32Master, AdamW8bit, AdamW4bit])
def test_optimizer_offload_prefetch_and_checkpoint_preserve_updates(device, tmp_path, mode, optimizer_cls):
    assert extension("npu").optimizer_implementation == "ascendc"
    initial = [torch.linspace(-1 + i, 1 + i, 256).reshape(16, 16).to(torch.bfloat16) for i in range(3)]
    params = [torch.nn.Parameter(value.to(device)) for value in initial]
    refs = [torch.nn.Parameter(value.to(device)) for value in initial]
    kwargs = dict(lr=0.001, betas=(0.8, 0.95), weight_decay=0.01, bucket_numel=256)
    candidate = optimizer_cls(params, **kwargs)
    reference = optimizer_cls(refs, **kwargs)
    directory = str(tmp_path) if mode == "disk" else None
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    try:
        with torch.npu.stream(stream):
            for step in range(3):
                candidate.configure_state_offload(mode=mode, directory=directory, batch_size=2)
                if mode == "disk" and step:
                    candidate.prefetch_state()
                    assert len(candidate._disk_prefetch_futures) == 2
                    for future in candidate._disk_prefetch_futures.values():
                        assert all(tensor.is_pinned() for tensor in future.result().values())
                for index, (param, ref) in enumerate(zip(params, refs, strict=True)):
                    gradient = torch.sin(torch.arange(256, device=device) * 0.1 + step + index).reshape(16, 16)
                    param.grad = gradient.to(torch.bfloat16)
                    ref.grad = param.grad.clone()
                candidate.step()
                reference.step()
                candidate.offload_state(mode=mode, directory=directory, batch_size=2)
                for param, ref in zip(params, refs, strict=True):
                    torch.testing.assert_close(param.cpu(), ref.cpu(), rtol=0, atol=0)
                if step == 1:
                    path = tmp_path / "optimizer.pt"
                    torch.save(candidate.state_dict(), path)
                    candidate.onload_state(device)
                    candidate = optimizer_cls(params, **kwargs)
                    candidate.load_state_dict(torch.load(path, weights_only=False))
                    candidate.offload_state(mode=mode, directory=directory, batch_size=2)
        stream.synchronize()
    finally:
        candidate.onload_state(device)
    assert not list(tmp_path.rglob("*.mmap"))
