"""Device graph lifecycle contracts with CPU storage and mocked accelerator APIs."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from areno.accel.utils import is_cuda_graph_capturing
from areno.engine import inference
from areno.engine.config import RuntimeConfig
from areno.engine.runtime import decode_graph
from tests.npu_stub import register_npu_device


@pytest.fixture(params=["cuda", "npu"])
def device(request):
    register_npu_device()
    return torch.device(request.param, 1)


@pytest.fixture
def cpu_storage(monkeypatch, device):
    # Keep real tensor copying/padding while recording the requested device.
    for name in ("zeros", "full", "arange", "tensor"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            requested = kwargs.pop("device", None)
            assert requested in (None, device)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)


def test_graph_uses_selected_device_for_warmup_capture_and_replay(monkeypatch, device, cpu_storage):
    events = []

    @contextmanager
    def context(name, value):
        events.append((name, value))
        yield

    class Stream:
        def wait_stream(self, other):
            events.append(("wait", self, other))

    current, side = Stream(), Stream()
    graph = SimpleNamespace(replay=lambda: events.append(("replay",)))
    api = SimpleNamespace(
        **{"CUDAGraph" if device.type == "cuda" else "NPUGraph": lambda: graph},
        device=lambda d: context("device", d),
        Stream=lambda **kw: side,
        stream=lambda s: context("stream", s),
        current_stream=lambda d: current,
        graph=lambda g, **kw: context("capture", (g, kw["stream"])),
        memory_allocated=lambda d: 100,
        reset_peak_memory_stats=lambda d: events.append(("reset_peak", d)),
        max_memory_allocated=lambda d: 164,
        synchronize=lambda d: events.append(("sync", d)),
    )
    # Only the selected accelerator supplies the required methods.
    monkeypatch.setattr(torch, device.type, api)

    def model(**kwargs):
        events.append(("model",))
        return SimpleNamespace(logits_shard=kwargs["input_ids"].float().unsqueeze(-1))

    captured = decode_graph.DecodeGraph(model, 4, 2, 9, 4, device)
    pointers = [t.data_ptr() for t in (captured.input_ids, captured.block_table, captured.recurrent_slots)]
    assert captured.warmup() == 64
    captured.capture()
    assert ("capture", (graph, side)) in events
    assert ("wait", side, current) in events and ("wait", current, side) in events
    assert events.count(("model",)) == 4
    for count in (4, 2):
        captured.replay_tensors(
            torch.arange(count) + 10,
            torch.arange(count) + 2,
            torch.arange(count, dtype=torch.int32) + 2,
            torch.arange(count, dtype=torch.int32).view(count, 1),
            torch.arange(count),
        )
    assert events.count(("model",)) == 4 and events.count(("replay",)) == 2
    assert captured.input_ids.tolist() == [[10, 11, 0, 0]]
    assert captured.block_table.tolist() == [[0, 9], [1, 9], [9, 9], [9, 9]]
    assert captured.recurrent_slots.tolist() == [0, 1, 4, 4]
    assert pointers == [t.data_ptr() for t in (captured.input_ids, captured.block_table, captured.recurrent_slots)]


def test_graph_sync_brackets_collective_on_selected_device(monkeypatch, device):
    events = []
    monkeypatch.setattr(torch, device.type, SimpleNamespace(synchronize=lambda d: events.append(("sync", d))))
    monkeypatch.setattr(decode_graph.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(decode_graph.dist, "barrier", lambda **kw: events.append(("barrier", kw)))
    group = object()
    decode_graph.sync_before_graph_capture(device, group)
    args = {"group": group, **({"device_ids": [1]} if device.type == "cuda" else {})}
    assert events == [("sync", device), ("barrier", args), ("sync", device)]


@pytest.mark.parametrize("free,peer_ready,expected", [(119, True, False), (121, False, False), (121, True, True)])
def test_graph_memory_vote_includes_other_tp_ranks(monkeypatch, device, cpu_storage, free, peer_ready, expected):
    monkeypatch.setattr(torch, device.type, SimpleNamespace(mem_get_info=lambda d: (free, 1000)))
    monkeypatch.setattr(decode_graph.dist, "is_initialized", lambda: True)
    tp_group = object()

    def vote(value, *, op, group):
        assert op is torch.distributed.ReduceOp.MIN
        assert group is tp_group
        if not peer_ready:
            value.zero_()

    monkeypatch.setattr(decode_graph.dist, "all_reduce", vote)
    assert decode_graph.has_graph_capture_memory(device, tp_group, 100) == expected


def test_cache_initialization_captures_graphs_and_reallocation_invalidates_them(monkeypatch, device):
    captured = []
    model = SimpleNamespace(
        reset_kv_caches=lambda: None,
        allocate_kv_caches=lambda *args: [],
        set_kv_caches=lambda *args, **kwargs: None,
        onload_train_weights=lambda *args: None,
        prepare_infer_weights=lambda: None,
        offload_train_weights=lambda: None,
    )
    worker = SimpleNamespace(
        device=device,
        model=model,
        _infer_cache_spec=None,
        _decode_graphs={},
        _decode_graph_skipped_buckets=set(),
        _decode_graph_init_attempted=False,
        _prepare_actor_onloaded=lambda: None,
    )
    manager = inference.InferenceManager(worker)

    def capture(self):
        if not self._decode_graphs:
            captured.append(object())
            self._decode_graphs[1] = captured[-1]

    monkeypatch.setattr(inference.InferenceManager, "_init_decode_graphs", capture)
    spec = inference.InferCacheSpec(
        max_running_seqs=1, max_cache_len=8, num_blocks=4, block_size=4, max_blocks_per_seq=2
    )
    manager._init_infer_cache(spec)
    assert len(captured) == 1
    manager._init_infer_cache(spec)
    assert len(captured) == 1 and worker._decode_graphs[1] is captured[0]
    manager._init_infer_cache(
        inference.InferCacheSpec(max_running_seqs=2, max_cache_len=8, num_blocks=8, block_size=4, max_blocks_per_seq=2)
    )
    assert len(captured) == 2 and worker._decode_graphs[1] is captured[1]


@pytest.mark.parametrize("result", ["success", "eager", "memory", "local_oom", "peer_oom", "error"])
def test_inference_graph_capture_and_failure_decisions(monkeypatch, device, result):
    events = []

    class Graph:
        def __init__(self, *args, **kwargs):
            events.append("create")
            self.graph = SimpleNamespace(reset=lambda: events.append("reset"))

        def warmup(self):
            events.append("warmup")
            return 100

        def capture(self):
            events.append("capture")
            if result == "local_oom":
                raise torch.OutOfMemoryError("test capture OOM")
            if result == "error":
                raise RuntimeError("test unsupported capture operation")

    def vote(device, group, ready):
        assert ready == (result != "local_oom")
        return ready and result != "peer_oom"

    api = SimpleNamespace(mem_get_info=lambda d: (1000, 2000), empty_cache=lambda: events.append("empty"))
    monkeypatch.setattr(torch, device.type, api)
    monkeypatch.setattr(inference, "DecodeGraph", Graph)
    monkeypatch.setattr(inference, "get_tp_context", lambda: SimpleNamespace(group=None, is_rank0=True))
    monkeypatch.setattr(inference, "sync_before_graph_capture", lambda *args: events.append("sync"))
    monkeypatch.setattr(inference, "has_graph_capture_memory", lambda *args: result != "memory")
    monkeypatch.setattr(inference, "all_ranks_graph_ready", vote)
    runtime = RuntimeConfig(device_type=device.type, eager_decode=result == "eager", decode_graph_buckets=[1])
    worker = SimpleNamespace(
        device=device,
        config=SimpleNamespace(runtime=runtime),
        model=SimpleNamespace(training=False),
        _decode_graph_init_attempted=False,
        _decode_graphs={},
        _decode_graph_skipped_buckets=set(),
        _infer_batch_size=1,
        _max_blocks_per_seq=2,
        _scratch_block=9,
        _scratch_recurrent_slot=1,
    )
    manager = inference.InferenceManager(worker)
    if result == "error":
        with pytest.raises(RuntimeError, match="test unsupported capture operation"):
            manager._init_decode_graphs()
        return
    manager._init_decode_graphs()
    before = list(events)
    manager._init_decode_graphs()
    assert events == before
    if result == "eager":
        assert events == [] and not worker._decode_graph_init_attempted
    elif result == "success":
        assert list(worker._decode_graphs) == [1]
        assert isinstance(manager._decode_graph_for_active_count(1), Graph)
    else:
        assert not worker._decode_graphs and worker._decode_graph_skipped_buckets == {1}
        assert manager._decode_graph_for_active_count(1) is None
        if result.endswith("oom"):
            assert events[-3:] == ["reset", "empty", "sync"]


def test_capture_guard_uses_tensor_device(monkeypatch, device):
    monkeypatch.setattr(torch, device.type, SimpleNamespace(is_current_stream_capturing=lambda: True))
    assert is_cuda_graph_capturing(SimpleNamespace(device=device))
    assert not is_cuda_graph_capturing(torch.empty(1))


def test_npu_decode_progress_reports_graph_replay(monkeypatch):
    register_npu_device()
    manager = inference.InferenceManager(SimpleNamespace(device=torch.device("npu", 0)))
    messages = []
    monkeypatch.setattr(inference, "get_tp_context", lambda: SimpleNamespace(dp_rank=0, dp_size=1))
    monkeypatch.setattr(inference.logger, "info", lambda msg, *args: messages.append(msg % args))
    manager._record_decode_progress(enabled=True, interval_s=0, rollout_key=1, active_count=1, token_delta=1)
    manager._decode_progress_graph = True
    manager._record_decode_progress(enabled=True, interval_s=0, rollout_key=1, active_count=1, token_delta=1)
    assert "npu_graph=True" in messages[0] and "cuda_graph" not in messages[0]
