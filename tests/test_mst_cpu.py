"""Mini-sequence memory boundaries, derivatives and configuration contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.runtime.mst import mini_sequence_forward


@pytest.fixture(autouse=True)
def single_rank():
    previous = get_tp_context()
    set_tp_context(TPContext(0, 1, torch.device("cpu"), None))
    yield
    set_tp_context(previous)


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_mst_forward_input_and_parameter_gradients_cpu(chunk):
    torch.manual_seed(19)
    block = nn.Sequential(nn.Linear(8, 31), nn.SiLU(), nn.Linear(31, 8)).double()
    other = copy.deepcopy(block)
    x = torch.randn(2, 9, 8, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = block(x)
    actual = mini_sequence_forward(other, y, chunk_size=chunk)
    upstream = torch.randn_like(actual)
    reference = torch.autograd.grad(expected, (x, *block.parameters()), upstream)
    got = torch.autograd.grad(actual, (y, *other.parameters()), upstream)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(got, reference, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_mst_does_not_save_expanded_activations_cpu():
    block = nn.Sequential(nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, 8))
    x = torch.randn(1, 23, 8, requires_grad=True)
    saved = []
    parameter_storages = {p.untyped_storage().data_ptr() for p in block.parameters()}
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append((t.shape, t.untyped_storage().data_ptr())) or t, lambda t: t
    ):
        result = mini_sequence_forward(block, x, chunk_size=5)
    assert not any(128 in shape and storage not in parameter_storages for shape, storage in saved)
    result.square().sum().backward()
    assert all(p.grad is not None for p in block.parameters())


def test_mst_fixed_routes_and_route_weight_gradients_cpu():
    torch.manual_seed(21)
    x = torch.randn(2, 7, 8, requires_grad=True)
    routes = torch.randint(0, 3, (14, 2))
    weights = torch.randn(14, 2, requires_grad=True)
    experts = torch.randn(3, 8, 8, requires_grad=True)

    def execute(states, indices, probabilities):
        values = states.reshape(-1, 8)
        out = torch.einsum("th,tkoh->tko", values, experts[indices])
        return (out * probabilities[..., None]).sum(1).view_as(states)

    reference = execute(x, routes, weights)
    actual = mini_sequence_forward(execute, x, routes, weights, chunk_size=3)
    torch.testing.assert_close(actual, reference)
    expected = torch.autograd.grad(reference.square().mean(), (x, weights, experts))
    got = torch.autograd.grad(actual.square().mean(), (x, weights, experts))
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_mst_head_preserves_packed_boundaries_and_fp32_cast_cpu():
    from areno.engine.runtime.logprobs import packed_next_token_logprobs, packed_next_token_logprobs_from_hidden

    torch.manual_seed(23)
    head = nn.Linear(8, 37, bias=False).float()
    hidden = torch.randn(1, 17, 8, dtype=torch.bfloat16, requires_grad=True)
    tokens = torch.randint(37, (1, 17))
    cu = torch.tensor([0, 1, 6, 17], dtype=torch.int32)
    expected = packed_next_token_logprobs(head(hidden.float()), tokens, cu)
    actual = packed_next_token_logprobs_from_hidden(hidden, tokens, cu, head, chunk_size=3)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    upstream = torch.linspace(-1, 1, actual.numel())
    a = torch.autograd.grad(expected, (hidden, head.weight), upstream)
    b = torch.autograd.grad(actual, (hidden, head.weight), upstream)
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    torch.testing.assert_close(a[1], b[1], atol=2e-6, rtol=2e-6)


def test_mst_head_preserves_model_projection_layout_cpu():
    from areno.engine.runtime.logprobs import packed_next_token_logprobs_from_hidden

    class Head(nn.Linear):
        def forward(self, states):
            assert states.ndim == 3
            return super().forward(states)

    head = Head(8, 17, bias=False)
    hidden = torch.randn(1, 11, 8, requires_grad=True)
    tokens = torch.randint(17, (1, 11))
    logps = packed_next_token_logprobs_from_hidden(hidden, tokens, torch.tensor([0, 11]), head, chunk_size=4)
    logps.sum().backward()
    assert hidden.grad is not None and head.weight.grad is not None


def test_mst_fp32_projection_accumulates_tied_weight_gradients_once_cpu():
    from areno.engine.runtime.logprobs import packed_next_token_logprobs, packed_next_token_logprobs_from_hidden

    class Head(nn.Linear):
        def forward(self, states):
            return torch.nn.functional.linear(states.float(), self.weight.float())

    torch.manual_seed(41)
    head = Head(8, 17, bias=False, dtype=torch.bfloat16)
    hidden = torch.randn(1, 19, 8, dtype=torch.bfloat16, requires_grad=True)
    tokens = torch.randint(17, (1, 19))
    cu = torch.tensor([0, 19])
    expected = packed_next_token_logprobs(head(hidden), tokens, cu)
    actual = packed_next_token_logprobs_from_hidden(hidden, tokens, cu, head, chunk_size=3)
    upstream = torch.linspace(-1, 1, actual.numel())
    reference = torch.autograd.grad((expected * upstream).sum(), head.weight)[0]
    got = torch.autograd.grad((actual * upstream).sum(), head.weight)[0]
    torch.testing.assert_close(got, reference, atol=0, rtol=0)


def test_mst_bf16_parameter_gradients_do_not_round_partial_sums_cpu():
    block = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    block.weight.data.fill_(1)
    states = torch.tensor([[[512.0], [1.0], [-512.0]]], dtype=torch.bfloat16, requires_grad=True)
    expected = torch.autograd.grad(block(states).float().sum(), block.weight)[0]
    output = mini_sequence_forward(block, states, chunk_size=1)
    actual = torch.autograd.grad(output.float().sum(), block.weight)[0]
    assert expected.item() == 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_mst_preserves_full_forward_without_saving_expansion_cpu(dtype):
    torch.manual_seed(43)
    block = nn.Sequential(nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, 8)).to(dtype)
    states = torch.randn(1, 23, 8, dtype=dtype, requires_grad=True)
    expected = block(states)
    saved = []
    lengths = []
    handle = block.register_forward_pre_hook(lambda module, inputs: lengths.append(inputs[0].shape[1]))
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value.shape) or value, lambda value: value
    ):
        output = mini_sequence_forward(block, states, chunk_size=5)
    assert lengths == [23]
    assert torch.Size([1, 23, 128]) not in saved
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.float().square().mean().backward()
    handle.remove()
    assert lengths[1:] == [5, 5, 5, 5, 3]
    assert states.grad is not None


def test_mst_bf16_stochastic_block_replays_full_rng_draw_cpu():
    block = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.3)).bfloat16().train()
    states = torch.randn(1, 23, 8, dtype=torch.bfloat16, requires_grad=True)
    torch.manual_seed(47)
    expected = block(states)
    reference = torch.autograd.grad(expected.float().sum(), (states, *block.parameters()))
    torch.manual_seed(47)
    output = mini_sequence_forward(block, states, chunk_size=5)
    rng = torch.get_rng_state()
    actual = torch.autograd.grad(output.float().sum(), (states, *block.parameters()))
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
    for value, baseline in zip(actual, reference, strict=True):
        torch.testing.assert_close(value, baseline, atol=0, rtol=0)


def test_mst_bf16_inside_outer_layer_checkpoint_cpu():
    from torch.utils.checkpoint import checkpoint

    block = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8)).bfloat16()
    states = torch.randn(1, 11, 8, dtype=torch.bfloat16, requires_grad=True)
    output = checkpoint(lambda value: mini_sequence_forward(block, value, chunk_size=4), states, use_reentrant=False)
    output.float().square().sum().backward()
    assert states.grad is not None and all(parameter.grad is not None for parameter in block.parameters())


@pytest.mark.parametrize(
    "device,tp,expected",
    [
        ("cuda", 1, 1024),
        ("cuda", 2, 0),
        ("npu", 1, 0),
        ("cpu", 1, 0),
    ],
)
def test_mst_automatic_schedule_cpu(device, tp, expected):
    from areno.engine.runtime.mst import training_chunk_size

    assert training_chunk_size(device, tp) == expected


def test_mst_internal_schedule_override_cpu(monkeypatch):
    from areno.engine.runtime.mst import training_chunk_size

    monkeypatch.setenv("ARENO_MST_CHUNK_SIZE", "8192")
    assert training_chunk_size("cuda", 1) == 8192
    assert training_chunk_size("cuda", 2) == 0
    assert training_chunk_size("cpu", 1) == 0
    monkeypatch.setenv("ARENO_MST_CHUNK_SIZE", "0")
    assert training_chunk_size("cuda", 1) == 0
    monkeypatch.setenv("ARENO_MST_CHUNK_SIZE", "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        training_chunk_size("cuda", 1)


def test_mst_large_chunks_keep_gradients_cpu():
    torch.manual_seed(19)
    block = nn.Sequential(nn.Linear(8, 31), nn.SiLU(), nn.Linear(31, 8)).double()
    other = copy.deepcopy(block)
    x = torch.randn(16391, 8, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = block(x)
    actual = mini_sequence_forward(other, y, chunk_size=8192)
    upstream = torch.randn_like(actual)
    reference = torch.autograd.grad(expected, (x, *block.parameters()), upstream)
    got = torch.autograd.grad(actual, (y, *other.parameters()), upstream)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(got, reference, strict=True):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-10)


def test_mst_has_no_public_option_cpu():
    from areno.api.trainer_config import TrainerConfig
    from areno.cli.train import train_command

    assert all("mst" not in option.name for option in train_command.params)
    assert "mst_chunk_size" not in TrainerConfig.__dataclass_fields__


def test_mst_train_metadata_cpu():
    from areno.engine.runtime.train_step import _train_meta

    meta = _train_meta(
        {"train_cu_seqlens": torch.tensor([0, 5]), "_mst_chunk_size": 3},
        torch.ones(1, 5),
        sequence_parallel=False,
    )
    assert meta.mst_chunk_size == 3


@pytest.mark.parametrize("family,size,expected", [("gemma4", 0, 256), ("gemma4", 1024, 256), ("qwen3", 1024, 1024)])
def test_mst_keeps_existing_head_memory_bound_cpu(monkeypatch, family, size, expected):
    from areno.engine import training

    class HeadReached(Exception):
        pass

    class Model(nn.Module):
        lm_head = None

        def forward(self, input_ids, *, train_meta, defer_lm_head, **kwargs):
            assert defer_lm_head
            assert train_meta.mst_chunk_size == size
            return SimpleNamespace(hidden_states=torch.ones(1, 3, 4))

    def check_head(*args, chunk_size, **kwargs):
        assert chunk_size == expected
        raise HeadReached

    worker = SimpleNamespace(
        device=torch.device("cpu"),
        model=Model(),
        _train_state_ready=True,
        config=SimpleNamespace(
            model=SimpleNamespace(model_type=family),
            runtime=SimpleNamespace(activation_checkpointing=True),
            effective_sequence_parallel=False,
        ),
    )
    monkeypatch.setattr(training, "_pack_train_data", lambda pack: pack)
    monkeypatch.setattr(training, "training_chunk_size", lambda *args: size)
    monkeypatch.setattr(training, "packed_next_token_logprobs_from_hidden", check_head)
    pack = {"input_ids": torch.tensor([[0, 1, 2]]), "train_cu_seqlens": torch.tensor([0, 3])}
    with pytest.raises(HeadReached):
        training.TrainingManager(worker)._train_step([pack], allow_step=False, grad_scale=1)


def test_mst_recompute_restores_parallel_context_cpu():
    from areno.engine.parallel.collectives import is_sequence_parallel_active, sequence_parallel_region

    calls = []

    def block(x):
        calls.append(is_sequence_parallel_active())
        return x.sin()

    x = torch.randn(1, 11, 3, requires_grad=True)
    with sequence_parallel_region(True):
        y = mini_sequence_forward(block, x, chunk_size=4)
    y.sum().backward()
    assert calls == [True] * 6
    assert not is_sequence_parallel_active()
    torch.testing.assert_close(x.grad, x.detach().cos())


def test_mst_routed_boundary_does_not_recompute_router_cpu():
    from areno.engine.runtime.metadata import TrainMeta
    from areno.engine.runtime.recompute import checkpoint_routed_moe_layer

    calls = []
    attention_lengths = []
    expert_lengths = []
    states = torch.randn(1, 11, 4, requires_grad=True)
    scale = nn.Parameter(torch.tensor(0.3))

    def attention(x):
        attention_lengths.append(x.shape[1])
        return x.cumsum(1)

    def route(x):
        calls.append(True)
        return torch.zeros(11, 1, dtype=torch.long), x.mean(-1).reshape(11, 1) * scale

    def expert(x, indices, weight):
        expert_lengths.append(x.shape[1])
        return x.sin() * weight.view(1, -1, 1)

    actual = checkpoint_routed_moe_layer(
        attention,
        nn.Identity(),
        route,
        expert,
        states,
        train_meta=TrainMeta(activation_checkpointing=True, mst_chunk_size=4),
    )
    attended = states.cumsum(1)
    expected = attended + attended.sin() * attended.mean(-1, keepdim=True) * scale
    torch.testing.assert_close(actual, expected)
    got = torch.autograd.grad(actual.square().mean(), (states, scale))
    ref = torch.autograd.grad(expected.square().mean(), (states, scale))
    for a, b in zip(got, ref, strict=True):
        torch.testing.assert_close(a, b)
    assert len(calls) == 1
    assert set(attention_lengths) == {11}
    assert max(expert_lengths) == 4


def test_mst_flat_tokens_and_inference_bypass_cpu():
    from areno.engine.runtime.metadata import InferMeta, TrainMeta
    from areno.engine.runtime.recompute import tokenwise_forward

    lengths = []

    def block(x):
        assert x.ndim == 2
        lengths.append(x.shape[0])
        return x.sin()

    x = torch.randn(11, 4, requires_grad=True)
    meta = TrainMeta(mst_chunk_size=4)
    y = tokenwise_forward(block, x, train_meta=meta)
    y.sum().backward()
    assert max(lengths) == 4
    torch.testing.assert_close(x.grad, x.detach().cos())
    lengths.clear()
    tokenwise_forward(block, x, train_meta=meta, infer_meta=InferMeta(mode="prefill"))
    assert lengths == [11]


def test_mst_disabled_preserves_phi4_full_sequence_mask_cpu():
    from types import SimpleNamespace

    from areno.engine.runtime.metadata import TrainMeta
    from areno.models.phi4mm.model import Phi4MMDecoderLayer

    full_mask = torch.ones(1, 10, dtype=torch.bool)

    class Attention(nn.Module):
        def forward(self, x, *args):
            return x.sin()

    class MaskedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = SimpleNamespace(vision_lora_mask=full_mask)

        def forward(self, x):
            assert self.gate_up_proj.vision_lora_mask is full_mask
            return x.cos()

    layer = Phi4MMDecoderLayer.__new__(Phi4MMDecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = layer.post_attention_layernorm = nn.Identity()
    layer.self_attn = Attention()
    layer.mlp = MaskedMLP()
    x = torch.randn(1, 5, 4, requires_grad=True)
    result = layer(x, torch.arange(10)[None], TrainMeta(sequence_parallel=True))
    expected = x + x.sin()
    expected = expected + expected.cos()
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert x.grad is not None


def _measure_mst_candidate(schedule, cost_per_token):
    size = schedule.size
    for _ in range(5):
        schedule.observe(0.4)
        schedule.record_timing(size, size * cost_per_token)


def test_mst_adaptive_budget_cpu(monkeypatch):
    monkeypatch.setenv("ARENO_MST_MEM_FRAC", "0.8")
    from areno.engine.runtime.mst import _AdaptiveChunks

    schedule = _AdaptiveChunks(torch.device("cpu"), 8192, 65536)
    for _ in range(12):
        schedule.observe(0.5)
    assert schedule.size == 1024  # Headroom alone must not increase chunks.
    _measure_mst_candidate(schedule, 10)
    assert schedule.size == 1536
    schedule.observe(0.81)
    assert schedule.size == 768
    for _ in range(12):
        schedule.observe(0.5)
    assert schedule.size == 768
    schedule.oom()
    assert schedule.size == 384
    assert schedule.best_size == 384 and schedule.best_cost is None


def test_mst_adaptive_unified_memory_pressure_cpu(monkeypatch):
    from areno.engine.runtime import mst

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (70, 100))
    monkeypatch.setattr(mst.psutil, "virtual_memory", lambda: SimpleNamespace(available=15, total=100))
    schedule = mst._AdaptiveChunks(torch.device("cpu"), 1024, 2048)
    assert schedule.pressure() == pytest.approx(0.85)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (10, 100))
    assert schedule.pressure() == pytest.approx(0.9)


@pytest.mark.parametrize("oom", [False, True])
def test_mst_adaptive_vjp_and_retry_cpu(monkeypatch, oom):
    from areno.engine.runtime import mst

    torch.manual_seed(73)
    block = nn.Sequential(nn.Linear(8, 31), nn.SiLU(), nn.Linear(31, 8)).double()
    other = copy.deepcopy(block)
    x = torch.randn(2, 23, 8, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    schedule = mst._AdaptiveChunks(x.device, 3, 46)
    pressure = iter([0.5] * 7 + [0.81] + [0.5] * 200)
    monkeypatch.setattr(schedule, "pressure", lambda: next(pressure))
    monkeypatch.setattr(mst, "_adaptive_chunks", lambda *args: schedule)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    original = mst._chunk_vjp
    calls = []
    failed = False

    def vjp(*args):
        nonlocal failed
        start, end = args[-2:]
        calls.append((start, end))
        grads = original(*args)
        if oom and not failed and start > 0:
            failed = True
            raise torch.cuda.OutOfMemoryError("simulated allocation failure before gradient commit")
        return grads

    monkeypatch.setattr(mst, "_chunk_vjp", vjp)
    expected = block(x)
    actual = mst.mini_sequence_forward(other, y, chunk_size=3)
    upstream = torch.randn_like(actual)
    reference = torch.autograd.grad(expected, (x, *block.parameters()), upstream)
    got = torch.autograd.grad(actual, (y, *other.parameters()), upstream)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(got, reference, strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    assert len({end - start for start, end in calls}) > 1
    if oom:
        assert failed
        assert calls[1][0] == calls[2][0]
        assert calls[2][1] < calls[1][1]


@pytest.mark.parametrize("budget", [0.5, 0.8, 0.95])
def test_mst_respects_memory_budget_cpu(monkeypatch, budget):
    from areno.engine.runtime.mst import _AdaptiveChunks

    monkeypatch.setenv("ARENO_MST_MEM_FRAC", str(budget))
    schedule = _AdaptiveChunks(torch.device("cpu"), 1024, 65536)
    for _ in range(4):
        schedule.observe(budget * 0.8)
    assert schedule.size == 1024
    schedule.observe(budget * 0.98)
    assert schedule.size == 1024
    schedule.observe(budget)
    assert schedule.size == 512


def test_mst_spark_uses_reclaimable_host_memory_cpu(monkeypatch):
    from areno.engine.runtime import mst

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="NVIDIA GB10"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (20, 100))
    monkeypatch.setattr(mst.psutil, "virtual_memory", lambda: SimpleNamespace(available=40, total=100))
    monkeypatch.setenv("ARENO_MST_MEM_FRAC", "0.8")
    schedule = mst._AdaptiveChunks(torch.device("cuda"), 1024, 65536)
    assert schedule.pressure() == pytest.approx(0.6)
    for _ in range(4):
        schedule.observe(schedule.pressure())
    assert schedule.size == 1024
    monkeypatch.setattr(mst.psutil, "virtual_memory", lambda: SimpleNamespace(available=15, total=100))
    schedule.observe(schedule.pressure())
    assert schedule.size == 512
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="NVIDIA H100"))
    discrete = mst._AdaptiveChunks(torch.device("cuda"), 1024, 65536)
    monkeypatch.setattr(mst.psutil, "virtual_memory", lambda: SimpleNamespace(available=40, total=100))
    assert discrete.pressure() == pytest.approx(0.8)


def test_mst_reuses_schedule_per_module_and_releases_dead_owners(monkeypatch):
    import gc
    import weakref

    from areno.engine.runtime import mst

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="NVIDIA GB10"))
    device = torch.device("cuda:0")
    owner, other = nn.Linear(2, 2), nn.Linear(2, 2)
    schedule = mst._adaptive_chunks(device, 1024, 65536, owner)
    for _ in range(8):
        schedule.observe(0.4)
    assert schedule.size == 1024
    assert mst._adaptive_chunks(device, 1024, 65536, owner) is schedule
    assert mst._adaptive_chunks(device, 1024, 65536, other).size == 1024
    assert mst._adaptive_chunks(device, 1024, 100, owner).size == 1024
    schedule.observe(0.4)
    assert schedule.size == 1024
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None


def test_mst_speed_search_selects_measured_optimum(monkeypatch):
    from areno.engine.runtime.mst import _AdaptiveChunks

    monkeypatch.setenv("ARENO_MST_MEM_FRAC", "0.8")
    schedule = _AdaptiveChunks(torch.device("cpu"), 1024, 65536)
    _measure_mst_candidate(schedule, 10)
    assert schedule.size == 1536
    _measure_mst_candidate(schedule, 8)
    assert schedule.size == 2048
    _measure_mst_candidate(schedule, 9)
    assert schedule.size == 1024  # Probe below the best, not the larger loser.
    _measure_mst_candidate(schedule, 10)
    assert schedule.size == schedule.best_size == 1536
    assert schedule._holding
    for _ in range(20):
        schedule.observe(0.79)
        schedule.record_timing(1536, 1536 * 8.1)
    assert schedule.size == 1536
    schedule.observe(0.81)
    assert schedule.size == 768


def test_mst_can_select_smaller_chunks_and_stops_probing_after_stabilizing(monkeypatch):
    from areno.engine.runtime.mst import _AdaptiveChunks

    schedule = _AdaptiveChunks(torch.device("cpu"), 1024, 65536)
    _measure_mst_candidate(schedule, 10)
    _measure_mst_candidate(schedule, 12)  # 1536 is slower.
    assert schedule.size == 512
    _measure_mst_candidate(schedule, 7)
    assert schedule.size == 1
    _measure_mst_candidate(schedule, 30)
    _measure_mst_candidate(schedule, 10)
    assert schedule.size == schedule.best_size == 512
    for _ in range(1000):
        schedule.observe(0.4)
        schedule.record_timing(512, 512 * 9)
    assert schedule.size == schedule.best_size == 512
    assert schedule._frozen
    assert schedule.start_timing(512) is None
    # Memory backoff remains available without restarting the throughput search.
    schedule.observe(1)
    assert schedule.size == 256
    assert schedule._frozen
    schedule.oom()
    assert schedule.size == 128
    for _ in range(1000):
        schedule.observe(0.4)
        schedule.record_timing(128, 128)
    # This microbatch's OOM ceiling prevents an immediate failed retry.
    assert schedule.size == 128
    assert schedule.throughput_baseline == 512
    assert schedule.start_timing(128) is None
    # A later microbatch can recover gradually as memory headroom returns.
    schedule.maximum = 65536
    for _ in range(8):
        schedule.observe(0.4)
    assert schedule.size == 512
    assert schedule.start_timing(512) is None
    assert schedule._frozen


def test_mst_warmup_noise_tails_and_unfinished_events_do_not_drive_search(monkeypatch):
    from areno.engine.runtime.mst import _AdaptiveChunks

    schedule = _AdaptiveChunks(torch.device("cpu"), 1024, 65536)
    schedule.observe(0.4)
    schedule.record_timing(1024, 1e8)  # Discard first execution.
    for cost in [10, 10.1, 9.9, 10]:
        schedule.observe(0.4)
        schedule.record_timing(1024, cost * 1024)
    assert schedule.size == 1536
    _measure_mst_candidate(schedule, 9.8)  # <5% gain is indistinguishable from noise.
    assert schedule.best_size == 1024
    old = schedule.size
    schedule.record_timing(17, 0.001)  # Partial tail cannot look artificially fast.
    assert schedule.size == old
    from unittest.mock import Mock

    begin = Mock()
    end = Mock()
    end.query.return_value = False
    schedule._pending.append((begin, end, old, schedule._generation))
    schedule.poll_timings()
    begin.elapsed_time.assert_not_called()
    end.synchronize.assert_not_called()
    end.query.return_value = True
    begin.elapsed_time.return_value = old * 11
    schedule.poll_timings()
    begin.elapsed_time.assert_called_once_with(end)
    end.synchronize.assert_not_called()


def test_mst_current_train_eta_and_log_throttle(monkeypatch, caplog):
    import logging

    from areno.engine.runtime import mst

    monkeypatch.setattr(mst.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(mst.logger, "propagate", False)
    clock = [100.0]
    monkeypatch.setattr(mst.time, "monotonic", lambda: clock[0])
    with mst.mst_training_progress():
        progress = mst._TRAIN_PROGRESS.get()
        progress.total = 400
        progress.begin()
        clock[0] = 110.0
        progress.advance(100, 1024)
        assert not caplog.records
        clock[0] = 120.0
        with caplog.at_level(logging.INFO, logger=mst.__name__):
            progress.advance(100, 1280)
        assert "remaining~30.0s" in caplog.text
        assert "MST_progress=200/400" in caplog.text
        assert "rough estimate" in caplog.text
        clock[0] = 125.0
        progress.advance(100, 1280)
        assert len(caplog.records) == 1
    assert mst._TRAIN_PROGRESS.get() is None


def test_mst_speed_driven_chunks_preserve_forward_and_gradients_cpu(monkeypatch):
    from areno.engine.runtime import mst

    torch.manual_seed(17)
    block = nn.Sequential(nn.Linear(8, 19), nn.SiLU(), nn.Linear(19, 8)).double()
    other = copy.deepcopy(block)
    x = torch.randn(1, 4096, 8, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    schedule = mst._AdaptiveChunks(x.device, 64, 4096)
    monkeypatch.setattr(mst, "_adaptive_chunks", lambda *args: schedule)
    monkeypatch.setattr(schedule, "pressure", lambda: 0.4)
    monkeypatch.setattr(schedule, "start_timing", lambda size: size if size == schedule.size else None)
    monkeypatch.setattr(
        schedule,
        "end_timing",
        lambda size: schedule.record_timing(size, size * (10 if size == 64 else 8)) if size else None,
    )
    expected = block(x)
    actual = mst.mini_sequence_forward(other, y, chunk_size=64)
    upstream = torch.randn_like(actual)
    reference = torch.autograd.grad(expected, (x, *block.parameters()), upstream)
    got = torch.autograd.grad(actual, (y, *other.parameters()), upstream)
    assert schedule.size > 64
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for observed, baseline in zip(got, reference, strict=True):
        torch.testing.assert_close(observed, baseline, atol=1e-10, rtol=1e-11)


def test_mst_pending_cuda_events_are_bounded_without_synchronization_cpu(monkeypatch):
    from unittest.mock import Mock

    from areno.engine.runtime import mst

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="NVIDIA GB10"))
    event = Mock()
    event.query.return_value = False
    factory = Mock(return_value=event)
    monkeypatch.setattr(torch.cuda, "Event", factory)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: "stream")
    schedule = mst._AdaptiveChunks(torch.device("cuda"), 1024, 4096)
    for _ in range(12):
        schedule.end_timing(schedule.start_timing(1024))
        schedule.poll_timings()
    assert len(schedule._pending) == 8
    assert factory.call_count == 16
    event.synchronize.assert_not_called()
    assert schedule.start_timing(17) is None
