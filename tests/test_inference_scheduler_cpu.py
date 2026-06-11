from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch

import areno.engine.inference as inference_mod
import areno.engine.worker as worker_mod
from areno.engine.api import ArenoEngine, _chunk_prompts_for_prefill_budget, _merge_async_dp_rollouts
from areno.engine.data import SamplingParams
from areno.engine.data.rollout_state import InferenceBatchState
from areno.engine.inference import InferenceManager
from areno.engine.protocol import Command, Op, RolloutPayload
from areno.engine.runtime.rollout import _build_rollout_from_rows, partial_tail_threshold
from tests.helpers import PatchedContext


class _FakeInferenceManager(InferenceManager):
    """CPU-only manager that exercises rollout scheduling without a model."""

    def __init__(self):
        super().__init__(SimpleNamespace())
        self.device = torch.device("cpu")

    def _infer_next_token_tensor(self, payload):
        count = int(payload.sample_indices.numel())
        return torch.ones(count, dtype=torch.long), torch.zeros(count, dtype=torch.float32)

    def _infer_decode_next_token_tensor(
        self,
        next_tokens,
        position_ids,
        cache_seqlens,
        block_table,
        active_count,
        sampling_params,
        sample_generator,
        *,
        sample_step,
        eos_token_id,
    ):
        del position_ids, cache_seqlens, block_table, active_count, sampling_params, sample_generator, eos_token_id
        return next_tokens + 1, torch.zeros_like(next_tokens, dtype=torch.float32) - float(sample_step)

def test_no_sync_rollout_continues_pending_prompts_beyond_running_slots():
    """A single rollout call should drain pending rows while respecting slot limits."""

    manager = _FakeInferenceManager()
    state = InferenceBatchState(
        prompts=[[10], [11], [12], [13]],
        max_new_tokens=3,
        max_running_seqs=2,
        max_cache_len=8,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=4,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            prompt_indices=[0, 1, 2, 3],
        )

    assert state.generated == [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert state.finish_reason == ["length", "length", "length", "length"]


def test_no_sync_rollout_drains_pending_prompts_when_first_token_hits_length_cap():
    """Rows with max_new_tokens=1 should finish during prefill and free slots."""

    manager = _FakeInferenceManager()
    state = InferenceBatchState(
        prompts=[[10], [11], [12], [13]],
        max_new_tokens=1,
        max_running_seqs=2,
        max_cache_len=4,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=2,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            prompt_indices=[0, 1, 2, 3],
        )

    assert state.generated == [[1], [1], [1], [1]]
    assert state.finish_reason == ["length", "length", "length", "length"]


def test_no_sync_rollout_marks_small_active_tail_partial():
    """Async batches can cut a tiny tail so it can be resumed with later requests."""

    class TailManager(_FakeInferenceManager):
        def _infer_decode_next_token_tensor(self, *args, **kwargs):
            active_count = int(args[4])
            next_tokens = args[0]
            values = next_tokens + 1
            # Finish every row except row 0 on the first decode step.
            if active_count == 4:
                values = torch.tensor([2, 99, 99, 99], dtype=torch.long)
            return values, torch.zeros_like(values, dtype=torch.float32)

    manager = TailManager()
    state = InferenceBatchState(
        prompts=[[10], [11], [12], [13]],
        max_new_tokens=6,
        max_running_seqs=4,
        max_cache_len=8,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=8,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(stop_token_ids=(99,)),
            eos_token_id=None,
            prompt_indices=[0, 1, 2, 3],
            partial_tail_threshold=1,
            partial_tail_ready_callback=lambda active_count: True,
        )

    assert state.generated[0] == [1, 2]
    assert state.finish_reason[0] == "partial"
    assert state.finish_reason[1:] == ["stop", "stop", "stop"]


def test_no_sync_rollout_keeps_tail_when_no_waiting_batch():
    """Small tails should continue when no compatible queued requests can refill them."""

    class TailManager(_FakeInferenceManager):
        def _infer_decode_next_token_tensor(self, *args, **kwargs):
            active_count = int(args[4])
            next_tokens = args[0]
            values = next_tokens + 1
            if active_count == 4:
                values = torch.tensor([2, 99, 99, 99], dtype=torch.long)
            return values, torch.zeros_like(values, dtype=torch.float32)

    manager = TailManager()
    state = InferenceBatchState(
        prompts=[[10], [11], [12], [13]],
        max_new_tokens=4,
        max_running_seqs=4,
        max_cache_len=8,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=8,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(stop_token_ids=(99,)),
            eos_token_id=None,
            prompt_indices=[0, 1, 2, 3],
            partial_tail_threshold=1,
            partial_tail_ready_callback=lambda active_count: False,
        )

    assert state.generated[0] == [1, 2, 3, 4]
    assert state.finish_reason[0] == "length"


def test_no_sync_rollout_respects_partial_tail_min_tokens():
    """A tail should not be cut before it has made enough decode progress."""

    manager = _FakeInferenceManager()
    state = InferenceBatchState(
        prompts=[[10]],
        max_new_tokens=3,
        max_running_seqs=4,
        max_cache_len=8,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=8,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            prompt_indices=[0],
            partial_tail_threshold=1,
            partial_tail_min_tokens=4,
        )

    assert state.generated == [[1, 2, 3]]
    assert state.finish_reason == ["length"]


def test_no_sync_rollout_respects_partial_tail_cooldown():
    """The same rollout row should not be cut again before its cooldown expires."""

    manager = _FakeInferenceManager()
    state = InferenceBatchState(
        prompts=[[10]],
        max_new_tokens=3,
        max_running_seqs=4,
        max_cache_len=8,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=8,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            prompt_indices=[0],
            partial_tail_threshold=1,
            partial_tail_ready_callback=lambda active_count: True,
            partial_tail_cooldown_until_s=[inference_mod.time.monotonic() + 100.0],
        )

    assert state.generated == [[1, 2, 3]]
    assert state.finish_reason == ["length"]


def test_partial_tail_threshold_uses_local_capacity():
    """Single-request async payloads can still use configured local capacity for tail cuts."""

    assert partial_tail_threshold(local_running=1, coalesce_timeout_s=5.0) == 0
    assert partial_tail_threshold(local_running=8, coalesce_timeout_s=0.0) == 0
    assert partial_tail_threshold(local_running=8, coalesce_timeout_s=5.0) == 1


def test_async_single_request_payload_uses_capacity_for_partial_tail():
    """Serve singletons should still allocate local capacity so long tails can be cut."""

    class ClusterStub:
        def __init__(self):
            self.payload = None

        async def call_async(self, op, payload):
            self.payload = payload
            assert op is Op.INFER_ROLLOUT
            output = _build_rollout_from_rows(
                [[1]],
                [[2]],
                ["stop"],
                [torch.tensor([-0.1], dtype=torch.float32)],
                metrics=None,
            )
            return [output, None]

    cluster = ClusterStub()
    engine = object.__new__(ArenoEngine)
    engine.cluster = cluster
    engine.config = SimpleNamespace(tp_size=1, dp_size=2, runtime=SimpleNamespace(kv_block_size=16))

    result = asyncio.run(
        engine._generate_rollout_async_once(
            [[1]],
            max_new_tokens=16,
            max_running_prompts=32,
            eos_token_id=None,
            sampling_params=SamplingParams(),
            coalesce_timeout_s=0.01,
        )
    )

    assert result.response_ids == [[2]]
    assert cluster.payload.max_running_seqs == 16
    assert cluster.payload.partial_tail_threshold == 2
    assert cluster.payload.partial_tail_min_tokens == 256


def test_partial_continuation_allows_repeated_tail_cuts_until_progress_completes():
    """A resumed old request may be cut again, but every cut must make progress."""

    engine = object.__new__(ArenoEngine)
    calls = []
    initial = _build_rollout_from_rows(
        [[10]],
        [[1, 2]],
        ["partial"],
        [torch.tensor([-0.1, -0.2], dtype=torch.float32)],
        metrics=None,
    )
    first_continuation = _build_rollout_from_rows(
        [[10, 1, 2]],
        [[3]],
        ["partial"],
        [torch.tensor([-0.3], dtype=torch.float32)],
        metrics=None,
    )
    second_continuation = _build_rollout_from_rows(
        [[10, 1, 2, 3]],
        [[4]],
        ["stop"],
        [torch.tensor([-0.4], dtype=torch.float32)],
        metrics=None,
    )
    continuations = [first_continuation, second_continuation]

    async def fake_once(prompts, **kwargs):
        calls.append((prompts, kwargs))
        return continuations.pop(0)

    engine._generate_rollout_async_once = fake_once
    result = asyncio.run(
        engine._complete_partial_async_rollout(
            initial,
            max_new_tokens=4,
            max_running_prompts=8,
            eos_token_id=None,
            sampling_params=SamplingParams(),
            decode_progress_interval_s=0.0,
            coalesce_timeout_s=5.0,
        )
    )

    assert result.prompt_ids == [[10]]
    assert result.response_ids == [[1, 2, 3, 4]]
    assert result.finish_reason == ["stop"]
    assert [call[0] for call in calls] == [[[10, 1, 2]], [[10, 1, 2, 3]]]


def test_prefill_chunking_uses_global_max_running_prompts():
    """A 256-row flat rollout should stay one chunk even when dp_size is 8."""

    prompts = [[idx] for idx in range(256)]

    chunks = _chunk_prompts_for_prefill_budget(
        prompts,
        max_running_prompts=256,
        dp_size=8,
        max_prefill_tokens=1024,
    )

    assert [len(chunk) for chunk in chunks] == [256]


def test_decode_progress_log_is_worker_aggregated(monkeypatch):
    """Concurrent batch progress should be throttled per worker."""

    manager = _FakeInferenceManager()
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)
    messages = []
    monkeypatch.setattr(inference_mod.logger, "info", lambda msg, *args: messages.append(msg % args))

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx):
        manager._record_decode_progress(
            enabled=True,
            interval_s=0.0,
            rollout_key=1,
            active_count=4,
            token_delta=4,
            cache_tokens=8,
            sample_step=2,
            max_new_tokens=16,
        )
        manager._record_decode_progress(
            enabled=True,
            interval_s=60.0,
            rollout_key=2,
            active_count=6,
            token_delta=6,
            cache_tokens=9,
            sample_step=2,
            max_new_tokens=16,
        )

    assert len(messages) == 1
    assert "concurrent_batches=2" in messages[0]
    assert "active=10" in messages[0]


def test_worker_coalesces_rollout_commands_until_local_capacity():
    """Worker-side rollout coalescing should stop once local capacity is full."""

    first = _rollout_command(1, [[1]], target=2)
    second = _rollout_command(2, [[2]], target=2)
    third = _rollout_command(3, [[3]], target=2)
    worker = object.__new__(worker_mod.ArenoWorker)
    worker._cmd_queue = _QueueDouble([second, third])
    worker._deferred_commands = []
    ctx = SimpleNamespace(dp_rank=0)

    with PatchedContext(worker_mod, get_tp_context=lambda: ctx):
        commands = worker.collect_rollout_commands(first)

    assert [command.request_id for command in commands] == [1, 2]
    assert [command.request_id for command in worker._deferred_commands] == []
    assert [command.request_id for command in worker._cmd_queue.items] == [third.request_id]


def test_worker_coalesced_rollout_distributes_single_prompt_requests_across_dp():
    """Coalesced one-prompt requests should fill all DP lanes, not only DP rank 0."""

    first = _rollout_command(1, [[1]], target=2, dp_size=2)
    second = _rollout_command(2, [[2]], target=2, dp_size=2)
    third = _rollout_command(3, [[3]], target=2, dp_size=2)
    fourth = _rollout_command(4, [[4]], target=2, dp_size=2)
    worker = object.__new__(worker_mod.ArenoWorker)
    worker._cmd_queue = _QueueDouble([second, third, fourth])
    worker._deferred_commands = []
    ctx = SimpleNamespace(dp_rank=1)

    with PatchedContext(worker_mod, get_tp_context=lambda: ctx):
        commands = worker.collect_rollout_commands(first)
        merged, counts = worker_mod._merge_rollout_payloads([command.payload for command in commands], current_dp_rank=1)

    assert [command.request_id for command in commands] == [1, 2, 3, 4]
    assert merged.prompts_by_dp == [[[1], [3]], [[2], [4]]]
    assert counts == [0, 1, 0, 1]


def test_worker_early_finished_rows_build_per_request_rollout():
    """A finished row can be converted to its request output before the batch ends."""

    generated = torch.tensor([[10, 11, 0], [20, 21, 22]], dtype=torch.long)
    logprobs = torch.tensor([[-0.1, -0.2, 0.0], [-0.3, -0.4, -0.5]], dtype=torch.float32)
    response_lens = torch.tensor([2, 3], dtype=torch.long)

    output = worker_mod._build_rollout_from_tensor_rows(
        prompts=[[1], [2]],
        generated=generated,
        logprobs=logprobs,
        response_lens=response_lens,
        finish_reasons=["stop", "length"],
        start=0,
        end=1,
        truncate_stop_token_ids=(),
    )

    assert output.prompt_ids == [[1]]
    assert output.response_ids == [[10, 11]]
    assert output.finish_reason == ["stop"]
    torch.testing.assert_close(output.logprobs[0, :2], torch.tensor([-0.1, -0.2]))


def test_worker_coalesced_rollout_returns_finished_request_early():
    """Worker coalescing should publish completed requests before the full batch ends."""

    first = _rollout_command(1, [[1]], target=2)
    second = _rollout_command(2, [[2]], target=2)
    generated = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)
    logprobs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], dtype=torch.float32)
    response_lens = torch.tensor([2, 2], dtype=torch.long)
    worker = object.__new__(worker_mod.ArenoWorker)
    worker._rank = 0
    worker._result_queue = _ResultQueueDouble()
    ctx = SimpleNamespace(dp_rank=0, is_rank0=True)

    def infer_rollout(payload, finished_callback=None, partial_tail_ready_callback=None):
        finished_callback(torch.tensor([0]), generated, logprobs, response_lens, "stop", ())
        return worker_mod._build_rollout_from_tensor_rows(
            payload.prompts_by_dp[0],
            generated,
            logprobs,
            response_lens,
            ["stop", "length"],
            0,
            2,
            (),
        )

    worker.infer_rollout = infer_rollout

    with PatchedContext(worker_mod, get_tp_context=lambda: ctx):
        remaining = worker.run_rollout_commands([first, second])

    assert [item[1].request_id for item in worker._result_queue.items] == [1]
    assert [request_id for request_id, _ in remaining] == [2]
    assert worker._result_queue.items[0][1].payload.response_ids == [[10, 11]]
    assert remaining[0][1].response_ids == [[20, 21]]


def test_async_single_prompt_merge_accepts_non_zero_dp_owner():
    """Async coalescing may assign a single request to any DP rank."""

    output = worker_mod._build_rollout_from_tensor_rows(
        prompts=[[2]],
        generated=torch.tensor([[20, 21]], dtype=torch.long),
        logprobs=torch.tensor([[-0.3, -0.4]], dtype=torch.float32),
        response_lens=torch.tensor([2], dtype=torch.long),
        finish_reasons=["stop"],
        start=0,
        end=1,
        truncate_stop_token_ids=(),
    )

    merged = _merge_async_dp_rollouts([worker_mod._empty_rollout(), output], total_count=1)

    assert merged.prompt_ids == [[2]]
    assert merged.response_ids == [[20, 21]]


def test_async_coalesced_single_prompt_requests_merge_from_each_dp_owner():
    """Each coalesced one-prompt request should merge from the DP rank that owns it."""

    outputs_by_request = []
    for owner_dp in range(4):
        outputs = [worker_mod._empty_rollout() for _ in range(4)]
        outputs[owner_dp] = worker_mod._build_rollout_from_tensor_rows(
            prompts=[[owner_dp]],
            generated=torch.tensor([[100 + owner_dp]], dtype=torch.long),
            logprobs=torch.tensor([[-0.1 * (owner_dp + 1)]], dtype=torch.float32),
            response_lens=torch.tensor([1], dtype=torch.long),
            finish_reasons=["stop"],
            start=0,
            end=1,
            truncate_stop_token_ids=(),
        )
        outputs_by_request.append(_merge_async_dp_rollouts(outputs, total_count=1))

    assert [output.prompt_ids for output in outputs_by_request] == [[[0]], [[1]], [[2]], [[3]]]
    assert [output.response_ids for output in outputs_by_request] == [[[100]], [[101]], [[102]], [[103]]]


def test_async_merge_falls_back_to_non_empty_outputs_when_dp_order_assumption_fails():
    """Async coalescing can break the original per-request DP split assumption."""

    outputs = [
        worker_mod._empty_rollout(),
        worker_mod._build_rollout_from_tensor_rows(
            prompts=[[1]],
            generated=torch.tensor([[101]], dtype=torch.long),
            logprobs=torch.tensor([[-0.1]], dtype=torch.float32),
            response_lens=torch.tensor([1], dtype=torch.long),
            finish_reasons=["stop"],
            start=0,
            end=1,
            truncate_stop_token_ids=(),
        ),
        worker_mod._empty_rollout(),
        worker_mod._build_rollout_from_tensor_rows(
            prompts=[[3]],
            generated=torch.tensor([[103]], dtype=torch.long),
            logprobs=torch.tensor([[-0.3]], dtype=torch.float32),
            response_lens=torch.tensor([1], dtype=torch.long),
            finish_reasons=["stop"],
            start=0,
            end=1,
            truncate_stop_token_ids=(),
        ),
    ]

    merged = _merge_async_dp_rollouts(outputs, total_count=2)

    assert merged.prompt_ids == [[1], [3]]
    assert merged.response_ids == [[101], [103]]


class _QueueDouble:
    """Small FIFO queue double for worker coalescing tests."""

    def __init__(self, items):
        self.items = list(items)

    def get(self, timeout=None):
        del timeout
        if not self.items:
            raise worker_mod.queue.Empty
        return self.items.pop(0)


class _ResultQueueDouble:
    """Capture worker result queue writes for early-return tests."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_worker_refill_target_requires_half_batch_refill():
    """Partial tail should wait for enough queued rows to rebuild a useful batch."""

    payload = _rollout_command(1, [[1]], target=16, dp_size=2).payload

    assert worker_mod._rollout_refill_target(payload, active_count=1) == 16
    assert worker_mod._rollout_refill_target(payload, active_count=4) == 16
    assert worker_mod._rollout_refill_target(payload, active_count=20) == 16


def _rollout_command(request_id: int, prompts: list[list[int]], *, target: int, dp_size: int = 1) -> Command:
    prompts_by_dp = [prompts[rank::dp_size] for rank in range(dp_size)]
    prompt_indices_by_dp = [list(range(len(prompts)))[rank::dp_size] for rank in range(dp_size)]
    payload = RolloutPayload(
        prompts_by_dp=prompts_by_dp,
        prompt_indices_by_dp=prompt_indices_by_dp,
        max_new_tokens=4,
        eos_token_id=None,
        sampling_params=SamplingParams(),
        max_running_seqs=len(prompts),
        max_cache_len=8,
        max_blocks_per_seq=2,
        max_prefill_tokens=16,
        num_blocks=2,
        block_size=4,
        coalesce_max_running_seqs=target,
        coalesce_timeout_s=1.0,
    )
    return Command(op=Op.INFER_ROLLOUT, payload=payload, request_id=request_id)
