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
from areno.engine.runtime.rollout import _build_rollout_from_rows
from tests.helpers import PatchedContext


class _FakeInferenceManager(InferenceManager):
    """CPU-only manager that exercises rollout scheduling without a model."""

    def __init__(self):
        super().__init__(SimpleNamespace())
        self.device = torch.device("cpu")
        self.prefill_only_chunks = 0
        self.ops = []

    def _infer_next_token_tensor(self, payload):
        count = int(payload.sample_indices.numel())
        self.ops.append(("sample_prefill", count))
        return torch.ones(count, dtype=torch.long), torch.zeros(count, dtype=torch.float32)

    def _run_prefill_payload(self, payload):
        assert int(payload.sample_indices.numel()) == 0
        self.prefill_only_chunks += 1
        self.ops.append(("prefill_chunk", int(payload.input_ids.numel())))

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
        self.ops.append(("decode", int(next_tokens.numel())))
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


def test_chunked_prefill_runs_intermediate_chunks_without_sampling():
    """Long prompts should fill paged KV in chunks and only sample on final chunk."""

    manager = _FakeInferenceManager()
    state = InferenceBatchState(
        prompts=[[10, 11, 12, 13, 14]],
        max_new_tokens=1,
        max_running_seqs=1,
        max_cache_len=8,
        max_prefill_tokens=2,
        kv_block_size=4,
        num_cache_blocks=2,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            prompt_indices=[0],
        )

    assert manager.prefill_only_chunks == 2
    assert state.generated == [[1]]
    assert state.finish_reason == ["length"]


def test_prefill_reserves_prompt_blocks_without_max_new_token_overreservation():
    """Paged KV should grow for decode instead of reserving full response length upfront."""

    state = InferenceBatchState(
        prompts=[[1, 2]],
        max_new_tokens=100,
        max_running_seqs=1,
        max_cache_len=128,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=2,
    )

    payload = state.build_prefill_payload()

    assert payload is not None
    assert state._seq_to_blocks == {0: [0]}
    assert state._free_blocks == [1]
    state.ensure_decode_blocks([0], [4])
    assert state._seq_to_blocks == {0: [0, 1]}


def test_chunked_prefill_interleaves_with_active_decode():
    """A pending long prompt should not run all prefill chunks before active rows decode."""

    manager = _FakeInferenceManager()
    state = InferenceBatchState(
        prompts=[[10], [20, 21, 22, 23, 24]],
        max_new_tokens=3,
        max_running_seqs=2,
        max_cache_len=8,
        max_prefill_tokens=2,
        kv_block_size=4,
        num_cache_blocks=4,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            prompt_indices=[0, 1],
        )

    assert manager.ops[:4] == [("sample_prefill", 1), ("decode", 1), ("prefill_chunk", 2), ("decode", 1)]


def test_async_single_request_payload_uses_configured_local_capacity():
    """Serve singletons should still allocate local capacity for dynamic refill."""

    class ClusterStub:
        def __init__(self):
            self.payload = None

        async def call_async(self, op, payload, **kwargs):
            del kwargs
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
        )
    )

    assert result.response_ids == [[2]]
    assert cluster.payload.max_running_seqs == 16


def test_async_single_prompt_requests_round_robin_across_dp_ranks():
    """Independent serve requests should not all land on DP rank 0."""

    class ClusterStub:
        def __init__(self):
            self.payloads = []

        async def call_async(self, op, payload, **kwargs):
            del kwargs
            self.payloads.append(payload)
            assert op is Op.INFER_ROLLOUT
            owner = next(rank for rank, rows in enumerate(payload.prompts_by_dp) if rows)
            outputs = [worker_mod._empty_rollout() for _ in range(4)]
            outputs[owner] = _build_rollout_from_rows(
                [[owner]],
                [[owner + 10]],
                ["stop"],
                [torch.tensor([-0.1], dtype=torch.float32)],
                metrics=None,
            )
            return outputs

    cluster = ClusterStub()
    engine = object.__new__(ArenoEngine)
    engine.cluster = cluster
    engine.config = SimpleNamespace(tp_size=1, dp_size=4, runtime=SimpleNamespace(kv_block_size=16))

    async def run_requests():
        return [
            await engine._generate_rollout_async_once(
                [[request_idx]],
                max_new_tokens=16,
                max_running_prompts=32,
                eos_token_id=None,
                sampling_params=SamplingParams(),
            )
            for request_idx in range(4)
        ]

    outputs = asyncio.run(run_requests())

    assert [[len(rows) for rows in payload.prompts_by_dp] for payload in cluster.payloads] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert [output.response_ids for output in outputs] == [[[10]], [[11]], [[12]], [[13]]]


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
        )
        manager._record_decode_progress(
            enabled=True,
            interval_s=60.0,
            rollout_key=2,
            active_count=6,
            token_delta=6,
        )

    assert len(messages) == 1
    assert "active=10" in messages[0]
    assert "cuda_graph=False" in messages[0]
    assert "step=" not in messages[0]
    assert "window_tokens=" not in messages[0]
    assert "cache_tokens=" not in messages[0]


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


def test_worker_split_rollout_output_by_explicit_rows():
    """Continuous refill demux must follow state row ids, not request ranges."""

    merged = _build_rollout_from_rows(
        [[1], [2], [3]],
        [[10], [20, 21], [30]],
        ["stop", "length", "stop"],
        [
            torch.tensor([-0.1], dtype=torch.float32),
            torch.tensor([-0.2, -0.3], dtype=torch.float32),
            torch.tensor([-0.4], dtype=torch.float32),
        ],
        metrics=None,
    )

    first, second = worker_mod._split_rollout_output_by_rows(merged, [[0, 2], [1]])

    assert first.prompt_ids == [[1], [3]]
    assert first.response_ids == [[10], [30]]
    assert first.finish_reason == ["stop", "stop"]
    torch.testing.assert_close(first.logprobs[0, :1], torch.tensor([-0.1]))
    torch.testing.assert_close(first.logprobs[1, :1], torch.tensor([-0.4]))
    assert second.prompt_ids == [[2]]
    assert second.response_ids == [[20, 21]]
    assert second.finish_reason == ["length"]
    torch.testing.assert_close(second.logprobs[0, :2], torch.tensor([-0.2, -0.3]))


def test_worker_continuous_rollout_refills_from_waiting_request_and_returns_finished_early():
    """Worker decode should admit queued requests without waiting for a new batch."""

    first = _rollout_command(1, [[1]], target=2)
    second = _rollout_command(2, [[2]], target=2)
    generated = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)
    logprobs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], dtype=torch.float32)
    response_lens = torch.tensor([2, 2], dtype=torch.long)
    worker = object.__new__(worker_mod.ArenoWorker)
    worker._rank = 0
    worker._result_queue = _ResultQueueDouble()
    worker._current_request_ids = [1]
    ctx = SimpleNamespace(dp_rank=0, is_rank0=True, world_size=1, device=torch.device("cpu"))

    class AliasedState:
        def __init__(self, prompts):
            self.prompts = prompts

        def append_prompts(self, prompts):
            start = len(self.prompts)
            self.prompts.extend(prompts)
            return list(range(start, start + len(prompts)))

    def infer_rollout(payload, finished_callback=None, refill_callback=None):
        assert refill_callback is not None
        refill_callback(AliasedState(payload.prompts_by_dp[0]))
        finished_callback(torch.tensor([0]), generated, logprobs, response_lens, "stop", ())
        return worker_mod._build_rollout_from_tensor_rows(
            payload.prompts_by_dp[0],
            generated,
            logprobs,
            response_lens,
            ["stop", "length"],
            0,
            len(payload.prompts_by_dp[0]),
            (),
        )

    worker.infer_rollout = infer_rollout

    worker._cmd_queue = _QueueDouble([second])
    worker._deferred_commands = []

    with PatchedContext(worker_mod, get_tp_context=lambda: ctx):
        remaining = worker.run_rollout_command(first)

    assert [item[1].request_id for item in worker._result_queue.items] == [1]
    assert [request_id for request_id, _ in remaining] == [2]
    assert worker._result_queue.items[0][1].payload.response_ids == [[10, 11]]
    assert remaining[0][1].response_ids == [[20, 21]]


def test_worker_refill_does_not_double_append_aliased_payload_prompts():
    """InferenceBatchState owns the payload prompt list, so refill must not append it twice."""

    first = _rollout_command(1, [[1]], target=3)
    second = _rollout_command(2, [[2]], target=3)
    third = _rollout_command(3, [[3]], target=3)
    generated = torch.tensor([[10], [20], [30]], dtype=torch.long)
    logprobs = torch.tensor([[-0.1], [-0.2], [-0.3]], dtype=torch.float32)
    response_lens = torch.tensor([1, 1, 1], dtype=torch.long)
    worker = object.__new__(worker_mod.ArenoWorker)
    worker._rank = 0
    worker._result_queue = _ResultQueueDouble()
    worker._current_request_ids = [1]
    worker._cmd_queue = _QueueDouble([second, third])
    worker._deferred_commands = []
    ctx = SimpleNamespace(dp_rank=0, is_rank0=True, world_size=1, device=torch.device("cpu"))

    class AliasedState:
        def __init__(self, prompts):
            self.prompts = prompts

        def append_prompts(self, prompts):
            start = len(self.prompts)
            self.prompts.extend(prompts)
            return list(range(start, start + len(prompts)))

    def infer_rollout(payload, finished_callback=None, refill_callback=None):
        assert refill_callback is not None
        refill_callback(AliasedState(payload.prompts_by_dp[0]))
        assert payload.prompts_by_dp[0] == [[1], [2], [3]]
        finished_callback(torch.tensor([0]), generated, logprobs, response_lens, "stop", ())
        return worker_mod._build_rollout_from_tensor_rows(
            payload.prompts_by_dp[0],
            generated,
            logprobs,
            response_lens,
            ["stop", "length", "length"],
            0,
            len(payload.prompts_by_dp[0]),
            (),
        )

    worker.infer_rollout = infer_rollout

    with PatchedContext(worker_mod, get_tp_context=lambda: ctx):
        remaining = worker.run_rollout_command(first)

    assert [item[1].request_id for item in worker._result_queue.items] == [1]
    assert [request_id for request_id, _part in remaining] == [2, 3]
    assert [part.response_ids for _request_id, part in remaining] == [[[20]], [[30]]]


def test_worker_sends_empty_ack_for_requests_without_local_dp_rows():
    """Non-owner DP ranks must not hold an async request open until their active rollout ends."""

    command = _rollout_command(1, [[1]], target=2, dp_size=2)
    worker = object.__new__(worker_mod.ArenoWorker)
    worker._rank = 1
    worker._result_queue = _ResultQueueDouble()
    worker._cmd_queue = _QueueDouble([])
    worker._deferred_commands = []
    worker._current_request_ids = [1]
    ctx = SimpleNamespace(dp_rank=1, is_rank0=True)

    def infer_rollout(payload, finished_callback=None, refill_callback=None):
        del payload, finished_callback, refill_callback
        return worker_mod._empty_rollout()

    worker.infer_rollout = infer_rollout

    with PatchedContext(worker_mod, get_tp_context=lambda: ctx):
        remaining = worker.run_rollout_command(command)

    assert [item[1].request_id for item in worker._result_queue.items] == [1]
    assert worker._result_queue.items[0][1].payload.prompt_ids == []
    assert remaining == []


def test_worker_refill_queue_is_drained_by_tp_rank0_decision():
    """Only TP-local rank 0 should decide whether the group consumes a queued command."""

    worker = object.__new__(worker_mod.ArenoWorker)
    worker._cmd_queue = _QueueDouble([])
    ctx = SimpleNamespace(is_rank0=True, world_size=2, dp_rank=0, device=torch.device("cpu"), group=object())

    with PatchedContext(
        worker_mod,
        get_tp_context=lambda: ctx,
        dist=SimpleNamespace(broadcast=lambda flag, src, group: flag),
    ):
        assert worker._next_refill_command() is None

    command = _rollout_command(2, [[2]], target=2)
    worker._cmd_queue = _QueueDouble([command])

    with PatchedContext(
        worker_mod,
        get_tp_context=lambda: ctx,
        dist=SimpleNamespace(broadcast=lambda flag, src, group: flag),
    ):
        assert worker._next_refill_command().request_id == 2


def test_worker_non_rank0_consumes_refill_only_after_tp_broadcast():
    """Sibling TP ranks should follow rank 0's refill decision instead of racing get_nowait."""

    command = _rollout_command(2, [[2]], target=2)
    worker = object.__new__(worker_mod.ArenoWorker)
    worker._cmd_queue = _QueueDouble([command])
    ctx = SimpleNamespace(is_rank0=False, world_size=2, dp_rank=0, device=torch.device("cpu"), group=object())

    with PatchedContext(
        worker_mod,
        get_tp_context=lambda: ctx,
        dist=SimpleNamespace(
            broadcast=lambda header, src, group: header.copy_(
                torch.tensor([1, Op.INFER_ROLLOUT.value, 2], dtype=header.dtype)
            )
        ),
    ):
        assert worker._next_refill_command().request_id == 2


def test_worker_non_rank0_defers_unmatched_refill_commands():
    """A TP sibling must not consume a different request than TP-rank0 chose."""

    stale = _rollout_command(99, [[9]], target=2)
    selected = _rollout_command(2, [[2]], target=2)
    worker = object.__new__(worker_mod.ArenoWorker)
    worker._cmd_queue = _QueueDouble([stale, selected])
    worker._deferred_commands = []
    ctx = SimpleNamespace(is_rank0=False, world_size=2, dp_rank=0, device=torch.device("cpu"), group=object())

    with PatchedContext(
        worker_mod,
        get_tp_context=lambda: ctx,
        dist=SimpleNamespace(
            broadcast=lambda header, src, group: header.copy_(
                torch.tensor([1, Op.INFER_ROLLOUT.value, 2], dtype=header.dtype)
            )
        ),
    ):
        cmd = worker._next_refill_command()

    assert cmd.request_id == 2
    assert [command.request_id for command in worker._deferred_commands] == [99]


def test_async_single_prompt_merge_accepts_non_zero_dp_owner():
    """Async single-request rollout may assign a request to any DP rank."""

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


def test_async_single_prompt_requests_merge_from_each_dp_owner():
    """Each one-prompt request should merge from the DP rank that owns it."""

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
    """Async single-request rollout can break the original DP order assumption."""

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
    """Small FIFO queue double for worker continuous-refill tests."""

    def __init__(self, items):
        self.items = list(items)

    def get(self, timeout=None):
        del timeout
        if not self.items:
            raise worker_mod.queue.Empty
        return self.items.pop(0)

    def get_nowait(self):
        return self.get()


class _ResultQueueDouble:
    """Capture worker result queue writes for early-return tests."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _rollout_command(request_id: int, prompts: list[list[int]], *, target: int, dp_size: int = 1) -> Command:
    prompts_by_dp = [prompts[rank::dp_size] for rank in range(dp_size)]
    prompt_indices_by_dp = [list(range(len(prompts)))[rank::dp_size] for rank in range(dp_size)]
    payload = RolloutPayload(
        prompts_by_dp=prompts_by_dp,
        prompt_indices_by_dp=prompt_indices_by_dp,
        max_new_tokens=4,
        eos_token_id=None,
        sampling_params=SamplingParams(),
        max_running_seqs=target,
        max_cache_len=8,
        max_blocks_per_seq=2,
        max_prefill_tokens=16,
        num_blocks=2,
        block_size=4,
    )
    return Command(op=Op.INFER_ROLLOUT, payload=payload, request_id=request_id)
