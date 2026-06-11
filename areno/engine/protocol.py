"""TPCluster process protocol used by `ArenoEngine`.

This module owns the wire protocol between the coordinator process (which
exposes `ArenoEngine`) and one worker process per device. A `TPCluster` owns
the worker subprocesses, broadcasts a single `Command` to all of them, and
waits for every rank to report a `WorkerResult` before returning. Workers run
the per-rank event loop in `_worker_entry`, which knows how to defer
request-id demux so multiple caller threads/tasks can have in-flight commands.
"""

from __future__ import annotations

import multiprocessing as mp
import asyncio
import queue
import socket
import threading
import time
import traceback
from dataclasses import dataclass, replace
from enum import Enum, auto
from itertools import count
from typing import Any

import torch

from areno.engine.config import EngineConfig
from areno.engine.data import SamplingParams
from areno.engine.parallel.context import destroy_process_group, get_tp_context, init_process_group
from areno.engine.runtime.rollout import partial_tail_threshold


class Op(Enum):
    """Worker commands understood by the rank process loop."""

    TRAIN = auto()
    INFER_ROLLOUT = auto()
    ENSURE_ROLES = auto()
    SCORE_LOGPROBS = auto()
    SCORE_VALUES = auto()
    SCORE_REWARDS = auto()
    TRAIN_VALUES = auto()
    ROLLOUT_SESSION_BEGIN = auto()
    ROLLOUT_SESSION_END = auto()
    SAVE_CHECKPOINT = auto()
    SHUTDOWN = auto()


@dataclass(slots=True)
class Command:
    """Message sent from coordinator to every rank worker."""

    op: Op
    payload: Any = None
    request_id: int | None = None


@dataclass(slots=True)
class WorkerResult:
    """Rank response payload or serialized traceback.

    `request_id` lets the coordinator demux multiple concurrent calls.
    """

    ok: bool
    payload: Any = None
    error: str | None = None
    request_id: int | None = None


@dataclass(slots=True)
class RolloutPayload:
    """Typed payload for Op.INFER_ROLLOUT."""

    prompts_by_dp: list[list[list[int]]]
    prompt_indices_by_dp: list[list[int]]
    max_new_tokens: int
    eos_token_id: int | tuple[int, ...] | None
    sampling_params: SamplingParams
    max_running_seqs: int
    max_cache_len: int
    max_blocks_per_seq: int
    max_prefill_tokens: int
    num_blocks: int
    block_size: int
    decode_progress_interval_s: float = 0.0
    cancel_flags: torch.Tensor | None = None
    cancel_indices_by_dp: list[list[int]] | None = None
    coalesce_max_running_seqs: int | None = None
    coalesce_timeout_s: float = 0.0
    coalesced_request_ids: list[int] | None = None
    coalesced_counts_by_dp: list[list[int]] | None = None
    partial_tail_threshold: int = 0
    partial_tail_min_tokens: int = 0
    partial_tail_cooldown_until_s_by_dp: list[list[float]] | None = None


@dataclass(slots=True)
class TrainPayload:
    """Typed payload for Op.TRAIN."""

    data_packs_by_dp: list[list[dict[str, Any]]]
    gradient_accumulation_steps: int | None = None


@dataclass(slots=True)
class RoleSpecPayload:
    """Serialized role specification sent to worker ranks."""

    path: str
    trainable: bool
    optimizer_lr: float | None = None


@dataclass(slots=True)
class EnsureRolesPayload:
    """Typed payload for Op.ENSURE_ROLES."""

    roles: dict[str, RoleSpecPayload]


@dataclass(slots=True)
class ScorePayload:
    """Typed payload for role score ops."""

    role: str
    token_rows_by_dp: list[list[list[int]]]
    pad_token_id: int
    microbatch_size: int = 8


@dataclass(slots=True)
class TrainValuesPayload:
    """Typed payload for Op.TRAIN_VALUES."""

    role: str
    data_packs_by_dp: list[list[dict[str, Any]]]
    gradient_accumulation_steps: int | None = None
    cliprange_value: float = 0.5
    value_loss_coef: float = 0.5


@dataclass(slots=True)
class SaveCheckpointPayload:
    """Typed payload for Op.SAVE_CHECKPOINT."""

    path: str


@dataclass(slots=True)
class _PendingClusterCall:
    """Coordinator-side accumulator for one in-flight cluster request."""

    op: Op
    results: list[Any]
    pending: set[int]
    event: threading.Event
    future: asyncio.Future | None = None
    loop: asyncio.AbstractEventLoop | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class _QueuedRolloutCall:
    request_id: int
    payload: RolloutPayload
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop


def find_free_port() -> int:
    """Reserve an available localhost TCP port for torch distributed init."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rollout_payload_coalesce_enabled(payload: Any) -> bool:
    """Return whether an async rollout payload should use coordinator batching."""

    return isinstance(payload, RolloutPayload) and float(payload.coalesce_timeout_s) > 0.0 and payload.cancel_flags is None


def _rollout_payload_count(payload: RolloutPayload) -> int:
    """Return total prompt rows in a rollout payload."""

    return sum(len(rows) for rows in payload.prompts_by_dp)


def _rollout_queue_count(items: list[_QueuedRolloutCall]) -> int:
    """Return total prompt rows queued for rollout batching."""

    return sum(_rollout_payload_count(item.payload) for item in items)


def _rollout_queue_target(payload: RolloutPayload) -> int:
    """Global queue target derived from per-DP local running prompt limit."""

    local_target = max(int(payload.coalesce_max_running_seqs or payload.max_running_seqs), 1)
    return local_target * len(payload.prompts_by_dp)


def _rollout_payloads_compatible(first: RolloutPayload, other: RolloutPayload) -> bool:
    """Return whether two rollout payloads can share one worker batch."""

    return (
        _rollout_payload_coalesce_enabled(other)
        and first.max_new_tokens == other.max_new_tokens
        and first.eos_token_id == other.eos_token_id
        and first.sampling_params == other.sampling_params
        and first.block_size == other.block_size
        and first.decode_progress_interval_s == other.decode_progress_interval_s
        and first.cancel_flags is None
        and other.cancel_flags is None
        and first.cancel_indices_by_dp is None
        and other.cancel_indices_by_dp is None
    )


def _merge_rollout_payloads_for_cluster(payloads: list[RolloutPayload], request_ids: list[int]) -> RolloutPayload:
    """Merge async rollout requests once in the coordinator before broadcast."""

    first = payloads[0]
    dp_size = len(first.prompts_by_dp)
    prompts_by_dp: list[list[list[int]]] = [[] for _ in range(dp_size)]
    prompt_indices_by_dp: list[list[int]] = [[] for _ in range(dp_size)]
    cooldown_until_by_dp: list[list[float]] = [[] for _ in range(dp_size)]
    counts_by_dp = [[0 for _ in payloads] for _ in range(dp_size)]
    row_idx = 0
    for request_idx, payload in enumerate(payloads):
        for prompt, prompt_index, cooldown_until_s in _iter_rollout_payload_rows(payload):
            dp_rank = row_idx % dp_size
            prompts_by_dp[dp_rank].append(prompt)
            prompt_indices_by_dp[dp_rank].append(prompt_index)
            cooldown_until_by_dp[dp_rank].append(cooldown_until_s)
            counts_by_dp[dp_rank][request_idx] += 1
            row_idx += 1
    local_capacity = max(int(first.coalesce_max_running_seqs or first.max_running_seqs), 1)
    max_running_seqs = max(max((len(rows) for rows in prompts_by_dp), default=0), local_capacity)
    max_cache_len = max(payload.max_cache_len for payload in payloads)
    max_blocks_per_seq = max(payload.max_blocks_per_seq for payload in payloads)
    max_prefill_tokens = max(payload.max_prefill_tokens for payload in payloads)
    tail_threshold = partial_tail_threshold(local_capacity, float(first.coalesce_timeout_s))
    return replace(
        first,
        prompts_by_dp=prompts_by_dp,
        prompt_indices_by_dp=prompt_indices_by_dp,
        max_running_seqs=max_running_seqs,
        max_cache_len=max_cache_len,
        max_blocks_per_seq=max_blocks_per_seq,
        max_prefill_tokens=max_prefill_tokens,
        num_blocks=max_running_seqs * int(max_blocks_per_seq),
        coalesce_timeout_s=0.0,
        coalesced_request_ids=list(request_ids),
        coalesced_counts_by_dp=counts_by_dp,
        partial_tail_threshold=tail_threshold,
        partial_tail_min_tokens=max(int(payload.partial_tail_min_tokens) for payload in payloads),
        partial_tail_cooldown_until_s_by_dp=cooldown_until_by_dp,
    )


def _coalesced_pending_ranks(counts_by_dp: list[list[int]], request_idx: int, *, tp_size: int, world_size: int) -> set[int]:
    """Ranks that must answer for one request in a coordinator-coalesced rollout."""

    ranks = {
        dp_rank * tp_size + tp_rank
        for dp_rank, counts in enumerate(counts_by_dp)
        if counts[request_idx] > 0
        for tp_rank in range(tp_size)
    }
    return ranks or set(range(world_size))


def _iter_rollout_payload_rows(payload: RolloutPayload):
    """Yield prompt rows in the payload's original pre-DP-split order."""

    dp_size = len(payload.prompts_by_dp)
    total = _rollout_payload_count(payload)
    for original_idx in range(total):
        dp_rank = original_idx % dp_size
        local_idx = original_idx // dp_size
        cooldown_until_s = 0.0
        if payload.partial_tail_cooldown_until_s_by_dp is not None:
            cooldown_until_s = float(payload.partial_tail_cooldown_until_s_by_dp[dp_rank][local_idx])
        yield payload.prompts_by_dp[dp_rank][local_idx], payload.prompt_indices_by_dp[dp_rank][local_idx], cooldown_until_s


class TPCluster:
    """Small process cluster used by ArenoEngine.

    Each rank is a long-lived process with one command queue. A cluster call is
    synchronous: broadcast one command to all ranks, then wait until every rank
    has reported success or one rank reports an error.
    """

    def __init__(self, config: EngineConfig, worker_cls: type):
        """Create an unstarted process cluster for a worker class."""

        self.config = config
        self.worker_cls = worker_cls
        # `spawn` start method is required by CUDA-aware workers; do not
        # inherit fds/CUDA state from the parent.
        self.ctx = mp.get_context("spawn")
        self.cmd_queues: list[mp.Queue] = []
        self.result_queue: mp.Queue = self.ctx.Queue()
        self.processes: list[mp.Process] = []
        self.started = False
        self._send_lock = threading.Lock()
        self._request_ids = count(1)
        self._pending_lock = threading.Lock()
        self._pending_calls: dict[int, _PendingClusterCall] = {}
        self._pump_stop = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self._rollout_queue: list[_QueuedRolloutCall] = []
        self._rollout_queue_lock: asyncio.Lock | None = None
        self._rollout_flush_task: asyncio.Task | None = None

    def start(self) -> None:
        """Spawn workers and wait until every rank has finished initialization."""

        if self.started:
            return
        # Reserve a unique TCP port for torch.distributed rendezvous; ranks
        # discover each other through this port over loopback.
        port = find_free_port()
        assert self.config.devices is not None
        devices = self.config.devices
        world_size = self.config.tp_size * int(self.config.dp_size)
        if len(devices) != world_size:
            raise ValueError("len(devices) must equal tp_size * dp_size")
        for rank in range(world_size):
            cmd_q = self.ctx.Queue()
            proc = self.ctx.Process(
                target=_worker_entry,
                args=(
                    self.worker_cls,
                    rank,
                    world_size,
                    devices[rank],
                    port,
                    self.config,
                    cmd_q,
                    self.result_queue,
                ),
                daemon=True,
            )
            proc.start()
            self.cmd_queues.append(cmd_q)
            self.processes.append(proc)
        try:
            self._wait_for_worker_ready(set(range(world_size)))
        except BaseException:
            for proc in self.processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in self.processes:
                proc.join(timeout=0)
            for q in self.cmd_queues:
                _close_queue(q)
            _close_queue(self.result_queue)
            self.cmd_queues = []
            self.processes = []
            raise
        else:
            self.started = True
            self._start_result_pump()

    def _start_result_pump(self) -> None:
        """Start the single result-demux thread for all concurrent calls."""

        if self._pump_thread is not None and self._pump_thread.is_alive():
            return
        self._pump_stop.clear()
        self._pump_thread = threading.Thread(target=self._result_pump_loop, name="areno-tpcluster-results", daemon=True)
        self._pump_thread.start()

    def _wait_for_worker_ready(self, pending: set[int]) -> None:
        """Block until every worker reports that model construction is complete."""

        while pending:
            try:
                rank, result = self.result_queue.get(timeout=0.2)
            except queue.Empty as exc:
                dead = self._dead_pending_workers(pending)
                if dead:
                    details = ", ".join(f"rank {rank} pid {pid} exitcode {exitcode}" for rank, pid, exitcode in dead)
                    raise RuntimeError(f"worker exited during startup: {details}") from exc
                continue
            if not result.ok:
                raise RuntimeError(f"rank {rank} failed during startup:\n{result.error}")
            pending.discard(rank)

    def call(
        self,
        op: Op,
        payload: Any = None,
        timeout: float | None = None,
    ) -> list[Any]:
        """Broadcast one command and collect one ordered result from every rank."""
        request_id = next(self._request_ids)
        pending = self._submit_call(op, payload, request_id=request_id)
        if not pending.event.wait(timeout=timeout):
            with self._pending_lock:
                self._pending_calls.pop(request_id, None)
            raise TimeoutError(f"timed out waiting for {op}")
        if pending.error is not None:
            raise pending.error
        return pending.results

    async def call_async(
        self,
        op: Op,
        payload: Any = None,
        timeout: float | None = None,
    ) -> list[Any]:
        """Async variant of :meth:`call` backed by the shared result pump."""

        request_id = next(self._request_ids)
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        if op is Op.INFER_ROLLOUT and _rollout_payload_coalesce_enabled(payload):
            await self._enqueue_rollout_call(request_id, payload, future, loop)
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except BaseException:
                with self._pending_lock:
                    self._pending_calls.pop(request_id, None)
                raise
        self._submit_call(op, payload, request_id=request_id, future=future, loop=loop)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except BaseException:
            with self._pending_lock:
                self._pending_calls.pop(request_id, None)
            raise

    async def _enqueue_rollout_call(
        self,
        request_id: int,
        payload: RolloutPayload,
        future: asyncio.Future,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Queue one async rollout for coordinator-side coalescing."""

        if self._rollout_queue_lock is None:
            self._rollout_queue_lock = asyncio.Lock()
        async with self._rollout_queue_lock:
            self._rollout_queue.append(_QueuedRolloutCall(request_id, payload, future, loop))
            if self._rollout_flush_task is None or self._rollout_flush_task.done():
                self._rollout_flush_task = asyncio.create_task(self._flush_rollout_queue_at_deadline(float(payload.coalesce_timeout_s)))
            first = self._rollout_queue[0].payload
            if _rollout_queue_count(self._rollout_queue) >= _rollout_queue_target(first):
                self._rollout_flush_task.cancel()
                self._rollout_flush_task = asyncio.create_task(self._flush_rollout_queue_now())

    async def _flush_rollout_queue_at_deadline(self, delay_s: float) -> None:
        """Flush queued rollout calls at the DDL if the batch did not fill."""

        try:
            await asyncio.sleep(max(delay_s, 0.0))
        except asyncio.CancelledError:
            return
        await self._flush_rollout_queue_now()

    async def _flush_rollout_queue_now(self) -> None:
        """Submit one or more coordinator-coalesced rollout batches."""

        if self._rollout_queue_lock is None:
            return
        async with self._rollout_queue_lock:
            queued = self._rollout_queue
            self._rollout_queue = []
        while queued:
            first = queued.pop(0)
            batch = [first]
            remaining = []
            target = _rollout_queue_target(first.payload)
            count = _rollout_payload_count(first.payload)
            for item in queued:
                if count < target and _rollout_payloads_compatible(first.payload, item.payload):
                    batch.append(item)
                    count += _rollout_payload_count(item.payload)
                else:
                    remaining.append(item)
            queued = remaining
            self._submit_coalesced_rollout_batch(batch)

    def _submit_coalesced_rollout_batch(self, batch: list[_QueuedRolloutCall]) -> None:
        """Broadcast a coordinator-coalesced rollout while preserving request futures."""

        if not self.started:
            self.start()
        world_size = self.config.tp_size * int(self.config.dp_size)
        payload = _merge_rollout_payloads_for_cluster([item.payload for item in batch], [item.request_id for item in batch])
        assert payload.coalesced_counts_by_dp is not None
        for request_idx, item in enumerate(batch):
            pending_ranks = _coalesced_pending_ranks(
                payload.coalesced_counts_by_dp,
                request_idx,
                tp_size=self.config.tp_size,
                world_size=world_size,
            )
            pending = _PendingClusterCall(
                op=Op.INFER_ROLLOUT,
                results=[None] * world_size,
                pending=pending_ranks,
                event=threading.Event(),
                future=item.future,
                loop=item.loop,
            )
            with self._pending_lock:
                self._pending_calls[item.request_id] = pending
        cmd = Command(op=Op.INFER_ROLLOUT, payload=payload, request_id=None)
        with self._send_lock:
            for q in self.cmd_queues:
                q.put(cmd)

    def _submit_call(
        self,
        op: Op,
        payload: Any = None,
        *,
        request_id: int,
        future: asyncio.Future | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> _PendingClusterCall:
        if not self.started:
            self.start()
        world_size = self.config.tp_size * int(self.config.dp_size)
        pending = _PendingClusterCall(
            op=op,
            results=[None] * world_size,
            pending=set(range(world_size)),
            event=threading.Event(),
            future=future,
            loop=loop,
        )
        with self._pending_lock:
            self._pending_calls[request_id] = pending
        cmd = Command(op=op, payload=payload, request_id=request_id)
        with self._send_lock:
            for q in self.cmd_queues:
                q.put(cmd)
        return pending

    def _result_pump_loop(self) -> None:
        """Read rank results and demux them to the matching pending call."""

        while not self._pump_stop.is_set():
            try:
                rank, result = self.result_queue.get(timeout=0.2)
            except queue.Empty:
                self._fail_dead_pending_calls()
                continue
            request_id = result.request_id
            if request_id is None:
                continue
            with self._pending_lock:
                pending = self._pending_calls.get(request_id)
            if pending is None:
                continue
            self._apply_result(request_id, rank, result, pending)

    def _apply_result(self, request_id: int, rank: int, result: WorkerResult, pending: _PendingClusterCall) -> None:
        """Apply one worker result to a pending call and complete it if done."""

        if not result.ok:
            self._finish_pending_call(request_id, pending, RuntimeError(f"rank {rank} failed during {pending.op}:\n{result.error}"))
            return
        pending.results[rank] = result.payload
        pending.pending.discard(rank)
        if not pending.pending:
            self._finish_pending_call(request_id, pending, None)

    def _finish_pending_call(self, request_id: int, pending: _PendingClusterCall, error: BaseException | None) -> None:
        """Mark a pending call complete and wake sync/async waiters."""

        with self._pending_lock:
            self._pending_calls.pop(request_id, None)
        pending.error = error
        pending.event.set()
        if pending.future is not None and pending.loop is not None:
            if error is None:
                pending.loop.call_soon_threadsafe(_set_async_result, pending.future, pending.results)
            else:
                pending.loop.call_soon_threadsafe(_set_async_exception, pending.future, error)

    def _fail_dead_pending_calls(self) -> None:
        """Fail pending calls when a worker dies without returning a result."""

        with self._pending_lock:
            calls = list(self._pending_calls.items())
        for request_id, pending in calls:
            dead = self._dead_pending_workers(pending.pending)
            if not dead:
                continue
            details = ", ".join(f"rank {rank} pid {pid} exitcode {exitcode}" for rank, pid, exitcode in dead)
            self._finish_pending_call(
                request_id,
                pending,
                RuntimeError(f"worker exited without reporting result during {pending.op}: {details}"),
            )

    def _dead_pending_workers(self, pending: set[int]) -> list[tuple[int, int | None, int | None]]:
        """Return pending ranks whose process exited before reporting."""

        dead = []
        for rank in pending:
            proc = self.processes[rank]
            exitcode = proc.exitcode
            if exitcode is not None:
                dead.append((rank, proc.pid, exitcode))
                # Reap the process so its resources are released promptly.
                proc.join(timeout=0)
        return dead

    def close(self) -> None:
        """Request shutdown and terminate workers that do not exit promptly."""

        if not self.started:
            return
        try:
            pump_stop = getattr(self, "_pump_stop", None)
            if pump_stop is not None:
                pump_stop.set()
            pump_thread = getattr(self, "_pump_thread", None)
            if pump_thread is not None:
                pump_thread.join(timeout=2)
                self._pump_thread = None
            # Polite shutdown: SHUTDOWN op lets workers tear down the
            # distributed context cleanly.
            for q in self.cmd_queues:
                q.put(Command(op=Op.SHUTDOWN))
            for proc in self.processes:
                proc.join(timeout=5)
        finally:
            # If a worker is still alive after the grace period, force it.
            for proc in self.processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in self.processes:
                proc.join(timeout=0)
            for q in self.cmd_queues:
                _close_queue(q)
            _close_queue(self.result_queue)
            self.started = False

    def __enter__(self) -> "TPCluster":
        """Start the cluster for context-manager usage."""

        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        """Close the cluster on context-manager exit."""

        self.close()


def _worker_entry(
    worker_cls: type,
    rank: int,
    world_size: int,
    device_id: int,
    port: int,
    config: EngineConfig,
    cmd_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    """Worker process main loop.

    The process initializes its TP/DP distributed context once, builds the rank
    worker, then serves synchronous commands from the coordinator until
    shutdown. Exceptions are serialized back so the coordinator can fail all
    ranks with the original traceback.
    """

    try:
        # Rendezvous on the shared port; this blocks until every rank arrives.
        init_process_group(
            rank=rank,
            world_size=world_size,
            master_addr="127.0.0.1",
            master_port=port,
            device_id=device_id,
            tp_size=config.tp_size,
        )
        torch.set_float32_matmul_precision("high")
        worker = worker_cls(config)
        # Inject coordinator-facing handles so worker methods can report
        # request-id-scoped results without re-importing this module.
        worker._rank = rank
        worker._result_queue = result_q
        worker._cmd_queue = cmd_q
        worker._deferred_commands = []
        # Signal readiness only after distributed init, model construction,
        # weight loading, and optimizer setup have completed. Without this
        # barrier, the first command pays lazy startup cost and rollout timing
        # incorrectly includes checkpoint loading.
        result_q.put((rank, WorkerResult(ok=True)))
        while True:
            cmd = worker._deferred_commands.pop(0) if worker._deferred_commands else cmd_q.get()
            if cmd.op is Op.SHUTDOWN:
                # Acknowledge shutdown so the coordinator's join completes.
                result_q.put((rank, WorkerResult(ok=True, request_id=cmd.request_id)))
                break
            worker._current_request_id = cmd.request_id
            if cmd.op is Op.INFER_ROLLOUT:
                if cmd.payload.coalesced_request_ids is not None:
                    ctx = get_tp_context()
                    request_ids = list(cmd.payload.coalesced_request_ids)
                    counts = list(cmd.payload.coalesced_counts_by_dp[ctx.dp_rank])
                else:
                    request_ids = None
                    counts = None
                if request_ids is not None and counts is not None:
                    worker._current_request_ids = request_ids
                    for request_id, payload in worker.run_coalesced_rollout_payload(cmd.payload, request_ids, counts):
                        result_q.put((rank, WorkerResult(ok=True, payload=payload, request_id=request_id)))
                    worker._current_request_ids = []
                    continue
                commands = worker.collect_rollout_commands(cmd)
                worker._current_request_ids = [command.request_id for command in commands]
                for request_id, payload in worker.run_rollout_commands(commands):
                    result_q.put((rank, WorkerResult(ok=True, payload=payload, request_id=request_id)))
                worker._current_request_ids = []
                continue
            payload = worker.handle(cmd)
            result_q.put((rank, WorkerResult(ok=True, payload=payload, request_id=cmd.request_id)))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        # Send the traceback up to the coordinator so it can raise on call().
        request_id = getattr(locals().get("cmd", None), "request_id", None)
        request_ids = getattr(locals().get("worker", None), "_current_request_ids", None) or [request_id]
        error = traceback.format_exc()
        for failed_request_id in request_ids:
            result_q.put((rank, WorkerResult(ok=False, error=error, request_id=failed_request_id)))
    finally:
        destroy_process_group()


def _close_queue(q: mp.Queue) -> None:
    """Close a multiprocessing queue and reap its feeder thread when present."""

    try:
        q.close()
    finally:
        q.join_thread()


def _set_async_result(future: asyncio.Future, value: Any) -> None:
    """Resolve an asyncio future from the result-pump thread."""

    if not future.done():
        future.set_result(value)


def _set_async_exception(future: asyncio.Future, exc: BaseException) -> None:
    """Fail an asyncio future from the result-pump thread."""

    if not future.done():
        future.set_exception(exc)
