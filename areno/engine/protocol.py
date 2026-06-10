"""TPCluster process protocol used by `ArenoEngine`.

This module owns the wire protocol between the coordinator process (which
exposes `ArenoEngine`) and one worker process per device. A `TPCluster` owns
the worker subprocesses, broadcasts a single `Command` to all of them, and
waits for every rank to report a `WorkerResult` before returning. Workers run
the per-rank event loop in `_worker_entry`, which knows how to defer
`INFER_ROLLOUT_ADD` commands so that they always arrive after the matching
`INFER_ROLLOUT` setup command.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import socket
import time
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import torch

from areno.engine.config import EngineConfig
from areno.engine.data import SamplingParams
from areno.engine.parallel.context import destroy_process_group, init_process_group


class Op(Enum):
    """Worker commands understood by the rank process loop."""

    TRAIN = auto()
    INFER_ROLLOUT = auto()
    INFER_ROLLOUT_ADD = auto()
    ENSURE_ROLES = auto()
    SCORE_LOGPROBS = auto()
    SCORE_VALUES = auto()
    SCORE_REWARDS = auto()
    TRAIN_VALUES = auto()
    SAVE_CHECKPOINT = auto()
    SHUTDOWN = auto()


@dataclass(slots=True)
class Command:
    """Message sent from coordinator to every rank worker."""

    op: Op
    payload: Any = None


@dataclass(slots=True)
class WorkerResult:
    """Rank response payload or serialized traceback.

    `final=False` indicates a partial result delivered mid-op (for example,
    streaming token batches from a rollout); `final=True` closes the op and
    counts toward the per-rank completion in `TPCluster.call`.
    """

    ok: bool
    payload: Any = None
    error: str | None = None
    final: bool = True


@dataclass(slots=True)
class RolloutPayload:
    """Typed payload for Op.INFER_ROLLOUT."""

    prompts_by_dp: list[list[list[int]]]
    prompt_indices_by_dp: list[list[int]]
    session_id: int | None
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


@dataclass(slots=True)
class RolloutAddPayload:
    """Typed payload for Op.INFER_ROLLOUT_ADD."""

    session_id: int
    prompts_by_dp: list[list[list[int]]]
    prompt_indices_by_dp: list[list[int]]
    cancel_flags: torch.Tensor | None = None
    cancel_indices_by_dp: list[list[int]] | None = None


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
class RolloutPartialPayload:
    """Typed worker-to-coordinator partial rollout event."""

    dp_rank: int
    rows: list[int]
    response_ids: list[list[int]]
    finish_reason: list[str]
    prompt_indices: list[int] | None = None


def find_free_port() -> int:
    """Reserve an available localhost TCP port for torch distributed init."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
        partial_callback: Callable[[int, Any], None] | None = None,
    ) -> list[Any]:
        """Broadcast one command and collect one ordered result from every rank."""
        if not self.started:
            self.start()
        # Drain any stale partial results left behind by a previous op so the
        # current op's final-result count is not inflated.
        while True:
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        cmd = Command(op=op, payload=payload)
        for q in self.cmd_queues:
            q.put(cmd)

        world_size = self.config.tp_size * int(self.config.dp_size)
        # Results are slotted by rank id so the returned list keeps a stable
        # ordering even though workers complete in arbitrary order.
        results: list[Any] = [None] * world_size
        pending = set(range(world_size))
        received = 0
        deadline = None if timeout is None else time.monotonic() + timeout
        while received < world_size:
            poll_timeout = 0.2
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for {op}")
                poll_timeout = min(poll_timeout, remaining)
            try:
                rank, result = self.result_queue.get(timeout=poll_timeout)
            except queue.Empty as exc:
                # No result arrived this tick; detect silently dead workers
                # so the caller does not hang forever on a crashed process.
                dead = self._dead_pending_workers(pending)
                if dead:
                    details = ", ".join(f"rank {rank} pid {pid} exitcode {exitcode}" for rank, pid, exitcode in dead)
                    raise RuntimeError(f"worker exited without reporting result during {op}: {details}") from exc
                continue
            if not result.ok:
                raise RuntimeError(f"rank {rank} failed during {op}:\n{result.error}")
            if not result.final:
                # Partial (streaming) result: forward to the callback but do
                # not count it as a completed rank.
                if partial_callback is not None:
                    partial_callback(rank, result.payload)
                continue
            results[rank] = result.payload
            pending.discard(rank)
            received += 1
        return results

    def broadcast(self, op: Op, payload: Any = None) -> None:
        """Broadcast a command without waiting for worker results."""
        if not self.started:
            self.start()
        cmd = Command(op=op, payload=payload)
        for q in self.cmd_queues:
            q.put(cmd)

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
        # Inject coordinator-facing handles so worker methods can send partial
        # results or pull follow-up commands without re-importing this module.
        worker._rank = rank
        worker._result_queue = result_q
        worker._cmd_queue = cmd_q
        # Buffer for INFER_ROLLOUT_ADD commands that arrived before their
        # matching INFER_ROLLOUT setup op had a chance to run.
        worker._deferred_commands = []
        # Signal readiness only after distributed init, model construction,
        # weight loading, and optimizer setup have completed. Without this
        # barrier, the first command pays lazy startup cost and rollout timing
        # incorrectly includes checkpoint loading.
        result_q.put((rank, WorkerResult(ok=True)))
        while True:
            # Prefer replaying a deferred ADD over reading a new command, so
            # batched additions get drained as soon as the worker is free.
            cmd = worker._deferred_commands.pop(0) if worker._deferred_commands else cmd_q.get()
            # Skip stray ADDs that arrived while we were idle: stash them and
            # keep pulling until we hit a real op.
            while cmd.op is Op.INFER_ROLLOUT_ADD:
                worker._deferred_commands.append(cmd)
                cmd = cmd_q.get()
            if cmd.op is Op.SHUTDOWN:
                # Acknowledge shutdown so the coordinator's join completes.
                result_q.put((rank, WorkerResult(ok=True)))
                break
            payload = worker.handle(cmd)
            result_q.put((rank, WorkerResult(ok=True, payload=payload)))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        # Send the traceback up to the coordinator so it can raise on call().
        result_q.put((rank, WorkerResult(ok=False, error=traceback.format_exc())))
    finally:
        destroy_process_group()


def _close_queue(q: mp.Queue) -> None:
    """Close a multiprocessing queue and reap its feeder thread when present."""

    try:
        q.close()
    finally:
        q.join_thread()
