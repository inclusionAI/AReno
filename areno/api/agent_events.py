"""Append-only agent observability; independent of completion console logging."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4


class AgentEvents:
    def __init__(self, directory, *, owner_pid=None, device=None):
        self.owner_pid = owner_pid or os.getpid()
        self.device = device
        suffix = f"{self.owner_pid}.{os.getpid()}" if owner_pid else str(os.getpid())
        self.path = Path(directory) / f"agent_events.{suffix}.jsonl"
        self.callback = None
        self.lock = threading.Lock()
        self.sequence = 0
        self.writer_id = uuid4().hex
        self.step = 0
        self.stop = threading.Event()
        self.monitor = threading.Thread(target=self._resources, daemon=True, name="areno-resources")
        self.monitor.start()

    def _resources(self):
        import sys

        import psutil

        while not self.stop.wait(2):
            host, swap = psutil.virtual_memory(), psutil.swap_memory()
            values = dict(
                host_used=host.total - host.available,
                host_total=host.total,
                host_available=host.available,
                swap_used=swap.used,
                swap_total=swap.total,
            )
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                values.update(
                    cuda_allocated=torch.cuda.memory_allocated(self.device),
                    cuda_reserved=torch.cuda.memory_reserved(self.device),
                )
            self.record("resource", step=self.step, **values)

    def close(self):
        self.stop.set()
        self.monitor.join(timeout=3)

    def record(self, kind, *, step=0, prompt_idx=None, sample_idx=None, **data):
        with self.lock:
            self.sequence += 1
            self.step = step
            event = dict(
                version=1,
                id=f"{os.getpid()}:{self.writer_id}:{self.sequence}",
                worker_pid=os.getpid(),
                pid=self.owner_pid,
                time=time.time(),
                kind=kind,
                step=step,
                prompt_idx=prompt_idx,
                sample_idx=sample_idx,
                data=data,
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            if self.callback:
                self.callback(event)


def record_agent_event(ctx, item, kind, **data):
    """Optional harness hook; usable by external agents without a dashboard dependency."""
    trainer = getattr(ctx, "_trainer", None)
    recorder = getattr(trainer, "record_agent_event", None)
    if recorder:
        recorder(kind, prompt_idx=item.prompt_index, sample_idx=item.sample_index, **data)


def learning_summary(groups):
    """Classify the actual scored reward groups, including flat nonbinary rewards."""
    counts = dict(zero=0, one=0, varying=0, flat_other=0, insufficient=0)
    for rewards in groups:
        if len(rewards) < 2:
            counts["insufficient"] += 1
        elif min(rewards) != max(rewards):
            counts["varying"] += 1
        elif rewards[0] == 0:
            counts["zero"] += 1
        elif rewards[0] == 1:
            counts["one"] += 1
        else:
            counts["flat_other"] += 1
    return counts


def source_label(record):
    if not isinstance(record, dict):
        return "Unlabeled source"
    source = record.get("source") or {}
    if isinstance(source, str):
        return source
    if not isinstance(source, dict):
        source = {}
    for key in ("dataset", "dataset_id", "repo"):
        value = source.get(key) or record.get(key)
        if isinstance(value, str) and value:
            return value
    return "Unlabeled source"


_turn_observer = ContextVar("areno_turn_observer", default=None)


@contextmanager
def capture_agent_turns(recorder):
    """Capture completed model turns as a harness constructs them, per rollout."""
    counts = {}

    def observe(turn):
        item = turn.item
        key = (item.prompt_index, item.sample_index)
        index = counts.get(key, 0)
        counts[key] = index + 1
        response = turn.response

        def get(obj, name):
            return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

        metadata = get(response, "areno")
        if metadata is None:
            metadata = (get(response, "model_extra") or {}).get("areno")
        if isinstance(metadata, dict) and metadata.get("_dashboard_turn_reported"):
            return
        choices = get(response, "choices") or []
        message = get(choices[0], "message") if choices else {}
        recorder("sample", prompt_idx=key[0], sample_idx=key[1], status="running")
        recorder(
            "turn",
            prompt_idx=key[0],
            sample_idx=key[1],
            turn=index,
            messages=turn.messages,
            tool_definitions=turn.tools,
            response={
                "role": "assistant",
                "content": get(message, "content"),
                "reasoning_content": get(message, "reasoning_content"),
                "tool_calls": turn.parsed_tool_calls,
            },
            input_tokens=len(turn.input_tokens),
            output_tokens=len(turn.response_tokens),
        )

    token = _turn_observer.set(observe)
    try:
        yield
    finally:
        _turn_observer.reset(token)


def report_agent_turn(turn):
    observer = _turn_observer.get()
    if observer is not None:
        observer(turn)
