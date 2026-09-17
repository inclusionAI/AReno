"""Decode incremental Modal output into bounded lines and structured events."""

from __future__ import annotations

import json
import math
import re
import time

from areno.dashboard.flow.remote import PREFIX


def lines(chunks):
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
        if len(buffer) > 65536:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def parse_line(line):
    event = {"type": "log", "message": line[-16000:], "time": time.time()}
    marker = line.find(PREFIX)
    if marker >= 0:
        try:
            candidate = json.loads(line[marker + len(PREFIX) :])
            if isinstance(candidate, dict) and isinstance(candidate.get("type"), str):
                if candidate["type"] == "metric":
                    if not isinstance(candidate.get("tag"), str) or not math.isfinite(float(candidate.get("value"))):
                        return event
                json.dumps(candidate, allow_nan=False)
                return candidate
        except (ValueError, TypeError):
            pass
    return event


def dashboard_state(event):
    """Read trainer state, including logs from already-running older sandboxes."""
    if event.get("type") == "dashboard_state":
        state = event.get("state")
        return state if isinstance(state, dict) and isinstance(state.get("stage"), str) else None
    if event.get("type") == "log":
        message = event.get("message", "")
        stage = re.search(r"\bstage=([a-z_]+)\b", message)
        epoch = re.search(r"\bepoch=(\d+)\b", message)
        if stage and epoch:
            state = {"stage": stage[1], "epoch": int(epoch[1])}
            step = re.search(r"\bstep=(\d+)\b", message)
            role = re.search(r"\brole=([a-z_]+)\b", message)
            if step:
                state["step"] = int(step[1])
            if role:
                state["role"] = role[1]
            return state
    return None
