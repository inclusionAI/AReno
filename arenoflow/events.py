"""Decode incremental Modal output into bounded lines and structured events."""

from __future__ import annotations

import json
import math
import time

from arenoflow.remote import PREFIX


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
