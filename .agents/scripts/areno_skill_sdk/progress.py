"""Progress event protocol for long-running skill scripts.

Issue #275 ships only the protocol plus a deterministic JSONL sink. TTY
in-place refresh, non-TTY line output, cancellation, and last-completed-stage
reporting belong to issue #276, which builds on this protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass
class ProgressEvent:
    """A single progress observation.

    ``stage`` is the current phase name (e.g. ``"load_dataset"``).
    ``fraction`` is completion in ``[0.0, 1.0]``. ``message`` is an optional
    human-readable detail.
    """

    stage: str
    fraction: float
    message: str = ""


class ProgressSink:
    """Base class for progress output.

    Subclasses implement :meth:`emit`. The base holds no resources; concrete
    sinks (TTY, non-TTY, JSONL) are added by #276 on top of this protocol.
    """

    def emit(self, event: ProgressEvent) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources. Default no-op."""


class JsonLinesSink(ProgressSink):
    """Write each event as one JSON object per line.

    Deterministic and trivially testable: the output is a sequence of JSON
    objects with fixed fields, suitable for programmatic consumption. This is
    the only sink shipped by #275.
    """

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def emit(self, event: ProgressEvent) -> None:
        payload: dict[str, Any] = {
            "type": "progress",
            "stage": event.stage,
            "fraction": event.fraction,
            "message": event.message,
        }
        self.stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self.stream.flush()

    def close(self) -> None:
        pass
