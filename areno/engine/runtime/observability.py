"""Optional, throttled stage reporting without device synchronization."""

import time
from contextvars import ContextVar
from functools import wraps

_REPORTER = ContextVar("areno_training_reporter", default=None)


def capture_training_events(function):
    @wraps(function)
    def call(self, *args, **kwargs):
        token = _REPORTER.set(getattr(self, "_event_reporter", None))
        try:
            return function(self, *args, **kwargs)
        finally:
            _REPORTER.reset(token)

    return call


def training_event_reporter():
    return _REPORTER.get()


def report_training_event(phase, **data):
    report = training_event_reporter()
    if report is not None:
        report(phase=phase, **data)


def event_reporter(recorder, step):
    last = 0

    def report(*, phase, **data):
        nonlocal last
        now = time.monotonic()
        if phase == "mst" and now - last < 2:
            return
        if phase == "mst":
            last = now
        recorder.record("progress", step=step, phase=phase, **data)

    return report
