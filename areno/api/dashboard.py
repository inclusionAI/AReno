"""Optional dashboard integration helpers."""

from __future__ import annotations

from typing import Any


def record_dashboard_state(instance: Any, **kwargs: Any) -> None:
    """Record dashboard state when the backend supports it.

    Unit-test fakes and third-party backend-like objects do not need to
    implement dashboard reporting. Keeping this optional avoids coupling the
    core trainer loops to the dashboard feature.
    """

    recorder = getattr(instance, "record_dashboard_state", None)
    if recorder is not None:
        recorder(**kwargs)
