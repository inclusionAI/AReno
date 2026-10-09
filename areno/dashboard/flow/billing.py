"""Actual Modal billing data. No local price tables or cost extrapolation."""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal


def wire(value):
    if dataclasses.is_dataclass(value):
        return wire(dataclasses.asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(item) for item in value]
    return value


def fetch_billing(provider):
    now = dt.datetime.now(dt.timezone.utc)
    workspace = provider.modal.Workspace.from_context(client=provider.client)
    result = {
        "fetched_at": now.isoformat(),
        "currency": "USD",
        "scope": "workspace",
        "cycle": now.strftime("%Y-%m"),
        "summary": None,
        "report": None,
        "errors": {},
    }
    try:
        result["summary"] = wire(workspace.billing.summary())
    except Exception as exc:
        result["errors"]["summary"] = str(exc)
    try:
        result["report"] = wire(
            workspace.billing.report(
                start=now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), end=now, resolution="h"
            )
        )
    except Exception as exc:
        result["errors"]["report"] = str(exc)
    return result
