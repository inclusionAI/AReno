"""Logging bootstrap for the areno package.

Installs one stream handler on the root `areno` logger so all submodules
share the same format and level. The handler is installed only once, and the
log level can be overridden via the ARENO_LOG_LEVEL environment variable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d - %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_logger = logging.getLogger("areno.retry")


def configure_default_logging() -> None:
    """Attach the areno stream handler once with a sensible default level."""

    logger = logging.getLogger("areno")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    logger.addHandler(handler)
    logger.setLevel(_log_level_from_env())
    # Stop the areno logger from re-emitting through the root logger.
    logger.propagate = False


def _log_level_from_env() -> int:
    """Resolve a logging level from ARENO_LOG_LEVEL, defaulting to INFO."""

    value = os.environ.get("ARENO_LOG_LEVEL", "INFO").upper()
    return getattr(logging, value, logging.INFO)


def log_agent_retry_event(
    *,
    attempt: int,
    max_attempts: int,
    delay_s: float,
    error_type: str,
    error_message: str,
    failure_category: str = "model_request",
    **extra: Any,
) -> None:
    """Emit a structured retry event for agent execution failures.

    Each call produces a JSON line on the ``areno.retry`` logger at INFO
    level so operators can grep, aggregate, or forward retry events without
    parsing free-form log lines.
    """
    event = {
        "event": "agent_retry",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "delay_s": round(delay_s, 3),
        "error_type": error_type,
        "error_message": str(error_message)[:200],
        "failure_category": failure_category,
    }
    event.update(extra)
    _logger.info("agent_retry %s", json.dumps(event, ensure_ascii=False, sort_keys=True))
