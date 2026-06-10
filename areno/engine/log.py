"""Logging bootstrap for the areno package.

Installs one stream handler on the root `areno` logger so all submodules
share the same format and level. The handler is installed only once, and the
log level can be overridden via the ARENO_LOG_LEVEL environment variable.
"""

from __future__ import annotations

import logging
import os


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d - %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


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
