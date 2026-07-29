"""File logging handler for ``--log-to-file``.

Kept in a separate module so CPU tests can verify the file-handler logic
without importing ``areno.cli.train`` (which pulls in torch and other
heavy GPU-only dependencies).

AReno uses two separate logger trees:

* The **root** logger — third-party libraries (httpx, transformers, etc.)
* The ``areno`` logger — AReno's own training engine, rollout, and CLI
  code.  This logger has ``propagate = False`` (see
  ``areno/engine/log.py``) so it does **not** bubble up to the root
  logger.

To capture both trees we attach the FileHandler to **both** the root
logger and the ``areno`` logger.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def install_file_handler(metrics_log_dir: str | None) -> Path:
    """Attach FileHandlers so training logs are persisted to disk.

    The file is written as ``areno_train.<pid>.log`` under the configured
    ``metrics_log_dir`` so ``areno logs`` can read it after the run.

    Returns the path to the log file.
    """

    log_dir = Path(metrics_log_dir) if metrics_log_dir else Path("/tmp/areno/tfevent")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"areno_train.{os.getpid()}.log"

    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    # Attach to the root logger (captures httpx, transformers, etc.)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    # Also attach to the "areno" logger which has propagate=False
    # and would otherwise bypass the root handler entirely.
    areno_logger = logging.getLogger("areno")
    areno_logger.setLevel(logging.INFO)
    areno_logger.addHandler(handler)

    logging.info("log-to-file enabled: writing to %s", log_path)
    return log_path


def remove_file_handlers() -> None:
    """Remove all FileHandler instances from root and areno loggers."""

    for logger_name in (None, "areno"):
        logger = logging.getLogger(logger_name)
        logger.handlers = [
            h for h in logger.handlers if not isinstance(h, logging.FileHandler)
        ]