"""File logging handler for ``--log-to-file``.

Kept in a separate module so CPU tests can verify the file-handler logic
without importing ``areno.cli.train`` (which pulls in torch and other
heavy GPU-only dependencies).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


def install_file_handler(metrics_log_dir: str | None) -> Path:
    """Attach a FileHandler so training logs are persisted to disk.

    The file is written as ``areno_train.<pid>.log`` under the configured
    ``metrics_log_dir`` so ``areno logs`` can read it after the run.

    Returns the path to the log file.
    """

    log_dir = Path(metrics_log_dir) if metrics_log_dir else Path("/tmp/areno/tfevent")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"areno_train.{os.getpid()}.log"

    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    logging.info("log-to-file enabled: writing to %s", log_path)
    return log_path


def remove_file_handlers() -> None:
    """Remove all FileHandler instances from the root logger (for cleanup)."""

    root_logger = logging.getLogger()
    root_logger.handlers = [
        h for h in root_logger.handlers if not isinstance(h, logging.FileHandler)
    ]