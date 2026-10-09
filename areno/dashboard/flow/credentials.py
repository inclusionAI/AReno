"""Private local Modal credentials; never returned to the browser or job store."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from uuid import uuid4


class Credentials:
    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "modal-credentials.json"

    @property
    def saved(self):
        return self.path.is_file() and not self.path.is_symlink()

    def load(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return None
        with os.fdopen(fd) as handle:
            mode = os.fstat(handle.fileno())
            if not stat.S_ISREG(mode.st_mode) or mode.st_mode & 0o077:
                raise ValueError("Saved Modal credentials must have owner-only permissions (0600)")
            if hasattr(os, "getuid") and mode.st_uid != os.getuid():
                raise ValueError("Saved Modal credentials must belong to the dashboard user")
            body = json.load(handle)
        values = body.get("token_id"), body.get("token_secret")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("Saved Modal credentials are incomplete; reconnect in Settings")
        return values

    def save(self, token_id, token_secret):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        temporary = self.path.with_name(".modal-credentials-" + uuid4().hex)
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump({"token_id": token_id, "token_secret": token_secret}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def forget(self):
        self.path.unlink(missing_ok=True)
