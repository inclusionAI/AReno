"""Reusable building blocks for registered Harbor agents and datasets.

These bases operate on explicit files and commands supplied by adapters. They
do not select a dataset, prescribe record fields, or register implementations.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.environments.base import ExecResult

from areno.api.harness import HarnessAttempt

WorkspaceFiles = Mapping[str, str | bytes]


def workspace_path(name: str, reserved_roots: Sequence[str] = ()) -> PurePosixPath:
    """Validate an adapter-provided path before writing into a fresh workspace."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] in reserved_roots:
        raise ValueError(f"invalid task file path: {name}")
    return relative


def write_workspace_files(root: Path, files: WorkspaceFiles, *, reserved_roots: Sequence[str] = ()) -> None:
    for name, content in files.items():
        target = root / workspace_path(name, reserved_roots)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode() if isinstance(content, str) else content)


@dataclass(slots=True)
class TaskVerification:
    """Trusted verification inputs, uploaded after the agent finishes."""

    command: Sequence[str]
    files: WorkspaceFiles = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


class FileWorkspaceDataset(ABC):
    """Share validated workspace transfer without prescribing a grader."""

    def __init__(self, *, reserved_roots: Sequence[str] = ()) -> None:
        self.reserved_roots = tuple(reserved_roots)

    def normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)

    @abstractmethod
    def workspace_files(self, record: dict[str, Any]) -> WorkspaceFiles:
        """Map dataset-specific fields to text or binary workspace files."""

    async def prepare(self, attempt: HarnessAttempt) -> None:
        handle = attempt.environment
        assert handle is not None
        snapshot = attempt.root / "snapshot"
        snapshot.mkdir()
        write_workspace_files(snapshot, self.workspace_files(attempt.item.record), reserved_roots=self.reserved_roots)
        await handle.environment.upload_dir(snapshot, handle.workspace)


class WorkspaceDataset(FileWorkspaceDataset):
    """Share command grading across dataset adapters.

    Subclasses map their own record schema to files and verification commands.
    Override ``score`` for non-binary rewards or ``verify`` for richer protocols.
    Datasets without file workspaces can implement the HarnessDataset protocol
    directly instead of inheriting this optional utility base.
    """

    @abstractmethod
    def verification(self, attempt: HarnessAttempt) -> TaskVerification:
        """Build the dataset's trusted grader without executing it."""

    def score(self, result: ExecResult) -> float:
        return float(result.return_code == 0)

    async def verify(self, attempt: HarnessAttempt) -> dict[str, Any]:
        handle = attempt.environment
        assert handle is not None
        verification = self.verification(attempt)
        if verification.files:
            snapshot = attempt.root / "verification"
            snapshot.mkdir(exist_ok=True)
            write_workspace_files(snapshot, verification.files)
            await handle.environment.upload_dir(snapshot, handle.control_dir)
        try:
            result = await handle.environment.exec(
                shlex.join(verification.command),
                cwd=verification.cwd or handle.workspace,
                env=verification.env,
                timeout_sec=attempt.verify_timeout,
            )
        except TimeoutError:
            return {"reward": 0.0, "verify_returncode": None, "verify_timed_out": True}
        return {
            "reward": self.score(result),
            "verify_returncode": result.return_code,
            "verify_timed_out": False,
            "verify_log": (result.stdout or "") + (result.stderr or ""),
        }
