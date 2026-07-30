"""Preflight output-directory writability probe.

Before expensive model or worker initialisation, ``probe_directory_writability``
verifies that a directory can actually be created, written to, flushed,
renamed within, and cleaned up.  This goes beyond a simple ``os.access`` check
(which can give false positives on NFS, read-only mounts, full disks, or
SELinux) by performing a real I/O round-trip with a uniquely-named probe file.

The probe file is always removed, even on failure or interruption, so user
data is never touched or overwritten.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreflightProbeResult:
    """Outcome of a single directory writability probe."""

    stage: str           # e.g. "checkpoint" or "metrics"
    path: str            # the path that was probed
    ok: bool
    operation: str = ""  # failing operation: "create"/"write"/"flush"/"rename"/"cleanup"
    error: str | None = None


@dataclass(frozen=True)
class PreflightConfig:
    """Controls preflight probe behaviour."""

    enabled: bool = True
    probe_prefix: str = ".areno_preflight_"


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------

_PROBE_PAYLOAD = b"areno preflight probe\n"


def probe_directory_writability(
    path: str | Path,
    *,
    stage: str,
    config: PreflightConfig | None = None,
) -> PreflightProbeResult:
    """Probe whether *path* is writable via a full create→write→flush→rename→cleanup cycle.

    Returns a :class:`PreflightProbeResult`; never raises.
    """

    cfg = config or PreflightConfig()

    if not cfg.enabled:
        return PreflightProbeResult(stage=stage, path=str(path), ok=True)

    path_str = str(path) if path is not None else ""
    if not path_str:
        return PreflightProbeResult(stage=stage, path=path_str, ok=True)

    resolved = Path(path_str).expanduser().resolve()

    # If the path exists but is a regular file, we cannot use it as a directory.
    if resolved.exists() and not resolved.is_dir():
        return PreflightProbeResult(
            stage=stage,
            path=str(resolved),
            ok=False,
            operation="create",
            error=f"exists but is a file, not a directory: {resolved}",
        )

    probe_name = f"{cfg.probe_prefix}{stage}_{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp"
    renamed_name = f"{cfg.probe_prefix}{stage}_{os.getpid()}_{uuid.uuid4().hex[:8]}.renamed"
    probe_file = resolved / probe_name
    renamed_file = resolved / renamed_name
    fh = None  # tracked so finally can close it on interruption

    try:
        # --- create ---
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as exc:
            return PreflightProbeResult(
                stage=stage,
                path=str(resolved),
                ok=False,
                operation="create",
                error=f"{type(exc).__name__}: {exc}",
            )

        # --- probe file create (exclusive) ---
        try:
            fh = open(probe_file, "xb")
        except FileExistsError:
            # UUID collision is extremely unlikely; try once more with a new UUID.
            probe_name = f"{cfg.probe_prefix}{stage}_{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp"
            renamed_name = f"{cfg.probe_prefix}{stage}_{os.getpid()}_{uuid.uuid4().hex[:8]}.renamed"
            probe_file = resolved / probe_name
            renamed_file = resolved / renamed_name
            try:
                fh = open(probe_file, "xb")
            except FileExistsError:
                return PreflightProbeResult(
                    stage=stage,
                    path=str(resolved),
                    ok=False,
                    operation="create",
                    error="concurrent creation conflict: probe file already exists",
                )
        except (OSError, PermissionError) as exc:
            return PreflightProbeResult(
                stage=stage,
                path=str(resolved),
                ok=False,
                operation="create",
                error=f"{type(exc).__name__}: {exc}",
            )

        # --- write / flush / close ---
        try:
            fh.write(_PROBE_PAYLOAD)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # fsync may fail on some filesystems (e.g. tmpfs); not fatal.
                pass
        except (OSError, PermissionError) as exc:
            fh.close()
            fh = None
            return PreflightProbeResult(
                stage=stage,
                path=str(resolved),
                ok=False,
                operation="write",
                error=f"{type(exc).__name__}: {exc}",
            )
        fh.close()
        fh = None

        # --- rename ---
        try:
            probe_file.replace(renamed_file)
        except (OSError, PermissionError) as exc:
            return PreflightProbeResult(
                stage=stage,
                path=str(resolved),
                ok=False,
                operation="rename",
                error=f"{type(exc).__name__}: {exc}",
            )

        # --- cleanup ---
        try:
            renamed_file.unlink(missing_ok=True)
        except OSError as exc:
            # Cleanup failure is a warning, not a hard failure.
            return PreflightProbeResult(
                stage=stage,
                path=str(resolved),
                ok=True,  # write succeeded; cleanup is best-effort
                operation="cleanup",
                error=f"warning: could not remove probe file: {exc}",
            )

        return PreflightProbeResult(stage=stage, path=str(resolved), ok=True)

    except KeyboardInterrupt:
        return PreflightProbeResult(
            stage=stage,
            path=str(resolved),
            ok=False,
            operation="interrupted",
            error="probe interrupted by user",
        )
    finally:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        # Best-effort cleanup of any residual probe files.
        _cleanup_probe_files(resolved, cfg.probe_prefix)


def _cleanup_probe_files(directory: Path, prefix: str) -> None:
    """Remove any leftover probe files matching *prefix*."""

    try:
        for f in directory.glob(f"{prefix}*"):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Batch probe
# ---------------------------------------------------------------------------

def probe_paths(
    paths: list[tuple[str, str | Path]],
    *,
    config: PreflightConfig | None = None,
) -> list[PreflightProbeResult]:
    """Probe multiple ``(stage, path)`` pairs.

    Returns one result per pair.  A ``None`` or empty path is skipped
    (returns ``ok=True``).
    """

    return [
        probe_directory_writability(p, stage=stage, config=config)
        for stage, p in paths
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_probe_results(
    results: list[PreflightProbeResult],
    *,
    color: bool = False,
) -> str:
    """Human-readable summary of probe results."""

    lines: list[str] = []
    for r in results:
        if r.ok:
            status = "OK"
            if color:
                status = f"\033[32m{status}\033[0m"
        else:
            status = "FAIL"
            if color:
                status = f"\033[31m{status}\033[0m"
        lines.append(f"  {status}  {r.stage} directory")
        lines.append(f"        path: {r.path}")
        if not r.ok:
            lines.append(f"        failed operation: {r.operation}")
            if r.error:
                lines.append(f"        error: {r.error}")
        elif r.error:
            lines.append(f"        note: {r.error}")
        lines.append("")
    if not any(r.ok for r in results) and results:
        lines.append("  Check directory permissions or choose a writable path.")
    elif any(not r.ok for r in results):
        lines.append("  Fix the failing directories above before starting training.")
    return "\n".join(lines)


def format_probe_results_json(results: list[PreflightProbeResult]) -> str:
    """Structured JSON summary of probe results."""

    payload = {
        "checks": [
            {
                "stage": r.stage,
                "path": r.path,
                "ok": r.ok,
                "operation": r.operation,
                "error": r.error,
            }
            for r in results
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)