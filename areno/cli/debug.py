"""Runtime failure evidence collector for AReno diagnostics.

Exposes ``areno debug`` commands for collecting environment snapshots,
wrapping commands with auto-collection on failure, and reconstructing
failure bundles from saved tracebacks.

Design rules:
- Every sub-collector uses ``_safe_*`` wrappers so a failure in one area
  never hides the original error or blocks the rest of the bundle.
- No new dependencies; all operations use stdlib plus existing areno
  contracts (``collect_env``, dataclass serialisation, JSON-lines patterns).
- Sensitive environment values are redacted by default.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

# -- Sensitive-data redaction --------------------------------------------------

DEFAULT_REDACT_KEYS: frozenset[str] = frozenset(
    {
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "MODELSCOPE_API_TOKEN",
        "OPENAI_API_KEY",
        "API_KEY",
        "TOKEN",
        "SECRET",
        "KEY",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "AUTH",
        "ACCESS_TOKEN",
        "REFRESH_TOKEN",
        "ARENO_INSTALL_LOG",
    }
)

_REDACT_PATTERNS: tuple[str, ...] = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL", "PRIVATE")


def _is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return upper in DEFAULT_REDACT_KEYS or any(pattern in upper for pattern in _REDACT_PATTERNS)


def _redact_value(value: str) -> str:
    if len(value) <= 4:
        return "****"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


# -- Data model ----------------------------------------------------------------

_FIELD_ORDER: tuple[str, ...] = (
    "timestamp",
    "areno_version",
    "python_version",
    "platform_info",
    "command",
    "resolved_config",
    "env_vars_redacted",
    "gpu_summary",
    "cuda_info",
    "error_type",
    "error_message",
    "error_traceback",
    "process_info",
    "worker_state",
    "collection_warnings",
    "extra",
)


@dataclass
class FailureBundle:
    """Structured failure evidence collected at runtime.

    Every field may be ``None`` when collection for that area failed or
    was disabled.  ``collection_warnings`` records sub-collector errors
    so operators can see what was skipped.
    """

    timestamp: str = ""
    areno_version: str | None = None
    python_version: str = ""
    platform_info: str = ""
    command: list[str] | None = None
    resolved_config: dict | None = None
    env_vars_redacted: dict[str, str] = field(default_factory=dict)
    gpu_summary: dict | None = None
    cuda_info: dict | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    process_info: dict = field(default_factory=dict)
    worker_state: list[dict] | None = None
    collection_warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_ordered_dict(self) -> dict:
        raw = dataclasses.asdict(self)
        ordered: dict[str, Any] = {}
        for key in _FIELD_ORDER:
            if key in raw:
                ordered[key] = raw[key]
        for key in raw:
            if key not in ordered:
                ordered[key] = raw[key]
        return ordered


# -- Safe collectors -----------------------------------------------------------
# Every collector returns None rather than raising, so a GPU or nccl failure
# inside gpu_info never prevents the rest of the bundle from being written.


def _safe_areno_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("areno")
    except (ImportError, PackageNotFoundError):
        return None


def _safe_env_collect(*, redact: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if redact and _is_sensitive_key(key):
            result[key] = _redact_value(value)
        else:
            result[key] = value
    return result


def _safe_gpu_info() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        devices = []
        for idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(idx)
            prop = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": f"{major}.{minor}",
                    "memory_total_gb": round(prop.total_memory / (1024**3), 1),
                }
            )
        return {"available": True, "device_count": len(devices), "devices": devices}
    except ImportError:
        # PyTorch not01 installed; not an error for CPU-only installs.
        return {"available": False, "pytorch_installed": False}
    except Exception:
        return None


def _safe_cuda_info() -> dict | None:
    try:
        nvcc = shutil.which("nvcc")
        cuda_home = os.environ.get("CUDA_HOME")
        return {"cuda_home": cuda_home, "nvcc_path": nvcc}
    except Exception:
        return None


def _safe_traceback(error: BaseException | None) -> str | None:
    if error is None:
        return None
    try:
        return "".join(traceback.format_exception(type(error), error, error.__traceback__))
    except Exception:
        return repr(error)


def _safe_process_info() -> dict:
    try:
        return {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": str(Path.cwd()),
            "executable": sys.executable,
        }
    except Exception:
        return {"pid": os.getpid()}


def _safe_config_dump(config: Any) -> dict | None:
    if config is None:
        return None
    try:
        if dataclasses.is_dataclass(config):
            return dataclasses.asdict(config)
        return {"config_type": type(config).__name__, "config_repr": repr(config)}
    except Exception:
        return None


def _safe_traceback_from_file(path: Path) -> str | None:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
        return text
    except (OSError, UnicodeDecodeError) as exc:
        raise click.UsageError(f"Cannot read traceback file {path}: {exc}") from exc


# -- Bundle collection ---------------------------------------------------------


def collect_failure_bundle(
    *,
    command: list[str] | None = None,
    config: Any = None,
    error: BaseException | None = None,
    include_env: bool = True,
    include_gpu: bool = True,
    redact_env: bool = True,
) -> FailureBundle:
    """Collect failure evidence into a :class:`FailureBundle`.

    Parameters
    ----------
    command:
        The original command-line that triggered the run.
    config:
        An optional ``TrainerConfig`` (or similar) dataclass to serialise.
    error:
        The caught exception, if any.
    include_env:
        When ``False``, environment variables are not collected.
    include_gpu:
        When ``False``, GPU and CUDA information are not collected.
    redact_env:
        When ``True`` (the default), sensitive environment values are redacted.
    """

    warnings: list[str] = []

    def _safe_call(fn, *args, label: str, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            warnings.append(f"{label}: {exc}")
            return None

    bundle = FailureBundle(
        timestamp=datetime.now(timezone.utc).isoformat(),
        areno_version=_safe_call(_safe_areno_version, label="areno_version"),
        python_version=sys.version,
        platform_info=platform.platform(),
        command=list(command) if command else (sys.argv[1:] if len(sys.argv) > 1 else None),
        resolved_config=_safe_call(_safe_config_dump, config, label="config_dump"),
        env_vars_redacted=(_safe_call(_safe_env_collect, redact=redact_env, label="env_collect") if include_env else {}),
        gpu_summary=_safe_call(_safe_gpu_info, label="gpu_info") if include_gpu else None,
        cuda_info=_safe_call(_safe_cuda_info, label="cuda_info") if include_gpu else None,
        error_type=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else None,
        error_traceback=_safe_call(_safe_traceback, error, label="traceback"),
        process_info=_safe_call(_safe_process_info, label="process_info"),
        worker_state=None,
        collection_warnings=warnings,
        extra={},
    )
    return bundle


# -- Bundle output -------------------------------------------------------------


def write_bundle(bundle: FailureBundle, output_dir: Path) -> Path:
    """Persist *bundle* to *output_dir* and return the bundle sub-directory path.

    Creates three files inside a timestamped sub-directory:

    * ``bundle.json`` -- machine-readable structured evidence.
    * ``summary.md``  -- human-readable diagnostic summary.
    * ``traceback.txt`` -- raw traceback (only when an error is present).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = bundle.timestamp.replace(":", "-").replace("T", "-").replace("+", "-")
    bundle_dir = output_dir / f"areno-failure-{safe_ts}"
    bundle_dir.mkdir(exist_ok=True)

    # 1. Machine-readable JSON
    (bundle_dir / "bundle.json").write_text(
        json.dumps(bundle.to_ordered_dict(), indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2. Human-readable Markdown summary
    (bundle_dir / "summary.md").write_text(
        _render_markdown(bundle),
        encoding="utf-8",
    )

    # 3. Raw traceback (for grep / editors)
    if bundle.error_traceback:
        (bundle_dir / "traceback.txt").write_text(
            bundle.error_traceback,
            encoding="utf-8",
        )

    return bundle_dir


def _render_markdown(bundle: FailureBundle) -> str:
    lines = [
        "# AReno Failure Bundle",
        "",
        f"**Timestamp**: {bundle.timestamp}",
        f"**AReno Version**: {bundle.areno_version or 'unknown'}",
        f"**Python**: {bundle.python_version.strip()}",
        f"**Platform**: {bundle.platform_info}",
        "",
    ]

    if bundle.command:
        lines.append("## Command")
        lines.append("```")
        lines.append(" ".join(bundle.command))
        lines.append("```")
        lines.append("")

    if bundle.error_type:
        lines.append("## Error")
        lines.append(f"**Type**: `{bundle.error_type}`")
        lines.append(f"**Message**: {bundle.error_message or ''}")
        lines.append("")
        if bundle.error_traceback:
            lines.append("### Traceback (most recent call last)")
            lines.append("```")
            lines.append(bundle.error_traceback.strip())
            lines.append("```")
            lines.append("")

    if bundle.gpu_summary:
        lines.append("## GPU")
        lines.append("```json")
        lines.append(json.dumps(bundle.gpu_summary, indent=2))
        lines.append("```")
        lines.append("")

    if bundle.cuda_info:
        lines.append("## CUDA")
        lines.append("```json")
        lines.append(json.dumps(bundle.cuda_info, indent=2))
        lines.append("```")
        lines.append("")

    if bundle.process_info:
        lines.append("## Process")
        lines.append("```json")
        lines.append(json.dumps(bundle.process_info, indent=2))
        lines.append("```")
        lines.append("")

    if bundle.collection_warnings:
        lines.append("## Collection Warnings")
        for w in bundle.collection_warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


# -- CLI command ---------------------------------------------------------------


@click.command(
    name="debug",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Collect runtime failure evidence for AReno diagnostics.",
)
@click.option(
    "--collect",
    is_flag=True,
    help="Collect an environment snapshot and write it to --output-dir.",
)
@click.option(
    "--wrap",
    is_flag=True,
    help="Execute a subcommand and auto-collect evidence on failure.",
)
@click.option(
    "--output-dir",
    default="./areno-debug",
    show_default=True,
    help="Directory for failure evidence bundles.",
)
@click.option(
    "--traceback-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Read traceback from a file (post-mortem mode).",
)
@click.option(
    "--redact/--no-redact",
    default=True,
    show_default=True,
    help="Redact sensitive environment variable values.",
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def debug_command(
    collect: bool,
    wrap: bool,
    output_dir: str,
    traceback_file: str | None,
    redact: bool,
    extra_args: tuple[str, ...],
) -> None:
    """Collect runtime failure evidence for AReno diagnostics.

    Three modes:

    \b
    1. Snapshot:  areno debug --collect [--output-dir ./bundles]
       Writes the current environment snapshot as a local bundle.

    \b
    2. Wrap:      areno debug --wrap -- areno train --algo gspo ...
       Runs the subcommand; on failure, auto-collects evidence.

    \b
    3. Post-mortem: areno debug --traceback-file /tmp/crash.txt
       Reconstructs a bundle from a saved traceback file.
    """

    output_path = Path(output_dir)

    if collect:
        _validate_exclusive_options(wrap=wrap, traceback_file=traceback_file, extra_args=extra_args)
        bundle = collect_failure_bundle(include_env=redact)
        result = write_bundle(bundle, output_path)
        click.echo(f"Environment snapshot written to: {result}")

    elif traceback_file:
        _validate_exclusive_options(wrap=wrap, extra_args=extra_args)
        tb_path = Path(traceback_file)
        tb_text = _safe_traceback_from_file(tb_path)
        bundle = collect_failure_bundle(
            error=RuntimeError("(reconstructed from traceback file)"),
            include_env=redact,
        )
        bundle.error_traceback = tb_text
        result = write_bundle(bundle, output_path)
        click.echo(f"Post-mortem bundle written to: {result}")

    elif wrap and extra_args:
        # Wrap mode: execute the subcommand and collect on failure.
        cmd = list(extra_args)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            click.echo(f"Command failed with exit code {exc.returncode}. Collecting evidence...", err=True)
            bundle = collect_failure_bundle(
                command=cmd,
                error=exc,
                include_env=redact,
            )
            result = write_bundle(bundle, output_path)
            click.echo(f"Failure bundle written to: {result}", err=True)
            raise SystemExit(exc.returncode) from None

    else:
        # Interactive / default: print current snapshot to stdout.
        _validate_exclusive_options(wrap=wrap, traceback_file=traceback_file)
        bundle = collect_failure_bundle(include_env=redact)
        click.echo(json.dumps(bundle.to_ordered_dict(), indent=2, default=str))


def _validate_exclusive_options(
    *,
    wrap: bool = False,
    traceback_file: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> None:
    """Reject conflicting options with a clear error."""

    if wrap and traceback_file:
        raise click.UsageError("--wrap and --traceback-file are mutually exclusive")
    if wrap and not extra_args:
        raise click.UsageError("--wrap requires a subcommand; e.g. `areno debug --wrap -- areno train ...`")
    if traceback_file and extra_args:
        raise click.UsageError("--traceback-file does not accept extra arguments")


def main() -> None:
    """Console-script entrypoint for ``areno debug`` (used in tests)."""

    debug_command.main(prog_name="areno debug")


if __name__ == "__main__":
    main()