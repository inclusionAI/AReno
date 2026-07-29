"""Setup diagnostics for the AReno command line.

These commands intentionally avoid importing AReno engine/model modules. They
only inspect the Python environment, optional dependencies, CUDA toolchain, and
the compiled extension importability.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import click

_ENV_VARS = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "MAX_JOBS",
    "TORCH_CUDA_ARCH_LIST",
    "ARENO_BUILD_EXT",
    "HF_HOME",
    "HF_HUB_CACHE",
)


@click.command(name="env", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON support report.")
def env_command(as_json: bool) -> None:
    """Print an AReno environment/support report."""

    report = collect_env()
    if as_json:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    _print_env_report(report)


@click.command(name="check", context_settings={"help_option_names": ["-h", "--help"]})
def check_command() -> None:
    """Check whether this machine is ready to run AReno."""

    report = collect_env()
    results = run_checks(report)
    failed = any(result.status == "FAIL" for result in results)
    click.echo(f"AReno check: {'not ready' if failed else 'ready'}")
    click.echo()
    for result in results:
        click.echo(f"{result.status:<4} {result.name}")
        if result.detail:
            click.echo(f"     {result.detail}")
    next_steps = [result.next_step for result in results if result.status in {"FAIL", "WARN"} and result.next_step]
    if next_steps:
        click.echo()
        click.echo("Next:")
        for step in next_steps:
            click.echo(f"  {step}")
    if failed:
        raise click.exceptions.Exit(1)


def collect_env() -> dict[str, Any]:
    """Collect a lightweight support report without initializing the engine."""

    torch_info = _torch_info()
    cuda_home = _cuda_home()
    nvcc_path = shutil.which("nvcc")
    return {
        "areno": {"version": _package_version("areno")},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "torch": torch_info,
        "cuda": {
            "cuda_home": os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
            "inferred_cuda_home": cuda_home,
            "nvcc": _nvcc_info(nvcc_path),
            "driver": _nvidia_smi_driver_info(),
        },
        "gpus": torch_info.get("gpus", []),
        "dependencies": {
            "flash_attn": _dependency_info("flash-attn", "flash_attn"),
            "flash_linear_attention": _dependency_info("flash-linear-attention", "fla"),
            "areno_accel": _dependency_info(None, "areno.accel._areno_accel"),
        },
        "install": {
            "build_ext_disabled": _build_ext_disabled(),
        },
        "env": {name: os.environ.get(name) for name in _ENV_VARS},
        "paths": {
            "metrics_log_dir": "/tmp/areno/tfevent",
            "hf_cache": os.environ.get("HF_HUB_CACHE") or str(Path.home() / ".cache" / "huggingface" / "hub"),
        },
    }


@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str = ""
    next_step: str = ""


def run_checks(report: dict[str, Any]) -> list[CheckResult]:
    """Return readiness checks with actionable statuses."""

    results: list[CheckResult] = []
    py_version = sys.version_info
    results.append(
        _result(
            py_version >= (3, 10),
            "Python >= 3.10",
            f"found {platform.python_version()}",
            "Use Python 3.10 or newer.",
        )
    )

    system = report["platform"]["system"]
    machine = report["platform"]["machine"]
    platform_ok = system == "Linux"
    results.append(
        _result(
            platform_ok,
            "Supported platform",
            f"{system} {machine}",
            "Run AReno on Linux with an NVIDIA CUDA GPU. On Windows, use WSL2.",
        )
    )

    torch_info = report["torch"]
    torch_imported = bool(torch_info["imported"])
    results.append(
        _result(
            torch_imported,
            "PyTorch import",
            torch_info.get("version") or torch_info.get("error", ""),
            "Install CUDA-enabled PyTorch matching your CUDA version.",
        )
    )
    results.append(
        _result(
            torch_imported and _version_at_least(torch_info.get("version"), (2, 6)),
            "PyTorch >= 2.6",
            torch_info.get("version") or "not importable",
            "Install PyTorch 2.6 or newer with CUDA support.",
        )
    )
    results.append(
        _result(
            torch_imported and bool(torch_info.get("cuda_build")),
            "PyTorch CUDA build",
            f"torch.version.cuda={torch_info.get('cuda_build')}",
            "Install a CUDA-enabled PyTorch build; CPU-only torch cannot run AReno.",
        )
    )
    results.append(
        _result(
            torch_imported and bool(torch_info.get("cuda_available")),
            "torch.cuda.is_available()",
            f"visible_gpus={torch_info.get('device_count')}",
            "Check NVIDIA driver installation, CUDA_VISIBLE_DEVICES, and PyTorch CUDA compatibility.",
        )
    )
    results.append(
        _result(
            bool(report["gpus"]),
            "NVIDIA GPU visibility",
            ", ".join(gpu["name"] for gpu in report["gpus"]) if report["gpus"] else "no GPUs reported by torch",
            "Make at least one NVIDIA GPU visible to the process.",
        )
    )

    for label in ("flash_attn", "flash_linear_attention"):
        dep = report["dependencies"][label]
        results.append(
            _result(
                bool(dep["imported"]),
                f"{label} import",
                dep.get("version") or dep.get("error", ""),
                f"Install {dep['distribution']} before installing AReno.",
                warn=True,
            )
        )

    accel = report["dependencies"]["areno_accel"]
    accel_imported = bool(accel["imported"])
    build_ext_disabled = bool(report.get("install", {}).get("build_ext_disabled"))
    if build_ext_disabled and not accel_imported:
        results.append(
            CheckResult(
                "FAIL",
                "ARENO_BUILD_EXT",
                "ARENO_BUILD_EXT=0 skipped the runtime CUDA extension",
                "Reinstall without ARENO_BUILD_EXT=0 before training or serving: pip install -e . --no-build-isolation",
            )
        )
    cuda_home = report["cuda"]["cuda_home"]
    results.append(_cuda_toolkit_result("CUDA_HOME", cuda_home or "not set", bool(cuda_home), accel_imported))
    nvcc = report["cuda"]["nvcc"]
    results.append(
        _cuda_toolkit_result(
            "nvcc",
            nvcc["version"] or nvcc["path"] or "not found",
            bool(nvcc["path"]),
            accel_imported,
        )
    )
    results.append(
        _result(
            accel_imported,
            "areno_accel import",
            accel.get("error", "imported") if not accel["imported"] else "imported",
            "Reinstall AReno from source with CUDA enabled: pip install -e . --no-build-isolation",
        )
    )

    for label, path in report["paths"].items():
        results.append(_writable_path_check(label, path))
    return results


def _cuda_toolkit_result(name: str, detail: str, present: bool, runtime_ready: bool) -> CheckResult:
    if present:
        return CheckResult("OK", name, detail)
    if runtime_ready:
        return CheckResult("OK", name, f"{detail} (not required for runtime; areno_accel imports)")
    next_step = (
        "export CUDA_HOME=/usr/local/cuda"
        if name == "CUDA_HOME"
        else "Add CUDA's bin directory to PATH, for example: export PATH=$CUDA_HOME/bin:$PATH"
    )
    return CheckResult("WARN", name, detail, next_step)


def _result(ok: bool, name: str, detail: str, next_step: str, *, warn: bool = False) -> CheckResult:
    if ok:
        return CheckResult("OK", name, detail)
    return CheckResult("WARN" if warn else "FAIL", name, detail, next_step)


def _writable_path_check(label: str, path_text: str) -> CheckResult:
    path = Path(path_text).expanduser()
    if path.exists() and not path.is_dir():
        return CheckResult(
            "WARN",
            f"{label} writable",
            f"{path} (exists but is a file, not a directory)",
            "Remove the file or choose a different directory path.",
        )
    parent = path if path.is_dir() else _nearest_existing_parent(path)
    ok = os.access(parent, os.W_OK)
    return _result(
        ok,
        f"{label} writable",
        str(path),
        f"Create the directory or choose a writable path: mkdir -p {parent}",
        warn=True,
    )


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _torch_info() -> dict[str, Any]:
    try:
        torch = import_module("torch")
    except Exception as exc:
        return {
            "imported": False,
            "error": f"{type(exc).__name__}: {exc}",
            "version": None,
            "cuda_build": None,
            "cuda_runtime": None,
            "cuda_runtime_error": None,
            "cuda_available": False,
            "device_count": 0,
            "gpus": [],
        }
    cuda_available = bool(torch.cuda.is_available())
    runtime_version = None
    runtime_error = None
    if cuda_available:
        try:
            runtime_version = _format_cuda_version(int(torch.cuda.cudart().cudaRuntimeGetVersion()))
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpus = []
    for idx in range(device_count):
        try:
            major, minor = torch.cuda.get_device_capability(idx)
            capability = f"{major}.{minor}"
            name = torch.cuda.get_device_name(idx)
        except Exception as exc:
            capability = None
            name = f"unavailable ({type(exc).__name__}: {exc})"
        gpus.append({"index": idx, "name": name, "capability": capability})
    return {
        "imported": True,
        "error": None,
        "version": torch.__version__,
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_runtime": runtime_version,
        "cuda_runtime_error": runtime_error,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpus": gpus,
    }


def _format_cuda_version(version: int) -> str:
    major = version // 1000
    minor = (version % 1000) // 10
    patch = version % 10
    return f"{major}.{minor}.{patch}"


def _version_at_least(version: str | None, minimum: tuple[int, int]) -> bool:
    if not version:
        return False
    parts: list[int] = []
    for piece in version.split("+", 1)[0].split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < len(minimum):
        parts.append(0)
    return tuple(parts[: len(minimum)]) >= minimum


def _dependency_info(package_name: str | None, module: str, *, distribution: str | None = None) -> dict[str, Any]:
    dist_name = distribution if distribution is not None else package_name
    version = _package_version(dist_name) if dist_name else None
    try:
        import_module(module)
    except Exception as exc:
        return {
            "distribution": dist_name,
            "module": module,
            "version": version,
            "imported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"distribution": dist_name, "module": module, "version": version, "imported": True, "error": None}


def _package_version(name: str | None) -> str | None:
    if not name:
        return None
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _build_ext_disabled() -> bool:
    value = os.environ.get("ARENO_BUILD_EXT")
    return value is not None and value.lower() in {"0", "false", "no", "off"}


def _cuda_home() -> str | None:
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            return value
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parents[1])
    return None


def _nvcc_info(path: str | None) -> dict[str, str | None]:
    if not path:
        return {"path": None, "version": None}
    try:
        proc = subprocess.run(
            [path, "--version"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except Exception as exc:
        return {"path": path, "version": f"{type(exc).__name__}: {exc}"}
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {"path": path, "version": lines[-1] if lines else None}


def _nvidia_smi_driver_info() -> dict[str, str | None]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"path": None, "driver_version": None, "cuda_version": None, "error": "nvidia-smi not found"}
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=driver_version,cuda_version", "--format=csv,noheader", "-i", "0"],
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return {"path": smi, "driver_version": None, "cuda_version": None, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"path": smi, "driver_version": None, "cuda_version": None, "error": proc.stderr.strip()}
    output = proc.stdout.strip()
    if not output:
        return {
            "path": smi,
            "driver_version": None,
            "cuda_version": None,
            "error": "nvidia-smi returned empty output",
        }
    values = [part.strip() for part in output.split(",", 1)]
    return {
        "path": smi,
        "driver_version": values[0] if values else None,
        "cuda_version": values[1] if len(values) > 1 else None,
        "error": None,
    }


def _print_env_report(report: dict[str, Any]) -> None:
    click.echo("AReno environment")
    click.echo(f"  AReno: {report['areno']['version'] or 'unknown'}")
    click.echo(f"  Python: {report['python']['version']} ({report['python']['executable']})")
    platform_info = report["platform"]
    click.echo(f"  Platform: {platform_info['platform']} [{platform_info['machine']}]")
    torch_info = report["torch"]
    click.echo(f"  PyTorch: {torch_info.get('version') or 'not importable'}")
    click.echo(f"  PyTorch CUDA build: {torch_info.get('cuda_build') or 'none'}")
    click.echo(f"  CUDA runtime: {torch_info.get('cuda_runtime') or 'unavailable'}")
    click.echo(f"  torch.cuda.is_available: {torch_info.get('cuda_available')}")
    click.echo(f"  CUDA_HOME: {report['cuda']['cuda_home'] or 'not set'}")
    if report["cuda"].get("inferred_cuda_home") and report["cuda"].get("inferred_cuda_home") != report["cuda"].get(
        "cuda_home"
    ):
        click.echo(f"  inferred CUDA_HOME: {report['cuda']['inferred_cuda_home']}")
    nvcc = report["cuda"]["nvcc"]
    click.echo(f"  nvcc: {nvcc['path'] or 'not found'}")
    if nvcc["version"]:
        click.echo(f"    {nvcc['version']}")
    driver = report["cuda"]["driver"]
    click.echo(f"  nvidia-smi: {driver['path'] or 'not found'}")
    if driver.get("driver_version") or driver.get("cuda_version"):
        click.echo(f"    driver={driver.get('driver_version')} cuda={driver.get('cuda_version')}")
    elif driver.get("error"):
        click.echo(f"    {driver['error']}")
    click.echo("  GPUs:")
    if report["gpus"]:
        for gpu in report["gpus"]:
            click.echo(f"    [{gpu['index']}] {gpu['name']} (cc {gpu['capability']})")
    else:
        click.echo("    none visible")
    click.echo("  Dependencies:")
    for name, dep in report["dependencies"].items():
        status = "ok" if dep["imported"] else "missing"
        version = dep["version"] or "unknown"
        click.echo(f"    {name}: {status} (version={version})")
        if dep["error"]:
            click.echo(f"      {dep['error']}")
    install = report.get("install", {})
    click.echo("  Install:")
    click.echo(f"    ARENO_BUILD_EXT disabled: {install.get('build_ext_disabled', False)}")
    click.echo("  Environment variables:")
    for name, value in report["env"].items():
        click.echo(f"    {name}={value if value is not None else '<unset>'}")


# ---------------------------------------------------------------------------
# Pre-flight checks for ``areno train``
#
# These checks run before training starts so that environment problems (missing
# GPU, insufficient disk, broken paths) are caught *before* a potentially long
# model download.  The three-level severity model lets users override
# non-fatal issues with ``--skip-check`` while hard failures always block:
#
#   CRITICAL — training is impossible (no GPU, no PyTorch).  Cannot be skipped.
#   ERROR    — training will almost certainly fail (wrong PyTorch version,
#              missing areno_accel, path not found).  Skippable with --skip-check.
#   WARNING  — training works but may be slower or limited (no flash_attn,
#              non-Linux platform).  Never blocks.
# ---------------------------------------------------------------------------

# Kept for backward compatibility with any external callers that inspect labels.
_CHECK_LABEL = "PASS", "WARN", "ERROR", "CRITICAL"


@dataclass(frozen=True)
class PreflightCheck:
    """A single pre-flight check result with severity level."""

    level: str  # "PASS", "WARN", "ERROR", "CRITICAL"
    name: str
    detail: str = ""
    next_step: str = ""


@dataclass(frozen=True)
class PreflightResult:
    """Aggregated pre-flight check results.

    ``critical_failures`` block training unconditionally (even with
    ``--skip-check``).  ``errors`` block training unless the user passes
    ``--skip-check``.  ``warnings`` never block.
    """

    checks: list[PreflightCheck]
    kaggle_detected: bool = False

    @property
    def passed(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.level == "PASS"]

    @property
    def warnings(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.level == "WARN"]

    @property
    def errors(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.level == "ERROR"]

    @property
    def critical_failures(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.level == "CRITICAL"]


def _is_kaggle() -> bool:
    """Detect whether we are running inside a Kaggle Notebook.

    Kaggle sets ``KAGGLE_KERNEL_RUN_TYPE`` and always mounts ``/kaggle/working``.
    We check both so the detection works even if one indicator is missing.
    """

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return Path("/kaggle/working").is_dir()


def _disk_free_gb(path: str) -> float | None:
    """Return available disk space in GB for the partition holding *path*."""

    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return usage.free / (1024**3)


_MIN_DISK_GB = 5.0


def run_preflight_checks(
    config,
    *,
    reward_ckpt: str | None = None,
) -> PreflightResult:
    """Run pre-flight environment and config checks before training.

    *config* is a ``TrainerConfig`` (or subclass).  The function reuses
    :func:`collect_env` for system-level inspection, then adds training-specific
    checks such as GPU/world_size matching and path reachability.
    """

    report = collect_env()
    checks: list[PreflightCheck] = []

    # --- CRITICAL: cannot train without these ---------------------------
    py_version = sys.version_info
    checks.append(
        PreflightCheck(
            "PASS" if py_version >= (3, 10) else "CRITICAL",
            "Python >= 3.10",
            f"found {platform.python_version()}",
            "Use Python 3.10 or newer.",
        )
    )

    torch_info = report["torch"]
    torch_imported = bool(torch_info["imported"])
    checks.append(
        PreflightCheck(
            "PASS" if torch_imported else "CRITICAL",
            "PyTorch import",
            torch_info.get("version") or torch_info.get("error", ""),
            "Install CUDA-enabled PyTorch matching your CUDA version.",
        )
    )

    cuda_available = bool(torch_info.get("cuda_available"))
    checks.append(
        PreflightCheck(
            "PASS" if cuda_available else "CRITICAL",
            "torch.cuda.is_available()",
            f"visible_gpus={torch_info.get('device_count')}",
            "Check NVIDIA driver installation and CUDA_VISIBLE_DEVICES.",
        )
    )

    gpu_count = len(report.get("gpus", []))
    checks.append(
        PreflightCheck(
            "PASS" if gpu_count >= 1 else "CRITICAL",
            "NVIDIA GPU visibility",
            ", ".join(g["name"] for g in report["gpus"]) if report["gpus"] else "no GPUs reported",
            "Make at least one NVIDIA GPU visible to the process.",
        )
    )

    # --- ERROR: training will almost certainly fail ----------------------
    checks.append(
        PreflightCheck(
            "PASS" if (torch_imported and _version_at_least(torch_info.get("version"), (2, 6))) else "ERROR",
            "PyTorch >= 2.6",
            torch_info.get("version") or "not importable",
            "Install PyTorch 2.6 or newer with CUDA support.",
        )
    )

    cuda_build = torch_info.get("cuda_build")
    checks.append(
        PreflightCheck(
            "PASS" if cuda_build else "ERROR",
            "PyTorch CUDA build",
            f"torch.version.cuda={cuda_build}" if cuda_build else "CPU-only build",
            "Install a CUDA-enabled PyTorch build; CPU-only torch cannot run AReno.",
        )
    )

    checks.append(
        PreflightCheck(
            "PASS" if gpu_count >= config.world_size else "ERROR",
            "GPU count >= world_size",
            f"gpus={gpu_count}, world_size={config.world_size}",
            f"Reduce --world-size to <= {gpu_count} or make more GPUs visible.",
        )
    )

    accel = report["dependencies"]["areno_accel"]
    checks.append(
        PreflightCheck(
            "PASS" if accel["imported"] else "ERROR",
            "areno_accel import",
            accel.get("error", "imported") if not accel["imported"] else "imported",
            "Reinstall AReno from source: pip install -e . --no-build-isolation",
        )
    )

    # --- ERROR: path reachability ----------------------------------------
    # Distinguish local paths from remote model repo IDs.  The order matters:
    # we check existence first, then use the leading character to decide
    # whether a non-existent path is a local file (absolute/home-relative) or
    # a remote repo ID (e.g. "Qwen/Qwen3-0.6B" contains "/" but is not a
    # filesystem path).  We cannot rely on Path.suffix alone because names
    # like "Qwen3.5-0.8B" have a suffix-like trailing segment.
    ckpt_path = Path(config.ckpt)
    if ckpt_path.exists() or ckpt_path.is_dir():
        # Path exists locally — no further checks needed.
        checks.append(
            PreflightCheck(
                "PASS",
                "Checkpoint path",
                f"{config.ckpt} (found)",
            )
        )
    elif config.ckpt.startswith(("/", "~")):
        # Absolute or home-relative path that doesn't exist — this is a user
        # error, not a remote download target.
        checks.append(
            PreflightCheck(
                "ERROR",
                "Checkpoint path",
                f"{config.ckpt} (not found)",
                f"Verify --ckpt path: {config.ckpt}",
            )
        )
    elif "/" in config.ckpt or "\\" in config.ckpt:
        # Relative path with a separator but no leading slash — convention for
        # HuggingFace/ModelScope repo IDs like "Qwen/Qwen3-0.6B".  These will
        # be fetched from the model hub at training time.
        checks.append(
            PreflightCheck(
                "PASS",
                "Checkpoint path",
                f"{config.ckpt} (remote, will download from {config.model_hub})",
            )
        )
    else:
        # Bare name without a path separator — could be a local file in the
        # current directory or a hub shorthand.  Check existence to decide.
        checks.append(
            PreflightCheck(
                "PASS" if ckpt_path.exists() else "ERROR",
                "Checkpoint path",
                f"{config.ckpt} ({'found' if ckpt_path.exists() else 'not found'})",
                f"Verify --ckpt path: {config.ckpt}",
            )
        )

    # Dataset paths come in three forms: local files/directories (check existence),
    # remote dataset refs like "AI-MO/NuminaMath-CoT" (will be fetched by the
    # datasets library), or local files with known suffixes that should exist.
    dataset_path = Path(config.dataset_path)
    dataset_supported = dataset_path.suffix.lower() in _SUPPORTED_DATASET_SUFFIXES
    if dataset_path.exists() or dataset_path.is_dir():
        checks.append(
            PreflightCheck(
                "PASS",
                "Dataset path",
                f"{config.dataset_path} (found)",
            )
        )
    elif dataset_supported and not dataset_path.exists():
        checks.append(
            PreflightCheck(
                "ERROR",
                "Dataset path",
                f"{config.dataset_path} (not found)",
                f"Verify --dataset-path: {config.dataset_path}",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "PASS",
                "Dataset path",
                f"{config.dataset_path} (remote, will download)",
            )
        )

    # --- ERROR: disk space -----------------------------------------------
    cwd_free = _disk_free_gb(os.getcwd())
    if cwd_free is not None:
        checks.append(
            PreflightCheck(
                "PASS" if cwd_free >= _MIN_DISK_GB else "ERROR",
                "Disk space (workdir)",
                f"{cwd_free:.1f} GB available in {os.getcwd()}",
                f"Free at least {_MIN_DISK_GB:.0f} GB in the working directory.",
            )
        )

    save_path = getattr(config, "save_path", None)
    if save_path:
        save_dir = Path(save_path)
        target_dir = save_dir if save_dir.is_dir() else save_dir.parent
        if not target_dir.exists():
            target_dir = _nearest_existing_parent(save_dir)
        save_free = _disk_free_gb(str(target_dir))
        if save_free is not None:
            checks.append(
                PreflightCheck(
                    "PASS" if save_free >= _MIN_DISK_GB else "ERROR",
                    "Disk space (save-path)",
                    f"{save_free:.1f} GB available for {save_path}",
                    f"Free at least {_MIN_DISK_GB:.0f} GB for --save-path.",
                )
            )

    # --- ERROR: metrics log dir writable ---------------------------------
    metrics_dir = getattr(config, "metrics_log_dir", None)
    if metrics_dir:
        checks.append(_writable_preflight_check("metrics_log_dir", metrics_dir))

    # --- WARNING: optional dependencies ----------------------------------
    # These never block training. flash_attn falls back to native attention;
    # CUDA_HOME/nvcc are only needed for source compilation, not runtime;
    # non-Linux platforms may work via WSL2.
    for label in ("flash_attn", "flash_linear_attention"):
        dep = report["dependencies"][label]
        checks.append(
            PreflightCheck(
                "PASS" if dep["imported"] else "WARN",
                f"{label} import",
                dep.get("version") or dep.get("error", "not installed"),
                f"Install {dep['distribution']} for better performance (optional).",
            )
        )

    cuda_home = report["cuda"]["cuda_home"]
    checks.append(
        PreflightCheck(
            "PASS" if cuda_home else "WARN",
            "CUDA_HOME",
            cuda_home or "not set",
            "export CUDA_HOME=/usr/local/cuda (optional; not required if areno_accel imports).",
        )
    )

    nvcc = report["cuda"]["nvcc"]
    checks.append(
        PreflightCheck(
            "PASS" if nvcc["path"] else "WARN",
            "nvcc",
            nvcc["version"] or nvcc["path"] or "not found",
            "Add CUDA's bin directory to PATH (optional; not required if areno_accel imports).",
        )
    )

    system = report["platform"]["system"]
    checks.append(
        PreflightCheck(
            "PASS" if system == "Linux" else "WARN",
            "Supported platform",
            f"{system} {report['platform']['machine']}",
            "Run AReno on Linux with an NVIDIA CUDA GPU. On Windows, use WSL2.",
        )
    )

    return PreflightResult(checks=checks, kaggle_detected=_is_kaggle())


def _writable_preflight_check(label: str, path_text: str) -> PreflightCheck:
    """Return a preflight check for path writability."""

    path = Path(path_text).expanduser()
    if path.exists() and not path.is_dir():
        return PreflightCheck(
            "ERROR",
            f"{label} writable",
            f"{path} (exists but is a file, not a directory)",
            "Remove the file or choose a different directory path.",
        )
    parent = path if path.is_dir() else _nearest_existing_parent(path)
    ok = os.access(parent, os.W_OK)
    return PreflightCheck(
        "PASS" if ok else "ERROR",
        f"{label} writable",
        str(path),
        f"Create the directory: mkdir -p {path}",
    )


_SUPPORTED_DATASET_SUFFIXES = {".json", ".jsonl", ".parquet", ".csv", ".tsv", ".arrow"}


def _style(text: str, *, color: bool, fg: str | None = None, bold: bool = False) -> str:
    return click.style(text, fg=fg, bold=bold) if color else text


def _preflight_level_str(level: str, *, color: bool) -> str:
    """Return a padded, optionally colored level label."""

    colors = {
        "PASS": "bright_green",
        "WARN": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }
    text = level.ljust(8)
    if color:
        return click.style(text, fg=colors.get(level, "white"), bold=level in {"ERROR", "CRITICAL"})
    return text


def print_preflight_result(result: PreflightResult, *, color: bool = True) -> None:
    """Print pre-flight check results in a readable format.

    The output is designed to be readable in both interactive terminals and
    non-interactive environments (Kaggle ``!`` cells, CI logs, piped output).
    We deliberately do *not* use ``click.confirm`` to prompt the user because
    stdin may not be a TTY; instead, train.py decides whether to proceed or
    exit based on the structured ``PreflightResult``.
    """

    bar = "=" * 80
    click.echo(_style(bar, color=color, fg="bright_black"))
    click.echo(_style("Pre-flight environment check", color=color, fg="bright_white", bold=True))
    click.echo(_style(bar, color=color, fg="bright_black"))

    for check in result.checks:
        level_str = _preflight_level_str(check.level, color=color)
        line = f"  {level_str}  {check.name}"
        if check.detail:
            line += f": {check.detail}"
        click.echo(line)

    # Summary line
    n_pass = len(result.passed)
    n_warn = len(result.warnings)
    n_err = len(result.errors)
    n_crit = len(result.critical_failures)

    click.echo()
    if n_crit:
        summary = _style("FAILED", color=color, fg="red", bold=True)
        summary += f" — {n_crit} critical, {n_err} errors, {n_warn} warnings, {n_pass} passed."
    elif n_err:
        summary = _style("FAILED", color=color, fg="red", bold=True)
        summary += f" — {n_err} errors, {n_warn} warnings, {n_pass} passed."
    else:
        summary = _style("OK", color=color, fg="bright_green", bold=True)
        summary += f" — {n_pass} passed, {n_warn} warning(s)."

    click.echo(f"  Result: {summary}")

    if result.kaggle_detected:
        click.echo()
        click.echo(f"  {_style('INFO', color=color, fg='bright_cyan')}  Kaggle environment detected.")
        click.echo(f"  {_style('INFO', color=color, fg='bright_cyan')}  If training is slow, try reducing --batch-size and --max-new-tokens.")

    # Next steps for failures
    next_steps = [c.next_step for c in result.checks if c.level in {"CRITICAL", "ERROR"} and c.next_step]
    if next_steps:
        click.echo()
        click.echo("  Next steps:")
        for step in next_steps:
            click.echo(f"    - {step}")

    click.echo(_style(bar, color=color, fg="bright_black"))
