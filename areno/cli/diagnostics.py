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
    click.echo("  Environment variables:")
    for name, value in report["env"].items():
        click.echo(f"    {name}={value if value is not None else '<unset>'}")
