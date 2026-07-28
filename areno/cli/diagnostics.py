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

# Resource demand estimates for multi-process train/serve runs. These are
# deliberately conservative upper bounds -- a false warning is cheap, a silent
# FD/shm exhaustion mid-run is not. Constants are module-level so they are easy
# to tune and surface in diagnostics output; they are not public API.
_BASE_FDS_PER_WORKER = 64
_SHM_BASELINE_BYTES = 1 << 30  # 1 GiB per tensor-parallel group; NCCL + CUDA IPC.
_SHMMAX_PATH = "/proc/sys/kernel/shmmax"

# Severity reused across the host-resource preflight; the diagnostics `check`
# command uses OK/WARN/FAIL, which maps cleanly to normal/warning/blocking.
RESOURCE_OK = "OK"
RESOURCE_WARN = "WARN"
RESOURCE_FAIL = "FAIL"


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


def collect_host_limits() -> dict[str, Any]:
    """Probe OS-level per-process resource limits without touching the engine.

    Returns a dict with `file_descriptors`, `processes`, and `shared_memory`
    entries. Each entry carries `available` (bool), `soft`/`hard` (for rlimits),
    `value` (the effective limit used for comparison), and `error` when a probe
    could not run. A probe that is unavailable on the current platform degrades
    to a warning rather than a failure -- the preflight must not block runs on
    platforms (e.g. macOS/Windows) where a limit simply does not exist.
    """

    return {
        "file_descriptors": _fd_limit(),
        "processes": _nproc_limit(),
        "shared_memory": _shmmax_limit(),
    }


def _fd_limit() -> dict[str, Any]:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError) as exc:
        return _unavailable(f"{type(exc).__name__}: {exc}")
    return _rlimit_entry(soft, hard)


def _nproc_limit() -> dict[str, Any]:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    except (OSError, ValueError) as exc:
        return _unavailable(f"{type(exc).__name__}: {exc}")
    except AttributeError:
        # RLIMIT_NPROC is not defined on every platform; degrade to a warning.
        return _unavailable("RLIMIT_NPROC unavailable")
    return _rlimit_entry(soft, hard)


def _rlimit_entry(soft: Any, hard: Any) -> dict[str, Any]:
    """Build a probe entry from a (soft, hard) rlimit pair.

    `RLIM_INFINITY` means the limit is unbounded -- that satisfies any demand,
    so the probe is `available` with `unbounded=True` rather than unavailable.
    A finite soft limit is the effective value used for comparison.
    """

    import resource

    soft_finite = soft != resource.RLIM_INFINITY
    hard_finite = hard != resource.RLIM_INFINITY
    if not soft_finite and not hard_finite:
        return {
            "available": True,
            "unbounded": True,
            "soft": soft,
            "hard": hard,
            "value": None,
            "error": None,
        }
    value = soft if soft_finite else hard
    return {
        "available": True,
        "unbounded": False,
        "soft": soft,
        "hard": hard,
        "value": int(value),
        "error": None,
    }


def _unavailable(error: str) -> dict[str, Any]:
    return {
        "available": False,
        "unbounded": False,
        "soft": None,
        "hard": None,
        "value": None,
        "error": error,
    }


def _shmmax_limit() -> dict[str, Any]:
    try:
        text = Path(_SHMMAX_PATH).read_text(encoding="utf-8").strip()
        value = int(text)
    except (OSError, ValueError) as exc:
        # /proc/sys/kernel/shmmax is Linux-only; absence is expected elsewhere.
        return _unavailable(f"{type(exc).__name__}: {exc}")
    return {"available": True, "unbounded": False, "soft": value, "hard": value, "value": value, "error": None}


def estimate_resource_demand(world_size: int, tp_size: int) -> dict[str, int]:
    """Return a documented upper-bound estimate of resource demand for one run.

    - file descriptors: per-worker base plus one socket per cross-rank peer for
      the NCCL/tensor-parallel mesh; sum across all `world_size` workers.
    - processes: `world_size` worker ranks plus the driver process.
    - shared memory: baseline per tensor-parallel group, scaled by `tp_size`.

    The estimate is intentionally conservative -- it triggers a warning rather
    than undershooting a real NCCL/CUDA IPC requirement mid-run.
    """

    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    cross_rank_peers = world_size * (world_size - 1)
    return {
        "file_descriptors": _BASE_FDS_PER_WORKER * world_size + cross_rank_peers,
        "processes": world_size + 1,
        "shared_memory": _SHM_BASELINE_BYTES * tp_size,
    }


def preflight_host_resources(
    world_size: int,
    tp_size: int,
    *,
    policy: str = "warn",
    limits: dict[str, Any] | None = None,
) -> list[CheckResult]:
    """Compare observed host limits against documented per-run demand.

    Returns one `CheckResult` per resource dimension (file descriptors,
    processes, shared memory). `policy` is one of skip/warn/block; it does not
    change the returned severities -- the caller decides whether a FAIL under
    `block` should abort. Probes that are unavailable degrade to WARN so the
    preflight never blocks a run on a platform that simply lacks the limit.

    `limits` is injectable for deterministic tests; production callers omit it
    so `collect_host_limits()` probes the real host.
    """

    if policy not in {"skip", "warn", "block"}:
        raise ValueError(f"resource-check policy must be skip/warn/block, got {policy!r}")
    if limits is None:
        limits = collect_host_limits()
    demand = estimate_resource_demand(world_size, tp_size)
    return [
        _fd_result(limits["file_descriptors"], demand["file_descriptors"]),
        _nproc_result(limits["processes"], demand["processes"]),
        _shmmax_result(limits["shared_memory"], demand["shared_memory"]),
    ]


def _fd_result(observed: dict[str, Any], required: int) -> CheckResult:
    return _resource_result(
        observed,
        required,
        name="file descriptors (RLIMIT_NOFILE)",
        unit="",
        adjust="raise the soft limit before launching workers, e.g. `ulimit -n 65536`",
    )


def _nproc_result(observed: dict[str, Any], required: int) -> CheckResult:
    return _resource_result(
        observed,
        required,
        name="processes (RLIMIT_NPROC)",
        unit="",
        adjust="raise the soft limit or run from a shell without a low nproc ulimit, e.g. `ulimit -u 32768`",
    )


def _shmmax_result(observed: dict[str, Any], required: int) -> CheckResult:
    return _resource_result(
        observed,
        required,
        name="shared memory (kernel.shmmax)",
        unit=" bytes",
        adjust="raise the system limit, e.g. `sudo sysctl -w kernel.shmmax=<required>`",
    )


def _resource_result(
    observed: dict[str, Any],
    required: int,
    *,
    name: str,
    unit: str,
    adjust: str,
) -> CheckResult:
    if not observed.get("available"):
        error = observed.get("error") or "probe unavailable"
        return CheckResult(
            RESOURCE_WARN,
            name,
            f"probe unavailable ({error}); required{unit}={required}",
            f"Limit not observable on this platform; {adjust} if the run fails with fd/shmem exhaustion.",
        )
    if observed.get("unbounded"):
        return CheckResult(RESOURCE_OK, name, f"observed=unbounded required{unit}={required}")
    value = int(observed["value"])
    delta = value - required
    detail = f"observed{unit}={value} required{unit}={required} delta{unit}={delta}"
    if value >= required:
        return CheckResult(RESOURCE_OK, name, detail)
    return CheckResult(RESOURCE_FAIL, name, detail, adjust)


def format_resource_preflight(results: list[CheckResult]) -> str:
    """Render resource preflight results as a concise human-readable block."""

    lines = ["Host resource preflight:"]
    for result in results:
        lines.append(f"  {result.status:<4} {result.name}")
        if result.detail:
            lines.append(f"       {result.detail}")
        if result.next_step and result.status in {RESOURCE_WARN, RESOURCE_FAIL}:
            lines.append(f"       -> {result.next_step}")
    return "\n".join(lines)


def should_block_on_resources(results: list[CheckResult]) -> bool:
    """True if any resource preflight result is a blocking failure."""

    return any(result.status == RESOURCE_FAIL for result in results)
