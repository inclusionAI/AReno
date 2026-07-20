#!/usr/bin/env bash

# Keep all work inside main so an incomplete copy of this script cannot run.
main() {
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

DRY_RUN=0
CREATED_VENV=0
CURRENT_STEP_ID="startup"
CURRENT_STEP_LABEL="Start installation"
if [[ -n "${XDG_STATE_HOME:-}" ]]; then
  DEFAULT_LOG_DIRECTORY="$XDG_STATE_HOME/areno"
elif [[ -n "${HOME:-}" ]]; then
  DEFAULT_LOG_DIRECTORY="$HOME/.local/state/areno"
else
  DEFAULT_LOG_DIRECTORY="${TMPDIR:-/tmp}/areno"
fi
LOG_FILE="${ARENO_INSTALL_LOG:-$DEFAULT_LOG_DIRECTORY/install.log}"

usage() {
  cat <<'EOF'
AReno installer

Usage:
  bash scripts/install.sh

Options:
  --dry-run   Show the installation plan without changing the environment.
  -h, --help  Show this help text.

The normal installation needs no options. It prepares the Python environment,
installs dependencies, builds AReno's CUDA extension, and verifies the result.
EOF
}

usage_error() {
  printf 'ERROR: %s\n\n' "$*" >&2
  usage >&2
  exit 2
}

startup_error() {
  local message="$1"
  local action="$2"
  printf '\nAReno installation could not start\n'
  printf '==================================\n'
  printf '%s\n\n' "$message"
  printf 'Suggested action: %s\n' "$action"
  exit 1
}

available() {
  command -v "$1" >/dev/null 2>&1
}

require_commands() {
  local command_name
  local missing=()

  for command_name in "$@"; do
    if ! available "$command_name"; then
      missing+=("$command_name")
    fi
  done

  if ((${#missing[@]})); then
    startup_error \
      "Required system tools are missing: ${missing[*]}." \
      "install the missing tools with the system package manager, then rerun the installer"
  fi
}

detect_wsl() {
  local kernel_release
  kernel_release="$(uname -r 2>/dev/null || true)"

  case "$kernel_release" in
    *[Mm]icrosoft*[Ww][Ss][Ll]2*)
      IS_WSL2=1
      ;;
    *[Mm]icrosoft*)
      startup_error \
        "AReno does not support WSL1." \
        "upgrade the distribution to WSL2, then rerun the installer"
      ;;
  esac
}

check_nvidia_host() {
  local failure_action
  local nvidia_smi_output
  local nvidia_smi_status

  if [[ "$IS_WSL2" == "1" ]]; then
    failure_action="install or repair the Windows NVIDIA driver until nvidia-smi -L works inside WSL2"
  else
    failure_action="install or repair the NVIDIA driver until nvidia-smi -L lists the GPU"
  fi

  if ! available nvidia-smi; then
    startup_error "The NVIDIA driver tool nvidia-smi was not found." "$failure_action"
  fi

  set +e
  nvidia_smi_output="$(nvidia-smi -L 2>&1)"
  nvidia_smi_status=$?
  set -e
  if ((nvidia_smi_status != 0)) ||
    [[ -z "$nvidia_smi_output" || "$nvidia_smi_output" == *"No devices"* ]]; then
    startup_error \
      "The NVIDIA driver is installed, but it did not report a usable GPU." \
      "$failure_action"
  fi

  NVIDIA_GPU_SUMMARY="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  if [[ -z "$NVIDIA_GPU_SUMMARY" ]]; then
    NVIDIA_GPU_SUMMARY="$nvidia_smi_output"
  fi
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage_error "unknown option: $1"
      ;;
  esac
  shift
done

if [[ ! -f "$REPO_ROOT/pyproject.toml" || ! -f "$REPO_ROOT/setup.py" ]]; then
  startup_error \
    "AReno must be installed from a complete source checkout." \
    "git clone https://github.com/inclusionAI/AReno.git && cd AReno && bash scripts/install.sh"
fi

cd "$REPO_ROOT"

require_commands cat date dirname env mkdir tee uname

SYSTEM_NAME="$(uname -s 2>/dev/null || printf unknown)"
MACHINE_NAME="$(uname -m 2>/dev/null || printf unknown)"
IS_WSL2=0
NVIDIA_GPU_SUMMARY=""

if [[ "$DRY_RUN" == "0" ]]; then
  if [[ "$SYSTEM_NAME" != "Linux" ]]; then
    startup_error \
      "AReno training and serving require Linux with NVIDIA CUDA; detected $SYSTEM_NAME $MACHINE_NAME." \
      "run this installer on a Linux NVIDIA system"
  fi
  case "$MACHINE_NAME" in
    x86_64|amd64|aarch64|arm64)
      ;;
    *)
      startup_error \
        "AReno does not support CUDA extension builds on $MACHINE_NAME." \
        "use Linux x86_64 or Linux aarch64"
      ;;
  esac
  detect_wsl
  check_nvidia_host
fi

find_python() {
  local candidate
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done
  return 1
}

select_bootstrap_python() {
  local requested="$1"
  local not_found_message="$2"
  local suggested_action="$3"

  if [[ -n "$requested" ]]; then
    BOOTSTRAP_PYTHON="$(command -v "$requested" 2>/dev/null || true)"
  else
    BOOTSTRAP_PYTHON="$(find_python || true)"
  fi

  if [[ -z "$BOOTSTRAP_PYTHON" ]]; then
    startup_error "$not_found_message" "$suggested_action"
  fi
}

check_torch() {
  local python_bin="$1"
  "$python_bin" -c '
import importlib.util
import re

if importlib.util.find_spec("torch") is None:
    raise SystemExit(10)

try:
    import torch
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    raise SystemExit(14)

parts = [int(value) for value in re.findall(r"\d+", torch.__version__.split("+", 1)[0])[:2]]
if tuple(parts + [0] * (2 - len(parts))) < (2, 6):
    print(torch.__version__)
    raise SystemExit(11)
if not getattr(torch.version, "cuda", None):
    print(torch.__version__)
    raise SystemExit(12)
if not torch.cuda.is_available():
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
    raise SystemExit(13)
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPUs {torch.cuda.device_count()}")
'
}

torch_status_for_python() {
  local python_bin="$1"
  local status

  if [[ ! -x "$python_bin" ]]; then
    printf '10\n'
    return
  fi

  if check_torch "$python_bin" >/dev/null 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status"
}

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
NEED_VENV=0

if [[ -n "${PYTHON:-}" ]]; then
  select_bootstrap_python \
    "$PYTHON" \
    "The Python selected by PYTHON was not found." \
    "set PYTHON to a Python 3.10 or newer executable and rerun the installer"
  ENVIRONMENT_DESCRIPTION="the Python selected by PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  select_bootstrap_python \
    "" \
    "Python was not found in the active environment." \
    "repair or deactivate the environment, then rerun the installer"
  ENVIRONMENT_DESCRIPTION="the active Python environment"
else
  select_bootstrap_python \
    "" \
    "Python was not found." \
    "install Python 3.10 or newer and rerun bash scripts/install.sh"

  VENV_TORCH_STATUS="$(torch_status_for_python "$VENV_PYTHON")"
  DETECTED_TORCH_STATUS="$(torch_status_for_python "$BOOTSTRAP_PYTHON")"

  if [[ -x "$VENV_PYTHON" && "$VENV_TORCH_STATUS" == "0" ]]; then
    BOOTSTRAP_PYTHON="$VENV_PYTHON"
    ENVIRONMENT_DESCRIPTION="$REPO_ROOT/.venv"
  elif [[ "$DETECTED_TORCH_STATUS" == "0" ]]; then
    ENVIRONMENT_DESCRIPTION="the detected PyTorch environment at $BOOTSTRAP_PYTHON"
  elif [[ -x "$VENV_PYTHON" && "$VENV_TORCH_STATUS" != "10" ]]; then
    BOOTSTRAP_PYTHON="$VENV_PYTHON"
    ENVIRONMENT_DESCRIPTION="$REPO_ROOT/.venv"
  elif [[ "$DETECTED_TORCH_STATUS" != "10" ]]; then
    ENVIRONMENT_DESCRIPTION="the detected PyTorch environment at $BOOTSTRAP_PYTHON"
  elif [[ -x "$VENV_PYTHON" ]]; then
    BOOTSTRAP_PYTHON="$VENV_PYTHON"
    ENVIRONMENT_DESCRIPTION="$REPO_ROOT/.venv"
  else
    NEED_VENV=1
    ENVIRONMENT_DESCRIPTION="$REPO_ROOT/.venv"
  fi
fi

PYTHON_BIN="$BOOTSTRAP_PYTHON"
if [[ "$NEED_VENV" == "1" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
fi

if ! "$BOOTSTRAP_PYTHON" -c \
  'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  startup_error \
    "$("$BOOTSTRAP_PYTHON" --version 2>&1 || printf 'The selected Python environment') is not supported." \
    "remove the existing .venv or select Python 3.10 or newer with PYTHON"
fi

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

start_step() {
  local number="$1"
  local total="$2"
  CURRENT_STEP_ID="$3"
  CURRENT_STEP_LABEL="$4"
  printf '\n[%s/%s] %s\n' "$number" "$total" "$CURRENT_STEP_LABEL"
}

print_failure_suggestions() {
  case "$CURRENT_STEP_ID" in
    environment)
      printf '%s\n' \
        '- Confirm Python includes the venv and ensurepip modules.' \
        '- Remove an incomplete .venv directory, then rerun the installer.'
      ;;
    packaging)
      printf '%s\n' \
        '- Check network, proxy, package-index, and disk-space availability.' \
        '- Confirm the selected Python environment is writable.'
      ;;
    pytorch)
      printf '%s\n' \
        "- Prepare CUDA-enabled PyTorch 2.6 or newer in the selected environment: $PYTHON_BIN" \
        '- Use the official selector: https://pytorch.org/get-started/locally/' \
        '- On DGX Spark, use the NVIDIA-provided Jupyter environment or a current NGC PyTorch development container.' \
        '- Rerun bash scripts/install.sh after PyTorch is ready.'
      ;;
    cuda)
      printf '%s\n' \
        '- Install the CUDA development toolkit, including nvcc and CUDA headers.' \
        '- Install a C++ compiler (for example, the build-essential package on Ubuntu).' \
        '- Ensure CUDA_HOME points to the toolkit root and CUDA_HOME/bin is on PATH.' \
        '- On DGX systems, apply current DGX OS updates or use an NVIDIA CUDA development container.'
      ;;
    dependencies)
      printf '%s\n' \
        '- Check that the named dependency supports the installed PyTorch, CUDA, Python, and CPU architecture.' \
        '- On DGX Spark, use the NVIDIA-provided Jupyter environment or a current NGC PyTorch development container.' \
        '- Review the final package error above and the complete installation log.'
      ;;
    build)
      printf '%s\n' \
        '- Confirm the PyTorch CUDA version and nvcc toolkit are compatible.' \
        '- Confirm the detected GPU architecture is supported by the installed toolkit.' \
        '- Attach the installation log and `areno env --json` output when requesting support.'
      ;;
    verify)
      printf '%s\n' \
        '- Review the `areno check` failure above for the missing runtime component.' \
        '- Rerun the installer after correcting the reported environment issue.'
      ;;
    *)
      printf '%s\n' \
        '- Review the final error above and the complete installation log.' \
        '- Correct the reported environment issue, then rerun the installer.'
      ;;
  esac
}

installation_failed() {
  local status="$1"
  local message="${2:-The command shown above did not complete successfully.}"
  printf '\nAReno installation failed\n'
  printf '=========================\n'
  printf 'Step: %s\n' "$CURRENT_STEP_LABEL"
  printf 'Exit code: %s\n' "$status"
  printf 'Reason: %s\n\n' "$message"
  printf 'Suggested actions:\n'
  print_failure_suggestions
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nNo changes were made.\n'
  else
    printf '\nInstallation log: %s\n' "$LOG_FILE"
  fi
  exit "$status"
}

run_current_step() {
  local failure_reason="$1"
  shift
  printf '+ '
  quote_command "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CURRENT_STEP_LABEL" >>"$LOG_FILE"
  printf '+ ' >>"$LOG_FILE"
  quote_command "$@" >>"$LOG_FILE"

  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" != "0" ]]; then
    installation_failed "$status" "$failure_reason"
  fi
}

check_python_packages() {
  "$PYTHON_BIN" - "$@" <<'PY'
import re
import sys
from importlib.metadata import PackageNotFoundError, version


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


installed = []
for requirement in sys.argv[1:]:
    name, separator, minimum = requirement.partition(">=")
    try:
        current = version(name)
    except PackageNotFoundError:
        raise SystemExit(1)
    if separator and version_tuple(current) < version_tuple(minimum):
        raise SystemExit(1)
    installed.append(f"{name} {current}")

print("Using installed " + ", ".join(installed))
PY
}

prepare_python_packages() {
  local failure_reason="$1"
  local package_output
  shift

  if [[ -x "$PYTHON_BIN" ]] && package_output="$(check_python_packages "$@" 2>/dev/null)"; then
    printf '%s\n' "$package_output"
    return
  fi

  run_current_step "$failure_reason" "$PYTHON_BIN" -m pip install "$@"
  if [[ "$DRY_RUN" == "0" ]]; then
    if ! package_output="$(check_python_packages "$@" 2>&1)"; then
      installation_failed 1 "$failure_reason"
    fi
    printf '%s\n' "$package_output"
  fi
}

initialize_log() {
  local log_directory
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  log_directory="$(dirname "$LOG_FILE")"
  if ! (umask 077 && mkdir -p "$log_directory" && : >"$LOG_FILE"); then
    startup_error \
      "The installation log cannot be written to $LOG_FILE." \
      "set ARENO_INSTALL_LOG to a writable path and rerun the installer"
  fi
  {
    printf 'AReno installation log\n'
    printf 'Platform: %s %s\n' "$SYSTEM_NAME" "$MACHINE_NAME"
    printf 'Repository: %s\n' "$REPO_ROOT"
    printf 'NVIDIA GPUs:\n%s\n' "$NVIDIA_GPU_SUMMARY"
  } >>"$LOG_FILE"
}

initialize_log

printf 'AReno installer\n'
printf '===============\n'
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Installation plan only; no commands will be executed.\n'
else
  printf 'AReno will prepare and verify the installation automatically.\n'
  printf 'Detected NVIDIA GPU(s):\n%s\n' "$NVIDIA_GPU_SUMMARY"
  printf 'Installation log: %s\n' "$LOG_FILE"
fi
TOTAL_STEPS=7

start_step 1 "$TOTAL_STEPS" environment "Check and Prepare Python environment"
if [[ "$NEED_VENV" == "1" ]]; then
  run_current_step "Python could not create the .venv environment." \
    "$BOOTSTRAP_PYTHON" -m venv "$REPO_ROOT/.venv"
  if [[ "$DRY_RUN" == "0" ]]; then
    CREATED_VENV=1
  fi
else
  printf 'Using %s\n' "$ENVIRONMENT_DESCRIPTION"
fi

if [[ "$DRY_RUN" == "0" ]]; then
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    run_current_step "pip could not be bootstrapped in the selected Python environment." \
      "$PYTHON_BIN" -m ensurepip --upgrade
  fi
elif [[ "$NEED_VENV" == "0" ]]; then
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    printf 'pip is available.\n'
  else
    printf 'pip will be bootstrapped if it is missing.\n'
  fi
fi

start_step 2 "$TOTAL_STEPS" pytorch "Check PyTorch"

PYTORCH_READY=1
if [[ -x "$PYTHON_BIN" ]]; then
  set +e
  TORCH_STATUS_OUTPUT="$(check_torch "$PYTHON_BIN" 2>&1)"
  TORCH_STATUS=$?
  set -e
else
  TORCH_STATUS_OUTPUT=""
  TORCH_STATUS=10
fi

case "$TORCH_STATUS" in
  0)
    printf 'Using installed %s\n' "$TORCH_STATUS_OUTPUT"
    ;;
  10)
    PYTORCH_FAILURE_REASON="PyTorch is not installed in the selected Python environment."
    ;;
  11)
    PYTORCH_FAILURE_REASON="PyTorch $TORCH_STATUS_OUTPUT is older than the required version 2.6."
    ;;
  12)
    PYTORCH_FAILURE_REASON="PyTorch $TORCH_STATUS_OUTPUT is CPU-only; AReno requires a CUDA-enabled build."
    ;;
  13)
    PYTORCH_FAILURE_REASON="$TORCH_STATUS_OUTPUT cannot access a usable NVIDIA GPU."
    ;;
  14)
    PYTORCH_FAILURE_REASON="PyTorch is installed but could not be imported: $TORCH_STATUS_OUTPUT"
    ;;
  *)
    PYTORCH_FAILURE_REASON="PyTorch could not be validated in the selected Python environment."
    ;;
esac

if [[ "$TORCH_STATUS" != "0" ]]; then
  PYTORCH_READY=0
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'PyTorch check: FAILED\n'
    printf 'Reason: %s\n\n' "$PYTORCH_FAILURE_REASON"
    printf 'Suggested actions:\n'
    print_failure_suggestions
  else
    installation_failed 1 "$PYTORCH_FAILURE_REASON"
  fi
fi

start_step 3 "$TOTAL_STEPS" packaging "Check and Prepare packaging tools"
prepare_python_packages "setuptools 69 or newer and wheel could not be prepared." \
  'setuptools>=69' wheel

start_step 4 "$TOTAL_STEPS" cuda "Check GPU support"

resolve_cuda_home() {
  local candidate=""
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf '%s\n' "$CUDA_HOME"
    return
  fi
  if [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
    printf '%s\n' "$CUDA_PATH"
    return
  fi
  if command -v nvcc >/dev/null 2>&1; then
    candidate="$(command -v nvcc)"
    candidate="$("$PYTHON_BIN" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$candidate")"
    candidate="$(cd "$(dirname "$candidate")/.." && pwd -P)"
    printf '%s\n' "$candidate"
    return
  fi
  for candidate in /usr/local/cuda /opt/cuda; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'The installer will locate nvcc, set CUDA_HOME, and detect visible GPU architectures.\n'
else
  DETECTED_CUDA_HOME="$(resolve_cuda_home || true)"
  if [[ -z "$DETECTED_CUDA_HOME" ]]; then
    installation_failed 1 "The CUDA compiler nvcc was not found."
  fi
  export CUDA_HOME="$DETECTED_CUDA_HOME"
  if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
    installation_failed 1 "A C++ compiler was not found."
  fi

  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST="$("$PYTHON_BIN" -c '
import torch

architectures = {
    f"{major}.{minor}"
    for index in range(torch.cuda.device_count())
    for major, minor in [torch.cuda.get_device_capability(index)]
}
print(";".join(sorted(architectures)))
')"
    if [[ -z "$TORCH_CUDA_ARCH_LIST" ]]; then
      installation_failed 1 "The GPU compute capability could not be detected."
    fi
    export TORCH_CUDA_ARCH_LIST
  fi
  printf 'CUDA toolkit: %s\n' "$CUDA_HOME"
  printf 'GPU architecture: %s\n' "$TORCH_CUDA_ARCH_LIST"
fi

start_step 5 "$TOTAL_STEPS" dependencies "Check and Prepare AReno dependencies"
prepare_python_packages "AReno's build helpers (psutil and ninja) could not be installed." \
  psutil ninja
prepare_python_packages "flash-linear-attention 0.2 or newer could not be installed." \
  'flash-linear-attention>=0.2'
if [[ "$DRY_RUN" == "1" ]]; then
  if [[ -x "$PYTHON_BIN" ]] && FLASH_ATTN_OUTPUT="$(check_python_packages 'flash-attn>=2.7' 2>/dev/null)"; then
    printf '%s\n' "$FLASH_ATTN_OUTPUT"
  else
    printf 'FlashAttention will be installed when every visible GPU supports the flash backend.\n'
    run_current_step "FlashAttention could not be installed for the detected GPU." \
      "$PYTHON_BIN" -m pip install 'flash-attn>=2.7' --no-build-isolation
  fi
else
  if "$PYTHON_BIN" -c '
import torch

raise SystemExit(
    0
    if torch.cuda.device_count() > 0
    and all(torch.cuda.get_device_capability(index) >= (8, 0) for index in range(torch.cuda.device_count()))
    else 1
)
'; then
    if FLASH_ATTN_OUTPUT="$(check_python_packages 'flash-attn>=2.7' 2>/dev/null)"; then
      printf '%s\n' "$FLASH_ATTN_OUTPUT"
    else
      run_current_step "FlashAttention could not be installed for the detected GPU." \
        "$PYTHON_BIN" -m pip install 'flash-attn>=2.7' --no-build-isolation
      if ! FLASH_ATTN_OUTPUT="$(check_python_packages 'flash-attn>=2.7' 2>&1)"; then
        installation_failed 1 "The installed FlashAttention does not satisfy version 2.7 or newer."
      fi
      printf '%s\n' "$FLASH_ATTN_OUTPUT"
    fi
  else
    printf 'The visible GPU uses AReno native attention; FlashAttention is not required.\n'
  fi
fi

start_step 6 "$TOTAL_STEPS" build "Build and install AReno"
if [[ "$DRY_RUN" == "1" ]]; then
  run_current_step "AReno's CUDA extension build failed." \
    env 'CUDA_HOME=<detected>' 'TORCH_CUDA_ARCH_LIST=<detected>' \
    "$PYTHON_BIN" -m pip install -e . --no-build-isolation
  printf '\n[7/7] Verify the installation\n'
  printf '+ areno check\n'
  if [[ "$PYTORCH_READY" == "1" ]]; then
    printf '\nInstallation plan complete. No changes were made.\n'
    exit 0
  fi
  printf '\nInstallation plan complete, but PyTorch is not ready. No changes were made.\n'
  exit 1
fi

BUILD_ENV=(env "CUDA_HOME=$CUDA_HOME" "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST")
if [[ -n "${MAX_JOBS:-}" ]]; then
  BUILD_ENV+=("MAX_JOBS=$MAX_JOBS")
fi
run_current_step "AReno's CUDA extension build failed." \
  "${BUILD_ENV[@]}" "$PYTHON_BIN" -m pip install -e . --no-build-isolation

CURRENT_STEP_ID="verify"
CURRENT_STEP_LABEL="Verify the installation"
printf '\n[7/7] %s\n' "$CURRENT_STEP_LABEL"
SCRIPTS_DIR="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
ARENO_BIN="$SCRIPTS_DIR/areno"
if [[ ! -x "$ARENO_BIN" ]]; then
  installation_failed 1 "The areno command was not created in the selected Python environment."
fi
run_current_step "The installed AReno runtime did not pass its readiness check." \
  "$ARENO_BIN" check

printf '\nAReno is ready\n'
printf '==============\n'
printf 'Installation and environment checks completed successfully.\n'
if [[ "$CREATED_VENV" == "1" ]]; then
  printf '\nActivate the prepared environment:\n'
  printf '  source %q\n' "$REPO_ROOT/.venv/bin/activate"
fi
printf '\nStart with:\n'
printf '  %q --help\n' "$ARENO_BIN"
printf '\nInstallation log: %s\n' "$LOG_FILE"
}

main "$@"
