#!/usr/bin/env bash
# Copyright 2025-2026 inclusionAI ASystem Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared installer functions, sourced by scripts/install.sh (Bash 3.2+).
# ARENO_INSTALL_* variables hold state shared by installation stages. Function
# temporaries are lowercase locals; immutable logging constants are readonly.
# Keep commands as argv arrays/"$@" rather than evaluating command strings.

# External configuration: preserve supplied values and default only when unset/empty.
# CLI switches can enable the four options; parse_args validates boolean values.
ARENO_INSTALL_DRY_RUN="${ARENO_INSTALL_DRY_RUN:-false}"
ARENO_INSTALL_NOTEBOOK="${ARENO_INSTALL_NOTEBOOK:-false}"
ARENO_INSTALL_FLASH="${ARENO_INSTALL_FLASH:-false}"
ARENO_INSTALL_BUILD_ONLY="${ARENO_INSTALL_BUILD_ONLY:-false}"
ARENO_INSTALL_LOG_PATH="${ARENO_INSTALL_LOG:-${XDG_STATE_HOME:-${HOME}/.local/state}/areno/install.log}"

# Only these owned ANSI sequences are added to terminal output. Tool/user text
# is printed as data, so percent signs and backslashes retain their meaning.
readonly ARENO_LOG_RESET=$'\033[0m'
readonly ARENO_LOG_INFO=$'\033[36m'
readonly ARENO_LOG_STATUS=$'\033[1;34m'
readonly ARENO_LOG_SUCCESS=$'\033[32m'
readonly ARENO_LOG_WARN=$'\033[33m'
readonly ARENO_LOG_ERROR=$'\033[31m'

# Reset internal state before parsing options or using installation/log functions.
# Do not inherit these values from the environment or a previous run: backend,
# interpreter, runtime pins and log readiness must come from this run's checks.
# Configuration above is deliberately preserved. This function does no file I/O.
areno::install::init_state() {
  ARENO_INSTALL_STEP="Parse arguments"

  # Environment selection records its result or plans a venv without creating it.
  ARENO_INSTALL_CREATE_VENV="false"
  ARENO_INSTALL_PROTECT_GLOBAL="false"
  ARENO_INSTALL_PYTHON_BIN=""
  ARENO_INSTALL_BOOTSTRAP_PYTHON=""
  ARENO_INSTALL_PACKAGES=""
  ARENO_INSTALL_BACKEND=""
  ARENO_INSTALL_HOST_SYSTEM=""
  ARENO_INSTALL_HOST_ARCH=""
  ARENO_INSTALL_PYTHON_VERSION=""
  ARENO_INSTALL_CUDA_VERSION=""

  # Arrays use array initialization; scalar state always uses double quotes.
  ARENO_INSTALL_PROFILE_PACKAGES=()
  ARENO_INSTALL_PROFILE_INDEX=""

  # Persistent files become available only after successful host preflight.
  ARENO_INSTALL_LOG_STARTED="false"
  ARENO_INSTALL_CONSTRAINTS=""

  # Cleanup owns only a venv created by this run; existing packages are not rolled back.
  ARENO_INSTALL_OWNED_VENV=""
  ARENO_INSTALL_PACKAGES_TOUCHED="false"
}

# Print supported options and required host setup without probing the environment.
areno::install::usage() {
  cat <<'USAGE'
Usage: bash scripts/install.sh [--dry-run] [--notebook] [--flash] [--build-only]

  --dry-run     Inspect prerequisites and print commands without changing files.
  --notebook    Use the kernel interpreter supplied in PYTHON; never select .venv.
  --flash       Install optional CUDA FlashAttention (CUDA only).
  --build-only  Build a CUDA image without device checks. Requires existing CUDA
                PyTorch and explicit TORCH_CUDA_ARCH_LIST. Does not verify readiness.

PYTHON selects an interpreter. Otherwise use an active virtualenv/conda, a prepared
runtime on PATH, or the checkout's .venv. Automatic global reuse requires a clean
pip check and preserves all installed package versions. Prepare system drivers/toolkits first;
Ascend also requires matching PyTorch/torch_npu and the sourced CANN environment.
The installer uses standard Python package CLIs, with no embedded Python code.
USAGE
}

# Return success only when the selected output descriptor can display color.
# Args: output fd (1 for normal output, 2 for warnings/errors).
areno::log::color_enabled() {
  [[ -t "$1" && "${TERM:-dumb}" != dumb && -z "${NO_COLOR:-}" ]]
}

# Remove terminal CSI sequences before writing command output or log records.
# Reads stdin, writes plain text to stdout. Flush each line so pip/build output
# stays live even when the filter feeds a pipe instead of a terminal.
areno::log::plain() {
  awk '
    BEGIN { csi = sprintf("%c", 27) "\\[[0-?]*[ -/]*[@-~]" }
    { gsub(csi, ""); print; fflush() }
  '
}

# Emit one record per line; terminal styling never enters the persistent log.
# Args: level, output fd, message(s). Reads ARENO_INSTALL_LOG_{STARTED,PATH}.
areno::log::write() {
  local level="$1"
  local fd="$2"
  local color=""
  local reset=""
  local timestamp
  local message
  local line
  local record
  shift 2

  timestamp="$(date '+%H:%M:%S')"
  if areno::log::color_enabled "${fd}"; then
    case "${level}" in
      INFO) color="${ARENO_LOG_INFO}" ;;
      STATUS) color="${ARENO_LOG_STATUS}" ;;
      SUCCESS) color="${ARENO_LOG_SUCCESS}" ;;
      WARN) color="${ARENO_LOG_WARN}" ;;
      ERROR) color="${ARENO_LOG_ERROR}" ;;
    esac
    reset="${ARENO_LOG_RESET}"
  fi
  for message in "$@"; do
    # Captured CLI output may contain several lines or forced terminal colors.
    message="$(printf '%s' "${message}" | areno::log::plain)"
    while IFS="" read -r line; do
      printf -v record '[%s] %-7s %s' "${timestamp}" "${level}" "${line}"
      printf '%s%s%s\n' "${color}" "${record}" "${reset}" >&"${fd}"
      if ${ARENO_INSTALL_LOG_STARTED}; then
        printf '%s\n' "${record}" >> "${ARENO_INSTALL_LOG_PATH}"
      fi
    done <<< "${message}"
  done
}

# Semantic entry points keep severity and output routing explicit at call sites.
areno::log::info() { areno::log::write INFO 1 "$@"; }
areno::log::status() { areno::log::write STATUS 1 "$@"; }
areno::log::success() { areno::log::write SUCCESS 1 "$@"; }
areno::log::warn() { areno::log::write WARN 2 "$@"; }
areno::log::error() { areno::log::write ERROR 2 "$@"; }

# Stream external tool output without changing its wording or exit status.
# The caller uses pipefail so a failing installer command cannot be hidden by tee.
areno::log::output() {
  areno::log::plain | tee -a "${ARENO_INSTALL_LOG_PATH}"
}

# Report an actionable preflight failure at the current stage and stop.
# Args: error message. Does not install packages or create a log file.
areno::install::fail() {
  areno::log::error "AReno installation failed (${ARENO_INSTALL_STEP}): $*"
  if ${ARENO_INSTALL_LOG_STARTED}; then
    areno::log::error "Installation log: ${ARENO_INSTALL_LOG_PATH}"
  fi
  exit 1
}

# Handle unexpected command failures while preserving their exit status.
# Args: exit status, source line. Disable the trap before reporting to avoid recursion.
areno::install::on_error() {
  local status="$1"
  local line="$2"
  trap - ERR
  areno::log::error "AReno installation failed (${ARENO_INSTALL_STEP}), line ${line}, exit ${status}." || :
  if ${ARENO_INSTALL_LOG_STARTED}; then
    areno::log::error "Installation log: ${ARENO_INSTALL_LOG_PATH}" || :
  fi
  exit "${status}"
}

# Handle catchable interruptions through the same parent EXIT cleanup path.
# Bash waits for a foreground pipeline before running this trap, so cleanup does
# not remove an environment while the active installer command is still using it.
areno::install::on_signal() {
  local signal="$1"
  local status="$2"
  areno::log::error "Installation interrupted by ${signal} (${ARENO_INSTALL_STEP})." || :
  exit "${status}"
}

# Remove only a venv claimed by this process, never an existing environment.
# Args: original exit status. Files outside the owned directory are not removed.
areno::install::cleanup() {
  local status="$1"
  local owned="${ARENO_INSTALL_OWNED_VENV}"
  local marker
  local owner
  local log_directory

  if [[ -n "${owned}" ]]; then
    marker="${owned}/.areno-install-owner"
    # Refuse replaced directories, symlinks, and changed/missing ownership marks.
    if [[ "${owned}" != "${ARENO_ROOT}/.venv" || ! -d "${owned}" || -L "${owned}" ||
          ! -f "${marker}" || -L "${marker}" ]]; then
      areno::log::warn "Cleanup retained ${owned}: ownership could not be verified. Inspect it manually."
      return
    fi
    owner="$(cat "${marker}")" || return
    if [[ "${owner}" != "$$" ]]; then
      areno::log::warn "Cleanup retained ${owned}: the ownership marker changed."
      return
    fi
    if [[ "${status}" == "0" ]]; then
      # Successful installations keep their environment, but release the marker.
      rm -f -- "${marker}" || areno::log::warn "Could not remove ${marker}; remove it before installing again."
      return
    fi

    # Defend against custom paths/aliases placing diagnostics inside the target.
    log_directory="$(cd "$(dirname "${ARENO_INSTALL_LOG_PATH}")" && pwd -P)" || return
    case "${log_directory}/" in
      "${owned}/"*)
        areno::log::warn "Cleanup retained ${owned}: installation diagnostics are stored inside it."
        return
        ;;
    esac
    if rm -rf -- "${owned}"; then
      areno::log::warn "Cleanup removed virtual environment created by this installation: ${owned}"
    else
      areno::log::warn "Cleanup could not remove ${owned}; inspect and clean it manually."
    fi
  elif [[ "${status}" != "0" && "${ARENO_INSTALL_PACKAGES_TOUCHED}" == "true" ]]; then
    areno::log::warn "Kept existing Python environment: ${ARENO_INSTALL_PYTHON_BIN}. Package changes were not rolled back."
  fi
}

# Run once on parent exit, including explicit fail(), ERR and signal exits.
# Cleanup/logging failures must never replace the installation's original status.
areno::install::on_exit() {
  local status="$1"
  [[ "${BASH_SUBSHELL}" == "0" ]] || return "${status}"
  trap - EXIT ERR
  trap '' INT TERM HUP
  set +o errexit
  areno::install::cleanup "${status}"
  if [[ "${status}" != "0" && "${ARENO_INSTALL_LOG_STARTED}" == "true" ]]; then
    areno::log::warn "Installation log retained: ${ARENO_INSTALL_LOG_PATH}"
    if [[ -f "${ARENO_INSTALL_CONSTRAINTS}" ]]; then
      areno::log::warn "Runtime constraints retained: ${ARENO_INSTALL_CONSTRAINTS}"
    fi
  fi
  exit "${status}"
}

# Claim an absent .venv atomically before asking Python to populate it. An
# incomplete existing directory is user-owned and must not be overwritten.
areno::install::claim_venv() {
  local target="${ARENO_ROOT}/.venv"
  if ! mkdir -- "${target}"; then
    areno::install::fail "Cannot create ${target}; any existing directory or symlink is left untouched."
  fi
  ARENO_INSTALL_OWNED_VENV="${target}"
  if ! printf '%s\n' "$$" > "${target}/.areno-install-owner"; then
    # Without a marker, only an empty-directory removal can be justified.
    rmdir -- "${target}" || areno::log::warn "Could not mark or remove ${target}; inspect it manually."
    ARENO_INSTALL_OWNED_VENV=""
    areno::install::fail "Could not record ownership of the new environment."
  fi
}

# Diagnose preexisting partial environments and log paths under a planned venv
# before start_log can create any directories. Cleanup checks real paths again.
areno::install::check_new_venv() {
  local parent
  local suffix=""
  local log_path
  [[ "${ARENO_INSTALL_CREATE_VENV}" == "true" ]] || return 0
  if [[ -e "${ARENO_ROOT}/.venv" || -L "${ARENO_ROOT}/.venv" ]]; then
    areno::install::fail 'The checkout .venv already exists but is not usable. Repair it or select another Python with PYTHON.'
  fi
  parent="$(dirname "${ARENO_INSTALL_LOG_PATH}")"
  while [[ ! -d "${parent}" ]]; do
    suffix="/$(basename "${parent}")${suffix}"
    parent="$(dirname "${parent}")"
  done
  log_path="$(cd "${parent}" && pwd -P)${suffix}/$(basename "${ARENO_INSTALL_LOG_PATH}")"
  case "${log_path}" in
    "${ARENO_ROOT}/.venv"|"${ARENO_ROOT}/.venv/"*)
      areno::install::fail 'ARENO_INSTALL_LOG must be outside a newly created .venv so failure diagnostics can be retained.'
      ;;
  esac
}

# Update ARENO_INSTALL_STEP so subsequent errors identify the failing phase.
# Args: human-readable stage name.
areno::install::stage() {
  ARENO_INSTALL_STEP="$1"
  areno::log::status "${ARENO_INSTALL_STEP}"
}

# Parse CLI switches into ARENO_INSTALL_* option flags; reject ambiguous input.
# Explicit notebooks require a kernel-bound interpreter. Colab keeps its current one.
areno::install::parse_args() {
  local option
  while (( $# )); do
    case "$1" in
      --dry-run) ARENO_INSTALL_DRY_RUN="true" ;;
      --notebook) ARENO_INSTALL_NOTEBOOK="true" ;;
      --flash) ARENO_INSTALL_FLASH="true" ;;
      --build-only) ARENO_INSTALL_BUILD_ONLY="true" ;;
      -h|--help) areno::install::usage; exit 0 ;;
      *) areno::install::fail "Unknown option: $1" ;;
    esac
    shift
  done
  # These values are used as boolean commands below, so accept only true/false.
  for option in ARENO_INSTALL_DRY_RUN ARENO_INSTALL_NOTEBOOK ARENO_INSTALL_FLASH ARENO_INSTALL_BUILD_ONLY; do
    case "${!option}" in
      true|false) ;;
      *) areno::install::fail "${option} must be true or false." ;;
    esac
  done
  if ${ARENO_INSTALL_NOTEBOOK} && [[ -z "${PYTHON:-}" ]]; then
    areno::install::fail '--notebook requires PYTHON set to the notebook kernel interpreter.'
  fi
  if compgen -e | grep '^COLAB_' >/dev/null; then
    ARENO_INSTALL_NOTEBOOK="true"
  fi
}

# Print a shell-escaped command, then execute its original argv unless dry-running.
# Args: command and arguments. Requires start_log before any mutating execution.
areno::install::run() {
  local command_text
  printf -v command_text '%q ' "$@"
  areno::log::info "+ ${command_text}"
  if ! ${ARENO_INSTALL_DRY_RUN}; then
    "$@" 2>&1 | areno::log::output
  fi
}

# Bind pip to the selected interpreter and constrain every install operation.
# Args: pip subcommand and arguments. Leave caller PIP_CONSTRAINT unchanged.
areno::install::pip() {
  if [[ "$1" == install ]]; then
    if [[ "${ARENO_INSTALL_DRY_RUN}" != "true" ]]; then
      # Even a failed pip operation may have changed the selected environment.
      ARENO_INSTALL_PACKAGES_TOUCHED="true"
    fi
    areno::install::run "${ARENO_INSTALL_PYTHON_BIN}" -m pip "$@" --constraint "${ARENO_INSTALL_CONSTRAINTS}"
  else
    areno::install::run "${ARENO_INSTALL_PYTHON_BIN}" -m pip "$@"
  fi
}

# Fail early when a host prerequisite is missing. Args: executable name.
areno::install::require_command() {
  command -v "$1" >/dev/null || areno::install::fail "Missing $1. Install the required system development tools."
}

# pip emits a stable name==version format. Normalize distribution names once.
areno::install::read_packages() {
  "$1" -m pip list --format=freeze --disable-pip-version-check |
    awk -F '==' 'NF == 2 {name=tolower($1); gsub(/[-_.]+/, "-", name); print name "==" $2}'
}

# Print one installed version from ARENO_INSTALL_PACKAGES, or empty if absent.
# Args: normalized distribution name. No Python imports or package mutations.
areno::install::package_version() {
  printf '%s\n' "${ARENO_INSTALL_PACKAGES}" | awk -F '==' -v name="$1" '$1 == name {print $2}'
}

# Compare major/minor components; suffixes in the patch version do not matter.
# Args: version string, minimum major, minimum minor. Returns 0 when compatible.
areno::install::version_at_least() {
  local major
  local minor
  local ignored
  IFS="." read -r major minor ignored <<< "$1"
  [[ "${major}" =~ ^[0-9]+$ && "${minor:-}" =~ ^[0-9]+$ ]] || return 1
  (( major > $2 || (major == $2 && minor >= $3) ))
}

# Probe implicit candidates without rejecting a usable checkout environment.
# An unsupported interpreter cannot take precedence merely because it has torch.
areno::install::probe_packages() {
  local python="$1"
  local version
  [[ -x "${python}" ]] || return 1
  version="$("${python}" --version 2>/dev/null)" || return 1
  areno::install::version_at_least "${version#Python }" 3 10 || return 1
  areno::install::read_packages "${python}"
}

# Validate the chosen interpreter and refuse environments owned by another run.
# Bootstrap Python needs no global pip: the new venv supplies and validates it.
# Args: optional "bootstrap" mode. Refreshes ARENO_INSTALL_PYTHON_VERSION.
areno::install::inspect_python() {
  local mode="${1:-existing}"
  local version_output
  local marker
  [[ -x "${ARENO_INSTALL_PYTHON_BIN}" ]] || areno::install::fail 'Python not found. Set PYTHON to a Python 3.10+ executable.'
  marker="$(dirname "${ARENO_INSTALL_PYTHON_BIN}")/../.areno-install-owner"
  if [[ -e "${marker}" || -L "${marker}" ]]; then
    areno::install::fail 'This environment has an installation ownership marker. Wait for the installer, or inspect an interrupted installation before removing the marker.'
  fi
  version_output="$("${ARENO_INSTALL_PYTHON_BIN}" --version 2>&1)"
  ARENO_INSTALL_PYTHON_VERSION="${version_output#Python }"
  areno::install::version_at_least "${ARENO_INSTALL_PYTHON_VERSION}" 3 10 || areno::install::fail 'Python 3.10+ is required.'
  if [[ "${mode}" != bootstrap ]]; then
    areno::install::inspect_pip
  fi
}

# Validate pip and interpreter architecture before the first package mutation.
# Also used immediately after creating our venv, which still has its owner marker.
# Package metadata does not prove device compatibility; build/CLI checks do that.
areno::install::inspect_pip() {
  local tags
  "${ARENO_INSTALL_PYTHON_BIN}" -m pip --version >/dev/null || areno::install::fail 'Selected Python needs pip. Prepare pip in that environment first.'
  tags="$("${ARENO_INSTALL_PYTHON_BIN}" -m pip debug --verbose 2>/dev/null)"
  case "${ARENO_INSTALL_HOST_SYSTEM}:${ARENO_INSTALL_HOST_ARCH}" in
    Darwin:arm64)
      [[ "${tags}" == *macosx_*_arm64* ]] || areno::install::fail 'Use native arm64 Python, not Rosetta.' ;;
    Linux:x86_64)
      [[ "${tags}" == *linux*x86_64* ]] || areno::install::fail 'Use a Linux x86_64 Python interpreter.' ;;
    Linux:aarch64)
      [[ "${tags}" == *linux*aarch64* ]] || areno::install::fail 'Use a Linux aarch64 Python interpreter.' ;;
  esac
  ARENO_INSTALL_PACKAGES="$(areno::install::read_packages "${ARENO_INSTALL_PYTHON_BIN}")"
}

# Choose the host backend and interpreter without creating files or environments.
# Updates ARENO_INSTALL_{HOST_*,PYTHON_*,PACKAGES,BACKEND,CREATE_VENV,BOOTSTRAP_PYTHON}.
# Explicit selections and prepared CANN environments take precedence over .venv.
areno::install::select_environment() {
  local release
  local current_python
  local current_torch
  local candidate_torch
  local explicit="false"
  ARENO_INSTALL_HOST_SYSTEM="$(uname -s)"
  ARENO_INSTALL_HOST_ARCH="$(uname -m)"
  case "${ARENO_INSTALL_HOST_SYSTEM}:${ARENO_INSTALL_HOST_ARCH}" in
    Linux:x86_64|Linux:aarch64|Darwin:arm64) ;;
    *) areno::install::fail "Unsupported platform: ${ARENO_INSTALL_HOST_SYSTEM} ${ARENO_INSTALL_HOST_ARCH}. Use Linux CUDA/Ascend or native Apple Silicon; Windows needs WSL2." ;;
  esac
  if [[ "${ARENO_INSTALL_HOST_SYSTEM}" == Linux ]]; then
    release="$(uname -r | tr '[:upper:]' '[:lower:]')"
    if [[ "${release}" == *microsoft* && "${release}" != *wsl2* ]]; then
      areno::install::fail 'WSL1 is unsupported. Upgrade to WSL2.'
    fi
  else
    areno::install::version_at_least "$(sw_vers -productVersion)" 14 0 || areno::install::fail 'MLX requires macOS 14+.'
  fi

  if [[ -n "${PYTHON:-}" ]]; then
    ARENO_INSTALL_PYTHON_BIN="$(command -v "${PYTHON}")"
    explicit="true"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    ARENO_INSTALL_PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    explicit="true"
  elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    ARENO_INSTALL_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    explicit="true"
  else
    ARENO_INSTALL_PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
  # A prepared runtime owns its interpreter, including vendor environments and
  # venvs invoked without activation. Do not swap it for a checkout-local venv.
  if ! ${explicit} && ! ${ARENO_INSTALL_NOTEBOOK} &&
     [[ -z "${ARENO_INSTALL_PYTHON_BIN}" || ! -f "$(dirname "${ARENO_INSTALL_PYTHON_BIN}")/../pyvenv.cfg" ]]; then
    current_python="${ARENO_INSTALL_PYTHON_BIN}"
    ARENO_INSTALL_PACKAGES="$(areno::install::probe_packages "${current_python}" 2>/dev/null)" || ARENO_INSTALL_PACKAGES=""
    if [[ "${ARENO_INSTALL_HOST_SYSTEM}" == Linux && -n "$(areno::install::package_version torch-npu)" ]]; then
      : # CANN owns its prepared interpreter, even if a checkout .venv exists.
    elif [[ -x "${ARENO_ROOT}/.venv/bin/python" ]]; then
      current_torch="$(areno::install::package_version torch)"
      ARENO_INSTALL_PYTHON_BIN="${ARENO_ROOT}/.venv/bin/python"
      ARENO_INSTALL_PACKAGES="$(areno::install::probe_packages "${ARENO_INSTALL_PYTHON_BIN}" 2>/dev/null)" || ARENO_INSTALL_PACKAGES=""
      candidate_torch="$(areno::install::package_version torch)"
      # Prefer the checkout environment, unless it would discard an existing
      # CUDA runtime for an empty or known CPU-only environment.
      if [[ "${ARENO_INSTALL_HOST_SYSTEM}" == Linux && -z "$(areno::install::package_version torch-npu)" &&
            -n "${current_torch}" && "${current_torch}" != *+cpu* &&
            ( -z "${candidate_torch}" || "${candidate_torch}" == *+cpu* ) ]] &&
          areno::install::version_at_least "${current_torch}" 2 6; then
        ARENO_INSTALL_PYTHON_BIN="${current_python}"
      fi
    elif [[ "${ARENO_INSTALL_HOST_SYSTEM}" == Linux && -n "$(areno::install::package_version torch)" ]]; then
      : # Reuse prepared CUDA environments even when no venv is activated.
    else
      ARENO_INSTALL_CREATE_VENV="true"
    fi
  fi
  # Strict checks apply only to the selected environment. A new venv's pip and
  # architecture are checked after creation, before installing any packages.
  if ${ARENO_INSTALL_CREATE_VENV}; then
    areno::install::inspect_python bootstrap
    ARENO_INSTALL_BOOTSTRAP_PYTHON="${ARENO_INSTALL_PYTHON_BIN}"
    ARENO_INSTALL_PYTHON_BIN="${ARENO_ROOT}/.venv/bin/python"
    ARENO_INSTALL_PACKAGES=""
    if ${ARENO_INSTALL_DRY_RUN}; then
      areno::log::info 'New virtualenv pip and architecture checks will run after creation.'
    fi
  else
    areno::install::inspect_python
  fi
  # Finding torch on PATH is not consent to replace globally installed packages.
  # Explicit Python/active environments and checkout venvs retain their existing
  # policy; automatic notebook/vendor interpreters get the same global guard.
  if ! ${explicit} && ! ${ARENO_INSTALL_CREATE_VENV} &&
     [[ "${ARENO_INSTALL_PYTHON_BIN}" != "${ARENO_ROOT}/.venv/bin/python" &&
        ! -f "$(dirname "${ARENO_INSTALL_PYTHON_BIN}")/../pyvenv.cfg" ]]; then
    ARENO_INSTALL_PROTECT_GLOBAL="true"
  fi
  if [[ "${ARENO_INSTALL_HOST_SYSTEM}" == Darwin ]]; then
    ARENO_INSTALL_BACKEND="mlx"
  elif [[ -n "$(areno::install::package_version torch-npu)" ]]; then
    ARENO_INSTALL_BACKEND="npu"
  else
    ARENO_INSTALL_BACKEND="cuda"
    if [[ -n "${ASCEND_HOME_PATH:-}${ASCEND_CANN_PACKAGE_PATH:-}" ]] || command -v npu-smi >/dev/null; then
      areno::install::fail 'Ascend environment found without torch_npu. Select its matching Python with PYTHON.'
    fi
  fi
}

# Reject a broken implicit global environment before logs or package mutations.
# pip check covers every installed distribution, including unrelated notebook tools.
areno::install::check_global_environment() {
  local check_output
  ${ARENO_INSTALL_PROTECT_GLOBAL} || return 0
  areno::log::info "Checking automatically selected global Python: ${ARENO_INSTALL_PYTHON_BIN}"
  if ! check_output="$("${ARENO_INSTALL_PYTHON_BIN}" -m pip check 2>&1)"; then
    areno::log::error "${check_output}"
    areno::install::fail 'The automatically selected global environment has broken dependencies; no packages were changed. Activate a dedicated venv/Conda environment, or explicitly set PYTHON to manage this environment.'
  fi
}

# Reject incompatible options and missing native build prerequisites before pip.
# Reads backend/options; CUDA preflight also records the detected toolkit version.
areno::install::check_host() {
  if ${ARENO_INSTALL_FLASH} && [[ "${ARENO_INSTALL_BACKEND}" != cuda ]]; then
    areno::install::fail '--flash is only supported by CUDA.'
  fi
  if ${ARENO_INSTALL_BUILD_ONLY} && [[ "${ARENO_INSTALL_BACKEND}" != cuda ]]; then
    areno::install::fail '--build-only is only supported by CUDA.'
  fi
  [[ "${ARENO_INSTALL_BACKEND}" != mlx ]] || return 0
  case "${ARENO_BUILD_EXT:-auto}" in
    0|false|no|off) areno::install::fail 'Unset ARENO_BUILD_EXT to build the native extensions.' ;;
  esac
  areno::install::require_command c++
  case "${ARENO_INSTALL_BACKEND}" in
    cuda) areno::install::check_cuda ;;
    npu) areno::install::check_npu ;;
  esac
}

# Check NVIDIA visibility and toolkit/compiler prerequisites; preserve existing torch.
# Updates ARENO_INSTALL_CUDA_VERSION and exports CUDA_HOME for the package builder.
# Build-only deliberately skips device visibility, but requires explicit target archs.
areno::install::check_cuda() {
  local nvcc=""
  local root=""
  local output
  local torch_version
  local target
  if ! ${ARENO_INSTALL_BUILD_ONLY}; then
    areno::install::require_command nvidia-smi
    output="$(nvidia-smi -L)"
    [[ "${output}" == *GPU* && "${output}" != *'No devices'* ]] || areno::install::fail 'No NVIDIA GPU is visible.'
    areno::log::info "${output}"
  fi
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    nvcc="${CUDA_HOME}/bin/nvcc"
    root="${CUDA_HOME}"
  elif [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
    nvcc="${CUDA_PATH}/bin/nvcc"
    root="${CUDA_PATH}"
  else
    nvcc="$(command -v nvcc || true)"
  fi
  [[ -n "${nvcc}" ]] || areno::install::fail 'nvcc not found. Install the CUDA toolkit and set CUDA_HOME or PATH.'
  output="$("${nvcc}" --version)"
  [[ "${output}" =~ release[[:space:]]([0-9]+\.[0-9]+) ]] || areno::install::fail 'Cannot determine the CUDA toolkit version.'
  ARENO_INSTALL_CUDA_VERSION="${BASH_REMATCH[1]}"
  # Preserve a valid explicit root; otherwise derive it from the selected PATH
  # compiler. Stale environment variables must not override a successful fallback.
  if [[ -z "${root}" ]]; then
    while [[ -L "${nvcc}" ]]; do
      target="$(readlink "${nvcc}")"
      case "${target}" in
        /*) nvcc="${target}" ;;
        *) nvcc="$(dirname "${nvcc}")/${target}" ;;
      esac
    done
    root="$(cd "$(dirname "${nvcc}")/.." && pwd -P)"
  fi
  CUDA_HOME="${root}"
  export CUDA_HOME
  torch_version="$(areno::install::package_version torch)"
  if [[ -n "${torch_version}" ]]; then
    areno::install::version_at_least "${torch_version}" 2 6 || areno::install::fail 'PyTorch >= 2.6 is required. Existing runtimes are not replaced.'
    [[ "${torch_version}" != *+cpu* ]] || areno::install::fail 'CPU-only PyTorch found. Select a CUDA environment with PYTHON.'
  fi
  if ${ARENO_INSTALL_BUILD_ONLY} && [[ -z "${torch_version}" || -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    areno::install::fail '--build-only requires existing CUDA PyTorch and TORCH_CUDA_ARCH_LIST.'
  fi
}

# Validate the sourced CANN development layout and management tools.
# Reuse CANN-matched torch/torch_npu instead of guessing a replacement runtime.
areno::install::check_npu() {
  local root="${ASCEND_HOME_PATH:-${ASCEND_CANN_PACKAGE_PATH:-}}"
  local layout
  local found="false"
  local tool
  [[ -n "$(areno::install::package_version torch)" ]] || areno::install::fail 'NPU requires CANN-matched PyTorch and torch_npu.'
  [[ -d "${root}" ]] || areno::install::fail 'Source CANN set_env.sh (ASCEND_HOME_PATH or ASCEND_CANN_PACKAGE_PATH must exist).'
  for layout in compiler/tikcpp/ascendc_kernel_cmake tools/tikcpp/ascendc_kernel_cmake ascendc_devkit/tikcpp/samples/cmake; do
    if [[ -f "${root}/${layout}/ascendc.cmake" ]]; then
      found="true"
      break
    fi
  done
  ${found} || areno::install::fail 'CANN development toolkit is missing ascendc.cmake.'
  for tool in cmake nm git npu-smi; do
    areno::install::require_command "${tool}"
  done
  # Keep this preflight read-only; runtime availability is checked again by AReno.
  npu-smi info
}

# Allocate this run's constraints independently, even when log directories match.
# Updates ARENO_INSTALL_CONSTRAINTS/LOG_STARTED. Dry runs never create either file.
areno::install::start_log() {
  ARENO_INSTALL_CONSTRAINTS="$(dirname "${ARENO_INSTALL_LOG_PATH}")/install-runtime-constraints.XXXXXX"
  if ! ${ARENO_INSTALL_DRY_RUN}; then
    mkdir -p "$(dirname "${ARENO_INSTALL_LOG_PATH}")"
    : > "${ARENO_INSTALL_LOG_PATH}"
    ARENO_INSTALL_LOG_STARTED="true"
    ARENO_INSTALL_CONSTRAINTS="$(mktemp "${ARENO_INSTALL_CONSTRAINTS}")"
    areno::log::info "Backend: ${ARENO_INSTALL_BACKEND}" "Python: ${ARENO_INSTALL_PYTHON_BIN}"
  fi
}

# Save accelerator versions, or every installed version for implicit global reuse.
# A clean baseline plus exact global pins prevents breaking unrelated dependents
# without parsing pip's JSON reports or implementing a dependency solver in Shell.
# Reads ARENO_INSTALL_PACKAGES; writes CONSTRAINTS only outside dry-run mode.
areno::install::save_pins() {
  local pins
  if ${ARENO_INSTALL_PROTECT_GLOBAL}; then
    pins="${ARENO_INSTALL_PACKAGES}"
    areno::log::info 'Protect all installed versions in the automatically selected global environment.'
  else
    pins="$(printf '%s\n' "${ARENO_INSTALL_PACKAGES}" | awk -F '==' '
    $1 ~ /^(torch|torchvision|torchaudio|torch-npu|triton|triton-ascend|mlx|mlx-metal|mlx-cpu|mlx-lm|mlx-vlm)$/ || $1 ~ /^nvidia-/ {print}')"
    areno::log::info 'Protect installed runtime versions:' "${pins:-none installed}"
  fi
  if ! ${ARENO_INSTALL_DRY_RUN}; then
    printf '%s\n' "${pins}" > "${ARENO_INSTALL_CONSTRAINTS}"
  fi
}

# Resolve all planned dependencies before the first install into a global Python.
# Use pip directly: its --dry-run must not mark the environment as modified.
# Installer --dry-run only prints this command; actual resolution may download
# metadata/build dependencies into temporary directories, but installs no target packages.
areno::install::check_global_plan() {
  local requirements="default.txt"
  local packages=('setuptools>=69' wheel)
  ${ARENO_INSTALL_PROTECT_GLOBAL} || return 0
  [[ "${ARENO_INSTALL_BACKEND}" != npu ]] || requirements="npu.txt"
  if [[ ${#ARENO_INSTALL_PROFILE_PACKAGES[@]} -gt 0 ]]; then
    packages+=("${ARENO_INSTALL_PROFILE_PACKAGES[@]}")
  fi
  if [[ -n "${ARENO_INSTALL_PROFILE_INDEX}" ]]; then
    packages+=(--extra-index-url "${ARENO_INSTALL_PROFILE_INDEX}")
  fi
  if [[ "${ARENO_INSTALL_BACKEND}" != mlx ]]; then
    packages+=(psutil ninja)
  fi
  if ${ARENO_INSTALL_FLASH}; then
    packages+=('flash-attn>=2.7' --no-build-isolation)
  fi
  areno::log::info 'Resolve dependencies without replacing any installed global package.'
  if ! areno::install::run "${ARENO_INSTALL_PYTHON_BIN}" -m pip install --dry-run \
      "${packages[@]}" -r "${ARENO_ROOT}/requirements/${requirements}" \
      --constraint "${ARENO_INSTALL_CONSTRAINTS}"; then
    areno::install::fail 'Cannot resolve the installation with all global versions protected; no packages were changed. Review the pip error above. Use a dedicated venv/Conda environment, or explicitly set PYTHON if existing versions may be changed.'
  fi
}

# Create a planned venv, protect existing runtimes, then install build tooling.
# Called only after host preflight; all mutations go through the dry-run wrapper.
areno::install::prepare_environment() {
  if ${ARENO_INSTALL_CREATE_VENV}; then
    if [[ "${ARENO_INSTALL_DRY_RUN}" != "true" ]]; then
      areno::install::claim_venv
    fi
    areno::install::run "${ARENO_INSTALL_BOOTSTRAP_PYTHON}" -m venv "${ARENO_ROOT}/.venv"
    if ! ${ARENO_INSTALL_DRY_RUN}; then
      areno::install::inspect_pip
    fi
  fi
  areno::install::save_pins
  areno::install::check_global_plan
  areno::install::pip install 'setuptools>=69' wheel
}

# Install only missing runtime packages from the selected bootstrap profile.
# Refresh constraints immediately so later dependency resolution cannot replace them.
areno::install::install_runtime() {
  if [[ ${#ARENO_INSTALL_PROFILE_PACKAGES[@]} -gt 0 ]]; then
    if [[ -n "${ARENO_INSTALL_PROFILE_INDEX}" ]]; then
      areno::install::pip install "${ARENO_INSTALL_PROFILE_PACKAGES[@]}" --index-url "${ARENO_INSTALL_PROFILE_INDEX}"
    else
      areno::install::pip install "${ARENO_INSTALL_PROFILE_PACKAGES[@]}"
    fi
  fi
  if ! ${ARENO_INSTALL_DRY_RUN}; then
    ARENO_INSTALL_PACKAGES="$(areno::install::read_packages "${ARENO_INSTALL_PYTHON_BIN}")"
    areno::install::save_pins
  fi
  if [[ "${ARENO_INSTALL_BACKEND}" != mlx ]]; then
    areno::install::pip install psutil ninja
  fi
}

# Select backend requirements and install optional CUDA FlashAttention.
# Refresh runtime pins for packages newly introduced by dependency resolution.
areno::install::install_dependencies() {
  local requirements="default.txt"
  if [[ "${ARENO_INSTALL_BACKEND}" == npu ]]; then
    requirements="npu.txt"
  fi
  areno::install::pip install -r "${ARENO_ROOT}/requirements/${requirements}"
  if ${ARENO_INSTALL_FLASH}; then
    areno::install::pip install 'flash-attn>=2.7' --no-build-isolation
  fi
  if ! ${ARENO_INSTALL_DRY_RUN}; then
    ARENO_INSTALL_PACKAGES="$(areno::install::read_packages "${ARENO_INSTALL_PYTHON_BIN}")"
    areno::install::save_pins
  fi
}

# Compile against the prepared environment with dependency resolution disabled.
# MLX has no native AReno extension; CUDA/NPU must build their local extensions.
areno::install::build() {
  if [[ "${ARENO_INSTALL_BACKEND}" == mlx ]]; then
    ARENO_BUILD_EXT="0"
  else
    ARENO_BUILD_EXT="1"
  fi
  export ARENO_BUILD_EXT
  areno::install::pip install -e "${ARENO_ROOT}" --no-build-isolation --no-deps
}

# Check package consistency and the existing AReno CLI in the selected interpreter.
# Build-only skips device checks. Neither mode claims model/tensor execution coverage.
areno::install::verify() {
  local check_output
  areno::install::pip check
  if ! ${ARENO_INSTALL_BUILD_ONLY}; then
    if ${ARENO_INSTALL_DRY_RUN}; then
      areno::install::run "${ARENO_INSTALL_PYTHON_BIN}" -m areno.cli.main check
    else
      # Use the existing public CLI. No installer-specific Python probes.
      if check_output="$("${ARENO_INSTALL_PYTHON_BIN}" -m areno.cli.main check 2>&1)"; then
        areno::log::info "${check_output}"
      else
        areno::log::error "${check_output}"
        areno::install::fail 'areno check failed in the selected environment.'
      fi
      if ${ARENO_INSTALL_FLASH} && ! printf '%s\n' "${check_output}" | grep -q '^OK *flash_attn import'; then
        areno::install::fail 'Explicitly requested FlashAttention did not pass its import check.'
      fi
    fi
  fi
  if ${ARENO_INSTALL_DRY_RUN}; then
    areno::log::success 'Installation plan complete. No files or packages were changed.'
  elif ${ARENO_INSTALL_BUILD_ONLY}; then
    areno::log::warn 'Build complete; device readiness was NOT verified. Run bash scripts/install.sh on the target GPU host.'
  else
    areno::log::success 'Installation and CLI checks passed.'
    areno::log::warn 'Model execution and device computation were not tested.'
    if ${ARENO_INSTALL_CREATE_VENV}; then
      areno::log::info "Activate: source \"${ARENO_ROOT}/.venv/bin/activate\""
    fi
    if ${ARENO_INSTALL_NOTEBOOK}; then
      areno::log::warn 'Restart the notebook kernel before importing AReno.'
    fi
  fi
  if ! ${ARENO_INSTALL_DRY_RUN}; then
    areno::log::info "Installation log: ${ARENO_INSTALL_LOG_PATH}" "Runtime constraints: ${ARENO_INSTALL_CONSTRAINTS}"
  fi
}
