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

# Offline integration tests: every package/device command is a shell stub.
set -o errexit
set -o nounset
set -o pipefail

TEST_ROOT=${1:?Usage: bash tests/test_install_shell.sh /path/to/AReno}
TEST_SUITE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/areno-shell-tests.XXXXXX")
TEST_SUITE_DIR=$(cd "${TEST_SUITE_DIR}" && pwd -P)
readonly TEST_ROOT TEST_SUITE_DIR
trap 'rm -rf "${TEST_SUITE_DIR}"' EXIT
TEST_PASSED=0

fail() {
  printf 'FAIL %s: %s\n' "${case_name}" "$*" >&2
  [[ ! -f ${case_dir}/output ]] || cat "${case_dir}/output" >&2
  exit 1
}

assert_contains() {
  grep -Fq -- "$2" "$1" || fail "missing '$2' in $1"
}

assert_absent() {
  if grep -Fq -- "$2" "$1"; then
    fail "unexpected '$2' in $1"
  fi
}

# Each invocation keeps its own retained constraints file beside its log.
constraints_file() {
  local files=("${case_dir}/state with spaces"/install-runtime-constraints.*)
  [[ ${#files[@]} == 1 && -f "${files[0]}" ]] || fail 'expected one retained constraints file'
  printf '%s\n' "${files[0]}"
}

# Prepare a disposable checkout and CLI stubs; never install packages or use devices.
new_case() {
  local command
  case_name=$1
  case_dir="${TEST_SUITE_DIR}/${case_name}"
  checkout="${case_dir}/checkout with spaces"
  mkdir -p "${checkout}" "${case_dir}/bin" "${case_dir}/home"
  cp -R "${TEST_ROOT}/scripts" "${TEST_ROOT}/requirements" "${checkout}/"
  cp "${TEST_ROOT}/pyproject.toml" "${TEST_ROOT}/setup.py" "${checkout}/"
  : > "${case_dir}/calls"
  : > "${case_dir}/mutations"
  printf '%s\n' 'torch==2.9.1+cu130' 'torchvision==0.24.1+cu130' 'triton==3.5.1' > "${case_dir}/packages"
  cat > "${case_dir}/bin/stub" <<'STUB'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
command_name=${0##*/}
printf '%s' "${command_name}" >> "${STUB_DIR}/calls"
printf ' <%s>' "$@" >> "${STUB_DIR}/calls"
printf '\n' >> "${STUB_DIR}/calls"
case "${command_name}" in
  uname)
    case "${1:--s}" in
      -s) printf '%s\n' "${STUB_SYSTEM}" ;;
      -m) printf '%s\n' "${STUB_MACHINE}" ;;
      -r) printf '%s\n' "${STUB_RELEASE}" ;;
      *) printf '%s %s %s\n' "${STUB_SYSTEM}" "${STUB_RELEASE}" "${STUB_MACHINE}" ;;
    esac
    ;;
  sw_vers) printf '%s\n' "${STUB_MACOS}" ;;
  sysctl) printf '%s\n' '0' ;;
  nvidia-smi)
    [[ ${STUB_NO_GPU:-0} != 1 ]] || exit 1
    case "$*" in
      *compute_cap*) printf '%s\n' '9.0' ;;
      *) printf '%s\n' 'GPU 0: NVIDIA H100 (UUID: GPU-test)' ;;
    esac
    ;;
  nvcc) printf 'Cuda compilation tools, release %s, V%s.0\n' "${STUB_CUDA}" "${STUB_CUDA}" ;;
  npu-smi)
    [[ ${STUB_NO_GPU:-0} != 1 ]] || exit 1
    printf '%s\n' '| NPU 0 | Ascend 910B | OK |'
    ;;
  rm)
    if [[ ${STUB_FAIL_REMOVE:-0} == 1 ]]; then
      for arg in "$@"; do
        [[ "${arg}" != "${STUB_DIR}/checkout with spaces/.venv" ]] || exit 29
      done
    fi
    /bin/rm "$@"
    ;;
  cmake|nm|git|c++|g++|ninja) : ;;
  python|python3|'python with spaces')
    python_version="${STUB_PYTHON_VERSION}"
    python_machine="${STUB_MACHINE}"
    if [[ "$0" != */.venv/bin/python && -n "${STUB_SYSTEM_PYTHON_VERSION:-}" ]]; then
      python_version="${STUB_SYSTEM_PYTHON_VERSION}"
    fi
    if [[ "$0" == */.venv/bin/python && -n "${STUB_VENV_MACHINE:-}" ]]; then
      python_machine="${STUB_VENV_MACHINE}"
    fi
    if [[ ${1:-} == --version ]]; then
      printf 'Python %s\n' "${python_version}"
      exit
    fi
    [[ ${1:-} == -m ]] || { printf 'Unexpected Python source execution: %s\n' "$*" >&2; exit 88; }
    module=${2:-}
    shift 2
    case "${module}" in
      pip)
        if [[ "$0" != */.venv/bin/python && ${STUB_NO_SYSTEM_PIP:-0} == 1 ]] ||
           [[ "$0" == */.venv/bin/python && ${STUB_NO_VENV_PIP:-0} == 1 ]]; then
          printf '%s\n' 'No module named pip' >&2
          exit 1
        fi
        case "${1:-}" in
          --version) printf 'pip 25.0 from %s/site-packages/pip (python 3.12)\n' "${STUB_DIR}" ;;
          list|freeze)
            if [[ "$0" == */.venv/bin/python && -f ${STUB_DIR}/venv-packages ]]; then
              cat "${STUB_DIR}/venv-packages"
            else
              cat "${STUB_DIR}/packages"
            fi
            ;;
          debug)
            printf 'pip version: pip 25.0\nsys.version: %s\nsys.executable: %s\nsys.platform: %s\n' "${python_version}" "$0" "${STUB_SYSTEM}"
            if [[ ${STUB_SYSTEM} == Darwin ]]; then
              printf 'Compatible tags: 1\n  cp312-cp312-macosx_14_0_%s\n' "${python_machine}"
            else
              printf 'Compatible tags: 1\n  cp312-cp312-linux_%s\n' "${python_machine}"
            fi
            ;;
          show)
            package=${2:-torch}
            version=$(awk -F '==' -v name="${package}" '$1 == name {print $2}' "${STUB_DIR}/packages")
            [[ -n ${version} ]] || exit 1
            printf 'Name: %s\nVersion: %s\nLocation: %s/site-packages\n' "${package}" "${version}" "${STUB_DIR}"
            ;;
          install)
            for arg in "$@"; do
              if [[ "${arg}" == --dry-run ]]; then
                [[ ${STUB_FAIL_PLAN:-0} != 1 ]] || { printf '%s\n' 'ResolutionImpossible: installed version is constrained'; exit 18; }
                printf '%s\n' 'Dependency resolution passed without installation.'
                exit 0
              fi
            done
            printf '%s\n' "${CUDA_HOME:-}" > "${STUB_DIR}/build-cuda-home"
            printf 'install' >> "${STUB_DIR}/mutations"
            printf ' <%s>' "$@" >> "${STUB_DIR}/mutations"
            printf '\n' >> "${STUB_DIR}/mutations"
            case "${STUB_TAMPER_OWNER:-}" in
              changed)
                printf '%s\n' 'different owner' > "${STUB_DIR}/checkout with spaces/.venv/.areno-install-owner"
                ;;
              symlink)
                /bin/rm "${STUB_DIR}/checkout with spaces/.venv/.areno-install-owner"
                printf '%s\n' "$PPID" > "${STUB_DIR}/external owner file"
                ln -s "${STUB_DIR}/external owner file" "${STUB_DIR}/checkout with spaces/.venv/.areno-install-owner"
                ;;
            esac
            if [[ ${STUB_REPLACE_VENV:-0} == 1 ]]; then
              mv "${STUB_DIR}/checkout with spaces/.venv" "${STUB_DIR}/original owned environment"
              mkdir -p "${STUB_DIR}/replacement target"
              printf '%s\n' 'replacement user data' > "${STUB_DIR}/replacement target/keep-me"
              ln -s "${STUB_DIR}/replacement target" "${STUB_DIR}/checkout with spaces/.venv"
            fi
            if [[ ${STUB_SIGNAL_INSTALL:-} != "" ]]; then
              kill -"${STUB_SIGNAL_INSTALL}" "$PPID"
              exit 0
            fi
            [[ ${STUB_FAIL_INSTALL:-0} != 1 ]] || exit 17
            for arg in "$@"; do
              case "${arg}" in
                torch==*|torchvision==*|mlx==*|mlx-lm==*|mlx-vlm==*) printf '%s\n' "${arg}" >> "${STUB_DIR}/packages" ;;
              esac
            done
            ;;
          check)
            # Preserve the post-install verification trace; preflight is read-only.
            if [[ -s "${STUB_DIR}/mutations" ]]; then
              printf '%s\n' 'pip check' >> "${STUB_DIR}/mutations"
            fi
            [[ ${STUB_FAIL_CHECK:-0} != 1 ]] || exit 19
            printf '%s\n' 'No broken requirements found.'
            ;;
          *) printf 'Unexpected pip command: %s\n' "$*" >&2; exit 89 ;;
        esac
        ;;
      venv)
        printf '%s\n' 'venv' >> "${STUB_DIR}/mutations"
        target=${!#}
        mkdir -p "${target}/bin"
        printf '%s\n' 'partially created virtualenv' > "${target}/partial-venv"
        [[ ${STUB_FAIL_VENV:-0} != 1 ]] || exit 23
        cp "$0" "${target}/bin/python"
        printf '%s\n' 'home = /stub' > "${target}/pyvenv.cfg"
        ;;
      ensurepip) printf '%s\n' 'ensurepip' >> "${STUB_DIR}/mutations" ;;
      areno.cli.main)
        [[ ${1:-} == check ]] || exit 90
        printf '%s\n' 'areno check' >> "${STUB_DIR}/mutations"
        [[ ${STUB_FAIL_ARENO:-0} != 1 ]] || exit 21
        printf '%s\n' 'AReno check: ready'
        printf '%s   flash_attn import\n' "${STUB_FLASH_IMPORT:-OK}"
        ;;
      *) printf 'Unexpected Python module: %s\n' "${module}" >&2; exit 91 ;;
    esac
    ;;
  *) printf 'Unexpected command %s\n' "${command_name}" >&2; exit 92 ;;
esac
STUB
  chmod +x "${case_dir}/bin/stub"
  for command in uname sw_vers sysctl nvidia-smi nvcc cmake nm git c++ g++ ninja python python3 'python with spaces'; do
    cp "${case_dir}/bin/stub" "${case_dir}/bin/${command}"
  done
  export STUB_DIR="${case_dir}"
  export STUB_SYSTEM=Linux STUB_MACHINE=x86_64 STUB_RELEASE=6.0 STUB_MACOS=15.0 STUB_CUDA=13.0 STUB_PYTHON_VERSION=3.12.0
  export STUB_NO_GPU=0 STUB_FAIL_INSTALL=0 STUB_FAIL_CHECK=0 STUB_FAIL_ARENO=0
  export STUB_FAIL_PLAN=0
  export STUB_EXPLICIT_PYTHON=1
  export STUB_FLASH_IMPORT=OK
  export STUB_FAIL_VENV=0 STUB_FAIL_REMOVE=0 STUB_SIGNAL_INSTALL=""
  export STUB_TAMPER_OWNER="" STUB_REPLACE_VENV=0 STUB_INSTALL_LOG=""
  export STUB_SYSTEM_PYTHON_VERSION="" STUB_NO_SYSTEM_PIP=0 STUB_NO_VENV_PIP=0
  export STUB_VENV_MACHINE=""
  export STUB_VIRTUAL_ENV="" STUB_CONDA_PREFIX="" STUB_CUDA_HOME="" STUB_CUDA_PATH=""
  unset ASCEND_HOME_PATH ASCEND_CANN_PACKAGE_PATH TORCH_CUDA_ARCH_LIST ARENO_BUILD_EXT
  unset ARENO_INSTALL_DRY_RUN ARENO_INSTALL_NOTEBOOK ARENO_INSTALL_FLASH ARENO_INSTALL_BUILD_ONLY
}

# Execute the real installer against stubs and capture its exact exit status.
run_installer() {
  local selected_python="${case_dir}/bin/python with spaces"
  if [[ ${STUB_EXPLICIT_PYTHON} == 0 ]]; then selected_python=''; fi
  result=0
  env -u VIRTUAL_ENV -u CONDA_PREFIX -u CUDA_HOME -u CUDA_PATH -u COLAB_RELEASE_TAG \
    HOME="${case_dir}/home" XDG_STATE_HOME="${case_dir}/state" \
    PATH="${case_dir}/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
    PYTHON="${selected_python}" \
    VIRTUAL_ENV="${STUB_VIRTUAL_ENV}" CONDA_PREFIX="${STUB_CONDA_PREFIX}" \
    CUDA_HOME="${STUB_CUDA_HOME}" CUDA_PATH="${STUB_CUDA_PATH}" \
    ARENO_INSTALL_LOG="${STUB_INSTALL_LOG:-${case_dir}/state with spaces/install.log}" \
    "${BASH}" "${checkout}/scripts/install.sh" "$@" > "${case_dir}/output" 2>&1 || result=$?
}

success() {
  [[ ${result} == 0 ]] || fail "exit status ${result}"
  TEST_PASSED=$((TEST_PASSED + 1))
  printf 'PASS %s\n' "${case_name}"
}

failure() {
  [[ ${result} != 0 ]] || fail 'expected failure'
  TEST_PASSED=$((TEST_PASSED + 1))
  printf 'PASS %s\n' "${case_name}"
}

# Preflight failures and dry runs must not create logs, packages, or environments.
no_mutations() {
  [[ ! -s ${case_dir}/mutations ]] || fail 'package or device mutations occurred'
  [[ ! -e ${case_dir}/state\ with\ spaces/install.log ]] || fail 'dry-run/preflight wrote log'
  [[ ! -d "${case_dir}/state with spaces" ]] || fail 'dry-run/preflight created diagnostics'
  [[ ! -d ${checkout}/.venv ]] || fail 'dry-run created virtual environment'
}

# Initialization must discard stale internal values without erasing configuration.
new_case internal_state_reset_preserves_configuration
result=0
(
  export ARENO_INSTALL_DRY_RUN="true"
  export ARENO_INSTALL_LOG="${case_dir}/configured log"
  export ARENO_INSTALL_BACKEND="stale-backend"
  export ARENO_INSTALL_LOG_STARTED="true"
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  [[ "${ARENO_INSTALL_BACKEND}" == "" ]] || exit 1
  [[ "${ARENO_INSTALL_LOG_STARTED}" == "false" ]] || exit 1
  [[ "${ARENO_INSTALL_DRY_RUN}" == "true" ]] || exit 1
  [[ "${ARENO_INSTALL_LOG_PATH}" == "${case_dir}/configured log" ]] || exit 1

  # Reset every scalar/array result after a simulated prior installation.
  for name in STEP PYTHON_BIN BOOTSTRAP_PYTHON PACKAGES BACKEND HOST_SYSTEM HOST_ARCH PYTHON_VERSION CUDA_VERSION PROFILE_INDEX CONSTRAINTS; do
    printf -v "ARENO_INSTALL_${name}" '%s' 'stale'
  done
  ARENO_INSTALL_CREATE_VENV="true"
  ARENO_INSTALL_PROTECT_GLOBAL="true"
  ARENO_INSTALL_LOG_STARTED="true"
  ARENO_INSTALL_PROFILE_PACKAGES=("stale==1.0")
  areno::install::init_state
  for name in PYTHON_BIN BOOTSTRAP_PYTHON PACKAGES BACKEND HOST_SYSTEM HOST_ARCH PYTHON_VERSION CUDA_VERSION PROFILE_INDEX CONSTRAINTS; do
    variable="ARENO_INSTALL_${name}"
    [[ "${!variable}" == "" ]] || exit 1
  done
  [[ "${ARENO_INSTALL_STEP}" == "Parse arguments" ]] || exit 1
  [[ "${ARENO_INSTALL_CREATE_VENV}" == "false" ]] || exit 1
  [[ "${ARENO_INSTALL_PROTECT_GLOBAL}" == "false" ]] || exit 1
  [[ "${ARENO_INSTALL_LOG_STARTED}" == "false" ]] || exit 1
  [[ "${#ARENO_INSTALL_PROFILE_PACKAGES[@]}" == "0" ]] || exit 1
  [[ "${ARENO_INSTALL_DRY_RUN}" == "true" ]] || exit 1
  areno::log::info 'state reset without opening the configured log'
) > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
[[ ! -e "${case_dir}/configured log" ]] || fail 'initialization opened a log file'
success

# Environment defaults use the same validated booleans as command-line switches.
new_case environment_dry_run_default
export ARENO_INSTALL_DRY_RUN=true
run_installer
assert_contains "${case_dir}/output" 'Installation plan complete'
no_mutations
success

new_case invalid_environment_boolean
export ARENO_INSTALL_DRY_RUN="touch ${case_dir}/unexpected"
run_installer
assert_contains "${case_dir}/output" 'ARENO_INSTALL_DRY_RUN must be true or false'
[[ ! -e "${case_dir}/unexpected" ]] || fail 'invalid boolean was executed'
no_mutations
failure

new_case cuda_dry_run
run_installer --dry-run
assert_contains "${case_dir}/output" 'cuda'
assert_contains "${case_dir}/output" 'default.txt'
assert_absent "${case_dir}/calls" 'install.py'
no_mutations
success

new_case mlx_dry_run
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64
printf '%s\n' 'mlx==0.32.3' 'mlx-lm==0.32.0' 'mlx-vlm==0.6.14' > "${case_dir}/packages"
run_installer --dry-run
assert_contains "${case_dir}/output" 'mlx'
assert_absent "${case_dir}/calls" 'nvidia-smi'
assert_absent "${case_dir}/calls" 'nvcc'
no_mutations
success

new_case npu_dry_run
cp "${case_dir}/bin/stub" "${case_dir}/bin/npu-smi"
printf '%s\n' 'torch==2.10.0+cpu' 'torch_npu==2.10.0.post2' 'triton-ascend==3.2.0' > "${case_dir}/packages"
export ASCEND_HOME_PATH="${case_dir}/CANN toolkit"
mkdir -p "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
: > "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
run_installer --dry-run
assert_contains "${case_dir}/output" 'npu.txt'
assert_absent "${case_dir}/calls" 'nvidia-smi'
assert_absent "${case_dir}/calls" 'nvcc'
no_mutations
success

new_case unsupported_windows
export STUB_SYSTEM=MINGW64_NT-10.0
run_installer --dry-run
no_mutations
failure

new_case intel_mac
export STUB_SYSTEM=Darwin STUB_MACHINE=x86_64
run_installer --dry-run
no_mutations
failure

new_case old_macos
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_MACOS=13.6
run_installer --dry-run
no_mutations
failure

new_case old_python
export STUB_PYTHON_VERSION=3.9.9
run_installer --dry-run
no_mutations
failure

# An incidental PATH interpreter must not prevent reuse of a healthy checkout.
for system_problem in old_version missing_pip; do
  new_case "checkout_venv_survives_system_${system_problem}"
  export STUB_EXPLICIT_PYTHON=0
  if [[ "${system_problem}" == old_version ]]; then
    export STUB_SYSTEM_PYTHON_VERSION=3.9.9
  else
    export STUB_NO_SYSTEM_PIP=1
  fi
  # The incidental Python even advertises CUDA torch; its old version or missing
  # pip must still exclude it from prepared-runtime priority.
  if [[ "${system_problem}" == old_version ]]; then
    : > "${case_dir}/venv-packages"
  else
    printf '%s\n' 'torch==2.9.1+cu130' > "${case_dir}/venv-packages"
  fi
  mkdir -p "${checkout}/.venv/bin"
  cp "${case_dir}/bin/stub" "${checkout}/.venv/bin/python"
  run_installer --dry-run
  assert_contains "${case_dir}/output" "Python: ${checkout}/.venv/bin/python"
  [[ ! -s ${case_dir}/mutations ]] || fail 'dry-run mutated environment'
  [[ ! -d "${case_dir}/state with spaces" ]] || fail 'dry-run created diagnostics'
  success
done

new_case bootstrap_venv_without_system_pip
export STUB_EXPLICIT_PYTHON=0 STUB_NO_SYSTEM_PIP=1 STUB_SYSTEM=Darwin STUB_MACHINE=arm64
: > "${case_dir}/packages"
run_installer
assert_contains "${case_dir}/mutations" 'venv'
assert_contains "${case_dir}/mutations" 'pip check'
assert_contains "${case_dir}/mutations" 'areno check'
[[ -x "${checkout}/.venv/bin/python" ]] || fail 'bootstrap did not keep its usable environment'
success

# Bootstrap must validate the venv's architecture before installing any package.
new_case bootstrap_wrong_architecture_fails_before_pip_install
export STUB_EXPLICIT_PYTHON=0 STUB_NO_SYSTEM_PIP=1 STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_VENV_MACHINE=x86_64
: > "${case_dir}/packages"
run_installer
[[ ${result} == 1 ]] || fail "expected architecture rejection, got ${result}"
assert_contains "${case_dir}/output" 'Use native arm64 Python'
assert_contains "${case_dir}/mutations" 'venv'
assert_absent "${case_dir}/mutations" 'install <'
[[ ! -e "${checkout}/.venv" ]] || fail 'wrong-architecture environment was retained'
failure

# Explicit selections must fail in place instead of silently switching Python.
new_case explicit_python_without_pip_is_rejected
export STUB_NO_SYSTEM_PIP=1
mkdir -p "${checkout}/.venv/bin"
cp "${case_dir}/bin/stub" "${checkout}/.venv/bin/python"
run_installer --dry-run
assert_contains "${case_dir}/output" 'Selected Python needs pip'
[[ ! -s ${case_dir}/mutations ]] || fail 'invalid explicit environment was mutated'
failure

for active_environment in VIRTUAL_ENV CONDA_PREFIX; do
  new_case "${active_environment}_without_pip_is_rejected"
  export STUB_EXPLICIT_PYTHON=0 STUB_NO_VENV_PIP=1
  mkdir -p "${checkout}/.venv/bin"
  cp "${case_dir}/bin/stub" "${checkout}/.venv/bin/python"
  export "STUB_${active_environment}=${checkout}/.venv"
  run_installer --dry-run
  assert_contains "${case_dir}/output" 'Selected Python needs pip'
  [[ ! -s ${case_dir}/mutations ]] || fail 'invalid activated environment was mutated'
  failure
done

# The compiler used for preflight and the root seen by pip/build must agree.
for cuda_source in cuda_path path_symlink explicit_home; do
  new_case "cuda_root_from_${cuda_source}"
  export STUB_CUDA_HOME="${case_dir}/missing toolkit"
  expected_cuda_home="${case_dir}/valid toolkit"
  mkdir -p "${expected_cuda_home}/bin"
  cp "${case_dir}/bin/stub" "${expected_cuda_home}/bin/nvcc"
  case "${cuda_source}" in
    cuda_path) export STUB_CUDA_PATH="${expected_cuda_home}" ;;
    path_symlink)
      rm "${case_dir}/bin/nvcc"
      ln -s '../valid toolkit/bin/nvcc' "${case_dir}/bin/nvcc"
      ;;
    explicit_home)
      export STUB_CUDA_HOME="${expected_cuda_home}" STUB_CUDA_PATH="${case_dir}/other toolkit"
      mkdir -p "${STUB_CUDA_PATH}/bin"
      cp "${case_dir}/bin/stub" "${STUB_CUDA_PATH}/bin/nvcc"
      ;;
  esac
  run_installer
  [[ "$(cat "${case_dir}/build-cuda-home")" == "${expected_cuda_home}" ]] || fail 'build used the wrong CUDA root'
  success
done

# Deterministically interleave A and B: B writes pins while A is still active.
# Distinct log names in the same directory must not share the constraints file.
new_case interleaved_installations_keep_independent_constraints
result=0
(
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  ARENO_INSTALL_LOG_PATH="${case_dir}/shared logs/install-A.log"
  ARENO_INSTALL_PACKAGES='torch==2.6.0'
  areno::install::start_log
  areno::install::save_pins
  first_constraints="${ARENO_INSTALL_CONSTRAINTS}"
  (
    areno::install::init_state
    ARENO_INSTALL_LOG_PATH="${case_dir}/shared logs/install-B.log"
    ARENO_INSTALL_PACKAGES='torch==2.9.1'
    areno::install::start_log
    areno::install::save_pins
    printf '%s\n' "${ARENO_INSTALL_CONSTRAINTS}" > "${case_dir}/second constraints path"
    cat "${ARENO_INSTALL_CONSTRAINTS}" > "${case_dir}/second pins"
  )
  [[ "${first_constraints}" != "$(cat "${case_dir}/second constraints path")" ]] || exit 1
  cat "${first_constraints}" > "${case_dir}/first pins after second install"
) > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
[[ ${result} == 0 ]] || fail 'interleaved installations shared their constraints'
assert_contains "${case_dir}/first pins after second install" 'torch==2.6.0'
assert_absent "${case_dir}/first pins after second install" 'torch==2.9.1'
assert_contains "${case_dir}/second pins" 'torch==2.9.1'
success

new_case wsl1
export STUB_RELEASE=4.4.0-Microsoft
run_installer --dry-run
no_mutations
failure

new_case mlx_flash_rejected
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64
run_installer --dry-run --flash
no_mutations
failure

new_case cuda_build_only
export TORCH_CUDA_ARCH_LIST=9.0 STUB_NO_GPU=1
run_installer --build-only
assert_absent "${case_dir}/calls" 'nvidia-smi'
assert_contains "${case_dir}/mutations" 'pip check'
assert_absent "${case_dir}/mutations" 'areno check'
assert_contains "$(constraints_file)" 'torch==2.9.1+cu130'
assert_contains "${case_dir}/mutations" "<$(constraints_file)>"
success

new_case build_only_requires_architecture
run_installer --build-only --dry-run
no_mutations
failure

new_case cuda_arm64_bootstrap
export STUB_MACHINE=aarch64
: > "${case_dir}/packages"
run_installer --dry-run
assert_contains "${case_dir}/output" 'torch==2.9.1'
assert_contains "${case_dir}/output" 'cu130'
no_mutations
success

new_case cuda_arm64_unsupported_bootstrap
export STUB_MACHINE=aarch64 STUB_CUDA=12.4
: > "${case_dir}/packages"
run_installer --dry-run
no_mutations
failure

new_case pip_check_failure
export STUB_FAIL_CHECK=1
run_installer
assert_contains "${case_dir}/mutations" 'pip check'
assert_absent "${case_dir}/output" 'AReno is ready'
failure

new_case install_failure
export STUB_FAIL_INSTALL=1
run_installer
assert_absent "${case_dir}/mutations" 'pip check'
assert_absent "${case_dir}/output" 'AReno is ready'
failure

new_case cli_check_failure
export STUB_FAIL_ARENO=1
run_installer
assert_contains "${case_dir}/mutations" 'areno check'
assert_absent "${case_dir}/output" 'AReno is ready'
failure

new_case no_visible_cuda_device
export STUB_NO_GPU=1
run_installer
no_mutations
failure

new_case npu_runtime_is_protected
cp "${case_dir}/bin/stub" "${case_dir}/bin/npu-smi"
printf '%s\n' 'torch==2.10.0+cpu' 'torch_npu==2.10.0.post2' 'triton-ascend==3.2.0' > "${case_dir}/packages"
export ASCEND_HOME_PATH="${case_dir}/CANN toolkit"
mkdir -p "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
: > "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
run_installer
assert_contains "$(constraints_file)" 'torch-npu==2.10.0.post2'
assert_contains "$(constraints_file)" 'triton-ascend==3.2.0'
assert_contains "${case_dir}/mutations" "<${checkout}/requirements/npu.txt>"
assert_contains "${case_dir}/mutations" 'areno check'
assert_absent "${case_dir}/calls" 'nvidia-smi'
success

new_case npu_missing_cann
cp "${case_dir}/bin/stub" "${case_dir}/bin/npu-smi"
printf '%s\n' 'torch==2.10.0+cpu' 'torch-npu==2.10.0.post2' > "${case_dir}/packages"
export ASCEND_HOME_PATH="${case_dir}/missing toolkit"
run_installer --dry-run
no_mutations
failure

new_case npu_flash_rejected
cp "${case_dir}/bin/stub" "${case_dir}/bin/npu-smi"
printf '%s\n' 'torch==2.10.0+cpu' 'torch-npu==2.10.0.post2' > "${case_dir}/packages"
run_installer --dry-run --flash
no_mutations
failure

new_case notebook_keeps_selected_python
mkdir -p "${checkout}/.venv/bin"
printf '#!/bin/bash\nexit 99\n' > "${checkout}/.venv/bin/python"
chmod +x "${checkout}/.venv/bin/python"
run_installer --notebook --dry-run
[[ ! -s ${case_dir}/mutations ]] || fail 'dry-run mutated environment'
success

new_case unknown_option
run_installer --unknown-option
no_mutations
failure

new_case implicit_mlx_dry_run_plans_isolated_environment
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0
: > "${case_dir}/packages"
run_installer --dry-run
assert_contains "${case_dir}/output" '.venv'
assert_contains "${case_dir}/output" '-m venv'
no_mutations
success

new_case implicit_cuda_keeps_vendor_environment
export STUB_EXPLICIT_PYTHON=0
mkdir -p "${checkout}/.venv/bin"
cp "${case_dir}/bin/stub" "${checkout}/.venv/bin/python"
printf '%s\n' 'torch==2.9.1+cpu' > "${case_dir}/venv-packages"
run_installer --dry-run
assert_absent "${case_dir}/output" '.venv/bin/python'
[[ ! -s ${case_dir}/mutations ]] || fail 'dry-run mutated environment'
success

new_case cpu_python_uses_cuda_checkout_environment
export STUB_EXPLICIT_PYTHON=0
printf '%s\n' 'torch==2.9.1+cpu' > "${case_dir}/packages"
printf '%s\n' 'torch==2.9.1+cu130' > "${case_dir}/venv-packages"
mkdir -p "${checkout}/.venv/bin"
cp "${case_dir}/bin/stub" "${checkout}/.venv/bin/python"
run_installer --dry-run
assert_contains "${case_dir}/output" '.venv/bin/python'
[[ ! -s ${case_dir}/mutations ]] || fail 'dry-run mutated environment'
success

# Implicit global reuse must prove safety before touching even build tooling.
for mode in install dry_run colab; do
  new_case "broken_implicit_global_${mode}_is_rejected"
  export STUB_EXPLICIT_PYTHON=0 STUB_FAIL_CHECK=1
  if [[ "${mode}" == colab ]]; then export COLAB_TEST_GLOBAL_GUARD=1; fi
  if [[ "${mode}" == dry_run ]]; then run_installer --dry-run; else run_installer; fi
  unset COLAB_TEST_GLOBAL_GUARD
  assert_contains "${case_dir}/output" 'global environment has broken dependencies'
  assert_absent "${case_dir}/calls" '<pip> <install>'
  no_mutations
  failure
done

new_case implicit_global_dependency_conflict_is_rejected_before_install
export STUB_EXPLICIT_PYTHON=0 STUB_FAIL_PLAN=1
printf '%s\n' 'rich==13.9.4' 'typing_extensions==4.12.2' >> "${case_dir}/packages"
run_installer
assert_contains "${case_dir}/output" 'ResolutionImpossible'
assert_contains "${case_dir}/output" 'no packages were changed'
assert_contains "$(constraints_file)" 'rich==13.9.4'
assert_contains "$(constraints_file)" 'typing-extensions==4.12.2'
[[ ! -s "${case_dir}/mutations" ]] || fail 'preflight changed packages'
[[ ! -e "${checkout}/.venv" ]] || fail 'preflight created a fallback environment'
failure

new_case clean_implicit_global_preserves_all_versions
export STUB_EXPLICIT_PYTHON=0
printf '%s\n' 'rich==13.9.4' 'typing_extensions==4.12.2' >> "${case_dir}/packages"
run_installer --flash
assert_contains "$(constraints_file)" 'rich==13.9.4'
assert_contains "$(constraints_file)" 'typing-extensions==4.12.2'
assert_contains "$(constraints_file)" 'torch==2.9.1+cu130'
plan=$(grep -F '<install> <--dry-run>' "${case_dir}/calls")
for arg in '<setuptools>=69>' '<wheel>' '<psutil>' '<ninja>' '<flash-attn>=2.7>' '<--no-build-isolation>' "<${checkout}/requirements/default.txt>" "<$(constraints_file)>"; do
  [[ "${plan}" == *"${arg}"* ]] || fail "resolver omitted ${arg}"
done
# A future refactor must not move the read-only resolution after a real install.
first_install=$(grep -F '<pip> <install>' "${case_dir}/calls" | head -1)
[[ "${first_install}" == "${plan}" ]] || fail 'package installation preceded resolution'
assert_contains "${case_dir}/mutations" 'areno check'
success

new_case implicit_global_dry_run_only_prints_resolution
export STUB_EXPLICIT_PYTHON=0 STUB_FAIL_PLAN=1
run_installer --dry-run
assert_contains "${case_dir}/output" 'install --dry-run'
assert_absent "${case_dir}/calls" '<pip> <install>'
no_mutations
success

new_case implicit_notebook_plan_includes_missing_runtime_profile
export STUB_EXPLICIT_PYTHON=0 STUB_FAIL_PLAN=1 COLAB_TEST_GLOBAL_GUARD=1
printf '%s\n' 'rich==13.9.4' > "${case_dir}/packages"
run_installer
unset COLAB_TEST_GLOBAL_GUARD
plan=$(grep -F '<install> <--dry-run>' "${case_dir}/calls")
for arg in '<torch==2.9.1>' '<torchvision==0.24.1>' '<--extra-index-url>' '<https://download.pytorch.org/whl/cu130>'; do
  [[ "${plan}" == *"${arg}"* ]] || fail "resolver omitted ${arg}"
done
[[ ! -s "${case_dir}/mutations" ]] || fail 'notebook preflight changed packages'
failure

# Explicit choices and actual venvs retain runtime-only protection.
for selection in explicit active_venv active_conda path_venv checkout_venv; do
  new_case "${selection}_does_not_use_global_guard"
  export STUB_FAIL_PLAN=1
  printf '%s\n' 'rich==13.9.4' >> "${case_dir}/packages"
  case "${selection}" in
    explicit) : ;;
    active_venv) export STUB_EXPLICIT_PYTHON=0 STUB_VIRTUAL_ENV="${case_dir}" ;;
    active_conda) export STUB_EXPLICIT_PYTHON=0 STUB_CONDA_PREFIX="${case_dir}" ;;
    path_venv)
      export STUB_EXPLICIT_PYTHON=0
      printf '%s\n' 'home = /stub' > "${case_dir}/pyvenv.cfg"
      ;;
    checkout_venv)
      export STUB_EXPLICIT_PYTHON=0
      mkdir -p "${checkout}/.venv/bin"
      cp "${case_dir}/bin/stub" "${checkout}/.venv/bin/python"
      ;;
  esac
  run_installer
  assert_absent "${case_dir}/calls" '<install> <--dry-run>'
  assert_absent "$(constraints_file)" 'rich=='
  assert_contains "$(constraints_file)" 'torch==2.9.1+cu130'
  success
done

new_case implicit_npu_global_is_protected
export STUB_EXPLICIT_PYTHON=0 STUB_FAIL_PLAN=1
cp "${case_dir}/bin/stub" "${case_dir}/bin/npu-smi"
printf '%s\n' 'torch==2.10.0+cpu' 'torch_npu==2.10.0.post2' 'rich==13.9.4' > "${case_dir}/packages"
export ASCEND_HOME_PATH="${case_dir}/CANN toolkit"
mkdir -p "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
: > "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
run_installer
assert_contains "${case_dir}/calls" "<${checkout}/requirements/npu.txt>"
assert_contains "$(constraints_file)" 'rich==13.9.4'
assert_contains "$(constraints_file)" 'torch-npu==2.10.0.post2'
[[ ! -s "${case_dir}/mutations" ]] || fail 'NPU preflight changed packages'
failure

new_case requested_flash_import_passes
run_installer --flash
assert_contains "${case_dir}/mutations" '<flash-attn>=2.7>'
assert_contains "${case_dir}/mutations" 'areno check'
success

new_case requested_flash_import_warning_fails
export STUB_FLASH_IMPORT=WARN
run_installer --flash
assert_contains "${case_dir}/mutations" 'areno check'
assert_absent "${case_dir}/output" 'Installation and CLI checks passed'
failure

new_case shell_source_only
for source_file in "${checkout}/scripts/install.sh" "${checkout}"/scripts/lib/*.sh; do
  "${BASH}" -n "${source_file}" || fail "invalid Shell syntax: ${source_file}"
  assert_absent "${source_file}" 'install.py'
  assert_absent "${source_file}" 'install_checks.py'
  if grep -Eq -- '(^|[[:space:]])-c([[:space:]]|$)|<<.*PY' "${source_file}"; then
    fail "embedded source execution in ${source_file}"
  fi
done
TEST_PASSED=$((TEST_PASSED + 1))
printf 'PASS %s\n' "${case_name}"

# Logger tests source the library in subshells so mocks and globals cannot leak.
new_case log_redirected_streams_are_plain
result=0
(
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  unset NO_COLOR
  export TERM=xterm
  areno::log::info 'info message'
  areno::log::status 'status message'
  areno::log::success 'success message'
  areno::log::warn 'warning message'
  areno::log::error 'error message'
) > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
assert_contains "${case_dir}/output" 'info message'
assert_contains "${case_dir}/output" 'status message'
assert_contains "${case_dir}/output" 'success message'
assert_absent "${case_dir}/output" 'warning message'
assert_absent "${case_dir}/output" 'error message'
assert_contains "${case_dir}/error" 'warning message'
assert_contains "${case_dir}/error" 'error message'
assert_absent "${case_dir}/error" 'info message'
assert_absent "${case_dir}/output" $'\033'
assert_absent "${case_dir}/error" $'\033'
success

new_case log_levels_use_distinct_colors
result=0
(
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  # Exercise formatting independently of the test runner's terminal allocation.
  areno::log::color_enabled() { return 0; }
  areno::log::info 'color-info'
  areno::log::status 'color-status'
  areno::log::success 'color-success'
  areno::log::warn 'color-warn'
  areno::log::error 'color-error'
) > "${case_dir}/output" 2>&1 || result=$?
: > "${case_dir}/colors"
for message in color-info color-status color-success color-warn color-error; do
  line=$(grep -F -- "${message}" "${case_dir}/output")
  [[ ${line} == *$'\033['* ]] || fail "no color for ${message}"
  color=${line#*$'\033['}
  color=${color%%m*}
  printf '%s\n' "${color}" >> "${case_dir}/colors"
done
[[ $(sort -u "${case_dir}/colors" | wc -l | tr -d ' ') == 5 ]] || fail 'log levels share color prefixes'
success

new_case log_file_stays_plain_with_colored_console
result=0
(
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  ARENO_INSTALL_LOG_STARTED=true
  ARENO_INSTALL_LOG_PATH="${case_dir}/log with spaces"
  areno::log::color_enabled() { return 0; }
  areno::log::success 'saved success'
  areno::log::warn 'saved warning'
) > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
assert_contains "${case_dir}/output" $'\033['
assert_contains "${case_dir}/error" $'\033['
assert_contains "${case_dir}/log with spaces" 'saved success'
assert_contains "${case_dir}/log with spaces" 'saved warning'
assert_absent "${case_dir}/log with spaces" $'\033'
success

new_case log_color_preferences_are_respected
result=0
(
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  export TERM=xterm NO_COLOR=1
  if areno::log::color_enabled 1; then exit 1; fi
  if areno::log::color_enabled 2; then exit 1; fi
  areno::log::info 'NO_COLOR output'
  unset NO_COLOR
  export TERM=dumb
  if areno::log::color_enabled 1; then exit 1; fi
  if areno::log::color_enabled 2; then exit 1; fi
  areno::log::warn 'dumb terminal output'
  export TERM=xterm
  if areno::log::color_enabled 1; then exit 1; fi
  areno::log::info 'redirected output'
) > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
assert_contains "${case_dir}/output" 'NO_COLOR output'
assert_contains "${case_dir}/output" 'redirected output'
assert_contains "${case_dir}/error" 'dumb terminal output'
assert_absent "${case_dir}/output" $'\033'
assert_absent "${case_dir}/error" $'\033'
success

new_case subprocess_output_strips_ansi_before_logging
result=0
(
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  ARENO_INSTALL_LOG_STARTED=true
  ARENO_INSTALL_LOG_PATH="${case_dir}/subprocess log"
  printf '\033[31mbuild failed\033[0m\n\033[2Kdependency message\n' | areno::log::output
) > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
assert_contains "${case_dir}/output" 'build failed'
assert_contains "${case_dir}/output" 'dependency message'
assert_contains "${case_dir}/subprocess log" 'build failed'
assert_contains "${case_dir}/subprocess log" 'dependency message'
assert_absent "${case_dir}/output" $'\033'
assert_absent "${case_dir}/subprocess log" $'\033'
success

new_case log_messages_are_literal_and_complete
result=0
(
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  areno::log::info '100% complete; literal %s and \n' 'second message'
) > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
assert_contains "${case_dir}/output" '100% complete; literal %s and \n'
assert_contains "${case_dir}/output" 'second message'
success

# A slow producer must not make the filter hold its first line until EOF.
new_case subprocess_output_streams_before_command_finishes
result=0
(
  source "${checkout}/scripts/lib/common.sh"
  areno::install::init_state
  ARENO_INSTALL_LOG_PATH="${case_dir}/streaming log"
  exec 3< <({ printf 'first line\n'; sleep 2; printf 'last line'; } | areno::log::output)
  IFS= read -r -t 1 first <&3 || exit 1
  [[ "${first}" == 'first line' ]] || exit 1
  IFS= read -r -t 5 last <&3 || exit 1
  [[ "${last}" == 'last line' ]] || exit 1
  exec 3<&-
) > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
assert_contains "${case_dir}/streaming log" 'first line'
assert_contains "${case_dir}/streaming log" 'last line'
success

# Cleanup owns only a venv created by this invocation. Existing resources survive.
new_case failed_new_venv_install_is_removed
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_INSTALL=1
: > "${case_dir}/packages"
run_installer
[[ ${result} == 17 ]] || fail "expected original pip exit 17, got ${result}"
[[ ! -e "${checkout}/.venv" ]] || fail 'failed new environment was retained'
assert_contains "${case_dir}/mutations" 'venv'
[[ -s "${case_dir}/state with spaces/install.log" ]] || fail 'failure log was removed'
[[ -f "$(constraints_file)" ]] || fail 'constraints were removed'
failure

new_case partial_new_venv_is_removed
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_VENV=1
: > "${case_dir}/packages"
run_installer
[[ ${result} == 23 ]] || fail "expected original venv exit 23, got ${result}"
[[ ! -e "${checkout}/.venv" ]] || fail 'partial new environment was retained'
assert_contains "${case_dir}/mutations" 'venv'
[[ -s "${case_dir}/state with spaces/install.log" ]] || fail 'failure log was removed'
failure

new_case failed_new_venv_check_is_removed
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_CHECK=1
: > "${case_dir}/packages"
run_installer
[[ ${result} == 19 ]] || fail "expected original check exit 19, got ${result}"
[[ ! -e "${checkout}/.venv" ]] || fail 'new environment survived failed dependency verification'
assert_contains "${case_dir}/mutations" 'pip check'
[[ -s "${case_dir}/state with spaces/install.log" ]] || fail 'verification log was removed'
failure

new_case failed_new_venv_cli_check_is_removed
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_ARENO=1
: > "${case_dir}/packages"
run_installer
[[ ${result} == 1 ]] || fail "expected CLI wrapper exit 1, got ${result}"
[[ ! -e "${checkout}/.venv" ]] || fail 'new environment survived failed runtime verification'
assert_contains "${case_dir}/mutations" 'areno check'
failure

new_case failed_new_venv_flash_check_is_removed
export STUB_EXPLICIT_PYTHON=0 STUB_FLASH_IMPORT=WARN
: > "${case_dir}/packages"
run_installer --flash
[[ ${result} == 1 ]] || fail "expected explicit flash verification exit 1, got ${result}"
[[ ! -e "${checkout}/.venv" ]] || fail 'new environment survived failed flash verification'
assert_contains "${case_dir}/mutations" '<flash-attn>=2.7>'
failure

new_case existing_checkout_venv_is_preserved
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_INSTALL=1
: > "${case_dir}/packages"
mkdir -p "${checkout}/.venv/bin"
cp "${case_dir}/bin/stub" "${checkout}/.venv/bin/python"
printf '%s\n' 'existing user data' > "${checkout}/.venv/keep-me"
run_installer
[[ ${result} == 17 ]] || fail "expected pip exit 17, got ${result}"
assert_contains "${checkout}/.venv/keep-me" 'existing user data'
[[ -x "${checkout}/.venv/bin/python" ]] || fail 'existing Python was deleted'
if grep -Fxq 'venv' "${case_dir}/mutations"; then fail 'existing environment was recreated'; fi
failure

new_case existing_symlink_venv_is_preserved
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_INSTALL=1
: > "${case_dir}/packages"
mkdir -p "${case_dir}/external environment/bin"
cp "${case_dir}/bin/stub" "${case_dir}/external environment/bin/python"
printf '%s\n' 'external user data' > "${case_dir}/external environment/keep-me"
ln -s "${case_dir}/external environment" "${checkout}/.venv"
run_installer
[[ ${result} == 17 ]] || fail "expected pip exit 17, got ${result}"
[[ -L "${checkout}/.venv" ]] || fail 'existing venv symlink was removed'
assert_contains "${case_dir}/external environment/keep-me" 'external user data'
if grep -Fxq 'venv' "${case_dir}/mutations"; then fail 'existing environment was recreated'; fi
failure

new_case preexisting_partial_venv_is_preserved
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0
: > "${case_dir}/packages"
mkdir -p "${checkout}/.venv"
printf '%s\n' 'unrelated partial environment' > "${checkout}/.venv/keep-me"
run_installer
assert_contains "${checkout}/.venv/keep-me" 'unrelated partial environment'
[[ ! -s "${case_dir}/mutations" ]] || fail 'existing incomplete directory was mutated'
failure

new_case dangling_venv_symlink_is_preserved
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0
: > "${case_dir}/packages"
ln -s "${case_dir}/nonexistent environment" "${checkout}/.venv"
run_installer
[[ -L "${checkout}/.venv" ]] || fail 'existing dangling symlink was removed'
[[ ! -s "${case_dir}/mutations" ]] || fail 'dangling symlink target was mutated'
failure

new_case successful_new_venv_is_preserved
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0
: > "${case_dir}/packages"
run_installer
[[ -x "${checkout}/.venv/bin/python" ]] || fail 'successful environment was removed'
[[ ! -e "${checkout}/.venv/.areno-install-owner" ]] || fail 'successful install retained ownership marker'
assert_contains "${case_dir}/mutations" 'areno check'
[[ -s "${case_dir}/state with spaces/install.log" ]] || fail 'success log was removed'
success

new_case term_cleans_new_venv_and_preserves_signal_status
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_SIGNAL_INSTALL=TERM
: > "${case_dir}/packages"
run_installer
[[ ${result} == 143 ]] || fail "expected TERM exit 143, got ${result}"
[[ ! -e "${checkout}/.venv" ]] || fail 'interrupted new environment was retained'
[[ -s "${case_dir}/state with spaces/install.log" ]] || fail 'signal log was removed'
failure

new_case cleanup_failure_preserves_install_exit_status
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_INSTALL=1 STUB_FAIL_REMOVE=1
: > "${case_dir}/packages"
cp "${case_dir}/bin/stub" "${case_dir}/bin/rm"
run_installer
[[ ${result} == 17 ]] || fail "cleanup replaced original exit 17 with ${result}"
[[ -d "${checkout}/.venv" ]] || fail 'rm failure fixture did not preserve environment'
assert_contains "${case_dir}/calls" 'rm <'
[[ -s "${case_dir}/state with spaces/install.log" ]] || fail 'failure log was removed'
failure

# Ownership changes must turn destructive cleanup into retention with diagnostics.
new_case changed_owner_marker_is_preserved
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_INSTALL=1 STUB_TAMPER_OWNER=changed
: > "${case_dir}/packages"
run_installer
[[ ${result} == 17 ]] || fail "expected original pip exit 17, got ${result}"
assert_contains "${checkout}/.venv/.areno-install-owner" 'different owner'
assert_contains "${case_dir}/output" 'ownership marker changed'
failure

new_case symlink_owner_marker_is_preserved
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_INSTALL=1 STUB_TAMPER_OWNER=symlink
: > "${case_dir}/packages"
run_installer
[[ ${result} == 17 ]] || fail "expected original pip exit 17, got ${result}"
[[ -L "${checkout}/.venv/.areno-install-owner" ]] || fail 'symlink owner marker was removed'
[[ -s "${case_dir}/external owner file" ]] || fail 'external marker target was removed'
assert_contains "${case_dir}/output" 'ownership could not be verified'
failure

new_case replacement_venv_symlink_is_preserved
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_FAIL_INSTALL=1 STUB_REPLACE_VENV=1
: > "${case_dir}/packages"
run_installer
[[ ${result} == 17 ]] || fail "expected original pip exit 17, got ${result}"
[[ -L "${checkout}/.venv" ]] || fail 'replacement venv symlink was removed'
assert_contains "${case_dir}/replacement target/keep-me" 'replacement user data'
[[ -d "${case_dir}/original owned environment" ]] || fail 'moved environment was unexpectedly deleted'
assert_contains "${case_dir}/output" 'ownership could not be verified'
failure

new_case marked_environment_is_not_reused
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0
: > "${case_dir}/packages"
mkdir -p "${checkout}/.venv/bin"
cp "${case_dir}/bin/stub" "${checkout}/.venv/bin/python"
printf '%s\n' 'another installation' > "${checkout}/.venv/.areno-install-owner"
run_installer
[[ ! -s "${case_dir}/mutations" ]] || fail 'owned environment was mutated by another install'
assert_contains "${checkout}/.venv/.areno-install-owner" 'another installation'
assert_contains "${case_dir}/output" 'installation ownership marker'
failure

new_case diagnostics_under_new_venv_are_rejected
export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0
export STUB_INSTALL_LOG="${checkout}/.venv/logs/install.log"
: > "${case_dir}/packages"
run_installer
no_mutations
[[ ! -e "${STUB_INSTALL_LOG}" ]] || fail 'unsafe log path was created'
assert_contains "${case_dir}/output" 'ARENO_INSTALL_LOG must be outside'
failure

for signal in INT HUP; do
  case "${signal}" in
    INT) signal_status="130" ;;
    HUP) signal_status="129" ;;
  esac
  new_case "${signal}_cleans_new_venv_and_preserves_signal_status"
  export STUB_SYSTEM=Darwin STUB_MACHINE=arm64 STUB_EXPLICIT_PYTHON=0 STUB_SIGNAL_INSTALL="${signal}"
  : > "${case_dir}/packages"
  run_installer
  [[ ${result} == "${signal_status}" ]] || fail "expected ${signal} exit ${signal_status}, got ${result}"
  [[ ! -e "${checkout}/.venv" ]] || fail 'interrupted new environment was retained'
  [[ -s "${case_dir}/state with spaces/install.log" ]] || fail 'signal log was removed'
  failure
done

# Even a broken log destination cannot replace the original command exit status.
new_case log_failure_preserves_original_status_and_runs_cleanup
cat > "${case_dir}/error trap.sh" <<'ERROR_TRAP'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
source "${TEST_CHECKOUT}/scripts/lib/common.sh"
areno::install::init_state
ARENO_INSTALL_LOG_STARTED="true"
ARENO_INSTALL_LOG_PATH="${STUB_DIR}"
areno::install::cleanup() { printf '%s\n' "$1" > "${STUB_DIR}/cleanup called"; }
trap 'areno::install::on_exit "$?"' EXIT
areno::install::on_error 17 123
ERROR_TRAP
result=0
TEST_CHECKOUT="${checkout}" "${BASH}" "${case_dir}/error trap.sh" > "${case_dir}/output" 2> "${case_dir}/error" || result=$?
[[ ${result} == 17 ]] || fail "log failure replaced original exit 17 with ${result}"
assert_contains "${case_dir}/cleanup called" '17'
failure

printf '\n%s Shell installer integration cases passed.\n' "${TEST_PASSED}"
