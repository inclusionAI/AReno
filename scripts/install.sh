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

# Install AReno with the native CUDA, Ascend NPU, or Apple Silicon backend.
# Requires Bash 3.2+; all orchestration and environment inspection live in Shell.
set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

# Resolve before declaring readonly so a failing command substitution stays fatal.
ARENO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ARENO_ROOT
source "${ARENO_ROOT}/scripts/lib/common.sh"
source "${ARENO_ROOT}/scripts/lib/runtime-profiles.sh"

# Run the six installation stages in order. Host preflight must finish before
# persistent logging, virtualenv creation, or any package changes are allowed.
areno::install::main() {
  areno::install::init_state
  trap 'areno::install::on_exit "$?"' EXIT
  trap 'areno::install::on_signal INT 130' INT
  trap 'areno::install::on_signal TERM 143' TERM
  trap 'areno::install::on_signal HUP 129' HUP
  areno::install::parse_args "$@"
  trap 'areno::install::on_error "$?" "$LINENO"' ERR

  areno::install::stage 'Inspect platform and Python environment'
  areno::install::select_environment
  areno::install::check_new_venv
  areno::install::check_host
  areno::install::select_profile
  areno::log::info "Platform: ${ARENO_INSTALL_HOST_SYSTEM} ${ARENO_INSTALL_HOST_ARCH}" "Backend: ${ARENO_INSTALL_BACKEND}" "Python: ${ARENO_INSTALL_PYTHON_BIN}"
  areno::install::check_global_environment
  areno::install::start_log

  areno::install::stage 'Prepare Python environment'
  areno::install::prepare_environment
  areno::install::stage 'Prepare platform runtime'
  areno::install::install_runtime
  areno::install::stage 'Install AReno dependencies'
  areno::install::install_dependencies
  areno::install::stage 'Build and install AReno'
  areno::install::build
  areno::install::stage 'Verify installation'
  areno::install::verify
}

areno::install::main "$@"
