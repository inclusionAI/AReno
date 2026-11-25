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

# Bootstrap only missing runtimes. Installed versions are protected by constraints.
# Keep package pairs and Python limits together; unknown combinations fail closed.
# Reads ARENO_INSTALL_{BACKEND,HOST_ARCH,CUDA_VERSION,PYTHON_VERSION,PACKAGES}.
# Writes ARENO_INSTALL_PROFILE_PACKAGES (argv array) and PROFILE_INDEX.
# Existing CUDA/CANN runtimes are reused; MLX defaults apply only to missing packages.
areno::install::select_profile() {
  local torch_version
  local torchvision_version
  local python_max
  local spec
  local name
  case "${ARENO_INSTALL_BACKEND}" in
    cuda)
      [[ -z "$(areno::install::package_version torch)" ]] || return 0
      case "${ARENO_INSTALL_HOST_ARCH}:${ARENO_INSTALL_CUDA_VERSION}" in
        x86_64:12.4)
          torch_version="2.6.0"
          torchvision_version="0.21.0"
          python_max="13"
          ;;
        x86_64:12.6|x86_64:12.8|x86_64:13.0|aarch64:13.0)
          torch_version="2.9.1"
          torchvision_version="0.24.1"
          python_max="14"
          ;;
        *) areno::install::fail "No PyTorch bootstrap profile for ${ARENO_INSTALL_HOST_ARCH} / CUDA ${ARENO_INSTALL_CUDA_VERSION}. Select a prepared CUDA environment with PYTHON." ;;
      esac
      if areno::install::version_at_least "${ARENO_INSTALL_PYTHON_VERSION}" 3 "$((python_max + 1))"; then
        areno::install::fail "This bootstrap profile supports Python through 3.${python_max}. Select an older Python."
      fi
      ARENO_INSTALL_PROFILE_PACKAGES=("torch==${torch_version}" "torchvision==${torchvision_version}")
      ARENO_INSTALL_PROFILE_INDEX="https://download.pytorch.org/whl/cu${ARENO_INSTALL_CUDA_VERSION//./}"
      ;;
    mlx)
      for spec in mlx==0.32.3 mlx-lm==0.32.0 mlx-vlm==0.6.14; do
        name="${spec%%==*}"
        if [[ -z "$(areno::install::package_version "${name}")" ]]; then
          ARENO_INSTALL_PROFILE_PACKAGES+=("${spec}")
        fi
      done
      ;;
    npu) : ;; # CANN selects the compatible runtime; never bootstrap a replacement.
  esac
}
