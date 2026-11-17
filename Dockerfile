FROM nvcr.io/nvidia/pytorch:25.09-py3

SHELL ["/bin/bash", "-lc"]

ARG PIP_INDEX_URL=https://pypi.org/simple
# Explicit targets are required because docker build has no visible GPU.
ARG TORCH_CUDA_ARCH_LIST="9.0"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
        apt-get install -y --no-install-recommends git && \
        rm -rf /var/lib/apt/lists/*

ARG ARENO_REPO_URL=https://github.com/inclusionAI/AReno.git
ARG ARENO_BRANCH=__local__

WORKDIR /workspace/areno
COPY . /workspace/areno

RUN if [[ "${ARENO_BRANCH}" != "__local__" ]]; then \
        cd /workspace && \
        rm -rf /workspace/areno && \
        git clone --depth 1 --branch "${ARENO_BRANCH}" "${ARENO_REPO_URL}" /workspace/areno; \
    fi

RUN PYTHON="$(command -v python)" bash scripts/install.sh --build-only

CMD ["/bin/bash"]
