FROM nvcr.io/nvidia/pytorch:25.09-py3

SHELL ["/bin/bash", "-lc"]

ARG TRANSFORMERS_VERSION=4.56.2
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
        apt-get install -y --no-install-recommends git && \
        rm -rf /var/lib/apt/lists/*

RUN python -m pip install --index-url "${PIP_INDEX_URL}" --upgrade pip setuptools wheel packaging ninja

RUN python -m pip install --index-url "${PIP_INDEX_URL}" \
        "psutil" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "datasets>=3.3.0" \
        "einops" \
        "safetensors>=0.4" \
        "accelerate" \
        "tqdm>=4.66" \
        "math-verify"

ARG ARENO_REPO_URL=https://github.com/inclusionAI/AReno.git
ARG ARENO_BRANCH=__local__

WORKDIR /workspace/areno
COPY . /workspace/areno

RUN if [[ "${ARENO_BRANCH}" != "__local__" ]]; then \
        cd /workspace && \
        rm -rf /workspace/areno && \
        git clone --depth 1 --branch "${ARENO_BRANCH}" "${ARENO_REPO_URL}" /workspace/areno; \
    fi

RUN python -m pip install -e . --no-build-isolation

CMD ["/bin/bash"]
