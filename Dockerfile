FROM nvcr.io/nvidia/pytorch:25.09-py3

SHELL ["/bin/bash", "-lc"]

ARG TRANSFORMERS_VERSION=4.56.2
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

RUN python -m pip install --index-url "${PIP_INDEX_URL}" --upgrade pip setuptools wheel packaging ninja

RUN python -m pip install --index-url "${PIP_INDEX_URL}" \
        "psutil" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "datasets>=3.3.0" \
        "einops" \
        "safetensors>=0.4" \
        "accelerate" \
        "tqdm>=4.66"

WORKDIR /workspace/areno
COPY . /workspace/areno

RUN python -m pip install -e . --no-build-isolation

CMD ["/bin/bash"]
