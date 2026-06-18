Build and installation
======================

This page covers the setup paths used by contributors and local operators:
Docker images, editable installs, source/wheel distributions, and local
installation.

Compatibility matrix
--------------------

.. list-table::
   :header-rows: 1

   * - Environment
     - Status
     - Notes
   * - Linux x86_64 + NVIDIA GPU
     - Supported
     - Primary training/serving target. Use CUDA-enabled PyTorch >= 2.6 and build ``areno_accel``.
   * - Linux aarch64 / Grace-Blackwell
     - Supported
     - Install a matching ``aarch64`` CUDA PyTorch build first, then build AReno with ``--no-build-isolation``.
   * - Windows WSL2 + NVIDIA GPU
     - Supported
     - Follow the Linux install path inside WSL2. Native Windows is not supported.
   * - macOS Apple Silicon
     - Metadata/docs only
     - Use ``ARENO_BUILD_EXT=0`` for docs or packaging checks. Training/serving is not supported.
   * - CPU-only environments
     - Metadata/docs/tests only
     - CPU-only PyTorch can run lightweight docs/tests, but cannot train or serve AReno models.

Docker
------

Docker is the setup escape hatch when you want to verify AReno before
debugging local Python, PyTorch, or CUDA build state. Build the CUDA runtime
image from the repository root, then run the same readiness check used by local
installs:

.. code-block:: bash

   docker build -t areno .
   docker run --gpus all --rm -it areno areno check

Use ``--build-arg PIP_INDEX_URL=...`` if your environment requires a package
mirror.

If you need local project files, model files, or a Hugging Face cache inside
the container, mount them explicitly:

.. code-block:: bash

   docker run --gpus all --rm -it \
     -v $PWD:/workspace \
     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
     areno \
     areno check

Host checklist:

.. code-block:: bash

   nvidia-smi
   docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   docker run --gpus all --rm areno areno check

Docker gives you a known-good Python/PyTorch/CUDA user-space environment. It
does not fix host-side requirements: the host still needs a working NVIDIA
driver, NVIDIA Container Toolkit support for ``--gpus all``, and a driver new
enough for the container CUDA runtime. Model downloads, Hugging Face tokens,
cache paths, network access, disk space, and multi-node or custom networking
remain user environment concerns and are outside the first Docker setup path.

Python distributions
--------------------

By default, package builds compile the ``areno_accel`` CUDA extension. Run the
build in an environment with PyTorch extension tooling and ``CUDA_HOME``:

.. code-block:: bash

   python -m pip install build
   python -m build --no-isolation

The generated artifacts are written to ``dist/``. That directory is ignored by
git.

For metadata or pure-Python packaging checks that should not require local
PyTorch/CUDA, explicitly skip extension compilation:

.. code-block:: bash

   ARENO_BUILD_EXT=0 python -m build --no-isolation

Installation
------------

Install a CUDA-enabled PyTorch environment first. Then install the project from
the repository root:

.. code-block:: bash

   pip install psutil
   pip install flash-linear-attention
   pip install -e . --no-build-isolation

.. note::

   ``--no-build-isolation`` uses the packages already installed in your
   environment. Install ``psutil`` first because PyTorch's CUDA extension
   builder imports it while sizing parallel compile jobs. CUDA and PyTorch
   must be ABI compatible. The editable install builds the ``areno_accel``
   CUDA extension used by local kernels.
   Install ``flash-attn`` before AReno only if you use the default
   ``--attn-backend flash`` high-throughput path. ``flash-attn`` is optional
   when running with ``--attn-backend native``; AReno automatically falls back
   to native attention on flash-attn-unsupported GPUs such as Tesla T4 and
   warns that native attention is a slower compatibility path. If building
   ``flash-attn`` from source is too slow for your environment, install a
   pre-built wheel from the
   `flash-attention releases <https://github.com/Dao-AILab/flash-attention/releases>`_
   that matches your Python, PyTorch, CUDA, and platform.
   When ``TORCH_CUDA_ARCH_LIST`` is not set, AReno targets the visible GPU
   architectures. Set it explicitly when cross-building or narrowing the build
   target. Common values include ``9.0`` for H100/H200, ``8.0`` for A100, and
   ``8.9`` for L40/RTX 4090:

   .. code-block:: bash

      TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=64 pip install -e . --no-build-isolation

   For iterative CUDA work, configure ``ccache`` with ``CC="ccache gcc"`` and
   ``CXX="ccache g++"`` before rebuilding.

Post-install checklist
----------------------

Run the readiness check after every fresh install:

.. code-block:: bash

   areno check

For setup reports, also collect a machine-readable environment bundle:

.. code-block:: bash

   areno env --json

``areno check`` reports common build-time and runtime setup problems with next
steps: missing or CPU-only PyTorch, unsupported PyTorch versions, missing
``CUDA_HOME`` or ``nvcc``, missing build-time dependencies such as ``psutil``,
unsupported platforms, and ``ARENO_BUILD_EXT=0`` installs that try to train or
serve without the compiled ``areno_accel`` extension.
