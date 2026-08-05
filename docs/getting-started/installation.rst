Installation
============

The installer is the single entry point for local AReno setup. Users do not
need to choose PyTorch packages, set CUDA build variables, or order Python
dependencies before running it.

Install AReno
-------------

Clone the repository and run one command:

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno
   bash scripts/install.sh

Before changing Python, the installer checks required system tools, rejects
WSL1, and verifies that ``nvidia-smi`` can see a GPU. It then uses an active
virtualenv or conda environment when available, reuses the repository's
``.venv`` when it is ready, or creates ``.venv`` automatically. If an IDE does
not expose environment activation metadata, the installer detects and reuses a
Python interpreter that already provides PyTorch instead of creating an empty
``.venv``. Finally, it checks for CUDA-enabled PyTorch 2.6 or newer, detects
CUDA build support, installs AReno's remaining dependencies, selects the
attention setup, builds the CUDA extension, and runs ``areno check``. The
installer never installs or upgrades PyTorch because the correct build depends
on the machine's CUDA platform. If PyTorch is missing or incompatible, it stops
with guidance for the selected Python environment. Other packages that already
satisfy AReno's requirements are reused; only missing or incompatible packages
are installed or updated.

Successful installation ends with ``AReno is ready`` and the exact command to
start using AReno. If installation stops, the same script reports the failed
stage, explains the immediate reason, prints targeted suggestions, and
preserves complete command output in the user state directory, usually
``~/.local/state/areno/install.log``.

To preview the plan without changing the environment:

.. code-block:: bash

   bash scripts/install.sh --dry-run

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
     - Start from a compatible CUDA-enabled ``aarch64`` PyTorch environment, such as a current NVIDIA NGC PyTorch development container. The installer validates it and builds AReno against it.
   * - Windows WSL2 + NVIDIA GPU
     - Supported
     - Follow the Linux install path inside WSL2. Native Windows is not supported.
   * - macOS Apple Silicon
     - Not supported
     - The installer requires Linux with NVIDIA CUDA.
   * - CPU-only environments
     - Not supported
     - AReno training and serving require an NVIDIA GPU and CUDA-enabled PyTorch.

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

Post-install checklist
----------------------

The installer runs the readiness check automatically. You can rerun it at any
time:

.. code-block:: bash

   areno check

For setup reports, also collect a machine-readable environment bundle:

.. code-block:: bash

   areno env --json

``areno check`` reports common build-time and runtime setup problems with next
steps: missing or CPU-only PyTorch, unsupported PyTorch versions, missing
``CUDA_HOME`` or ``nvcc``, missing build-time dependencies such as ``psutil``,
and unsupported platforms.
