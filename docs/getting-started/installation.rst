Installation
============

AReno has native CUDA and MLX installation paths. Linux and WSL2 use the CUDA
installer; Apple Silicon uses a normal pip source install with platform-marked
MLX dependencies.

The :doc:`backends` guide compares both native runtimes. Its platform pages
describe the complete, parallel train/serve setup:

* :doc:`cuda` for Linux, WSL2, NVIDIA CUDA, distributed topology, and CUDA
  checkpoints.
* :doc:`mlx` for Apple Silicon, unified memory, MLX checkpoints, and the
  single-process runtime.

Install CUDA AReno
------------------

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

Continue with :doc:`cuda` for CUDA training, serving, memory controls,
checkpoints, model support, and SDK configuration.

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
     - Supported
     - Use native ``arm64`` Python and the MLX pip installation below. The CUDA installer does not apply.
   * - CPU-only environments
     - Not supported
     - Training and serving require either NVIDIA CUDA or Apple Silicon MLX.

Install on Apple Silicon
------------------------

The repository installer currently prepares the CUDA toolchain, so do not run
``scripts/install.sh`` on macOS. Use a native ``arm64`` virtual environment:

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

The project metadata installs MLX dependencies only on Apple Silicon and CUDA
dependencies only on Linux. Verify that Python is not running through Rosetta:

.. code-block:: bash

   python -c "import platform; from areno.api import DefaultBackend; print(platform.machine(), DefaultBackend)"

Expected output contains ``arm64`` and ``BackendType.MLX``. Continue with
:doc:`mlx` for training, serving, checkpoints, memory controls, and
multimodal support.

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

On CUDA, ``areno check`` reports common build-time and runtime setup problems
such as missing or CPU-only PyTorch, unsupported PyTorch versions, missing
``CUDA_HOME`` or ``nvcc``, and missing build dependencies. On Apple Silicon,
verify the backend and MLX device from the active environment before loading a
checkpoint.
