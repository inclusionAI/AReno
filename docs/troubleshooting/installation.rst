:orphan:

Installation issues
===================

Most installation issues come from mismatched Python, PyTorch, CUDA, compiler,
or optional acceleration packages.

Start by rerunning the installer. It preserves the full command output in the
user state directory, usually ``~/.local/state/areno/install.log``, and ends
failures with the stage, reason, and suggested actions:

.. code-block:: bash

   bash scripts/install.sh

Failures before Python setup usually mean a required system tool is missing,
WSL1 was detected, or ``nvidia-smi`` cannot see a usable GPU. Correct the
reported host issue and rerun the same command.

PyTorch is checked but never installed or upgraded automatically. If the
installer reports missing, outdated, CPU-only, or unusable PyTorch, prepare
CUDA-enabled PyTorch 2.6 or newer in the Python environment named in the error,
then rerun the installer. Use the `official PyTorch selector
<https://pytorch.org/get-started/locally/>`_ for standard Linux GPU systems.
For DGX Spark, start from NVIDIA's provided Jupyter environment or follow the
`DGX Spark PyTorch playbook
<https://build.nvidia.com/spark/pytorch-fine-tune/instructions>`_ to select a
current NGC PyTorch development container.

Some IDE terminals provide a preconfigured Python environment without setting
``VIRTUAL_ENV`` or ``CONDA_PREFIX``. The installer detects a PyTorch-providing
Python on ``PATH`` and reuses it automatically. To select an interpreter
explicitly, rerun with ``PYTHON=/path/to/python bash scripts/install.sh``.

If installation completes but a later runtime check fails, collect:

.. code-block:: bash

   areno check
   areno env --json

Use Docker when the installer reports a host CUDA or system-toolchain problem
and a clean reproduction path is preferable.

See :doc:`/getting-started/installation` for supported setup paths.
