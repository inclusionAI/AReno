Diagnostics CLI reference
=========================

``areno env`` and ``areno check`` help diagnose setup problems before a user
hits low-level Python, CUDA, or PyTorch errors.

``areno env`` is a descriptive support report. It does not initialize the AReno
engine or load model weights. Use it when collecting information for an issue.

.. code-block:: bash

   areno env

For machine-readable issue reports:

.. code-block:: bash

   areno env --json

The report includes:

* AReno version
* Python version and executable
* OS, platform, and architecture
* PyTorch version, CUDA build, CUDA runtime, and CUDA availability
* CUDA driver information from ``nvidia-smi`` when available
* visible GPU count, names, and compute capability
* ``CUDA_HOME`` and inferred CUDA toolkit location
* ``nvcc`` path and version
* ``flash-attn`` import status and version
* ``flash-linear-attention`` import status and version
* ``areno_accel`` import status
* selected environment variables such as ``MAX_JOBS``,
  ``CUDA_VISIBLE_DEVICES``, and ``TORCH_CUDA_ARCH_LIST``

areno check
-----------

``areno check`` validates whether the machine is ready to run AReno training
and serving. It classifies each check as ``OK``, ``WARN``, or ``FAIL`` and
prints concrete next steps for failures.

.. code-block:: bash

   areno check

Example output:

.. code-block:: text

   AReno check: not ready

   OK   Python >= 3.10
        found 3.11.8
   OK   PyTorch CUDA build
        torch.version.cuda=12.4
   OK   CUDA_HOME
        not set (not required for runtime; areno_accel imports)

``CUDA_HOME`` and ``nvcc`` are only warnings when AReno needs to build its CUDA
extension. If the installed ``areno_accel`` extension imports successfully,
they are not required for runtime readiness.

Checks include:

* Python version
* supported platform
* PyTorch import and version
* PyTorch CUDA build
* ``torch.cuda.is_available()``
* NVIDIA GPU visibility
* ``CUDA_HOME`` and ``nvcc``
* optional runtime dependency imports
* ``areno_accel`` import
* writable cache/log locations

``WARN`` items usually indicate degraded or incomplete setup. ``FAIL`` items
mean AReno is not ready to run the CUDA training/inference engine.
