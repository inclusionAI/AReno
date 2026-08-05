:orphan:

areno_accel issues
==================

``areno_accel`` contains AReno-owned CUDA extension paths. Build failures
usually point to compiler, CUDA, PyTorch, or architecture mismatch.

Rerun the installer to rebuild the extension against the selected environment:

.. code-block:: bash

   bash scripts/install.sh

For a targeted developer build, pass build overrides to the same installer:

.. code-block:: bash

   TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=64 bash scripts/install.sh
