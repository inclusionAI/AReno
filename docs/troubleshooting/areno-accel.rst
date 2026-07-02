:orphan:

areno_accel issues
==================

``areno_accel`` contains AReno-owned CUDA extension paths. Build failures
usually point to compiler, CUDA, PyTorch, or architecture mismatch.

Useful setup commands:

.. code-block:: bash

   pip install psutil
   pip install -e . --no-build-isolation

For targeted extension builds, set the architecture list explicitly:

.. code-block:: bash

   TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=64 pip install -e . --no-build-isolation

For metadata-only checks on a CPU machine:

.. code-block:: bash

   ARENO_BUILD_EXT=0 pip install -e . --no-build-isolation

Do not treat a metadata-only install as a runnable training environment.
