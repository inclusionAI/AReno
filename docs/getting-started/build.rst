Build and installation
======================

This page covers the setup paths used by contributors and local operators:
Docker images, editable installs, source/wheel distributions, and local
installation.

Docker
------

Build the CUDA runtime image from the repository root:

.. code-block:: bash

   docker build -t areno .

Use ``--build-arg PIP_INDEX_URL=...`` if your environment requires a package
mirror.

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
