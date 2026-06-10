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

   pip install -e . --no-build-isolation
   pip install flash-attn flash-linear-attention

.. note::

   ``flash-attn``, CUDA, and PyTorch must be ABI compatible. The editable
   install builds the ``areno_accel`` CUDA extension used by local kernels.
