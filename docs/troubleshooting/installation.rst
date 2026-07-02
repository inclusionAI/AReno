:orphan:

Installation issues
===================

Most installation issues come from mismatched Python, PyTorch, CUDA, compiler,
or optional acceleration packages.

First checks:

.. code-block:: bash

   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   areno check
   areno env --json

Common fixes:

* Install AReno inside an environment that already has the intended PyTorch
  and CUDA stack.
* Use ``pip install -e . --no-build-isolation`` so extension builds use the
  installed PyTorch.
* Set ``ARENO_BUILD_EXT=0`` only for docs, metadata, or CPU-only package
  checks.
* Use Docker when you need a cleaner reproduction path.

See :doc:`/getting-started/installation` for supported setup paths.
