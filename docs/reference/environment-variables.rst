:orphan:

Environment variables
=====================

AReno uses a small number of environment variables for build and runtime
control.

``ARENO_BUILD_EXT``
   Set ``ARENO_BUILD_EXT=0`` to skip CUDA extension compilation for
   metadata-only installs, docs builds, or CPU-only packaging checks.

``TORCH_CUDA_ARCH_LIST``
   Set this when narrowing CUDA extension builds to a target GPU architecture,
   for example ``TORCH_CUDA_ARCH_LIST="9.0"`` for H100/H200-only builds.

``MAX_JOBS``
   Set this to control parallel compilation jobs during editable installs.

For environment inspection, use :doc:`/cli/diagnostics`.
