:orphan:

Environment variables
=====================

AReno uses a small number of environment variables for build and runtime
control.

``ARENO_BUILD_EXT``
   Set ``ARENO_BUILD_EXT=0`` to skip CUDA/HPU extension compilation for
   metadata-only installs, docs builds, or CPU-only packaging checks.
   Dependency selection still detects the installed HPU bridge. Use
   ``--no-build-isolation`` so source installs see the existing PyTorch environment.

``TORCH_CUDA_ARCH_LIST``
   Set this when narrowing CUDA extension builds to a target GPU architecture,
   for example ``TORCH_CUDA_ARCH_LIST="9.0"`` for H100/H200-only builds.

``MAX_JOBS``
   Set this to control parallel compilation jobs during editable installs.

``PT_HPU_LAZY_MODE``
   Defaults to ``1`` when the HPU bridge is installed. An explicit ``0`` enables
   eager mode; use the same value for native extension build and execution.

``PT_ENABLE_INT64_SUPPORT``
   Defaults to ``1`` for HPU, as required by the native index kernels. Import
   ``areno`` before ``torch`` in Python scripts to apply the defaults before
   the bridge loads.

``ARENO_HPU_ARCH``
   Automatically detected from ``hl-smi`` for HPU builds. Set ``gaudi2`` or
   ``gaudi3`` to override detection or build without visible hardware.

For environment inspection, use :doc:`/cli/diagnostics`.
