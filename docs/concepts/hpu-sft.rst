HPU backend integration
=======================

HPU reuses the CUDA backend workflows. ``api/backend/hpu`` contains only
``__init__.py`` and ``backend.py``. ``HpuBackend`` inherits ``CudaBackend``;
the shared ``ArenoEngine`` selects ``HpuWorker`` through ``RuntimeConfig.device_type``.
HPU specializes device initialization, memory probes, and execution boundaries while retaining
the existing engine managers and process protocol. The CUDA and MLX backend
directories retain their original files and implementations.

Shared implementation
---------------------

* Configuration: ``HpuConfig`` is an alias of ``CudaConfig``.
* Training, generation and losses: the existing CUDA backend and engine managers.
* Optimizers: the existing FP32-master, 8-bit and 4-bit classes in ``engine/optim``.
* Checkpoint: the existing engine model checkpoint format and model adapters.
  The previous HPU-specific optimizer-resume format has been removed.
* Serving: the existing serving adapter with ``device_type="hpu"``.
* Multiprocessing: the existing ``TPCluster`` and worker command protocol.

Device differences
------------------

The HPU process lifecycle registers HCCL, selects ``torch.device("hpu")``,
and passes that device and backend into the shared distributed initializer.
The same code creates TP/DP groups and train/rollout policy-relay groups,
including partitions with different TP sizes. HPU execution
uses ``mark_step`` boundaries. CUDA graph capture and CUDA compiler settings
for model compilation are disabled for HPU workers.

Training and serving use one coordinator with engine-owned worker processes,
matching CUDA. ``world_size = tp_size * dp_size``. The previous torchrun-specific
HPU CLI and separate serving process pool have been removed.

Status and verification
-----------------------

The Python accel wrappers retain their original module paths and autograd
contracts. Each call selects the native extension using its tensor device:
``_areno_accel`` for CUDA and ``_areno_accel_hpu`` for HPU. There are no separate
``accel/cuda`` or ``accel/hpu`` Python implementations, and model files do not
need device-specific imports. Shared metadata in ``accel.ops`` remains usable
without importing Triton; CUDA Triton kernels load only when called.

Native source implementations live in ``accel/csrc/hpu`` with TPC-C kernels,
graph-compiler glue, and C++ PyTorch bindings. Their public entry points match
the CUDA extension. Implemented source families are:

* SiLU, sigmoid, softplus, SiLU-and-multiply, and GELU-tanh-and-multiply,
  including backward.
* RMSNorm with mandatory or optional scale, and RMSNorm with a SiLU gate,
  including input, gate, and weight gradients.
* Dense linear forward and backward through the Synapse MME ``gemm`` node,
  with TPC bias addition and bias-gradient reduction. The MME CustomOp route
  still needs validation against the installed bridge.
* FP32-state AdamW and compact BF16/FP32-master AdamW, including packed
  rounding carries and state offsets.
* Block-wise 8-bit and packed 4-bit AdamW. The 4-bit matrix path includes
  factored row/column statistics and the factored update. Invalid gradient
  squares are excluded from statistics and set the validity flag, as in CUDA.
  Updates preserve an invalid block's weights and quantized state, including
  NaN storage bits and unused nibbles. Existing engine code still owns factor
  finalization and distributed reduction.

* Embedding gather and atomic gradient accumulation, with TP vocabulary ranges
  and actual int64 index storage.
* Dense, packed/GQA, and paged causal attention. Dense and packed attention
  include backward; paged decode updates caller-owned KV caches.
* Grouped linear forward/backward through the existing native MME linear path.
* Dense/packed causal convolution with SiLU and backward, plus history-based decode.
* Softmax top-k with backward, grouped sigmoid routing, MoE alignment,
  permute/unpermute/gather and route-weight gradients.
* Fused expert inference using native routing, MME, activation, weighted down
  projection, and reduction. Weighted down projections round before reduction,
  matching CUDA's operation order. Both SiLU and GELU-tanh gates are exposed.
* Grouped RMSNorm with sigmoid gate, matching the existing forward-only entry point.
* KDA chunk forward/backward, including raw gate, A_log, dt_bias, beta, initial
  state gradients, packed sequences and repeated state indices. Recurrent KDA
  updates the caller's state in place through a native state-copy kernel.
* Segmented linear attention prefill/decode, MTP state snapshots, and tree-mask
  verification. MTP and tree verification preserve the persistent state, matching
  the CUDA dispatcher; MTP writes the separate snapshot buffer.

All 49 exports in the CUDA C++ extension have matching HPU binding source.
The additional five entry points used by the shared KDA/Triton wrappers are
also present. This inventory is source coverage, not device validation.

Activations, normalization, and linear support FP32/BF16/FP16 storage;
optimizers support FP32/BF16 weights and gradients, including mixed dtypes.
The build targets Gaudi 2 or Gaudi 3 and embeds 160 TPC binaries. Arithmetic
uses native kernels; there is no Torch composite optimizer implementation
in the HPU backend.

This is an implementation awaiting Gaudi validation, **not working end-to-end
HPU training or serving support**. Known gaps are:

* Bailing/Bailing V3 directly import FLA lightning attention. Qwen3.5 and
  MiniCPM call FLA convolution and gated-delta helpers through model wrappers.
  These paths bypass the shared accel device dispatch. Model files are unchanged,
  so those adapters remain rejected by the HPU worker. Completing them requires
  moving those operator entry points into shared accel and adjusting the model
  imports, which needs resolution of the existing no-model-changes constraint.
* Native KDA and segmented attention currently require key/head dimension at
  most 512. KDA saves FP32 per-token states for backward; memory use is
  proportional to tokens * heads * key_dim * value_dim. It is not yet a
  memory-efficient chunk algorithm. CUDA model compilation/graph capture remains
  disabled on HPU; there is no HPU graph-replay implementation yet.
* Float64 storage and Gaudi 1 are not implemented.
* The CustomOp interface allocates outputs. Output-buffer and optimizer entry
  points copy the native result back to the caller's tensors. This preserves
  aliases but adds temporary storage and copies. Compact-master and quantized
  optimizer kernels, routing and recurrent kernels currently use scalar loops
  within programs and need device profiling/vectorization. Paged decode creates
  full temporary KV caches before copying them back. Its ``num_splits`` argument
  is accepted but does not yet split computation. Fused MoE uses additional FP32
  casts for the down MME accumulator. Performance and memory parity with CUDA
  have not been established.
* TPC optimizer indexing is limited to signed 32-bit sizes. Factored matrix
  dimensions and their product must fit this range; unsupported sizes fail
  explicitly at the C++ boundary.
* No HCCL, TP/DP, checkpoint, training, or serving integration has run on HPU.

The development host has no Gaudi device, SynapseAI bridge, or TPC compiler.
CPU tests check activation formulas and interpret the actual TPC normalization,
bias, optimizer, attention, convolution, routing, MoE and recurrent source
through host primitives. Optimizer cases cover
multiple updates, mixed dtypes, quantization tails and ties, compact carries,
matrix-shard offsets, accumulated statistics, and invalid-block preservation.
Device dispatch and shared workflows have separate regression coverage.
Host C++ syntax checks use published bridge/SDK headers; they do not link the
bridge or compile TPC code.
These checks do not establish HPU numerical accuracy or performance.

Building and validating on Gaudi
--------------------------------

Use a Linux Gaudi environment with matching SynapseAI, its PyTorch bridge,
and TPC SDK. Keep the bridge-provided PyTorch installation. From the repository
root, install the common Python dependencies and build the separate HPU extension:

.. code-block:: bash

   python -m pip install -r requirements/hpu.txt
   ARENO_BUILD_EXT=0 python -m pip install -e . --no-deps --no-build-isolation
   export PT_HPU_LAZY_MODE=1
   export PT_ENABLE_INT64_SUPPORT=1
   export ARENO_HPU_ARCH=gaudi2
   python areno/accel/csrc/hpu/setup.py build_ext --inplace
   python -m pytest -q tests/test_hpu_activation.py tests/test_hpu_dense.py tests/test_hpu_optimizer.py tests/test_hpu_operator_integration.py

Run model acceptance in a fresh process, separately from the kernel tests,
using an existing local model checkpoint. This exercises SFT with all three
optimizer modes, rollout, checkpoint export and reload. To exercise separate
train/rollout workers, additionally set ``ARENO_HPU_TEST_ROLLOUT_DEVICES`` and
``ARENO_HPU_TEST_ROLLOUT_TP_SIZE`` to the desired layout.

.. code-block:: bash

   ARENO_HPU_TEST_MODEL=/path/to/local/model \
   ARENO_HPU_TEST_WORLD_SIZE=4 ARENO_HPU_TEST_TP_SIZE=2 \
   python -m pytest -q tests/test_hpu_end_to_end.py

Set ``ARENO_HPU_ARCH=gaudi3`` for Gaudi 3. ``PT_HPU_LAZY_MODE`` must be explicitly
set to ``0`` (eager) or ``1`` (lazy) at build and execution, with the same value
for both. Rebuild when changing architecture or bridge mode. The original CUDA
``setup.py`` remains unchanged.

``TPC_COMPILER`` may override ``tpc-clang``. ``TPC_INCLUDE_DIR`` may override
``/usr/lib/habanatools/include``; it must contain ``gc_interface.h`` and
``tpc_kernel_lib_interface.h``.

The worker registers the built TPC library in ``GC_KERNEL_PATH`` before
initializing the bridge. Direct accel users must call
``areno.accel._extension.configure_hpu_kernel_library()`` before initializing
HPU. Set ``PT_ENABLE_INT64_SUPPORT=1`` before importing the bridge or creating
HPU tensors; otherwise the bridge can silently store ``torch.long`` in int32
memory. AReno's loader also sets this flag when absent and rejects an explicit
disabled value. Existing compiler libraries are retained, including the standard Gaudi
library when no explicit list is configured. See Intel's
`Multiple Kernel Libraries <https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/Multiple_Kernels_Library.html>`_
and `PyTorch CustomOp API <https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_CustomOp_API/page_index.html>`_.

Hardware tests check forward/backward, storage dtypes, vector tails, transposed
inputs, output aliases/strides, empty tensors, and optimizer state updates.
Optimizer tests also cover packed-buffer offsets and compact-master carries.
Tests skip without a bridge. Once a bridge is installed, missing hardware or
native libraries are failures, so an incomplete installation cannot appear to pass.
