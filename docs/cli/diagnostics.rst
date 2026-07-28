:orphan:

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

Host resource preflight
-----------------------

Before ``areno train`` and ``areno serve`` spawn worker ranks, AReno reads
process-level limits and compares them with a documented demand estimate for
the requested ``world_size``/``tp_size``. The preflight never changes the host;
it only reports and, optionally, blocks.

The ``--resource-check`` option (on both ``train`` and ``serve``) selects
behavior:

* ``warn`` (default) -- emit a stderr diagnostic only when a probed limit is
  below demand, then continue. Existing runs are unaffected.
* ``block`` -- raise a ``UsageError`` and abort before any worker starts if a
  probed limit is below demand.
* ``skip`` -- disable the preflight entirely.

Probed limits:

* file descriptors via ``RLIMIT_NOFILE``
* process count via ``RLIMIT_NPROC``
* shared-memory ceiling via ``/proc/sys/kernel/shmmax`` (Linux)

Demand estimate (conservative upper bound, see
``areno.cli.diagnostics.estimate_resource_demand``):

* file descriptors: ``64 * world_size + world_size * (world_size - 1)`` --
  per-worker base plus one socket per cross-rank peer for the
  NCCL/tensor-parallel mesh.
* processes: ``world_size + 1`` worker ranks plus the driver.
* shared memory: ``1 GiB * tp_size`` for NCCL/CUDA IPC per tensor-parallel
  group.

Each probe produces one of three severities, mirroring ``areno check``:

* ``OK`` -- observed limit meets demand (an unbounded ``RLIM_INFINITY`` limit
  counts as meeting demand).
* ``WARN`` -- the probe is unavailable on this platform (for example
  ``/proc/sys/kernel/shmmax`` is Linux-only, so macOS/Windows report this). A
  ``WARN`` never blocks, even under ``--resource-check block``.
* ``FAIL`` -- the observed limit is below demand; the result carries the exact
  ``observed``/``required``/``delta`` values and a remediation hint such as
  ``ulimit -n 65536`` or ``sudo sysctl -w kernel.shmmax=<required>``.

Because a probe may be unavailable per platform, a run is only blocked when a
limit was actually observed and is below demand -- the preflight degrades
cleanly on platforms without one of the probes.

Minimal example (success path, limits sufficient):

.. code-block:: bash

   areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
     --reward-fn-path examples/math/math_verify_reward.py --algo gspo \
     --tp-size 4 --world-size 8

Under the default ``warn`` policy this prints nothing to stdout when limits are
sufficient; a failing probe prints to stderr, for example:

.. code-block:: text

   Host resource preflight:
     FAIL file descriptors (RLIMIT_NOFILE)
          observed=1024 required=568 delta=-456
          -> raise the soft limit before launching workers, e.g. `ulimit -n 65536`

Boundary/invalid input: with ``--resource-check block`` and a below-demand
``RLIMIT_NOFILE``, the run aborts before worker initialization with a
``UsageError`` naming the failing probe and the exact observed/required values.
Re-run with ``--resource-check warn`` to proceed despite the warning, or raise
the limit first.
