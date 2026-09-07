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

``ARENO_LOG_LEVEL``
   Control the log level of the ``areno`` logger. Default is ``INFO``.
   Set to ``DEBUG`` for per-request decode progress and fine-grained
   engine diagnostics. Accepts standard Python log level names
   (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``). Resolved once at
   startup by ``areno.engine.log``.

``ARENO_LOG_COMPLETIONS``
   Number of decoded rollout samples to persist per training step (default
   ``1``). Samples are written to ``rollout_samples.{pid}.jsonl`` in the
   metrics log directory. Set to ``0`` to disable. For agentic rollouts
   the samples include rendered prompts, message lists, tool calls, and
   loss-mask summaries.

For environment inspection, use :doc:`/cli/diagnostics`.
