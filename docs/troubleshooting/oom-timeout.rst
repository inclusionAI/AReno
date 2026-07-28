:orphan:

Training OOM and timeout
========================

Out-of-memory and timeout failures usually come from rollout volume, sequence
length, tensor parallelism, model size, or slow external agent work.

First reductions:

* Lower ``--batch-size``.
* Lower rollout or sequence length settings.
* Use a smaller checkpoint for the first reproduction.
* Reduce agent concurrency for tool or environment tasks.
* Confirm no unrelated GPU process is consuming memory.

For agentic tasks, distinguish model time from environment time. Tool calls,
tests, sleeps, browser work, and sandbox actions can dominate rollout wall
time before the model is called again.

See :doc:`/cli/observability` for timing and metric interpretation.

Disk space protection
---------------------

AReno can estimate disk usage before training starts and monitor free
space during execution to prevent silent data corruption from a full disk.

Preflight estimation:

.. code-block:: bash

   areno check --disk-budget \
     --save-path /tmp/areno-outputs \
     --metrics-log-dir /tmp/areno/metrics \
     --max-steps 1000 --save-interval 100

The estimator computes per-step overhead (TensorBoard scalars ~2 KB +
rollout samples JSONL ~10 KB per step) and multiplies by total steps.
Checkpoint sizes are **excluded by default**; pass ``--include-checkpoints``
to include them (the estimator reads ``*.safetensors`` file sizes from
the checkpoint directory).

Runtime monitoring:

.. code-block:: bash

   areno train --ckpt Qwen/Qwen3-0.6B ... --disk-monitor

Thresholds default to percentages of total disk capacity:

* **Warn** at 5% free (logs a one-time warning)
* **Stop** at 1% free (triggers a controlled shutdown via the normal
  ``max_steps_reached`` exit path)

Override with absolute values if needed:

.. code-block:: bash

   areno train ... --disk-monitor --disk-warn-gb 20 --disk-stop-gb 5

Both human-readable and JSON output are supported:

.. code-block:: bash

   areno check --disk-budget --save-path /tmp/outputs --disk-json

The ``--disk-monitor`` flag is disabled by default; existing behavior is
unchanged when it is not provided.
