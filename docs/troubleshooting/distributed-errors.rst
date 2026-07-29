:orphan:

Distributed worker errors
=========================

In distributed training runs (``tp_size > 1`` or ``dp_size > 1``), a failure
on one worker rank typically triggers a cascade of NCCL collective timeouts
and communication errors on the remaining ranks.  These secondary errors can
flood the logs and obscure the **root cause**—the first worker's original
application error.

AReno provides an opt-in feature, ``preserve_first_error``, that captures the
earliest causal failure and presents subsequent errors as a grouped secondary
summary while retaining raw logs.

Enabling the feature
--------------------

The feature is **off by default** to preserve backward-compatible behavior.
You can enable it through the SDK or the CLI.

**SDK:**

.. code-block:: python

   from areno.api.config import ArenoConfig

   config = ArenoConfig(
       model_path="/path/to/model",
       runtime={"preserve_first_error": True},
   )

**CLI:**

.. code-block:: bash

   areno train --ckpt /path/to/model --dataset-path /data --algo grpo \
       --preserve-first-error

Output format
-------------

When the feature is enabled and a worker fails, AReno raises a
``FirstWorkerError`` exception whose message has two sections:

.. code-block:: text

   === First Worker Failure (root cause) ===
   rank=2  stage=TRAIN  timestamp=1234567.890123
   Traceback (most recent call last):
     File "/path/to/worker.py", line 123, in handle
       ...
   RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.

   === Secondary Errors (3 ranks, shown as summary) ===
   [rank=0  stage=TRAIN] RuntimeError: NCCL communicator was aborted
   [rank=1  stage=TRAIN] RuntimeError: NCCL communicator was aborted
   [rank=3  stage=TRAIN] RuntimeError: worker watchdog timeout

* **First Worker Failure**: the earliest error with full traceback, rank,
  stage (operation name), and coordinator-side timestamp.  This is the root
  cause to investigate first.
* **Secondary Errors**: subsequent failures shown as one-line summaries
  (exception class and message only).  If there are more than 5 secondary
  errors, only the first 5 are displayed and the total count is noted.

A WARNING-level log line is also emitted:

.. code-block:: text

   first worker failure: rank=2 stage=TRAIN | 3 secondary error(s) recorded

Limitations
-----------

* The feature adds a small latency to failure propagation because the
  coordinator waits for all ranks to report (or be detected as dead) before
  raising.  In the default (disabled) mode, the first error is raised
  immediately as before.
* Timestamps are coordinator-side ``time.monotonic()`` values (relative),
  not wall-clock times, because distributed clocks may be unsynchronized.

See also :doc:`oom-timeout` for memory-related failures and
:doc:`/cli/observability` for interpreting training metrics and logs.