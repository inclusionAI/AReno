:orphan:

Reward profiling and slow-sample detection
===========================================

AReno can measure the time each reward function call takes, flag samples
that exceed a configurable threshold, and enforce an optional per-batch
wall-clock timeout.  This helps identify slow reward hooks and samples
without changing the training loop.

CLI options
-----------

All three options are off by default, so existing training runs are
unaffected.

``--reward-profile``
    Enable per-sample reward timing and slow-sample detection.

``--reward-slow-threshold-s <seconds>``
    Flag reward samples whose execution time is greater than or equal to
    this many seconds.  When omitted, no slow-sample flagging is
    performed.

``--reward-batch-timeout-s <seconds>``
    Per-batch wall-clock budget for all reward calls.  If exceeded, a
    ``RewardTimeoutError`` is raised immediately with the hook name
    (``reward_fn``), the current sample's ``prompt_index`` and
    ``sample_index``, and the elapsed time.  When omitted, no timeout is
    enforced.

Example
-------

.. code-block:: bash

   areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
     --reward-fn-path examples/math/math_verify_reward.py --algo gspo \
     --tp-size 4 --reward-profile \
     --reward-slow-threshold-s 0.5 --reward-batch-timeout-s 10.0

Observable output
-----------------

TensorBoard scalars
    When reward profiling is enabled, the following scalars are written
    under the ``timing/`` namespace:

    - ``timing/reward_profile_slow_count`` — number of samples that
      exceeded the slow threshold in the current step.
    - ``timing/reward_profile_max_s`` — maximum per-sample reward time
      in the current step.

JSONL artifact
    Per-sample timing records are written to
    ``reward_profile.{pid}.jsonl`` in the metrics log directory.  Each
    line contains only:

    - ``prompt_index``
    - ``sample_index``
    - ``duration_s``
    - ``timed_out``

    Prompt and completion text are never logged.

Console logs
    When slow samples are detected, a ``WARNING``-level log line is
    emitted:

    .. code-block:: text

       reward_timing hook=reward_fn n=8 slow_count=1 slowest_sample_idx=3 slowest_time_s=0.5234

Limitations
-----------

- **Timeout is a soft enforcement.** The reward function runs in a
  background thread via ``ThreadPoolExecutor``.  On timeout, the main
  thread raises immediately, but the background thread cannot be
  killed — it will finish naturally and its resources are released when
  the process exits.  This is sufficient for pure-Python reward
  functions; reward functions that hold locks or network connections
  may need their own timeout handling.

- **Thread safety.** Because reward calls execute in a
  ``ThreadPoolExecutor`` thread, user ``reward_fn`` implementations must
  be thread-safe.

- **No hook name configuration.** AReno has a single reward function;
  the hook name is the internal constant ``reward_fn`` and cannot be
  overridden.

Copyable example
----------------

A self-contained, deterministic demo that runs without GPU, network, or
sandbox is provided at ``examples/reward_profiling/demo_cpu.py``:

.. code-block:: bash

   python3 examples/reward_profiling/demo_cpu.py