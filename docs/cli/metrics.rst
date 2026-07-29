:orphan:

Metrics CLI reference
=====================

``areno metrics``

Query metric history from local run artifacts -- read the ``events.out.tfevents.*``
files a training run writes and summarize one metric into ``last`` / ``min`` /
``max`` / ``recent`` values / a compact ``trend``. The command is read-only: it
never writes, starts, or contacts training/serving processes. It is the
command-line counterpart of the dashboard's metric view and shares the same
read-side code, so the two never drift.

.. code-block:: bash

   areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
     --reward-fn-path examples/math/math_verify_reward.py --algo gspo \
     --metrics-log-dir /tmp/areno/tfevent

   areno metrics --metrics-dir /tmp/areno/tfevent --name rollout/rewards_mean

Output (text table with a hand-written UTF-8 sparkline trend):

.. code-block:: text

   metric   rollout/rewards_mean
   count    5
   last     1
   min      0.5
   max      1
   trend    ▁▃▅▇█
   recent   0.5, 0.625, 0.75, 0.875, 1

For structured output (piped into ``jq`` or other tools):

.. code-block:: bash

   areno metrics --metrics-dir /tmp/areno/tfevent \
     --name rollout/rewards_mean --json

.. code-block:: json

   {
     "count": 5,
     "last": 1.0,
     "max": 1.0,
     "min": 0.5,
     "name": "rollout/rewards_mean",
     "recent": [0.5, 0.625, 0.75, 0.875, 1.0],
     "trend": [0.0, 0.25, 0.5, 0.75, 1.0]
   }

areno metrics
-------------

Summarize one metric from local run artifacts.

Options:

``--metrics-dir TEXT``
   Directory holding the run's ``events.out.tfevents.*`` artifacts. Omit to use
   areno's default metrics log dir (the same ``--metrics-log-dir`` training
   writes to by default).

``--pid INTEGER``
   Filter event files by the ``pid`` suffix embedded in the filename
   (``events.out.tfevents.<host>.<pid>.*``). Omit to merge every run writing
   into the same directory.

``--name TEXT``
   Metric tag to summarize, for example ``rollout/rewards_mean``. Omit to list
   the available tags found in the directory. When the tag is not found, the
   command exits non-zero and prints the available tags so you can pick one.

``--limit INTEGER``
   Number of recent values to include in ``recent`` and ``trend``. Defaults to
   ``20``.

``--json``
   Emit a machine-readable JSON object instead of the text table. The ``trend``
   field is the normalized ``[0, 1]`` series; the text table renders the same
   series as a UTF-8 sparkline.

Input contract
--------------

* Source: TensorBoard scalar events (``events.out.tfevents.*``). Each tag is
  truncated to its last ``500`` points (``EventAccumulator`` with
  ``size_guidance={"scalars": 10000}``), ``NaN`` values are skipped, and
  ``(name, step, value)`` triples are de-duplicated.
* ``min`` / ``max`` are computed in a single streaming pass over the bounded
  series, so memory stays bounded regardless of run length.
* ``rollout_samples.*.jsonl`` files are **not** read as metrics; only
  ``events.out.tfevents.*`` is.

Limitations (first version)
---------------------------

* Only TensorBoard scalars are read. A jsonl fallback source and a friendly
  ``--run <id>`` selector backed by the dashboard jobs registry are planned as
  follow-ups; use ``--pid`` today to disambiguate runs sharing one directory.
* The command is read-only and CPU-only; it does not require CUDA.

Examples
--------

List the metrics available from the default log dir::

   areno metrics

Query one metric from a specific run identified by pid::

   areno metrics --metrics-dir /tmp/areno/tfevent --pid 12345 \
     --name rollout/rewards_mean --limit 50