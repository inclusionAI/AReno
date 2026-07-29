:orphan:

Quarantine of failing samples
=============================

When individual training samples fail during reward scoring, agent execution,
or generation, AReno can isolate them to a local file instead of aborting the
entire training step. This is useful for reproduction and debugging.

Enable quarantine
-----------------

Add ``--quarantine-enabled`` to the train command:

.. code-block:: bash

   areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
     --reward-fn-path examples/math/math_verify_reward.py --algo gspo --tp-size 4 \
     --quarantine-enabled

Options
-------

All options have safe defaults and are disabled by default.

* ``--quarantine-enabled`` — Write failing samples to ``quarantine.<pid>.jsonl``
  in the metrics log directory.
* ``--quarantine-max-entries N`` (default 200) — Freeze the file after N entries.
* ``--quarantine-max-file-bytes N`` (default 10485760) — Freeze the file after
  it exceeds N bytes.
* ``--quarantine-failure-rate-threshold F`` (default 0.5) — Stop training and
  propagate the original error when the failure rate over the last
  ``--quarantine-failure-rate-window`` samples exceeds this fraction.
* ``--quarantine-failure-rate-window N`` (default 20) — Sliding window size for
  failure-rate detection.

Output format
-------------

Each line in ``quarantine.<pid>.jsonl`` is a JSON object:

.. code-block:: json

   {
     "phase": "reward",
     "reason": "ValueError: could not parse answer",
     "pid": 12345,
     "step": null,
     "epoch": null,
     "prompt_index": 0,
     "sample_index": 2,
     "prompt_len": 280,
     "completion_len": 1670,
     "prompt_hash": "01c3012f7cb6942b",
     "completion_hash": "af7479e170ffa91",
     "timestamp": "2026-07-28T08:21:43+00:00"
   }

Fields:

* ``phase`` — Where the failure occurred: ``reward``, ``agent``, or ``generation``.
* ``reason`` — Exception type and message.
* ``prompt_index`` / ``sample_index`` — Position within the batch.
* ``prompt_len`` / ``completion_len`` — Token lengths (not content).
* ``prompt_hash`` / ``completion_hash`` — Truncated SHA-256 hashes (16 hex chars).
  The raw prompt and completion text are **never** stored.

Limitations
-----------

* Quarantine is for reproduction, not data sharing. Sensitive content is
  redacted to hashes.
* Each rank writes its own file (``quarantine.<pid>.jsonl``).
* When the failure-rate threshold is exceeded, the original error propagates
  and training stops. Quarantine does not mask systemic failures.
* SFT and DPO trainers do not use quarantine because they have no per-sample
  reward or rollout stage.