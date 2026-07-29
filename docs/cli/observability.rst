:orphan:

Observability
=============

AReno exposes training state through plain logs, the per-step ``train_stats``
dictionary, TensorBoard scalar metrics, and optional agentic trajectory
diagnostics. The implementation is intentionally local and lightweight:
metrics are recorded by ``areno.api.metrics`` and the CLI exposes the output
directory through ``--metrics-log-dir``. AReno does not currently provide a
built-in wandb integration.

Console logs
------------

The trainer logs a compact lifecycle for every epoch and step. For rollout
algorithms, ``areno.api.trainers.policy_only.PolicyOnlyTrainer`` emits:

* ``epoch=<n> stage=epoch_start`` and ``epoch=<n> stage=epoch_end``.
* ``role=policy stage=rollout_start`` and ``stage=rollout_end`` around
  sampling or agentic execution.
* ``metric=reward_mean`` after rewards are computed.
* ``metric=rollout_logprob_mean`` when rollout logprobs are available.
* ``role=policy stage=train_start`` and ``stage=train_end`` around the
  optimizer step.
* ``train_stats={...}`` with the scalar dictionary returned by the backend and
  loss function.

The local AReno backend also logs step timing from
``areno.api.backend.areno.backend``:

.. code-block:: text

   time rollout=213.929721 train=17.327612 total=231.297132

Those numbers mean the backend measured roughly 214 seconds in rollout work,
17 seconds in the training step, and 231 seconds end-to-end for that step. The
same values are copied into ``train_stats`` as ``step_rollout_time_s``,
``step_train_time_s``, and ``step_e2e_time_s``.

During rollout, ``areno.engine.inference`` logs decode progress per data
parallel shard:

.. code-block:: text

   rollout decode progress: dp=0/2 active=6 cuda_graph=True tokens_per_second=60.8

``active`` is the number of currently scheduled decode requests on that shard
at the time of the progress log. In agentic runs it can dip to zero while the
external agent code is executing tools, tests, sleeps, background commands, or
other non-model work before asking the OpenAI-compatible proxy for the next
model response.

``train_stats``
---------------

``train_stats`` is the easiest place to inspect one completed optimizer step.
It is logged as a Python dictionary and also passed to TensorBoard by
``areno.api.metrics.MetricsRecorder`` when metrics recording is enabled.

Common fields include:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``loss`` / ``policy_loss`` / ``total_loss``
     - Loss scalars returned by the selected loss function.
   * - ``advantage_mean``
     - Mean policy advantage over trainable response tokens.
   * - ``response_len``
     - Mean response-token count for rows used in the step.
   * - ``rollout_logprobs_mean``
     - Mean rollout-time logprob over trainable response tokens.
   * - ``train_logprobs_mean``
     - Mean current-policy logprob during the training forward pass.
   * - ``logp_diff_mean`` / ``logp_abs_diff_mean``
     - Difference between rollout and train logprobs, useful for stale-policy
       or masking debugging.
   * - ``ratio_mean`` / ``ratio_std``
     - Policy-ratio diagnostics used by rollout policy losses such as GSPO,
       GRPO, and PPO.
   * - ``grad_norm``
     - Global gradient norm after clipping/accounting.
   * - ``grad_zero_ratio`` / ``grad_nonzero_ratio``
     - Fraction of parameter-gradient entries that are zero/non-zero.
   * - ``lr``
     - Current optimizer learning rate.
   * - ``step_rollout_time_s`` / ``step_train_time_s`` / ``step_e2e_time_s``
     - Per-step wall-clock timing from the local backend.

For example, if a debugging log shows:

.. code-block:: text

   metric=reward_mean value=0.125000
   train_stats={'loss': 0.0, 'advantage_mean': 0.0, 'response_len': 912.5625,
                'rollout_logprobs_mean': -0.17456013709306717,
                'train_logprobs_mean': -0.2500366196036339,
                'logp_diff_mean': 0.07547648251056671,
                'step_rollout_time_s': 213.92972119152546,
                'step_train_time_s': 17.327912725508213}

read it as: the sampled batch achieved low positive reward on average, the
training rows were long, rollout dominated wall time, and the current policy
assigned lower logprob than the rollout policy on average. ``loss`` can print
as ``0.0`` for some policy-gradient batches when normalized advantages cancel
in the scalar value at the current ratio, while gradients can still be non-zero
because the derivative depends on the logprob term.

TensorBoard metrics
-------------------

Pass ``--metrics-log-dir`` to control where TensorBoard event files are written.
The default is shown by ``areno train --help`` and is also surfaced in the
training config printout.

.. code-block:: bash

   areno train ... --metrics-log-dir /tmp/areno/tfevent
   tensorboard --logdir /tmp/areno/tfevent

The writer lives in ``areno.api.metrics``. It records three namespaces:

``rollout/*``
   Sample-side statistics computed from the train batch, including
   ``rollout/rewards_mean``, ``rollout/rewards_std``,
   ``rollout/rewards_max``, ``rollout/rewards_min``,
   ``rollout/accuracy``, ``rollout/advantages_mean``,
   ``rollout/advantages_std``, ``rollout/logprobs_mean``,
   ``rollout/seq_len_mean``, ``rollout/prompt_len_mean``,
   ``rollout/response_len_mean``, ``rollout/num_sequences``,
   ``rollout/skipped_long``, and ``rollout/total_skipped_long``.

``train/*``
   Every scalar returned in ``train_stats``. Typical examples are
   ``train/loss``, ``train/policy_loss``, ``train/total_loss``,
   ``train/ratio_mean``, ``train/ratio_std``, ``train/grad_norm``,
   ``train/lr``, ``train/rollout_logprobs_mean``, and
   ``train/train_logprobs_mean``.

``time/*``
   Stage timings when available: ``time/rollout``, ``time/reward``,
   ``time/advantage``, and ``time/train``.

Agentic diagnostics
-------------------

Agentic rollout adds two diagnostic surfaces.

First, the trainer logs batch-level information before and after agent
execution:

.. code-block:: text

   agentic rollout batch prompts=2 n_samples=8 expected_requests=16 max_running_prompts=16
   agentic train batch built samples=16 tokens=223308 messages=242 tool_calls=133 tool_results=97

These lines are useful for checking whether the configured concurrency,
trajectory length, and tool-call volume match expectations.

Second, set ``ARENO_LOG_COMPLETIONS`` to a positive integer to log a bounded
number of prompt/sample trajectories. For agentic rollouts this includes the
rendered prompt, message list, final answer, parsed tool calls, sampled tool
results, token row prefix, and loss-mask summary:

.. code-block:: bash

   ARENO_LOG_COMPLETIONS=2 areno train ...

When a trajectory is dropped for exceeding the model context window,
``PolicyOnlyTrainer`` logs ``agentic trajectory filtered: ...`` with token
counts, message counts, assistant turn counts, tool-result counts, and a short
prompt preview. This is the fastest way to debug overlong agentic examples
without dumping every token in every trajectory.

Startup health check (Issue #249)
---------------------------------

AReno can run automatic pass/warn/fail health checks over the first training
updates to catch silent degeneration early (zero effective tokens, constant
reward where variation is required, non-changing loss, excessive skipped
batches). It is off by default and reuses the existing metric stream
(``collect_train_batch_stats`` for sample-side signals, the backend train result
for ``loss`` / ``grad_zero_ratio``) — no parallel collection path.

Enable it from the CLI:

.. code-block:: bash

   areno train --algo gspo --ckpt ... --dataset-path ... \
     --health-check-enabled --health-check-window 20 --health-check-on-fail warn

Flags:

* ``--health-check-enabled`` — turn on the check (default off; full backward
  compatibility).
* ``--health-check-window`` — startup window length in training updates
  (default 20). The check evaluates once when this many updates complete.
* ``--health-check-on-fail`` — ``warn`` (default, log only) or ``fail`` (raise
  and abort the run when the summary is FAIL).
* ``--health-check-require-reward-variation`` / ``--health-check-allow-constant-reward`
  — whether reward must vary. Disable for tasks with legitimately constant
  reward (anti-false-positive).

Invalid thresholds are rejected at config validation time, before workers
spawn.

Output channels (all written only when the check is enabled):

* Console logs under the ``areno.health_check`` logger — one line per check
  (``name``, ``status``, ``stage``, ``msg``, ``metric_ref``) and a trailing
  ``SUMMARY=...`` line.
* A structured artifact JSON at
  ``<metrics-log-dir>/health_check/<run_id>.json`` with ``summary``, per-check
  ``stage``/``status``/``message``/``metric_ref``/``input``, window metadata,
  and ``original_errors``.
* ``health/*`` TensorBoard scalars (``health/summary``, plus one per check)
  written through the same ``MetricsRecorder`` writer as ``rollout/*`` and
  ``train/*``.

The four checks and their signals:

* ``effective_tokens`` (stage=trainer) — sum of prompt-mask-filtered response
  lengths; zero across the window fails, a low per-batch mean warns.
* ``reward_variance`` (stage=trainer) — reward std; with variation required,
  ``std==0`` fails and a low std warns. ``require_variation=false`` passes
  constant reward, but an empty reward list still warns (data-pipeline anomaly).
  NaN reward fails and records the original error.
* ``loss_change`` (stage=trainer) — first/last loss delta; ``delta <=
  min_abs_delta_fail`` (default 0, i.e. unchanged) with ≥2 samples fails; a
  delta below ``min_abs_delta_warn`` warns. NaN loss fails. A single-step
  window warns (unreliable delta).
* ``skipped_batches`` (stage=rollout) — combines the rollout ``skipped_long``
  ratio and the backend ``grad_zero_ratio`` proxy; reaching either threshold
  triggers (comparisons are threshold-inclusive ``>=``). The more severe
  sub-signal wins. A zero-batch window fails pointing at the data/input
  contract.

Failure messages reference only ``stage``, the triggering config field
(``input``), and a ``metric_ref`` into the existing TensorBoard namespace —
never training-sample text. The original error (e.g. a NaN detection) is kept
on the report rather than swallowed.

Example artifact:

.. code-block:: json

   {
     "run_id": "a1b2c3d4e5f6",
     "window": {"updates": 20, "completed_at_step": 20},
     "summary": "WARN",
     "checks": [
       {"name": "effective_tokens", "stage": "trainer", "status": "PASS",
        "message": "effective tokens ok (min_batch=512, mean=512.0)",
        "metric_ref": "metrics/rollout/response_len_mean", "input": "-"},
       {"name": "reward_variance", "stage": "trainer", "status": "WARN",
        "message": "low reward std=3.0e-07 < min_std_warn=1.0e-06",
        "metric_ref": "metrics/rollout/rewards_std",
        "input": "reward_variance.min_std_warn"}
     ],
     "original_errors": []
   }
