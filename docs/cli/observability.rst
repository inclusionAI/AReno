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

Multi-component reward analysis
-------------------------------

``areno reward-analysis`` is a focused, read-only command that inspects a
metrics directory for per-step reward *components* (for example
``correctness``, ``format``, ``length_penalty``) and reports how each component
evolves, how often it is zero or an outlier, and how it weights into the total
reward. It performs no model or worker initialization — inputs are validated up
front and the analysis runs purely over local artifacts.

.. code-block:: bash

   areno reward-analysis --metrics-dir /tmp/areno/tfevent
   areno reward-analysis --metrics-dir /tmp/areno/tfevent --json

A missing path is rejected up front with the affected stage and input:

.. code-block:: text

   Error: stage=artifact resolution: path not found (input=/tmp/does-not-exist)

Input contract
^^^^^^^^^^^^^^

The analyzer reuses AReno's existing ``{name, value, step}`` JSONL metric row
convention — the same schema the dashboard already ingests generically — scoped
to ``reward_components.<pid>.jsonl`` files in the metrics directory (naming
mirrors ``rollout_samples.<pid>.jsonl``). Each row names one component for one
step; rows are grouped by step before analysis:

.. code-block:: json

   {"step": 0, "name": "correctness", "value": 0.8}
   {"step": 0, "name": "format", "value": 0.2}
   {"step": 1, "name": "correctness", "value": 0.0}
   {"step": 1, "name": "format", "value": 0.2}
   {"step": 2, "name": "correctness", "value": 1.0}
   {"step": 2, "name": "format", "value": null}

``step`` is an integer and ``name`` a non-empty string. A ``value`` may be
``null`` (present-but-unknown, counted as *missing*) or non-finite
(``NaN``/``inf``, counted separately) and is never treated as zero. The set of
component names may grow over steps — a name first seen at step *k* is not
reported as missing before *k*. The per-step total defaults to the sum of the
step's finite component values.

Because the rows follow the existing metric convention, a producer only needs to
append these rows from a reward function (plain Python) or any metrics hook — no
trainer or SDK change is required. Malformed rows are skipped and reported by
file, line number, and step/component name only; prompt or completion text from
samples is never carried into errors.

Defaults and flags
^^^^^^^^^^^^^^^^^^

``--metrics-dir`` (required)
   Directory (or single file) to analyze.
``--json``
   Emit a machine-readable report instead of the human table.
``--history`` (default ``200``)
   Bounded per-component history length (a rolling window).
``--outlier-z`` (default ``3.0``)
   Z-score threshold used for the outlier fraction.

Output fields
^^^^^^^^^^^^^

Each component in the report carries:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``current``
     - Most recent finite value (``null`` if none).
   * - ``mean`` / ``std``
     - Welford mean and sample std over finite observations only.
   * - ``zero_fraction``
     - Fraction of present observations equal to zero.
   * - ``outlier_fraction``
     - Fraction of finite observations beyond ``--outlier-z`` standard deviations.
   * - ``non_finite_fraction``
     - Fraction of present observations that are ``NaN``/``inf``.
   * - ``missing_count``
     - Times the component was expected but absent or ``null``.
   * - ``weighted_contribution``
     - Component finite sum divided by the total finite sum (share of total reward).
   * - ``contribution_fraction``
     - Component finite sum divided by the sum of all components (relative weight).
   * - ``history`` / ``distribution``
     - Bounded recent values and a fixed-bucket histogram for drill-down.

A bounded ``steps`` list gives a per-update drill-down: each entry carries the
step's ``total``, the components present, and the names that were non-finite or
missing. Large batches are aggregated server-side: only per-step aggregates are
retained, never raw per-sample rewards, so memory stays bounded regardless of
batch size.

Limitations
^^^^^^^^^^^

This is a post-hoc, read-only view. It consumes whatever component artifact
exists in the metrics directory; producing that artifact from a reward function
is a separate concern tracked independently. The command needs no GPU, no
external database, and no sandbox. When the feature is not used, existing
training and serving behavior is unchanged.

The dashboard surfaces the same data: select a job and the *Reward Components*
panel polls ``/api/jobs/<id>/reward-components`` for the table and per-step
drill-down.
