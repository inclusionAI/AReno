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

Dashboard: plotting multiple metrics together
---------------------------------------------

The AReno dashboard (``areno dashboard --start``) renders TensorBoard scalars
with hand-written SVG, no charting dependency. The metrics panel lets you plot
several metrics on one chart so post-training runs are easier to compare.

User-facing controls (all default to the previous single-metric behavior — the
feature is strictly opt-in):

- **Metric multi-select** (top-right of the panel): toggle any number of metrics
  onto one chart. With exactly one selected, the panel renders exactly as
  before (backward compatible).
- **legend**: each series is labeled with its name, axis (``L`` left / ``R``
  right), and true value range. Legend entries are click-to-toggle.
- **hover values**: moving the pointer across the chart snaps to the nearest
  step and shows a card listing every series' value at that step.
- **dual axes**: when selected series differ in scale by more than an order of
  magnitude, the largest-scale series uses a right Y-axis and the rest use the
  left. The assignment is automatic.
- **normalize** (checkbox, off by default): view-only scaling of every series
  into ``[0, 1]`` so differently-scaled series can be compared by shape. This
  only affects the rendered pixels — stored metric data is never modified, and
  the legend keeps showing the true min/max range.
- **smooth** (existing slider): EMA smoothing, unchanged from before.
- **downsampling**: large series (more than 480 points) are downsampled with
  largest-triangle-three-buckets for display only. Stored data and the footer
  point count are unaffected.

Colors use the colorblind-safe Okabe-Ito palette, assigned in selection order.

Input contract and limitations:

- The dashboard reuses the existing read-only endpoint
  ``GET /api/jobs/<id>/metric?name=<metric>&limit=<n>``. Multi-metric rendering
  issues one such request per selected metric; there is no new endpoint and no
  backend change.
- A failed fetch for one metric is reported by name (for example
  ``train/lr: fetch failed: ...``) without blanking the rest of the chart.

CLI export
~~~~~~~~~~

``areno metrics`` exports one or more metrics for a job, reusing the same
``metric_series`` contract as the dashboard. It validates inputs (job id,
metric names, limit) before any heavy work and reports failures by naming the
affected input. Default output is human-readable; ``--json`` gives structured
output. No GPU or server process is required.

.. code-block:: text

   usage: areno metrics --job JOB --names NAMES [--limit N] [--json]

Input contract:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Option
     - Meaning
   * - ``--job`` (required)
     - Job id whose metrics to export.
   * - ``--names`` (required)
     - Comma-separated metric names, e.g. ``train/loss,train/reward``.
   * - ``--limit``
     - Max points per metric, clamped to ``1..5000`` (default ``500``).
   * - ``--json``
     - Emit structured JSON instead of human-readable text.

Defaults: ``--limit 500``, human-readable text. Selecting a single metric
reproduces the existing single-metric series exactly.

Output fields (``--json``):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``job_id``
     - The job id passed in.
   * - ``limit``
     - The clamped limit used.
   * - ``metrics[].name``
     - Metric name (one entry per requested name, in request order).
   * - ``metrics[].point_count``
     - Number of points returned for that metric.
   * - ``metrics[].points[].step`` / ``value`` / ``time``
     - One point per recorded step (same shape as the dashboard endpoint).

Example (human-readable):

.. code-block:: text

   $ areno metrics --job <id> --names train/loss,train/reward --limit 3
   job: <id>  limit: 3

   # train/loss  (3 points)
     step     57  0.6957
     step     58  0.7113
     step     59  0.6816

   # train/reward  (3 points)
     step     57  4.9981
     step     58  5.0
     step     59  5.0

Failure paths identify the affected input:

.. code-block:: text

   $ areno metrics --job nope --names train/loss
   Error: job not found: nope

   $ areno metrics --job <id> --names train/loss,typo/tag
   Error: unknown metric names for job <id>: typo/tag

   $ areno metrics --job <id> --names train/loss --limit 99999
   Error: --limit must be between 1 and 5000, got 99999

Limitations:

- Metrics are read from the job's ``metrics_dir`` (TensorBoard scalars and
  ``*.jsonl`` sidecar files). The CLI reads the local dashboard state; it does
  not connect to a running dashboard server.
- Downsampling and normalization are dashboard-display transforms; the CLI
  exports the raw recorded points up to ``--limit``.

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
