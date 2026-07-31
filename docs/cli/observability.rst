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

Dashboard trajectory detail page
--------------------------------

The AReno dashboard includes a **Trajectory** page that renders the full
contents of an agentic rollout sample in chronological order. Open the
dashboard, select a train job that has agentic samples, pick a step and
sample, and the page renders:

* **Events timeline** — model messages (user / assistant / tool), tool calls
  attached to the originating assistant message, and interleaved tool results.
  Long tool output is collapsed by default; click to expand.
* **Final answer** — the last assistant response extracted from the sample.
* **Token counts** — prompt tokens, response tokens, and total token-row
  length.
* **Training-mask status** — ``loss_mask_true`` / ``loss_mask_total``,
  ``first_loss_idx``, a visual mask bar, and a ``truncated`` flag when the
  stored mask is shorter than the declared total.
* **End reason** — completion cause (defaults to ``completed`` when missing).

Tool calls are displayed as read-only text; the dashboard never executes a
displayed call. Long tool results are truncated to 200 characters in the inline
view and can be expanded in the standalone tool-result card.

A **Raw JSON** toggle is available for inspecting the full sample dict
(messages, tokens, loss_mask, etc.). The structured view omits raw token and
loss-mask sequences — only counts are surfaced — to keep the default output
privacy-safe.

API endpoint
~~~~~~~~~~~~

.. code-block:: text

   GET /api/jobs/<job_id>/trajectory?step=<int>&prompt_idx=<int>&sample_idx=<int>&include_raw=<bool>

Query parameters:

``step``
   Training step of the desired sample (integer, required).

``prompt_idx``
   Prompt index within the step (integer, required).

``sample_idx``
   Sample index within the prompt (integer, required).

``include_raw``
   Whether to attach the full sample dict under the ``raw`` key. Defaults to
   ``false``. Set to ``true`` to inspect tokens, loss_mask, and other fields
   not present in the structured detail.

Response fields (when ``valid`` is ``true``):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``valid``
     - ``true`` if the sample is a valid agentic trajectory.
   * - ``step`` / ``prompt_idx`` / ``sample_idx``
     - Locating indices from the sample.
   * - ``events``
     - Chronologically ordered list of message and tool-result entries.
   * - ``final_answer``
     - Last assistant response string (may be empty).
   * - ``token_counts``
     - ``prompt_tokens``, ``response_tokens``, ``loss_mask_true``,
       ``loss_mask_total``, ``token_row_len``.
   * - ``training_mask``
     - ``loss_mask_true``, ``loss_mask_total``, ``first_loss_idx``,
       ``truncated``.
   * - ``end_reason``
     - Completion cause string; defaults to ``"completed"``.
   * - ``tool_call_count`` / ``tool_result_count``
     - Number of tool calls and results in the sample.
   * - ``raw``
     - Full original sample dict. **Only present when ``include_raw=true``.**

When ``valid`` is ``false``, the response contains an ``error`` field with a
human-readable message instead of the fields above.

Example
~~~~~~~

.. code-block:: bash

   curl "http://127.0.0.1:8765/api/jobs/<job_id>/trajectory?step=1&prompt_idx=0&sample_idx=0"

Returns the structured detail without raw training data:

.. code-block:: json

   {
     "valid": true,
     "step": 1,
     "prompt_idx": 0,
     "sample_idx": 0,
     "kind": "agentic",
     "events": [
       {"type": "message", "role": "user", "content": "What is 2+2?"},
       {"type": "message", "role": "assistant", "content": "", "tool_calls": [...]},
       {"type": "message", "role": "tool", "content": "4", "tool_result": {"name": "calculate", "ok": true, "result": 4}},
       {"type": "message", "role": "assistant", "content": "The answer is 4."}
     ],
     "final_answer": "The answer is 4.",
     "token_counts": {"prompt_tokens": 4, "response_tokens": 8, "loss_mask_true": 8, "loss_mask_total": 12, "token_row_len": 12},
     "training_mask": {"loss_mask_true": 8, "loss_mask_total": 12, "first_loss_idx": 4, "truncated": false},
     "end_reason": "completed",
     "tool_call_count": 1,
     "tool_result_count": 1
   }

Limitations
~~~~~~~~~~~

* Only ``kind == "agentic"`` samples are supported. Rollout samples without
  agent messages return a validation error.
* The page reads existing local rollout-sample artifacts; it does not execute
  model inference or tool calls.
* Raw token and loss-mask sequences are omitted from the structured response
  by default. Use ``include_raw=true`` when full inspection is needed.
