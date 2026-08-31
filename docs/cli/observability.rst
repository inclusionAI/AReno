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
``areno.api.backend.cuda.backend``:

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
   * - ``tower_grad_norm`` / ``projector_grad_norm``
     - Gradient norms for enabled multimodal parameter groups.
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

Dataset loader resource limits
------------------------------

When a custom dataset loader (``--dataset-loader-fn``) or the default loader
takes too long or returns too many records, it can block training or exhaust
memory.  Two CLI flags bound the loader's resource usage:

``--loader-timeout-s SECONDS``
   Abort the loader if it runs longer than *SECONDS* wall-clock time
   (default: ``0``, disabled).  On Unix this uses ``SIGALRM``; on platforms
   without ``SIGALRM`` the timeout is silently skipped and a warning is
   logged.

``--max-loader-records N``
   Truncate the dataset to the first *N* records when the loader returns more
   than *N* rows (default: ``0``, disabled).  The cap materialises generators
   and other non-sized iterables so their length can be measured.

Both limits are independent and can be combined.  When neither is set (the
default), the loader still runs through the guard so timing diagnostics are
emitted, but no timeout or truncation is enforced.  Generators are passed
through unchanged when ``--max-loader-records`` is disabled.

Minimal example
~~~~~~~~~~~~~~~

Cap the loader to 100 records with a 30-second timeout:

.. code-block:: bash

   areno train \
     --algo sft \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path yahma/alpaca-cleaned \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --model-hub hf \
     --max-loader-records 100 \
     --loader-timeout-s 30

When the loader finishes, AReno logs a diagnostic line and, if
``--metrics-log-dir`` is set, writes ``areno_loader_diagnostics.json`` in that
directory:

.. code-block:: text

   dataset loader finished: duration=3.250s records=100 mem_delta=124800KB truncated=True

Observable fields
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``duration``
     - Wall-clock time the loader took to return (seconds).
   * - ``records``
     - Number of records after any truncation.
   * - ``mem_delta``
     - Change in peak process RSS during the loader call (KB).
   * - ``truncated``
     - ``True`` when the loader returned more than ``--max-loader-records`` and
       the result was truncated.

Timeout behaviour
~~~~~~~~~~~~~~~~~

When the loader exceeds ``--loader-timeout-s``, AReno raises
``DatasetLoaderTimeout`` (a subclass of ``TimeoutError``) and aborts the run
before model or worker initialisation.  The original traceback is preserved.

.. code-block:: text

   DatasetLoaderTimeout: Dataset loader exceeded the configured timeout

User exceptions from the loader itself (e.g. ``ValueError``,
``FileNotFoundError``) are re-raised unchanged—only the timeout produces
``DatasetLoaderTimeout``.

Platform limitations and side effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The timeout mechanism relies on ``SIGALRM`` and ``setitimer``, which are
available on Unix (Linux, macOS).  Sub-second precision is supported via
``setitimer``.  On Windows or other platforms without ``SIGALRM``, the
timeout is not enforced and a warning is logged.  ``--max-loader-records``
works on all platforms because it only inspects the returned object.
The ``resource`` module is imported lazily; if it is unavailable (e.g.
Windows), memory diagnostics report 0.

Because ``signal.signal`` only works in the main thread, ``--loader-timeout-s``
is silently skipped with a warning when the loader is invoked from a
background thread.  If your custom loader spawns its own threads or runs in a
worker thread, the timeout will not interrupt it.

Installing a ``SIGALRM`` handler is a *process-wide* operation.  While AReno
saves and restores the previous handler around each guarded loader call, any
other component in the same process that also uses ``SIGALRM`` may be briefly
affected.  If you have code that depends on a custom ``SIGALRM`` handler, keep
it outside of the loader invocation, or disable ``--loader-timeout-s`` and rely
on ``--max-loader-records`` instead.

To disable both limits and use the loader as-is:

.. code-block:: bash

   areno train ...   # no --loader-timeout-s or --max-loader-records flags

Invalid parameters
~~~~~~~~~~~~~~~~~~

Negative values are rejected at config validation time:

.. code-block:: text

   ValueError: loader_timeout_s must be non-negative
   ValueError: max_loader_records must be non-negative

Deterministic local example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a tiny JSONL file and a minimal loader script to test without any
external dataset:

.. code-block:: bash

   # 1. Create a 3-row dataset
   echo '{"prompt":"hello","response":"world"}' > /tmp/tiny.jsonl
   echo '{"prompt":"foo","response":"bar"}' >> /tmp/tiny.jsonl
   echo '{"prompt":"test","response":"case"}' >> /tmp/tiny.jsonl

   # 2. Create a minimal loader
   cat > /tmp/loader.py << 'EOF'
   import json

   def load_training_dataset(path, **kw):
       with open(path) as f:
           return [json.loads(line) for line in f]
   EOF

   # 3. Run with record cap and timeout
   areno train --algo sft --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/tiny.jsonl \
     --dataset-loader-fn /tmp/loader.py \
     --max-loader-records 2 --loader-timeout-s 10 --summary
