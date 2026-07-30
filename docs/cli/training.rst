:orphan:

Training CLI reference
======================

``areno train``

Run SFT, DPO, GSPO, GRPO, or PPO training with the local Areno backend. The
command owns the full loop: dataset loading, optional normalization, rollout,
reward scoring, loss computation, optimizer steps, metrics, and checkpoint
saving.

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 2 \
     --n-samples 2 \
     --mini-bs 1

areno train
-----------

Start a training run.

Options are grouped into sections that match ``areno train --help``, following
the RL training loop: **Basic** (what to run plus devices), **Rollout**
(generate and score completions), **Train** (consume rollouts and update
weights), **Checkpoint** (produced artifacts), and **Observability** (logs).

Basic
~~~~~

Inputs, dataset loader, the algorithm and run length, and device counts.

``--ckpt TEXT``
   Actor model/tokenizer checkpoint path or remote model repo ID.

``--dataset-path TEXT``
   Training dataset path, Hugging Face ``save_to_disk`` directory, or remote
   dataset reference.

``--model-hub [hf|modelscope]``
   Remote hub used when ``--ckpt``, role checkpoints, or ``--dataset-path`` are
   not local paths. Use ``--model-hub hf`` for Hugging Face and
   ``--model-hub modelscope`` for ModelScope. Default: ``modelscope``.

Dataset references use ``repo/name``, ``repo/name:config``, or
``repo/name:config:split``. Examples: ``gsm8k:main`` and
``AI-MO/NuminaMath-TIR``.

``--dataset-path`` also accepts JSON/JSONL, Parquet, CSV/TSV, Arrow, and
``datasets.save_to_disk(...)`` directories.

``--dataset-loader-fn TEXT``
   Optional Python dataset loader function as ``file.py`` or
   ``file.py:function``.

Use ``--dataset-loader-fn`` when the raw dataset does not already match the
trainer schema. Without a loader, Areno passes dataset rows through unchanged,
except for SFT where a loader is required. See :doc:`dataset_loaders` for the
per-algorithm loader contracts.

``--dataset-mix-config PATH``
   JSON manifest for deterministic weighted SFT dataset mixing. This option is
   mutually exclusive with ``--dataset-path`` and ``--dataset-source``.

``--dataset-source NAME=PATH:WEIGHT``
   Command-line shorthand for a weighted SFT source. Repeat the option at least
   twice. It is mutually exclusive with ``--dataset-path`` and
   ``--dataset-mix-config``:

   .. code-block:: bash

      areno train \
        --algo sft \
        --ckpt Qwen/Qwen3-0.6B \
        --model-hub hf \
        --dataset-source alpaca_cleaned=yahma/alpaca-cleaned:0.6 \
        --dataset-source stanford_alpaca=tatsu-lab/alpaca:0.3 \
        --dataset-source alpaca_gpt4=vicgalle/alpaca-gpt4:0.1 \
        --dataset-mix-samples-per-epoch 10000 \
        --dataset-loader-fn examples/sft/alpaca/dataset_loader.py

   Any number of sources (two or more) can be repeated. Weights are normalized
   across all sources and mean probabilities of selecting a **row**, not a
   token or a batch. The shorthand uses seed ``42``, ``cycle`` exhaustion, and
   source shuffling. Override these with ``--dataset-mix-seed``,
   ``--dataset-mix-exhaustion``, and
   ``--dataset-mix-samples-per-epoch``.

``--dataset-mix-samples-per-epoch INTEGER``
   Fixed row-sampling budget for a command-line ``cycle`` mix. If omitted,
   AReno uses the sum of the loaded source row counts. A fixed budget keeps
   many-source training bounded while preserving the requested distribution;
   small sources repeat as necessary.

``--dataset-mix-exhaustion [cycle|stop|renormalize]``
   Exhaustion policy for command-line sources. ``cycle`` is the default because
   it keeps weights meaningful for the whole fixed budget. ``stop`` and
   ``renormalize`` do not accept a sample budget.

``--dataset-mix-seed INTEGER``
   Seed for source selection and per-source shuffling. Default: ``42``.

Both mixing forms currently support map-style datasets only, and the shared
``--dataset-loader-fn`` is applied independently to each source.

The manifest requires ``version: 1``, an integer ``seed``, an explicit
``exhaustion`` policy, and at least two uniquely named sources with finite,
positive weights. Unknown version-1 manifest or source fields are rejected so
misspelled settings cannot silently fall back to defaults. Weights are
normalized automatically:

.. code-block:: json

   {
     "version": 1,
     "seed": 42,
     "exhaustion": "cycle",
     "shuffle_within_sources": true,
     "samples_per_epoch": 10000,
     "sources": [
       {"name": "math", "path": "math.jsonl", "weight": 0.7},
       {"name": "code", "path": "code.jsonl", "weight": 0.3}
     ]
   }

Existing relative local paths, explicit ``./`` or ``../`` paths, and supported
local file suffixes are resolved from the manifest directory. Other values are
handled as remote dataset references using ``--model-hub``. The exhaustion
policies have deliberately different consumption behavior:

* ``stop`` ends when one selected source is exhausted. Records are not
  repeated, but other sources may remain partially unused.
* ``cycle`` restarts exhausted sources and emits exactly
  ``samples_per_epoch`` rows. When the field/flag is omitted, both manifest and
  command-line forms resolve the budget to the sum of source row counts.
* ``renormalize`` removes an exhausted source and renormalizes the remaining
  weights. Every source record is emitted exactly once.

For ``K`` sources and budget ``N``, source ``i`` receives about
``N * normalized_weight_i`` rows. This is stochastic: a source whose expected
count is below one can receive zero rows in an epoch, so AReno emits a plan
warning. Use a larger budget when every low-weight source needs practical
coverage.

The seed controls source selection and per-source ordering. AReno derives a
different deterministic ordering for each epoch. Source identity is attached
under the reserved ``__areno_meta__`` field. The run prints a sample-free
summary, logs the plan for each epoch, and records both a mix-spec hash and
schedule hash. These hashes identify sampler configuration and index ordering
only; they do not prove that a mutable remote dataset still has identical
contents.
The cumulative ``stage=dataset_mix_progress`` event reports scheduled,
filtered, and successfully trained rows plus trained target-token counts and
both row/token proportions. Each epoch plan is also written as
``dataset_mix_plan.<pid>.epoch-<n>.json`` to ``--metrics-log-dir``.

V1 guarantees deterministic replay from the beginning of an epoch. AReno
checkpoints currently do not persist an optimizer state plus mid-epoch dataset
cursor, so exact mid-epoch resume is outside this contract.

The map-style implementation precomputes an index schedule for the current
epoch. Very large mixes therefore require memory proportional to the planned
row count; streaming datasets are not supported in this first version.
See ``examples/sft/mixed`` for a copyable local example and an invalid-weight
boundary case.

``--algo TEXT``
   Training algorithm registered in ``areno.api``. Default: ``gspo``.

Built-in algorithms: ``sft``, ``dpo``, ``gspo``, ``grpo``, ``ppo``.

``--epochs INTEGER``
   Number of dataset epochs to train. Default: ``10``.

``--max-steps INTEGER``
   Optional global trainer step cap. Training stops after this many step
   indices have completed, even if the current epoch still has more batches.

``--world-size INTEGER``
   Total device count for the backend. Default: ``8``.

``--tp-size INTEGER``
   Tensor parallel size for the backend. Default: ``4``.

``world-size`` must be divisible by ``tp-size``.

Rollout
~~~~~~~

Everything that generates and scores completions: batch volume, sequence
limits, sampling, decode runtime, the agentic-rollout hooks, and the reward
signal.

``--batch-size INTEGER``
   Prompt or pair batch size. Default: ``32``.

``--n-samples INTEGER``
   Rollout samples per prompt for RL algorithms. Default: ``8``.

``--max-running-prompts INTEGER``
   Override global concurrent rollout prompts. Defaults to
   ``batch-size * n-samples`` for rollout algorithms.

``--max-prompt-tokens INTEGER``
   Maximum tokenized prompt length. Default: ``1024``.

``--max-new-tokens INTEGER``
   Maximum generated or supervised response tokens. Default: ``3071``.

``--max-context-len INTEGER``
   Maximum total context tokens for agentic rollout trajectories. This counts
   the original prompt plus all generated assistant turns concatenated into
   the training row. Defaults to the model context limit.

``--temperature FLOAT``
   Rollout sampling temperature. Default: ``1.0``.

``--top-k INTEGER``
   Rollout top-k; ``-1`` disables top-k filtering. Default: ``-1``.

``--top-p FLOAT``
   Rollout top-p. Default: ``1.0``.

``--greedy``
   Use greedy rollout decoding.

``--eager-decode``
   Disable decode CUDA graph and run rollout decode eagerly.

``--drop-rollout-state``
   Drop rollout state after each step to save GPU memory. By default, Areno
   keeps rollout state on GPU between steps for lower rollout setup overhead.

``--attn-backend [flash|native]``
   Attention backend. Default: ``flash``. Use ``native`` to run without
   ``flash-attn`` on the areno_accel native compatibility path. AReno
   automatically falls back to ``native`` on flash-attn-unsupported GPUs such
   as Tesla T4 and prints a warning. ``native`` is slower than ``flash`` on
   supported GPUs.

``--disable-thinking``
   Pass ``enable_thinking=False`` to tokenizer chat templates when supported.
   This is useful for models whose tokenizer template exposes an explicit
   thinking-mode switch, such as some reasoning/chat checkpoints. Tokenizers
   that do not accept ``enable_thinking`` automatically fall back to their
   normal chat-template call.

Training rollouts run inside a rollout session. The session owns actor
onload/offload, rollout cache state, CUDA graph state, and cleanup between
rollout and train phases. Direct prompt rollout and agentic rollout both use
the same session lifecycle.

``--agent-fn TEXT``
   Python file defining ``async def run_agent(ctx, batch)``. When provided,
   online RL algorithms use agentic rollout mode instead of direct prompt
   completion. The agent receives a local OpenAI-compatible base URL from
   ``ctx.get_base_url()`` and can call ``/v1/chat/completions`` with tools.
   Use ``batch.iter_samples()`` to iterate the expanded
   ``batch-size * n-samples`` agent tasks. The function returns explicit
   trajectories: one ``AgentTrajectoryTurn``, one ``AgentTrajectory``, or an
   iterable of either. Each turn must carry its ``AgentItem``, message list,
   and OpenAI response.

``--agent-timeout-s FLOAT``
   Timeout for agentic proxy requests and the agent function. Default:
   ``300.0``.

``--train-tool-results``
   Include tool-result spans in policy loss. Disabled by default because tool
   results are environment observations rather than policy actions. Assistant
   text and assistant tool-call spans are trainable by default.

Agentic trajectories can contain multiple chat-completion turns for the same
prompt/sample pair. The agent owns the OpenAI-style message list and returns
trajectory turns with the model response; Areno converts those turns into token
rows, rollout logprobs, parsed assistant tool calls, reward records, and loss
masks.
Tool-result/context spans are included in the token row for correct scoring
but are masked from policy loss unless ``--train-tool-results`` is set.
When ``--max-context-len`` is set, filtering uses the full concatenated token
row for the whole agentic trajectory, not only the latest chat-completion turn.

``--reward-fn-path TEXT``
   Python file defining ``reward_fn(record)``.

Reward files should expose:

.. code-block:: python

   def reward_fn(record):
       return 0.0

``--reward-ckpt TEXT``
   Optional PPO reward model checkpoint path or remote model repo ID.

Parameter tuning
~~~~~~~~~~~~~~~~

``--tune-params``
   Probe rollout and training memory before starting the real run, then fill
   safe values for ``--max-running-prompts``, ``--batch-size`` and
   ``--mini-bs``. This is intended for rollout-based algorithms such as GSPO,
   GRPO and PPO, including agentic rollouts where the right concurrency is
   hard to estimate by hand.

The tuner uses dummy-loaded model weights and synthetic token rows, so it
measures the selected model architecture, ``--world-size``/``--tp-size``,
sequence lengths, CUDA graph setup, optimizer state, and train microbatch
memory without consuming a real dataset row or writing checkpoints. It keeps
the user-provided ``--tp-size``, ``--n-samples``, ``--adam-8bit``, sequence
limits, model path, algorithm, and backend settings.

Search is deliberately conservative:

* rollout candidates are tried from larger to smaller
  ``--max-running-prompts`` values;
* if ``--max-running-prompts`` is explicitly provided, rollout probing is
  skipped and that concurrency is used directly for training-parameter tuning;
* training uses the rollout-selected concurrency to derive a batch size, then
  tries larger to smaller ``--mini-bs`` values;
* ``--drop-rollout-state`` is enabled for the tuned run so rollout memory does
  not remain resident during the training probe or optimizer step.

``--mem-frac FLOAT``
   Target maximum GPU memory fraction for tuning. Default: ``0.9``. Lower this
   when sharing a node or when the real reward/agent path has additional GPU
   users.

``--tune-max-samples INTEGER``
   Upper bound for sampled rollout/train rows considered during tuning.
   Default: ``256``. The rollout search does not try
   ``--max-running-prompts`` above this value, and the derived train batch size
   is capped by ``tune-max-samples / n-samples``.

Example:

.. code-block:: bash

   areno train \
     --ckpt /path/to/Qwen3.5-4B \
     --dataset-path examples/agentic/coding/dataset.jsonl \
     --dataset-loader-fn examples/agentic/coding/dataset_loader.py \
     --reward-fn-path examples/agentic/coding/reward.py \
     --agent-fn examples/agentic/coding/run_agent.py \
     --algo gspo \
     --world-size 8 \
     --tp-size 4 \
     --n-samples 8 \
     --max-new-tokens 2048 \
     --max-context-len 32768 \
     --tune-params \
     --mem-frac 0.9 \
     --tune-max-samples 256

Train
~~~~~

Everything that consumes rollouts and updates weights: training batching and
memory, the policy optimizer, reference/critic models, and the per-algorithm
loss knobs. Each algorithm-specific flag applies only to the algorithms named
in its description; flags for other algorithms are ignored.

``--mini-bs INTEGER``
   Backend training microbatch size. Default: ``16``.

``--gradient-accumulation-steps INTEGER``
   Optimizer step interval in microbatches. Defaults to accumulating all
   mini-batches in one train call.

``--activation-checkpointing / --no-activation-checkpointing``
   Enable decoder-layer activation recompute during training. Default:
   enabled.

``--lr FLOAT``
   Policy optimizer learning rate. Default: ``1.0e-6``.

``--min-lr FLOAT``
   Policy optimizer minimum learning rate. Default: ``1.0e-7``.

``--lr-decay-steps INTEGER``
   Policy LR decay steps. Default: ``1000``.

``--lr-decay-style TEXT``
   Policy LR decay style. Default: ``cosine``.

``--adam-beta1 FLOAT``
   Policy optimizer Adam beta1. Default: ``0.9``.

``--adam-beta2 FLOAT``
   Policy optimizer Adam beta2. Default: ``0.999``.

``--adam-8bit``
   Use 8-bit Adam moment states instead of FP32 Adam states.

``--weight-decay FLOAT``
   Policy optimizer weight decay. Default: ``1.0e-2``.

``--grad-clip-norm FLOAT``
   Policy gradient clipping norm. Default: ``1.0``.

``--ref-ckpt TEXT``
   Optional PPO/DPO reference model checkpoint path or remote model repo ID.

``--critic-ckpt TEXT``
   Optional PPO critic model checkpoint path or remote model repo ID.

``--critic-lr FLOAT``
   PPO critic optimizer learning rate. Default: ``1.0e-5``.

``--critic-warmup-steps INTEGER``
   PPO critic-only warmup steps before actor updates. Default: ``20``.

``--gspo-clip-eps FLOAT``
   GSPO sequence-ratio clipping epsilon. Default: ``3.0e-4``.

``--grpo-clip-eps FLOAT``
   GRPO token-ratio clipping epsilon. Default: ``0.2``.

``--dpo-beta FLOAT``
   DPO preference margin temperature. Default: ``0.1``.

``--use-kl-loss / --no-use-kl-loss``
   Enable PPO actor KL loss. Default: enabled.

``--kl-loss-coef FLOAT``
   PPO actor KL loss coefficient. Default: ``0.001``.

``--kl-loss-type TEXT``
   PPO actor KL loss type. Default: ``low_var_kl``.

``--clip-eps FLOAT``
   PPO policy clipping epsilon. Default: ``0.2``.

``--clip-ratio-c FLOAT``
   PPO lower policy clipping bound multiplier. Default: ``3.0``.

``--value-clip-eps FLOAT``
   PPO value clipping epsilon. Default: ``0.5``.

``--value-loss-coef FLOAT``
   PPO value loss coefficient. Default: ``0.5``.

``--gamma FLOAT``
   PPO GAE discount. Default: ``1.0``.

``--lam FLOAT``
   PPO GAE lambda. Default: ``0.95``.

Checkpoint
~~~~~~~~~~

``--save-path TEXT``
   Optional checkpoint output directory.

``--save-interval INTEGER``
   Save checkpoint every N train steps. Default: ``100``.

Observability
~~~~~~~~~~~~~~

``--metrics-log-dir TEXT``
   TensorBoard metrics log directory. See :doc:`observability` for the
   ``rollout/*``, ``train/*``, and ``time/*`` metric namespaces and debugging
   log examples.

Examples
--------

Tiny training smoke test
~~~~~~~~~~~~~~~~~~~~~~~~

Use this command when you only want to check that a machine can run one small
official training task end to end:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1

This verifies the training wiring; it is not intended to measure final model
quality.

GSPO math training
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 2 \
     --n-samples 2 \
     --mini-bs 1

SFT instruction tuning
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path yahma/alpaca-cleaned \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --algo sft \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 2 \
     --mini-bs 1

SFT loaders must normalize raw rows to ``prompt`` and ``response`` dictionaries.
The trainer performs tokenization and trains on the response suffix.

DPO preference training
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno train \
     --ckpt /path/to/policy \
     --ref-ckpt /path/to/reference \
     --dataset-path /path/to/dpo.jsonl \
     --dataset-loader-fn /path/to/dpo_dataset_loader.py \
     --algo dpo \
     --tp-size 1 \
     --world-size 1

The DPO loader should normalize each row to ``prompt``, ``chosen``, and
``rejected``.

PPO with reward and critic roles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno train \
     --ckpt /path/to/policy \
     --ref-ckpt /path/to/reference \
     --reward-ckpt /path/to/reward-model \
     --critic-ckpt /path/to/critic \
     --dataset-path /path/to/data \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --algo ppo \
     --tp-size 4 \
     --world-size 8

Agentic Tic-Tac-Toe
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/agentic/tictactoe/dataset_generator.py \
     --output /tmp/areno-tictactoe.jsonl \
     --count 256 \
     --seed 2026

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/areno-tictactoe.jsonl \
     --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 32 \
     --n-samples 8 \
     --max-new-tokens 32

The agent function can use the OpenAI Python client against
``ctx.get_base_url()`` and returns explicit ``AgentTrajectory`` or
``AgentTrajectoryTurn`` objects. Areno converts them to tokens, rollout
logprobs, parsed ``tool_calls``, rewards, and loss masks, then feeds the
resulting batch to the same policy trainer used by non-agentic rollouts.

Agentic DuelGrid
~~~~~~~~~~~~~~~~

DuelGrid is a turn-based grid tactics example for agentic RLVR. The user
controls ``U`` and the LLM controls ``A`` with JSON action sequences such as
``MOVE``, ``ATTACK``, ``RANGED_ATTACK``, ``PICKUP``, and ``SHIELD``.

.. code-block:: bash

   python examples/agentic/duelgrid/dataset_generator.py \
     --count 256 \
     --output /tmp/areno-duelgrid.jsonl

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/areno-duelgrid.jsonl \
     --dataset-loader-fn examples/agentic/duelgrid/dataset_loader.py \
     --reward-fn-path examples/agentic/duelgrid/reward.py \
     --agent-fn examples/agentic/duelgrid/run_agent.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1

The browser UI can replay the same rule engine:

.. code-block:: bash

   python examples/agentic/duelgrid/web_ui.py \
     --base-url http://127.0.0.1:8000/v1 \
     --api-key EMPTY \
     --model policy

Before GSPO/RLVR post-training, Gemma-E2B-it performs poorly in DuelGrid and
often oscillates between nearby tiles. After training, it learns to collect
health and energy pickups, chase the user, attack when it has position, and
avoid trap tiles while spending its turn energy. The reward curve improves
quickly early in training and then stabilizes after the policy has learned the
game loop.

.. list-table::
   :header-rows: 1
   :widths: 1 1 1

   * - Train before
     - Reward
     - Train after
   * - .. image:: ../../examples/agentic/duelgrid/images/train_before.gif
          :alt: Gemma-E2B-it before DuelGrid training
          :width: 260px
     - .. image:: ../../examples/agentic/duelgrid/images/train_reward.jpg
          :alt: DuelGrid training reward curve
          :width: 260px
     - .. image:: ../../examples/agentic/duelgrid/images/train_after.gif
          :alt: Gemma-E2B-it after DuelGrid training
          :width: 260px

Help
----

.. code-block:: bash

   areno train --help
