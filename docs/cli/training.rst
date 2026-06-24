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
   Actor model/tokenizer checkpoint path or Hugging Face repo ID.

``--dataset-path TEXT``
   Training dataset path, Hugging Face ``save_to_disk`` directory, or Hugging
   Face dataset reference.

Dataset references use ``repo/name``, ``repo/name:config``, or
``repo/name:config:split``. Examples: ``gsm8k:main`` and
``AI-MO/NuminaMath-TIR``.

``--dataset-path`` also accepts JSON/JSONL, Parquet, CSV/TSV, Arrow, and
``datasets.save_to_disk(...)`` directories.

``--dataset-loader-fn TEXT``
   Optional Python dataset loader function as ``file.py`` or
   ``file.py:function``.

Use ``--dataset-loader-fn`` when the raw dataset does not already match the
trainer schema. Without a loader, Areno passes dataset rows through unchanged.

``--algo TEXT``
   Training algorithm registered in ``areno.api``. Default: ``gspo``.

Built-in algorithms: ``sft``, ``dpo``, ``gspo``, ``grpo``, ``ppo``.

``--epochs INTEGER``
   Number of dataset epochs to train. Default: ``10``.

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

``--reward-fn-path TEXT``
   Python file defining ``reward_fn(record)``.

Reward files should expose:

.. code-block:: python

   def reward_fn(record):
       return 0.0

``--reward-ckpt TEXT``
   Optional PPO reward model checkpoint path or Hugging Face repo ID.

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
   Optional PPO/DPO reference model checkpoint path or Hugging Face repo ID.

``--critic-ckpt TEXT``
   Optional PPO critic model checkpoint path or Hugging Face repo ID.

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
   TensorBoard metrics log directory.

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
