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

Required inputs
~~~~~~~~~~~~~~~

Options:

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

Dataset and reward hooks
~~~~~~~~~~~~~~~~~~~~~~~~

``--dataset-loader-fn TEXT``
   Optional Python dataset loader function as ``file.py`` or
   ``file.py:function``.

``--reward-fn-path TEXT``
   Python file defining ``reward_fn(example, completions)``.

Use ``--dataset-loader-fn`` when the raw dataset does not already match the
trainer schema. Without a loader, Areno passes dataset rows through unchanged.

Reward files should expose:

.. code-block:: python

   def reward_fn(example, completions):
       return [0.0 for _ in completions]

Algorithm selection
~~~~~~~~~~~~~~~~~~~

``--algo TEXT``
   Training algorithm registered in ``areno.api``. Default: ``gspo``.

Built-in algorithms: ``sft``, ``dpo``, ``gspo``, ``grpo``, ``ppo``.

Parallelism and batch shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--tp-size INTEGER``
   Tensor parallel size for the backend. Default: ``4``.

``--world-size INTEGER``
   Total device count for the backend. Default: ``8``.

``--batch-size INTEGER``
   Prompt or pair batch size. Default: ``32``.

``--n-samples INTEGER``
   Rollout samples per prompt for RL algorithms. Default: ``8``.

``--mini-bs INTEGER``
   Backend training microbatch size. Default: ``16``.

``--gradient-accumulation-steps INTEGER``
   Optimizer step interval in microbatches. Defaults to accumulating all
   mini-batches in one train call.

``--max-running-prompts INTEGER``
   Override concurrent rollout prompts. Defaults to
   ``batch-size * n-samples // dp-size``.

``world-size`` must be divisible by ``tp-size``.

Sequence and rollout sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Optimizer
~~~~~~~~~

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

Runtime memory and speed
~~~~~~~~~~~~~~~~~~~~~~~~

``--activation-checkpointing / --no-activation-checkpointing``
   Enable decoder-layer activation recompute during training. Default:
   enabled.

``--keep-rollout-state``
   Keep rollout state on GPU between steps for speed. This trades extra GPU
   memory for lower rollout setup overhead.

``--eager-decode``
   Disable decode CUDA graph and run rollout decode eagerly.

Checkpointing and metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

``--save-path TEXT``
   Optional checkpoint output directory.

``--save-interval INTEGER``
   Save checkpoint every N train steps. Default: ``100``.

``--metrics-log-dir TEXT``
   TensorBoard metrics log directory.

``--epochs INTEGER``
   Number of dataset epochs to train. Default: ``10``.

Algorithm-specific options
--------------------------

GSPO
~~~~

``--gspo-clip-eps FLOAT``
   GSPO sequence-ratio clipping epsilon. Default: ``3.0e-4``.

GRPO
~~~~

``--grpo-clip-eps FLOAT``
   GRPO token-ratio clipping epsilon. Default: ``0.2``.

DPO
~~~

``--ref-ckpt TEXT``
   Optional DPO reference model checkpoint path or Hugging Face repo ID.

``--dpo-beta FLOAT``
   DPO preference margin temperature. Default: ``0.1``.

PPO
~~~

``--ref-ckpt TEXT``
   Optional PPO reference model checkpoint path or Hugging Face repo ID.

``--reward-ckpt TEXT``
   Optional PPO reward model checkpoint path or Hugging Face repo ID.

``--critic-ckpt TEXT``
   Optional PPO critic model checkpoint path or Hugging Face repo ID.

``--critic-warmup-steps INTEGER``
   PPO critic-only warmup steps before actor updates. Default: ``20``.

``--critic-lr FLOAT``
   PPO critic optimizer learning rate. Default: ``1.0e-5``.

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

Examples
--------

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

Help
----

.. code-block:: bash

   areno train --help
