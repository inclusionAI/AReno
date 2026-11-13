OPD training
============

This walkthrough runs a minimal On-Policy Distillation job and shows how to
record held-out numbers. For the underlying objective see
:doc:`../concepts/opd`.

Prerequisites
-------------

* A CUDA, NPU, or MLX backend. OPD is declared in the CUDA backend capabilities;
  NPU inherits that capability set, and MLX registers ``opd`` and the ``teacher``
  role with native losses. For a short Apple Silicon run, use
  ``examples/opd/run_mlx_smoke.sh``.
* ``areno check`` passes.
* Student and teacher share the same tokenizer. The trainer verifies this at
  start-up, so choose two checkpoints from the same model family.

Smoke run
---------

.. code-block:: bash

   cd /path/to/AReno
   sh examples/opd/run_smoke.sh

The script expands to:

.. code-block:: bash

   areno train \
     --algo opd \
     --ckpt Qwen/Qwen3-0.6B \
     --teacher-ckpt Qwen/Qwen3-0.6B-Instruct \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1 \
     --n-samples 2 \
     --mini-bs 1 \
     --max-prompt-tokens 256 \
     --max-new-tokens 256 \
     --max-steps 20 \
     --metrics-log-dir ./outputs/opd-smoke

There is deliberately no ``--reward-fn-path``. ``opd`` needs rollout but no
reward function. A successful run prints rollout logs, ``train_stats=...``
lines, and the ``opd_loss`` family of metrics.

Flags
-----

``--teacher-ckpt TEXT``
   Frozen teacher checkpoint or remote repository id. Required for OPD.

``--opd-coef FLOAT``
   Distillation loss scale. Default: ``1.0``.

``--opd-loss-type [pg|k3]``
   ``pg`` uses a clipped dense advantage; ``k3`` uses the low-variance KL
   estimator. Default: ``pg``.

``--opd-clip FLOAT``
   Dense-advantage clipping range for ``pg``. Default: ``5.0``.

``--opd-teacher-micro-bs INTEGER``
   Teacher scoring microbatch size. Defaults to ``--score-micro-bs``. Lower it
   if teacher scoring is the memory bottleneck.

Held-out evaluation
-------------------

OPD reports two kinds of held-out metrics on the same ``--eval-*`` schedule.

Task accuracy
~~~~~~~~~~~~~

Pass a verifier with ``--reward-fn-path``. OPD has no reward term in training,
so this callable is used only when evaluation fires: the trainer samples
held-out prompts with the current student policy (greedy, so repeated runs are
comparable) and scores the completions through the standard
``reward_fn(record)`` contract. ``examples/math/math_verify_reward.py`` works
as-is for GSM8K-style data.

.. code-block:: bash

   areno train \
     --algo opd \
     --ckpt Qwen/Qwen3-0.6B \
     --teacher-ckpt Qwen/Qwen3-0.6B-Instruct \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --validation-dataset-path gsm8k:main \
     --validation-dataset-loader-fn examples/math/dataset_loader.py \
     --eval-interval 5 \
     --eval-batches 1 \
     --eval-before-step \
     --tp-size 1 --world-size 1 \
     --batch-size 1 --n-samples 2 --mini-bs 1 \
     --max-steps 20

This produces ``generation_accuracy``, ``generation_reward_mean``,
``generation_samples`` and ``generation_batches``. With a binary verifier,
``generation_accuracy`` is the task-level number to compare before and after
training. Without ``--reward-fn-path`` the generation pass is skipped entirely.

Distillation progress
~~~~~~~~~~~~~~~~~~~~~

The trainer also scores fixed held-out sequences with both the student and the
frozen teacher, reporting ``teacher_student_gap`` and per-token NLL. This is a
cheap training-progress monitor rather than a task metric, and it is always
available when a validation set is configured.

All metrics go to ``validation_metrics.jsonl`` under ``--metrics-log-dir``,
prefixed with ``before_`` or ``after_``. ``--eval-before-step`` records the
initial state so the first comparison has a baseline. ``--eval-batches`` limits
the number of evaluation batches; held-out rows need ``prompt`` plus
``solutions`` (or ``answer``), which ``examples/math/dataset_loader.py``
already produces.

Troubleshooting
---------------

``OPD teacher and student vocab sizes differ``
   The teacher is from a different model family. Use a matching checkpoint.

``OPD requires teacher logprobs in packed_ref_logprobs``
   The teacher role did not score the batch. Confirm ``--teacher-ckpt`` is set
   and the backend supports the ``teacher`` role.

``teacher logprobs contain non-finite values``
   The teacher checkpoint produced NaN or Inf. Shorten the sequence or check
   the checkpoint.

Loss is exactly zero for the whole run
   Expected when student and teacher are the same checkpoint with no LoRA:
   their logprobs agree, so the dense advantage is zero. Use an instructed
   teacher or enable LoRA on the student.

Teacher scoring runs out of memory
   Lower ``--opd-teacher-micro-bs``. CUDA lazily loads the teacher and offloads
   it after each scoring pass, so it does not stay resident alongside the
   trainable weights. MLX has no role offload: both models stay resident, so
   also reduce ``--max-new-tokens`` and ``--n-samples`` on a small-memory Mac.
