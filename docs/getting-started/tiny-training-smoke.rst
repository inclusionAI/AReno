Tiny training smoke tutorial
============================

This walkthrough runs the smallest official AReno training task end to end.
It checks the CLI, dataset loader, reward function, rollout path, training
step, checkpoint writing, and TensorBoard metrics export. It is a wiring
check, not a benchmark for final model quality.

Prerequisites
-------------

* A CUDA-capable NVIDIA GPU with CUDA-enabled PyTorch 2.6+.
* A working AReno install.
* ``areno check`` should pass or at least report a usable GPU stack.

CPU-only machines can still install AReno for documentation, packaging, and
metadata checks, but they cannot run the training engine used in this
tutorial.

Step 1: Prepare an output directory
-----------------------------------

.. code-block:: bash

   mkdir -p outputs/tiny-smoke

Step 2: Run the smoke command
-----------------------------

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1 \
     --max-steps 1 \
     --save-path outputs/tiny-smoke \
     --save-interval 1 \
     --metrics-log-dir outputs/tiny-smoke/metrics

The example dataset loader and reward function come from this repository, so
the only external fetches are the model checkpoint and dataset.

Step 3: Read the expected logs
------------------------------

Look for a short chain of lifecycle logs:

.. code-block:: text

   epoch=0 stage=epoch_start
   role=policy stage=rollout_start
   metric=reward_mean value=...
   epoch=0 step=0 train_stats={...}
   epoch=0 step=0 stage=save_checkpoint_start path=outputs/tiny-smoke/step_000001
   epoch=0 step=0 stage=save_checkpoint_end path=...

The exact numbers will vary. The important part is that the run reaches
rollout, training, and checkpoint saving without raising an exception.

Step 4: Inspect the artifacts
-----------------------------

After the run finishes, you should have:

.. code-block:: text

   outputs/tiny-smoke/step_000001/
   outputs/tiny-smoke/metrics/

``step_000001/`` is the checkpoint written by ``--save-interval 1``. The metrics
directory contains TensorBoard event files and the same scalar series the
dashboard reads.

Step 5: Open TensorBoard
------------------------

.. code-block:: bash

   tensorboard --logdir outputs/tiny-smoke/metrics

Then confirm that the run exposes ``rollout/*``, ``train/*``, and ``time/*``
scalar series.

Troubleshooting
---------------

* If ``areno check`` reports no usable CUDA setup, stop here.
* If the model or dataset download fails, retry once network access or hub
  credentials are available.
* If the command cannot import ``examples/math/dataset_loader.py`` or
  ``examples/math/math_verify_reward.py``, check the paths relative to the
  repository root.
* If the run stops before ``train_stats=...``, use ``areno env --json`` and the
  final log lines to narrow down the failure.
