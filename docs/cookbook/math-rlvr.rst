Math RLVR recipe
================

This recipe runs a small math RLVR task with a GSM8K-style dataset loader and
math verification reward function.

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1

Key files:

* ``examples/math/dataset_loader.py`` normalizes the dataset.
* ``examples/math/math_verify_reward.py`` scores completions.
* :doc:`/cli/training` documents rollout, loss, and checkpoint flags.

Adapt this recipe by replacing the dataset loader and reward function first.
Keep the batch size small until the environment and reward signal are stable.
