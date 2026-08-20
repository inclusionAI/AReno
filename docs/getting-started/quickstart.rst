Quickstart
==========

Use these commands after installation to validate the main AReno paths. Full
training and agentic rollout require either a CUDA-capable NVIDIA GPU on Linux
or Apple Silicon with MLX. The CLI selects the native backend automatically.
Use :doc:`backends` for the shared contract and platform-specific setup.

Training smoke test
-------------------

Run the smallest official training task to verify that the CLI can load a
model, build batches, execute the training loop, and write outputs locally.

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

RLVR path
---------

RLVR connects a dataset, model rollout, reward function, and policy loss. The
math example is the fastest way to see that path end to end.

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1

Read :doc:`/concepts/training-loop` for the mental model and
:doc:`/cookbook/math-rlvr` for the runnable recipe shape.

On Apple Silicon, use a checkpoint supported by ``mlx-lm`` and add
``--world-size 1 --tp-size 1 --mini-bs 1`` for the first run. See
:doc:`mlx` for memory controls and MLX-specific checkpoint behavior.

Agentic rollout path
--------------------

Agentic rollout is for tasks where the model interacts with tools, games,
services, or an environment before AReno scores the trajectory.

.. code-block:: bash

   areno train \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --algo gspo

Read :doc:`/reference/agentic-rollout-api` for the agentic rollout boundary and
:doc:`/cookbook/tictactoe-agentic-rl` for the first recipe.
