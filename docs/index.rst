.. rst-class:: landing-page

areno
=====

.. raw:: html

   <div class="hero">
     <h1>areno</h1>
     <p>A lightweight CUDA-native stack for local LLM post-training and serving, including agentic RL. areno brings rollout, scoring, inference, and optimizer steps into one engine, so post-training loops stay compact and fast.</p>
     <div class="badges">
       <span class="badge">Lightweight runtime</span>
       <span class="badge">Unified train-infer engine</span>
       <span class="badge">Agentic rollouts</span>
       <span class="badge">Minimal dependencies</span>
       <span class="badge">SFT / DPO / GSPO / GRPO / PPO</span>
     </div>
   </div>

.. raw:: html

   <div class="areno-feature-grid">
     <div class="areno-feature-card"><strong>Lightweight by design</strong><p>The core stack stays small: PyTorch plus focused CUDA/attention dependencies, without a separate serving framework or trainer framework in the hot path.</p></div>
     <div class="areno-feature-card"><strong>Train and infer together</strong><p>Rollout, scoring, optimizer steps, CUDA graph handling, agentic proxying, and checkpoint I/O live in one local engine for a direct post-training loop.</p></div>
     <div class="areno-feature-card"><strong>Kernel-first runtime</strong><p>Fused CUDA paths cover routing, token movement, top-k, embedding, activation, normalization, and MoE hot paths.</p></div>
     <div class="areno-feature-card"><strong>Agentic by default</strong><p>Agent functions call a local OpenAI-compatible proxy and return explicit trajectories; areno converts model responses to completions, tokens, logprobs, rewards, and loss masks for training.</p></div>
   </div>

Quick start
-----------

Install in an existing CUDA + PyTorch environment:

.. code-block:: bash

   pip install psutil
   pip install flash-linear-attention
   pip install -e . --no-build-isolation

Install ``flash-attn`` as an optional extra only when using the default
``--attn-backend flash`` path. ``flash-attn`` is not required for
``--attn-backend native`` or for GPUs such as Tesla T4 where AReno falls back
to native attention.

Check whether the environment is ready:

.. code-block:: bash

   areno check

Use ``areno env --json`` when filing setup issues.

Run a tiny training smoke test when you only want to verify the wiring:

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

This is a smoke/sanity task for wiring, not a quality benchmark. It requires a
CUDA-capable NVIDIA GPU; CPU-only machines cannot run the AReno training
engine.

Run GSPO on a GSM8K-style dataset:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1

Start an OpenAI-compatible server:

.. code-block:: bash

   areno serve \
     --model-path /path/to/model \
     --tp-size 1 \
     --world-size 1 \
     --port 8000

Run an agentic rollout task:

.. code-block:: bash

   python examples/agentic/tictactoe/dataset_generator.py \
     --output /tmp/areno-tictactoe.jsonl \
     --count 2048 \
     --seed 2026

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/areno-tictactoe.jsonl \
     --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1

DuelGrid is a larger agentic demo with a browser game UI and multi-action
turns. Before GSPO/RLVR post-training, Gemma-E2B-it often moves back and forth
without progress. After training, it learns to collect pickups, chase the user,
attack when in range, and avoid trap tiles.

.. list-table::
   :header-rows: 1
   :widths: 1 1 1

   * - Train before
     - Reward
     - Train after
   * - .. image:: ../examples/agentic/duelgrid/images/train_before.gif
          :alt: DuelGrid before training
          :width: 260px
     - .. image:: ../examples/agentic/duelgrid/images/train_reward.jpg
          :alt: DuelGrid reward curve
          :width: 260px
     - .. image:: ../examples/agentic/duelgrid/images/train_after.gif
          :alt: DuelGrid after training
          :width: 260px

See ``examples/agentic/duelgrid`` for the rule engine, fixed-path dataset
loader, reward function, OpenAI-compatible agent, and browser UI.

What areno owns
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Layer
     - Scope
   * - Kernels
     - Fused CUDA paths in ``areno_accel`` for runtime hot paths.
   * - Engine
     - Tensor-parallel workers, KV/cache layout, CUDA graph support, rollout state, scoring, optimizer steps, and checkpoint I/O.
   * - Algorithms
     - SFT, DPO, GSPO, GRPO, PPO, and agentic rollouts are implemented inside the project rather than delegated to a separate trainer framework.
   * - Checkpoints
     - Hugging Face-compatible load/save adapters for supported model families.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting-started/build
   models/supported

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli/training
   cli/dataset_loaders
   cli/observability
   cli/inference
   cli/diagnostics
   sdk/trainer
