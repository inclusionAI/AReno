.. rst-class:: landing-page

areno
=====

.. raw:: html

   <div class="hero">
     <h1>areno</h1>
     <p>A lightweight CUDA-native post-training and inference stack for local LLMs. areno keeps rollout, inference, scoring, and training inside one engine for a compact train-infer workflow.</p>
     <div class="badges">
       <span class="badge">Lightweight runtime</span>
       <span class="badge">Unified train-infer engine</span>
       <span class="badge">Minimal dependencies</span>
       <span class="badge">SFT / DPO / GSPO / GRPO / PPO</span>
     </div>
   </div>

.. raw:: html

   <div class="card-grid">
     <div class="feature-card"><strong>Lightweight by design</strong><p>The core stack stays small: PyTorch plus focused CUDA/attention dependencies, without a separate serving framework or trainer framework in the hot path.</p></div>
     <div class="feature-card"><strong>Train and infer together</strong><p>Rollout, scoring, optimizer steps, CUDA graph handling, and checkpoint I/O live in one local engine for a direct post-training loop.</p></div>
     <div class="feature-card"><strong>Kernel-first runtime</strong><p>Fused CUDA paths cover routing, token movement, top-k, embedding, activation, normalization, and MoE hot paths.</p></div>
   </div>

Quick start
-----------

Install in an existing CUDA + PyTorch environment:

.. code-block:: bash

   pip install -e . --no-build-isolation
   pip install flash-attn flash-linear-attention

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
     - SFT, DPO, GSPO, GRPO, and PPO are implemented inside the project rather than delegated to a separate trainer framework.
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
   cli/inference
   sdk/trainer
