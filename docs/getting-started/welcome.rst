.. rst-class:: landing-page

AReno documentation
===================

.. raw:: html

   <div class="areno-hero">
     <div class="areno-hero-copy">
       <p class="areno-eyebrow">Local post-training and serving</p>
       <h1>Train and serve local LLMs with one CUDA-native loop.</h1>
       <p class="areno-lede">AReno keeps rollout, reward scoring, inference, optimizer steps, and checkpoint I/O in one compact engine for SFT, DPO, GSPO, GRPO, PPO, and agentic RL workflows.</p>
       <div class="areno-hero-actions" aria-label="Primary documentation links">
        <a class="areno-button areno-button-primary" href="installation.html">Get started</a>
        <a class="areno-button" href="../cli/training.html">Train a model</a>
       </div>
     </div>
   </div>

Start
-----

.. raw:: html

   <div class="areno-command-grid">
     <div class="areno-command-card">
       <p class="areno-card-kicker">Install</p>
       <h3>Build against your CUDA PyTorch environment.</h3>
      <pre><code>pip install psutil
   pip install flash-linear-attention
   pip install -e . --no-build-isolation</code></pre>
       <p>Use <code>ARENO_BUILD_EXT=0</code> for metadata-only docs or package checks on CPU-only machines.</p>
     </div>
     <div class="areno-command-card">
       <p class="areno-card-kicker">Check</p>
       <h3>Verify the local runtime before training.</h3>
      <pre><code>areno check
   areno env --json</code></pre>
       <p><code>areno check</code> reports common CUDA, PyTorch, extension, and platform setup issues with next steps.</p>
     </div>
   </div>

``flash-attn`` is optional unless you use the default ``--attn-backend flash``
path. Use ``--attn-backend native`` when you want to run without FlashAttention
or when the local GPU is unsupported by FlashAttention.

Core workflows
--------------

.. raw:: html

   <div class="areno-doc-grid">
    <a class="areno-doc-card" href="../cli/training.html">
       <span class="areno-icon">01</span>
       <strong>Train</strong>
       <p>Run SFT, DPO, GSPO, GRPO, or PPO from the CLI with dataset loading, rollout, reward scoring, and checkpoint saving in one loop.</p>
     </a>
    <a class="areno-doc-card" href="../cli/inference.html">
       <span class="areno-icon">02</span>
       <strong>Serve</strong>
       <p>Start an OpenAI-compatible chat-completions server backed by the local AReno inference engine.</p>
     </a>
    <a class="areno-doc-card" href="../sdk/trainer.html">
       <span class="areno-icon">03</span>
       <strong>Customize</strong>
       <p>Use <code>from areno import Trainer</code> for custom rollout, reward, loss, and checkpoint loops.</p>
     </a>
    <a class="areno-doc-card" href="../models/supported.html">
       <span class="areno-icon">04</span>
       <strong>Load models</strong>
       <p>Review the checkpoint families currently supported by AReno model adapters.</p>
     </a>
   </div>

.. raw:: html

   <div class="areno-command-grid areno-command-grid-wide">
     <div class="areno-command-card">
       <p class="areno-card-kicker">Training</p>
       <h3>Run a small GSPO smoke task.</h3>
      <pre><code>areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1</code></pre>
     </div>
     <div class="areno-command-card">
       <p class="areno-card-kicker">Serving</p>
       <h3>Open a local chat-completions endpoint.</h3>
      <pre><code>areno serve \
     --model-path /path/to/model \
     --tp-size 1 \
     --world-size 1 \
     --port 8000</code></pre>
     </div>
   </div>

Training and serving require a CUDA-capable NVIDIA GPU. CPU-only machines can
run docs, packaging checks, and lightweight CPU tests, but cannot run the AReno
training or serving engine.

Agentic rollout
---------------

.. raw:: html

   <div class="areno-split">
     <div>
       <p class="areno-eyebrow">Agentic RL</p>
       <h3>Collect trajectories through a local OpenAI-compatible proxy.</h3>
       <p>Agent functions call the local server, return explicit trajectory turns, and let AReno convert responses into completions, tokens, logprobs, rewards, and loss masks.</p>
     </div>
     <div class="areno-terminal areno-terminal-compact" aria-label="Agentic rollout command">
       <div class="areno-terminal-bar"><span></span><span></span><span></span></div>
      <pre><code>areno train \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --algo gspo</code></pre>
     </div>
   </div>

DuelGrid is a browser-game demo with multi-action turns. Before GSPO/RLVR
post-training, Gemma-E2B-it often moves back and forth without progress. After
training, it learns to collect pickups, chase the user, attack when in range,
and avoid trap tiles.

.. rst-class:: areno-showcase-table

.. list-table::
   :header-rows: 1
   :widths: 1 1 1

   * - Train before
     - Reward
     - Train after
   * - .. image:: ../../examples/agentic/duelgrid/images/train_before.gif
          :alt: DuelGrid before training
          :width: 260px
     - .. image:: ../../examples/agentic/duelgrid/images/train_reward.jpg
          :alt: DuelGrid reward curve
          :width: 260px
     - .. image:: ../../examples/agentic/duelgrid/images/train_after.gif
          :alt: DuelGrid after training
          :width: 260px

See ``examples/agentic/duelgrid`` for the rule engine, fixed-path dataset
loader, reward function, OpenAI-compatible agent, and browser UI.

What AReno owns
---------------

.. raw:: html

   <div class="areno-layer-list">
     <div><span>Kernels</span><p>Fused CUDA paths in <code>areno_accel</code> for runtime hot paths.</p></div>
     <div><span>Engine</span><p>Tensor-parallel workers, KV/cache layout, CUDA graph support, rollout state, scoring, optimizer steps, and checkpoint I/O.</p></div>
     <div><span>Algorithms</span><p>SFT, DPO, GSPO, GRPO, PPO, and agentic rollouts implemented inside the project rather than delegated to a separate trainer framework.</p></div>
     <div><span>Checkpoints</span><p>Hugging Face-compatible load/save adapters for supported model families.</p></div>
   </div>
