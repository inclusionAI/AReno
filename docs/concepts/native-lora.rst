Native LoRA
===========

AReno can train and serve LoRA adapters directly in its CUDA engine. The base
model stays frozen while the LoRA A and B parameters participate in the same
tensor-parallel, data-parallel, sequence-parallel, rollout, and optimizer
paths as full-parameter training. No external PEFT runtime is required during
training or inference.

Support
-------

Native LoRA currently supports these CUDA model adapters:

* Qwen3
* Qwen3-MoE
* Bailing-MoE V3 checkpoints with ``no_kda_lora=true``

The default target modules are ``q_proj``, ``k_proj``, ``v_proj``,
``o_proj``, ``gate_proj``, ``up_proj``, and ``down_proj``. Select a subset
with ``--lora-target-modules``. Bailing-MoE V3 additionally supports its
native attention projection names, including ``q_a_proj``, ``q_b_proj``,
``kv_a_proj_with_mqa``, and ``kv_b_proj``.

LoRA dropout must currently be zero. Standard PEFT LoRA adapters are accepted,
but options that change the adapter structure, such as DoRA, RS-LoRA, bias
training, rank patterns, alpha patterns, or ``modules_to_save``, are rejected
with a configuration error. Bailing-MoE V3 router-bias updates must also be
disabled so the base policy remains frozen.

Train
-----

Set ``--lora-rank`` to enable native LoRA. ``--lora-alpha`` defaults to 16 and
the default target list covers attention and MLP projections:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --world-size 1 \
     --tp-size 1 \
     --lora-rank 8 \
     --lora-alpha 16 \
     --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
     --save-path outputs/qwen3-lora \
     --save-interval 100

The resolved LoRA rank, alpha, dropout, target modules, and adapter path are
shown in the configuration summary printed before model loading.

Agentic LoRA uses the normal agent hooks. For example, this trains the
Tic-Tac-Toe tool-calling policy:

.. code-block:: bash

   python examples/agentic/tictactoe/dataset_generator.py \
     --output /tmp/areno-tictactoe.jsonl \
     --count 2048 \
     --seed 2026

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/areno-tictactoe.jsonl \
     --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --algo gspo \
     --batch-size 1 \
     --n-samples 8 \
     --max-running-prompts 8 \
     --max-new-tokens 3071 \
     --lora-rank 8 \
     --lora-alpha 16 \
     --save-path outputs/tictactoe-lora \
     --save-interval 100

Save and reload
---------------

Each LoRA save directory contains PEFT-compatible
``adapter_config.json`` and ``adapter_model.safetensors`` files. The save is
adapter-only: continue to pass the original base checkpoint with ``--ckpt``.
To initialize a new training run from a saved adapter:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --lora-adapter-path outputs/qwen3-lora/step_000100 \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --save-path outputs/qwen3-lora-continued

Adapter metadata is authoritative when ``--lora-adapter-path`` is present, so
its rank, alpha, dropout, and target modules replace the corresponding CLI
defaults. Adapter-only saves do not contain optimizer, scheduler, or RNG state;
loading one initializes the policy weights for a new run rather than exactly
resuming the old trainer state.

Serve
-----

Serve the frozen base and saved adapter together; merging is not required:

.. code-block:: bash

   areno serve \
     --model-path Qwen/Qwen3-0.6B \
     --lora-adapter-path outputs/qwen3-lora/step_000100 \
     --world-size 1 \
     --tp-size 1 \
     --port 8000

The endpoint remains OpenAI compatible. ``/v1/models`` reports the base model,
and chat completion requests use the loaded adapter.

Reference model reuse
---------------------

For algorithms that require a frozen reference policy, use
``--reference-mode reuse_actor_base`` when the reference is exactly the
actor's frozen base checkpoint. AReno temporarily disables the adapter to
evaluate the base policy, avoiding a second model copy. Keep the default
``independent`` mode when the reference checkpoint is different.
