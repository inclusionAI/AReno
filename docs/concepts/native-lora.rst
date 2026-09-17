Native LoRA
===========

AReno can train and serve LoRA adapters directly in its CUDA engine. By
default the base model stays frozen while the LoRA A and B parameters participate in the same
tensor-parallel, data-parallel, sequence-parallel, rollout, and optimizer
paths as full-parameter training. No external PEFT runtime is required during
training or inference.

Choosing what to train
----------------------

LoRA learns an additive change to a projection without updating its original
weight. For a weight matrix ``W``, the effective weight is
``W + (alpha / rank) * B @ A``. The optimizer updates A and B; W stays frozen.
This reduces gradient and optimizer-state memory, but does not remove the
base weights or the activations needed for backpropagation.

Start with pure LoRA when you want to adapt attention and MLP projections
while leaving other parameters unchanged. Sometimes you want more: a MoE
router, for example, may also need to learn how to dispatch tokens. You can
keep LoRA on the large projections and train the router directly.

Choose the training policy explicitly:

* **Pure LoRA:** set ``--lora-rank`` and ``--lora-target-modules``. Only the
  selected adapters are trainable; other base parameters remain frozen.
* **LoRA plus selected full parameters:** also set
  ``--full-parameter-targets``. These original parameters are updated directly,
  alongside the adapters. Everything else remains frozen.
* **Selected full parameters only:** set ``--full-parameter-targets`` without
  ``--lora-rank`` or ``--lora-adapter-path``. No adapters are created.
* **Normal full-parameter training:** omit all three options. This uses the
  model's normal training path, including its existing multimodal freezing rules.

LoRA mode has no implicit full-parameter defaults. Norms, embeddings, heads,
and routers stay frozen unless explicitly selected. Selecting one fullweight
module does not make the rest of the model trainable.

Rank and alpha
~~~~~~~~~~~~~~

Rank controls adapter size; alpha controls its scale through ``alpha / rank``.
Specify both when comparing runs. Keeping alpha fixed while increasing rank
changes both capacity and scaling. Keeping the ratio fixed removes that
particular scaling change, but does not guarantee equal gradient magnitudes
or learning speed.

A sufficiently large rank can represent a full-rank matrix update. This does
not mean LoRA and direct full-parameter Adam training follow the same trajectory:
they optimize different parameterizations. Matching RL rewards or validation
accuracy is an experimental question, not an API guarantee. AReno supports
standard LoRA scaling, not RS-LoRA.

Support
-------

Native LoRA currently supports these CUDA model adapters:

* Qwen3
* Qwen3-MoE
* Bailing-MoE V3 checkpoints with ``no_kda_lora=true``
* OLMo2 and Phi4MM
* Gemma4
* Qwen3.5 dense and MoE adapters, including the VLM language trunk
* MiniCPM-V 4.6 language trunk
* Legacy Bailing-MoE linear V2

For multimodal models, this support covers native language projections only;
vision/audio towers and projectors are not LoRA targets. Model-family support
does not imply that every checkpoint variant or TP/DP layout is qualified.
Gemma4 and Qwen3.5 routed-MoE adapters use grouped execution during rollout
when expert LoRA is active; their performance is not qualified by dense-model
tests.

This feature does not automatically change model precision. FlashAttention
requires FP16 or BF16 execution tensors. A checkpoint configuration declaring
FP32, including the official OLMo-2-0425-1B release, is not directly qualified
for the FlashAttention path without an explicitly prepared compatible
execution configuration. Checkpoint storage precision and execution precision
must not be confused.

The default target modules are ``q_proj``, ``k_proj``, ``v_proj``,
``o_proj``, ``gate_proj``, ``up_proj``, and ``down_proj``. Select a subset
with ``--lora-target-modules``. Bailing-MoE V3 additionally supports its
native attention projection names, including ``q_a_proj``, ``q_b_proj``,
``kv_a_proj_with_mqa``, and ``kv_b_proj``. Flash V3 checkpoints may select
their fused routed-expert projections as ``linear_fc1`` and ``linear_fc2``.
Concrete projection paths such as ``layers.2.mlp.experts.linear_fc1`` are
accepted when only specific layers should receive adapters.

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

The resolved LoRA rank, alpha, dropout, LoRA/full targets, and adapter path are
shown in the configuration summary printed before model loading.

Selecting full parameters
-------------------------

Use ``--full-parameter-targets`` to update original parameters rather than
attach adapters to them. This is useful for parameters outside the supported
LoRA projections, or modules you deliberately want to train in full. Full
training allocates gradients and optimizer state for those parameters:
selecting a large embedding or expert subtree can substantially increase memory.

The option takes comma-separated selectors, not regexes or wildcards.
Each selector is resolved against the native AReno model:

* An **exact parameter path** selects only that parameter. In a Bailing V3
  model with this layout, ``layers.2.mlp.gate.weight`` selects one router weight.
* An **exact module path** selects its entire parameter subtree.
  ``layers.2.mlp.gate`` includes any other parameters owned by that router,
  not just its weight.
* An **unqualified name** matches parameter names or immediate parent-module
  names across the model. This is intentionally broad: ``weight`` selects
  every parameter named ``weight``. Prefer exact paths for scoped experiments.

These are native parameter/module paths, not necessarily Hugging Face
checkpoint keys. LoRA selectors can identify logical projections inside fused
weights. Full-parameter selectors identify actual physical parameters. If q,
k, and v share a weight, selecting that weight trains all three components;
it does not select just the q slice.

Example: adapters plus a fullweight router
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a compatible Bailing V3 checkpoint, append the following to your normal
training command. The paths match the scoped Flash case in the shared E2E;
check your model layout before reusing them:

.. code-block:: bash

   --lora-rank 64 \
   --lora-alpha 64 \
   --lora-target-modules layers.0.attention.q_proj,layers.0.attention.k_proj,layers.2.mlp.experts.linear_fc1,layers.2.mlp.experts.linear_fc2 \
   --full-parameter-targets layers.2.mlp.gate.weight

Only the listed attention and expert projections receive adapters; one router
weight is trained directly. This does not adapt all layers or train all routers.

Example: train only selected full parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train only that router, append:

.. code-block:: bash

   --full-parameter-targets layers.2.mlp.gate.weight

Do not pass a LoRA rank or adapter path for this mode. The attention, experts,
and all other unselected parameters remain frozen.

Check the selection before a long run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Review the configuration summary and target information emitted during
initialization. Confirm the intended layers and trainable-parameter count.
Unknown selectors fail rather than being silently ignored.

Overlapping full-parameter selectors fail too. Do not select both
``layers.2.mlp.gate`` and ``layers.2.mlp.gate.weight``: the module already
includes the weight. The same base parameter cannot be both LoRA-adapted and
fully trained, including when logical names refer to shared fused storage.

For a comparison with full training, account for every intended parameter.
Increasing LoRA rank does not unfreeze uncovered norms, routers, embeddings,
or heads. Select them explicitly if they are trainable in your baseline.
Use an independent frozen reference when any actor base parameters are trainable.

Agentic training
----------------

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

Each LoRA save directory contains ``adapter_config.json`` and
``adapter_model.safetensors`` files. Pure LoRA saves are standard PEFT
artifacts. Saves containing explicit full parameters use the versioned
``ARENO_HYBRID`` metadata contract and store those canonical tensors beside
LoRA A/B. Hybrid saves contain the complete current values of selected full
parameters, not their differences from the base. Neither format is a complete
standalone model: continue to pass the original base checkpoint with ``--ckpt``.
On reload, saved full parameters replace their base values, while unselected
parameters still come from the base checkpoint. ``ARENO_HYBRID`` is an AReno
format, not a standard PEFT adapter that other serving engines can necessarily
load. Selected-full-parameter-only runs also use this format.
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
defaults; hybrid metadata also restores its full-parameter selectors. Adapter
saves do not contain optimizer, scheduler, or RNG state;
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
This reuse mode is rejected when ``full_parameter_targets`` is non-empty,
because the actor base is then trainable and cannot serve as a frozen reference.
