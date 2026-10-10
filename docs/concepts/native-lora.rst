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

* Llama, OLMo2, Qwen3 and Qwen3-MoE
* Qwen3.5 dense, MoE and vision-language variants
* Gemma4, Phi4MM and MiniCPM-V 4.6 language projections
* Legacy Bailing-MoE and Bailing-MoE V3 native projections

Bindings follow the working full-parameter model's native projection layout,
including packed and replicated projections. A checkpoint architecture must
first be supported by that model's FFT constructor and loader. LoRA does not
add a separate ``no_kda_lora`` flag requirement; it does not introduce a missing
two-stage KDA base architecture. Native-module coverage is broader than the
real-checkpoint profiles qualified by an individual experiment.

On Apple Silicon, the MLX backend supports dense Qwen3 LoRA and includes an
experimental Ling-3.0-tiny attention-only resolver. Ling MLX numerical and
real-checkpoint validation is pending. Its target subset differs from CUDA:
see :doc:`../getting-started/mlx` for the explicit target list, MLX-LM
requirements and validation entry points.

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
with a configuration error. Dropout on the adapter is a LoRA-specific option
rather than an FFT compatibility requirement.

Existing FFT multimodal tower/projector unfreeze options can accompany language
LoRA. Those explicitly selected media parameters remain trainable; other base
parameters stay frozen. Existing adaptive router-bias updates also remain
available. The engine publishes this additional policy state alongside A/B,
and the base-only reference view restores its original checkpoint values.
Routed-expert rollout consumes merged inference tiles through the native fused
path, with graph buffers refreshed after updates.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Each LoRA save directory contains
``adapter_config.json`` and ``adapter_model.safetensors`` files. A/B-only saves
are standard PEFT LoRA artifacts. When media unfreeze or adaptive router state
is present, the artifact uses ``peft_type=ARENO_LORA_POLICY`` and adds
``areno_policy_state.safetensors``; reload these combined policies through
AReno. Step-local routing counters are excluded. A standard PEFT adapter can
also initialize a new run with these FFT options enabled. The save is
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

Hybrid artifacts
----------------

``full_parameter_targets`` may be set in ``LoraConfig`` or through
``--full-parameter-targets``. Selected full parameters are saved as canonical
checkpoint tensors alongside A/B in a versioned ``ARENO_HYBRID`` artifact.
Their complete current values are restored on reload; unselected parameters
come from the original base checkpoint. Adaptive router buffers and media
state retain their existing policy-state sidecar. This is an AReno artifact,
not an external PEFT codec or an optimizer/scheduler/RNG resume checkpoint.

Hybrid training requires an independent frozen reference: ``reuse_actor_base``
is rejected because selected actor base parameters are trainable. Native NF4
QLoRA remains available for LoRA; combining QLoRA with explicit full targets
is currently rejected because its quantized base has no dense checkpoint layout.
