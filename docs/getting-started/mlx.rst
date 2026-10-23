:orphan:

Apple Silicon and MLX
=====================

AReno uses its native MLX backend on Apple Silicon. The same ``areno train``,
``areno serve``, dataset-loader, reward-function, agentic-rollout, and
``Trainer`` interfaces are shared with CUDA; only the backend implementation
changes. Backend selection is automatic and does not fall back across
platforms:

* Linux selects CUDA.
* macOS on ``arm64`` selects MLX.
* Other host/platform combinations fail early with an explicit error.

Requirements and installation
-----------------------------

Use an Apple Silicon Mac running a native ``arm64`` Python 3.10 or newer. A
Python process launched through Rosetta reports ``x86_64`` and is not a
supported MLX runtime. Unified-memory capacity limits the checkpoint size,
optimizer state, KV cache, and media features that can be resident together.

The CUDA-oriented ``scripts/install.sh`` is not the macOS installation path.
Install from the source checkout with pip:

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

Platform markers install ``mlx``, ``mlx-lm``, and ``mlx-vlm`` on Apple
Silicon without installing the Linux-only PyTorch, CUDA, Flash Linear
Attention, torchvision, or media-decoding dependency set. Verify the selected
runtime from Python:

.. code-block:: bash

   python -c "from areno.api import DefaultBackend; print(DefaultBackend)"

The result is ``BackendType.MLX``.

Training
--------

The CLI selects MLX automatically; there is no backend flag:

.. code-block:: bash

   areno train \
     --ckpt /path/to/model \
     --dataset-path /path/to/train.jsonl \
     --dataset-loader-fn /path/to/dataset_loader.py \
     --reward-fn-path /path/to/reward.py \
     --algo gspo \
     --batch-size 1 \
     --n-samples 8 \
     --mini-bs 1 \
     --max-running-prompts 8

MLX supports the built-in SFT, DPO, GRPO, GSPO, and PPO trainer paths. It runs
in one process on the Mac's unified-memory device, so tensor/data parallelism,
CUDA device lists, a separate rollout device partition, NCCL policy sync,
CUDA graphs, and FlashAttention selection do not apply. The CLI normalizes its
default ``world-size`` and ``tp-size`` values to one on MLX; when specifying
them explicitly, use ``--world-size 1 --tp-size 1``.

``--mini-bs`` has the same meaning on both backends: it is the number of
training rows in one gradient microbatch. ``--gradient-accumulation-steps``
controls how many such microbatches contribute to an optimizer update.

LoRA
~~~~

Dense Qwen3 text checkpoints support the same standard PEFT-compatible LoRA
configuration used by the CUDA backend. The base model stays frozen and only
the requested adapter A/B arrays are optimized:

.. code-block:: bash

   areno train \
     --ckpt /path/to/qwen3-model \
     --dataset-path /path/to/train.jsonl \
     --dataset-loader-fn /path/to/dataset_loader.py \
     --algo sft \
     --lora-rank 8 \
     --lora-alpha 16 \
     --save-path /path/to/adapters \
     --save-interval 100

Use ``--lora-adapter-path /path/to/peft-adapter`` to initialize training from
an existing standard PEFT directory. The adapter metadata is authoritative for
rank, alpha, dropout, and target modules. The same option can serve an exported
adapter with ``areno serve`` while ``--model-path`` continues to identify its
base checkpoint.

MLX LoRA requires a text checkpoint, ``--lora-dropout 0``, and
``--reference-mode independent``. Ordinary LoRA also supports MLX quantized
layers, including the online quantization path selected by
``--q4``. The explicit QLoRA path below adds stricter format
validation and base-checkpoint fingerprints. Qwen3 MoE, multimodal LoRA, multiple adapters,
and ``reuse_actor_base`` are not supported.

QLoRA with a prequantized base (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--qlora`` supports SFT on dense Qwen3 and Ling-3.0-tiny with a local MLX
4-bit affine checkpoint, group size 64. This uses MLX quantized matmuls with
floating LoRA parameters; it is not CUDA NF4, double quantization or a paged
optimizer. Other quantization formats, DPO/RL and auxiliary roles are rejected.

Convert a local floating checkpoint once, before training:

.. code-block:: bash

   python -m mlx_lm convert --hf-path "$MODEL_DIR" --mlx-path "$QUANTIZED_MODEL_DIR" \
     -q --q-bits 4 --q-group-size 64 --q-mode affine

Use the resulting directory as ``--ckpt`` and add ``--qlora`` to the LoRA SFT
command. SDK callers use ``MlxConfig(lora=LoraConfig(qlora=True, ...))``.
A floating checkpoint is not automatically quantized at training startup.
Do not combine ``--qlora`` with ``--q4``: online quantization
uses ordinary LoRA and its own adapter metadata. Quantized inference without
an adapter does not require ``--qlora``.
The converter respects the model's quantization predicate: Ling keeps its
embeddings, LM head and protected MLA/special projections floating. The LoRA
target list is unchanged, including adapters on those floating MLA projections.

Adapter exports add ``areno_quantization_config.json`` to the standard PEFT
files. This records the actual per-layer quantization and SHA-256 hashes of
the base config and safetensors files. Reload with the same base and
``--lora-adapter-path``; QLoRA is inferred from the metadata. Files may be
relocated, but changing, reformatting or resharding the base requires a new
adapter identity. Startup hashes the base files to detect accidental mismatch.
Do not remove the sidecar or treat an MLX affine adapter as a CUDA NF4 artifact.
Only adapter weights are restored; continuing training creates fresh optimizer
state, rather than resuming Adam moments or the original step counter.

Quantization reduces frozen-weight storage. LoRA parameters, gradients,
optimizer state and activations still consume memory; long-context capacity
must be measured for the selected adapter scope and batch size.

The implementation at ``ebe415f`` was validated on a 48 GiB M4 Pro Mac mini
with MLX 0.32.3 and MLX-LM 0.32.0 on 2026-10-21. Qwen's standalone test entry
point was corrected at ``36bff8f`` without changing runtime code. Both real
models used rank 8, alpha 16, batch/microbatch 1, AdamW at learning rate 0.001,
gradient checkpointing enabled and compilation disabled. Inputs were bounded
repetitions of "The capital of France is Paris.", with the first token masked.
These are correctness/capacity checks, not convergence or accuracy evaluations.

.. list-table:: Real-checkpoint MLX allocation peaks (GiB)
   :header-rows: 1

   * - Model
     - Tokens
     - QLoRA
     - Prior floating LoRA
   * - Ling-3.0-tiny
     - 512
     - 10.59
     - 20.49
   * - Ling-3.0-tiny
     - 1024
     - 15.52
     - 25.32
   * - Ling-3.0-tiny
     - 1536
     - 22.96
     - 32.73
   * - Ling-3.0-tiny
     - 2048
     - 32.93
     - First-step Metal OOM
   * - Qwen3-0.6B
     - 512
     - 2.19
     - Not measured

All five QLoRA cases passed generation, two finite-gradient training steps,
bitwise frozen-base checks, adapter export/reload with zero maximum absolute
logprob difference, and another training step. Ling retained the expanded
279-module scope (149,671,680 adapter parameters), with 255 quantized base
modules. Qwen covered all seven projection targets over 28 layers (196 adapter
modules, 5,046,272 adapter parameters). Ling's 2048-token losses were
0.39899 and 0.06770, followed by 0.02743 after reload. Qwen's were 0.24959 and
0.05839, followed by 0.16546 with fresh optimizer state after reload.

The floating comparison is from the earlier ``a811115`` validation on the
same host and adapter scope. Peaks are MLX allocations, not whole-machine
memory. Five-second host samples showed no swap increase during any Ling
QLoRA case (740.31 MiB throughout). The results do not establish sustained
stability, larger batches, 4096-token support, or quantization quality parity.

Reproduce with local checkpoints; the tests never download models:

.. code-block:: bash

   ARENO_E2E_MLX_QLORA_SYNTHETIC=1 \
     python -m pytest -q tests/test_mlx_qlora_e2e.py -k upstream

   ARENO_E2E_MLX_QLORA=1 \
   ARENO_E2E_MLX_BAILING_EXPANDED=1 \
   ARENO_E2E_MLX_BAILING_MODEL="$QUANTIZED_LING_DIR" \
   ARENO_E2E_MLX_BAILING_TOKENS=2048 \
     python -m pytest -q -s tests/test_mlx_bailing_v3_lora_e2e.py -k real \
       --basetemp outputs/ling-qlora-validation-new

   ARENO_E2E_MLX_QWEN_QLORA_MODEL="$QUANTIZED_QWEN_DIR" \
     python -m pytest -q -s tests/test_mlx_qlora_e2e.py -k real \
       --basetemp outputs/qwen-qlora-validation-new

Ling-3.0-tiny LoRA (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MLX LoRA resolver also implements Ling-3.0-tiny attention and MLP projections for
``model_type=bailing_hybrid`` with architecture ``BailingMoeV3ForCausalLM``.
Attention-only and expanded adapters have passed reduced-model numerical,
training and reload checks on Apple Silicon, including all four combinations
of compilation and gradient checkpointing. This is not a long-context capacity
or convergence guarantee.

Use an MLX-LM version containing ``mlx_lm.models.bailing_moe_v3`` and its
architecture mapping. The implementation was designed against upstream
revision ``1840cbba9cb3f8125b8cf7e137606256a2f162d1``. MLX-LM 0.31.3 in the
development environment did not contain this module; satisfying AReno's
dependency lower bound alone does not establish Ling compatibility.

Explicitly select a nonempty subset of these targets:

* KDA: ``q_proj``, ``k_proj``, ``v_proj``, ``f_proj``, ``o_proj``, ``g_proj``.
* MLA: ``q_a_proj``, ``q_b_proj``, ``kv_a_proj_with_mqa``, ``dense``, ``g_proj``.
* Ordinary MLP, shared experts and routed experts: ``gate_proj``, ``up_proj``,
  ``down_proj``. Each suffix selects all three kinds of MLP; selecting only
  shared or only routed experts is not supported.

For example, after the validation gates have passed, use a local non-quantized
model and short Alpaca-format training data:

.. code-block:: bash

   areno train \
     --ckpt "$MODEL_DIR" \
     --dataset-path "$TRAIN_JSONL" \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --algo sft --world-size 1 --tp-size 1 \
     --batch-size 1 --mini-bs 1 \
     --lora-rank 8 --lora-alpha 16 \
     --lora-target-modules q_proj,k_proj,v_proj,f_proj,o_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,dense \
     --activation-checkpointing --max-steps 2 \
     --save-path outputs/ling-tiny-mlx-lora --save-interval 1

All applicable layers must contain the requested ordinary linear projections
with the expected dimensions, or packed SwitchLinear projections for routed
experts. The example above retains attention-only training. To enable the
extended scope, append ``g_proj,gate_proj,up_proj,down_proj`` to its target list.
Routed-expert adapters can be much larger than attention adapters; initialization
logs module counts, adapter parameter count and trainable percentage.

``b_proj`` and ``kv_b_proj`` remain excluded. In MLX, ``kv_b_proj`` is transformed
into two head-specific modules; ordinary Linear LoRA cannot wrap it directly.
The architectural setting ``no_kda_lora=true`` does not prohibit adding
post-training adapters.

Only adapter arrays are trainable. All base weights, including routers, expert biases, expert weights,
embeddings, norms and convolutions remain frozen. Adapter saving and loading
use the same PEFT two-file format and metadata precedence as Qwen3. Actual
CUDA/Transformers interoperability remains a separate validation gate. Packed
expert adapters are exported as per-expert ``mlp.experts.<id>`` A/B tensors.
The file format alone does not imply CUDA target support: in particular the
current CUDA Ling resolver does not inject MLA ``g_proj`` adapters.

The opt-in validation entry points do not download assets:

.. code-block:: bash

   ARENO_E2E_MLX_BAILING_SYNTHETIC=1 \
     python -m pytest -q tests/test_mlx_bailing_v3_lora_e2e.py -k reduced

   ARENO_E2E_MLX_BAILING_EXPANDED=1 \
   ARENO_E2E_MLX_BAILING_MODEL="$MODEL_DIR" \
     python -m pytest -q -s tests/test_mlx_bailing_v3_lora_e2e.py -k real \
       --basetemp outputs/ling-lora-validation-new

Choose a fresh ``--basetemp`` directory: pytest clears an existing directory
at that path. The real-model gate defaults to 128 tokens and two optimizer
steps, then reloads the adapter and trains another step. Set
``ARENO_E2E_MLX_BAILING_STEPS=10`` for the stability check or
``ARENO_E2E_MLX_BAILING_TOKENS`` for the bounded synthetic input length.
Each real-model test directory contains ``ling-validation.json`` with step
metrics and cumulative memory peaks. Memory-pressure/swap monitoring, a
coherent-generation assessment, and the serve API check remain separate
manual acceptance checks. These tests do not certify a memory budget by
themselves.

The expanded real-checkpoint gate passed on a 48 GiB Mac mini with MLX 0.32.3
and MLX-LM 0.32.0 on 2026-10-21: all 24 layers, rank 8, alpha 16,
gradient checkpointing enabled, compilation disabled, and 32 input tokens.
It covered 279 runtime projection modules (including 69 packed routed-expert
projections), with 149,671,680 trainable adapter parameters. Two optimizer
steps produced losses 2.81108 and 1.27752; export/reload had zero maximum
absolute logprob difference, and a further optimizer step completed with loss
0.46292. Frozen base weights were unchanged. The cumulative MLX allocation
peak was approximately 17.58 GiB; this is not total system memory usage or a
budget for longer sequences. Cross-backend CUDA/Transformers loading was not
exercised by this gate.

Memory controls
~~~~~~~~~~~~~~~

Start with conservative sequence lengths and concurrency. The following
options have the largest effect on MLX unified-memory use:

``--mini-bs 1``
   Limits temporary training activations to one row at a time. It does not
   change the rollout sample count.

``--max-running-prompts N``
   Caps active rollout sequences. Reduce it when KV cache or multimodal
   prefill features dominate memory.

``--drop-rollout-state``
   Releases completed rollout KV/cache state at the rollout-session boundary
   instead of retaining it for the next rollout.

``--adam-8bit``
   Stores non-embedding Adam moments in the same block-wise dynamic 8-bit
   representation used by the CUDA backend. Token-embedding optimizer moments
   stay FP32, selected by parameter identity rather than name matching; model
   weights, gradients, and forward behavior are unchanged. This reduces
   optimizer memory for the remaining parameters; validate convergence for the
   target task.

``--activation-checkpointing``
   Recomputes supported decoder activations during backward. It is enabled by
   default.

For multimodal models, towers and projectors/mergers are frozen by default.
Use ``--unfreeze-mm-tower`` or ``--unfreeze-mm-projector`` only when the task
requires those parameters to learn; each adds gradients and optimizer state.
Their learning-rate schedules can be controlled independently with
``--mm-tower-lr`` and ``--mm-projector-lr`` plus the corresponding
``*-min-lr``, ``*-lr-steps``, and ``*-lr-style`` options.

Serving and continuous batching
-------------------------------

Start the same OpenAI-compatible server used on CUDA:

.. code-block:: bash

   areno serve \
     --model-path /path/to/model \
     --world-size 1 \
     --tp-size 1 \
     --max-running-prompts 16 \
     --port 8000

The MLX runtime keeps one rollout scheduler alive for the server lifetime.
Compatible requests submitted while decoding is active are admitted into the
continuous batch. Requests with different generation settings are scheduled
separately. ``--eager-decode``, ``--attn-backend``, and CUDA graph progress
fields are CUDA-specific and do not configure MLX execution.

To verify refill rather than only concurrent HTTP success, submit long
requests followed by short probes. Continuous batching is demonstrated when
the probes are submitted while long requests remain active and at least one
probe completes before the earlier long-request group has drained.

Checkpoints
-----------

The MLX providers can load compatible Hugging Face-layout and MLX-native
checkpoints supported by ``mlx-lm`` or ``mlx-vlm``. Full-weight training saves
the policy in native MLX format, including model configuration, safetensors,
tokenizer or processor assets, and ``areno_mlx_state.json``. Reload that saved
directory directly with either ``--ckpt`` or ``--model-path``.

LoRA training instead saves an adapter-only standard PEFT directory containing
``adapter_config.json`` and ``adapter_model.safetensors``. The frozen base
checkpoint is not copied. Continue adapter training with the base checkpoint
as ``--ckpt`` plus the saved directory as ``--lora-adapter-path``; use the same
pair with ``--model-path`` and ``--lora-adapter-path`` for serving. SDK callers
can write the same artifact explicitly through ``Trainer.export_adapter``.
Optimizer and scheduler state restart with the new trainer process.

An MLX full-weight checkpoint is not advertised as a portable
CUDA/Transformers export. Some unquantized model layouts may happen to be
compatible, but MLX model sanitization, tensor layout, and quantization are
model-family concerns. Test the target loader instead of renaming metadata.
LoRA tensors use standard PEFT layout, but model targets and base formats still
need to match the destination runtime. MLX QLoRA additionally requires its exact
quantized base and metadata sidecar; it is not a CUDA NF4 checkpoint. Its
validated generation path is the Trainer rollout API; serving is a separate
validation gate. Optimizer and scheduler state are not part of the current MLX
checkpoint round trip.

Models and multimodal input
---------------------------

Text model availability follows ``mlx-lm`` and multimodal availability
follows ``mlx-vlm``. A model must have both a supported MLX implementation and
a compatible tokenizer or processor. This differs from CUDA, where AReno's
own model adapters define support; support on one backend does not imply
support on the other.

The shared message schema accepts image, audio, video, and combined media.
Actual modality support still depends on the loaded ``mlx-vlm`` model and its
processor. MLX uses processor-native NumPy features before converting them to
MLX arrays, so a Torch installation is not required for supported PIL image
processors. Validate each audio/video codec in the same environment before a
long run.

SDK configuration
-----------------

Omitting ``backend_type`` is recommended because it selects the native
backend. Advanced SDK users can configure MLX explicitly without changing the
shared Trainer API:

.. code-block:: python

   from areno import Trainer
   from areno.api import MLX, MlxConfig

   trainer = Trainer(
       world_size=1,
       model_path="/path/to/model",
       backend_type=MLX,
       custom_config=MlxConfig(
           max_running_prompts=8,
           prefill_batch_size=2,
           completion_batch_size=8,
           max_kv_size=4096,
           keep_rollout_state=False,
           logits_chunk_size=2048,
           gradient_checkpointing=True,
       ),
   )
   trainer.init()

Do not pass ``CudaConfig`` with ``backend_type=MLX`` or ``MlxConfig``
with ``backend_type=CUDA``; typed configuration mismatches fail during
construction.
