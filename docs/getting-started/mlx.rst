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
   Stores Adam moment state in the MLX backend's 8-bit representation. This
   reduces optimizer memory; validate convergence for the target task.

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
checkpoints supported by ``mlx-lm`` or ``mlx-vlm``. AReno saves the trained
policy in native MLX format, including model configuration, safetensors,
tokenizer or processor assets, and ``areno_mlx_state.json``. Reload the saved
directory directly with either ``--ckpt`` or ``--model-path``.

An MLX checkpoint is not advertised as a portable CUDA/Transformers export.
Some unquantized model layouts may happen to be compatible, but MLX model
sanitization, tensor layout, and quantization are model-family concerns. Test
the target loader instead of renaming metadata. Optimizer and scheduler state
are not part of the current MLX checkpoint round trip.

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
