:orphan:

Linux and CUDA
==============

AReno uses its native CUDA backend on Linux. The same ``areno train``,
``areno serve``, dataset-loader, reward-function, agentic-rollout, and
``Trainer`` interfaces are shared with MLX; only the backend implementation
changes. Backend selection is automatic and does not fall back across
platforms:

* Linux selects CUDA.
* macOS on ``arm64`` selects MLX.
* Other host/platform combinations fail early with an explicit error.

Requirements and installation
-----------------------------

Use Linux ``x86_64`` or ``aarch64`` with an NVIDIA GPU, a compatible NVIDIA
driver and CUDA toolkit, and CUDA-enabled PyTorch 2.6 or newer. WSL2 follows
the same Linux path. Native Windows and WSL1 are not supported.

The installer validates the environment, reuses or creates a Python virtual
environment, builds ``areno_accel``, selects the attention setup, and runs the
readiness checks:

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno
   bash scripts/install.sh

AReno does not install or upgrade PyTorch automatically because the wheel must
match the host driver and CUDA toolkit. Install a compatible CUDA-enabled
PyTorch build before running the installer. To inspect the plan without
changing the environment:

.. code-block:: bash

   bash scripts/install.sh --dry-run

After installation, collect the runtime checks and environment report:

.. code-block:: bash

   areno check
   areno env --json

Training
--------

The CLI selects CUDA automatically; there is no backend flag:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --world-size 1 \
     --tp-size 1 \
     --batch-size 1 \
     --n-samples 8 \
     --mini-bs 1

CUDA supports the built-in SFT, DPO, GRPO, GSPO, and PPO trainer paths. It can
run a colocated rollout/training engine or place rollout on a separate CUDA
device partition. ``--world-size`` selects the total training ranks,
``--tp-size`` selects training tensor parallelism, and the world size must be
divisible by TP size.

For a separate rollout engine, set ``--train-devices``,
``--rollout-devices``, and ``--rollout-tp-size``. AReno streams updated policy
weights directly between GPU process groups after an optimizer step; it does
not stage the policy through a checkpoint file.

``--mini-bs`` has the same meaning on both backends: it is the number of
training rows in one gradient microbatch. ``--gradient-accumulation-steps``
controls how many such microbatches contribute to an optimizer update.

Memory controls
~~~~~~~~~~~~~~~

The main CUDA controls are:

``--mini-bs N``
   Limits temporary training activations per worker. Reduce it before reducing
   the logical rollout batch.

``--max-running-prompts N``
   Caps active rollout sequences and their KV-cache demand.

``--drop-rollout-state``
   Releases completed rollout state at the session boundary rather than
   retaining reusable cache and graph state for the next rollout.

``--activation-checkpointing``
   Recomputes supported decoder activations during backward. It is enabled by
   default.

``--optimizer-state-offload cpu``
   Moves optimizer state to host memory between train calls.

``--optimizer-state-offload disk``
   Keeps optimizer state in process-private persistent raw-mmap files and
   lazily copies buckets back for updates. Also pass
   ``--optimizer-state-offload-dir /path/to/local-nvme``. Disk offload is
   runtime scratch storage rather than a checkpoint. The default
   ``--optimizer-state-offload-batch-size 1`` groups mmap files and flushes;
   reduce it to lower CPU staging memory or increase it to reduce I/O calls.

``--attn-backend flash``
   Uses FlashAttention when the model and GPU support it. Use ``native`` for
   compatibility diagnostics or unsupported GPUs.

``--eager-decode``
   Disables CUDA graph replay for rollout decode. Use it to isolate graph
   capture issues; normal serving and training should retain graph replay when
   the model supports it.

For multimodal models, towers and projectors/mergers are frozen by default.
Use ``--unfreeze-mm-tower`` or ``--unfreeze-mm-projector`` only when required.
Their independent learning-rate schedules use ``--mm-tower-lr`` and
``--mm-projector-lr`` plus the corresponding ``*-min-lr``, ``*-lr-steps``,
and ``*-lr-style`` options.

Serving and continuous batching
-------------------------------

Start the OpenAI-compatible server with the required worker topology:

.. code-block:: bash

   areno serve \
     --model-path /path/to/model \
     --world-size 4 \
     --tp-size 4 \
     --max-running-prompts 32 \
     --port 8000

The CUDA runtime keeps a rollout session open for the server lifetime and
reuses model, KV-cache, and CUDA graph state. Compatible requests submitted
while decoding is active can refill the continuous batch. Requests with
different generation settings are scheduled separately.

Set ``--decode-progress-interval-s`` to report scheduled decode throughput,
active sequences, and whether CUDA graph replay was used. To verify refill
from the client, submit short probes while earlier long requests are active
and confirm a probe completes before the earlier group has drained.

Checkpoints
-----------

The CUDA backend loads model families implemented by AReno's adapters and
saves the trained policy in AReno's Hugging Face-oriented checkpoint layout.
Reload the saved directory directly with either ``--ckpt`` or
``--model-path``. When training and rollout use separate device partitions,
online policy synchronization remains an in-memory NCCL operation; checkpoint
saving is independent of synchronization.

Models and multimodal input
---------------------------

CUDA model availability is listed in :doc:`../models/supported` and is defined
by AReno's model adapters and kernels. This differs from MLX, where
``mlx-lm`` and ``mlx-vlm`` define model construction and weight conversion;
support on one backend does not imply support on the other.

The shared message schema accepts image, audio, video, and combined media.
Actual modality support depends on the loaded model adapter and processor.
Media preprocessing may use Torch, torchvision, PyAV, librosa, or
model-specific processor dependencies installed by the Linux package path.

SDK configuration
-----------------

Omitting ``backend_type`` is recommended because Linux selects CUDA
automatically. Advanced SDK users can configure CUDA explicitly:

.. code-block:: python

   from areno import Trainer
   from areno.api import CUDA, CudaConfig

   trainer = Trainer(
       world_size=4,
       model_path="/path/to/model",
       backend_type=CUDA,
       custom_config=CudaConfig(
           tp_size=4,
           devices=[0, 1, 2, 3],
           max_running_prompts=32,
           runtime={"attn_backend": "flash"},
       ),
   )
   trainer.init()

Do not pass ``MlxConfig`` with ``backend_type=CUDA`` or ``CudaConfig``
with ``backend_type=MLX``; typed configuration mismatches fail during
construction.
