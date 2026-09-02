Backends
========

AReno exposes one training, serving, checkpoint, dataset, reward, and agentic
rollout interface over two native execution backends. The CLI selects the
backend from the host platform and never silently falls back to another
runtime.

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Host
     - Backend
     - Execution model
     - Native checkpoint
   * - Linux or WSL2 with NVIDIA GPU
     - CUDA and PyTorch
     - Multi-process, TP/DP, optional separate rollout GPUs
     - Hugging Face-oriented AReno layout
   * - Apple Silicon macOS
     - MLX
     - Single-process unified-memory runtime
     - MLX safetensors and processor assets

Linux and CUDA
--------------

The CUDA backend owns AReno's process workers, tensor/data parallel layout,
fused ``areno_accel`` kernels, optional FlashAttention, CUDA graph decode, and
NCCL policy synchronization. Use it for Linux ``x86_64`` or ``aarch64`` with
a CUDA-enabled PyTorch 2.6 or newer installation.

Read :doc:`cuda` for installation, distributed training, memory controls,
continuous-batch serving, checkpoint behavior, model support, and
``CudaConfig``.

Apple Silicon and MLX
---------------------

The MLX backend owns one in-process model for rollout, scoring, and training,
plus a long-lived continuous-batch scheduler. It installs only the Apple
Silicon dependency set and does not require Torch or CUDA. MLX runs with
``world-size=1`` and ``tp-size=1`` and saves native MLX checkpoints.

Read :doc:`mlx` for installation, unified-memory controls, continuous-batch
serving, checkpoint behavior, ``mlx-lm``/``mlx-vlm`` model support, and
``MlxConfig``.

Shared contract
---------------

Both backends support the built-in SFT, DPO, GRPO, GSPO, and PPO trainer paths
through the same CLI and ``Trainer`` methods. They share algorithm schemas,
``TrainSequence`` batches, microbatch semantics, role APIs, multimodal message
format, rollout-session lifecycle, and OpenAI-compatible serving API.

Backend-native model implementations, kernels, distributed topology, and
checkpoint formats remain intentionally separate. A model supported by an
AReno CUDA adapter is not automatically supported by ``mlx-lm`` or
``mlx-vlm``, and an MLX checkpoint is not automatically a portable CUDA
checkpoint.
