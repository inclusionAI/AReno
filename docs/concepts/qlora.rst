QLoRA
=====

Add ``--qlora`` to a CUDA training command to train native LoRA adapters on
an NF4 base with double quantization and a paged optimizer::

   areno train --algo gspo --ckpt inclusionAI/Ling-3.0-tiny \
     --model-hub modelscope --qlora --adam-4bit \
     --dataset-path tictactoe.jsonl \
     --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --batch-size 4 --n-samples 4 --mini-bs 1 \
     --lr 1e-5 --min-lr 1e-5 --world-size 1 --tp-size 1

``--qlora`` enables rank 8, alpha 16 LoRA unless overridden with the existing
``--lora-*`` options. The SDK equivalent is ``LoraConfig(qlora=True)``.
It supports Qwen3 dense and Bailing-MoE V3 models with native LoRA support.
QLoRA is opt-in and independent of CCE.

The three components
--------------------

The implementation follows `QLoRA: Efficient Finetuning of Quantized LLMs
<https://arxiv.org/abs/2305.14314>`_:

* **NF4:** frozen native parallel projections and Bailing routed-expert
  projections use a 16-value normal-quantile codebook, packed two weights per
  byte, with an absolute-maximum scale per 64 weights.
* **Double quantization:** scales are mean-centered and encoded using the
  signed dynamic 8-bit floating codebook, with one FP32 scale per 256 scale
  values. Weight storage is approximately 4.127 bits per weight, plus small
  codebooks and the mean offset.
* **Paged optimizer:** persistent moments use ``cudaMallocManaged`` storage.
  CUDA controls page residency. FP32-master AdamW is used by default;
  ``--adam-8bit`` and ``--adam-4bit`` retain their update rules with managed
  state storage. Explicit CPU/disk optimizer offload cannot be combined with
  QLoRA.

Forward and input-gradient computation reconstruct the current projection
in its computation dtype; the full dequantized base is not retained.
Adapters remain trainable. Embeddings, output head, normalization, routers,
convolutions and non-native replicated projections retain their original
precision. Bailing quantized experts use grouped eager rollout execution,
sharing the packed base with training instead of retaining dense fused
inference weights.

Loading currently constructs the original model before quantizing its
projections. Loading-time capacity must therefore accommodate that model.
Saved adapters include ``areno_quantization_config.json`` so reloading the
adapter restores the same base quantization mode.

Memory accounting and validation
--------------------------------

On DGX Spark, CPU and GPU share physical DRAM. Managed allocations enable
paging semantics but do not create additional memory capacity. They are
outside PyTorch's caching allocator; comparisons must include managed bytes
as well as ``torch.cuda.max_memory_allocated()``. NF4 and double quantization
reduce actual storage. Paging alone does not compress optimizer state.

Run the numerical contracts on a CUDA host::

   NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
     python -m pytest -q tests/test_qlora.py tests/test_qlora_cpu.py

Use separate processes with the same checkpoint and workload for the model
training-step memory comparison::

   python scripts/benchmark_qlora.py --model /path/to/checkpoint \
     --adam-4bit --microbatch 4 --sequence-length 512 --output lora.json
   python scripts/benchmark_qlora.py --model /path/to/checkpoint \
     --adam-4bit --microbatch 4 --sequence-length 512 --qlora --output qlora.json

The script measures full-model training steps, including output logits and
optimizer state. It reports loading-independent allocated memory, explicitly
adds managed optimizer storage, and separates warmup from subsequent steps.
It does not claim to measure total system DRAM or rollout-cache peaks.

NF4 is lossy. Validate both the quantization error against the original
weights and operator/loss/gradient correctness against the same dequantized
weights. A successful numerical test does not establish reward convergence;
use actual task metrics for that comparison.
