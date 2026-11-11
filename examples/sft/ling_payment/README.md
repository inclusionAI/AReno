# Ling payment SFT on Apple Silicon

This recipe uses the original BF16 Ling-3.0-tiny checkpoint, online MLX affine
4-bit quantization (group size 64), and rank-4 LoRA on all 24 Attention and MLP
layers. AReno downloads the [model](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
and [prepared dataset](https://huggingface.co/datasets/AsystemAReno/Ling-Ting-SFT-Augment)
from Hugging Face on the first run and reuses its cache on subsequent runs.

From the AReno repository root:

```bash
areno train \
  --algo sft --model-hub hf \
  --ckpt inclusionAI/Ling-3.0-tiny \
  --dataset-path AsystemAReno/Ling-Ting-SFT-Augment::train \
  --dataset-loader-fn examples/sft/ling_payment/dataset_loader.py \
  --validation-dataset-path AsystemAReno/Ling-Ting-SFT-Augment::validation \
  --eval-batches 25 --eval-before-step \
  --shuffle --seed 0 \
  --q4 --lora-rank 4 --lora-alpha 32 --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,f_proj,o_proj,g_proj,b_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,dense,gate_proj,up_proj,down_proj \
  --batch-size 1 --mini-bs 1 --score-micro-bs 1 \
  --epochs 1 --max-steps 500 \
  --max-new-tokens 1024 \
  --lr 2e-5 --min-lr 2e-5 --lr-decay-style constant --weight-decay 0 \
  --no-adam-bias-correction --no-grad-clip --no-mlx-compile-train-step \
  --drop-rollout-state \
  --save-path runs/payment/adapters \
  --metrics-log-dir runs/payment/events
```

On Apple Silicon, AReno selects MLX with one device automatically. The omitted
defaults enable activation checkpointing, limit prompts to 1,024 tokens,
validate every 50 steps, and save every 100 steps.

`repo::split` selects a dataset split using its default configuration. Training
loads 1,115 rows from `train`; held-out validation loads 100 rows from
`validation`. The 500 `test` rows are reserved for generation evaluation.
Set `HF_HOME` to change the Hugging Face cache directory, or `HF_TOKEN` when
accessing a private or gated repository. Local paths remain supported for
`--ckpt`, `--dataset-path`, and `--validation-dataset-path`.

The loader receives the resolved local checkpoint as `model_path`, so its
tokenizer always matches `--ckpt`. Direct Python callers can supply `model_path`
or use the legacy `LING_MODEL_DIR` environment variable.

The loader enables thinking and masks prompt tokens. It preserves the reference
MLX-LM padding convention, including its one supervised padding token when the
1024-token cap permits. AReno shuffles training and validation using the same
seeded NumPy sampler. `--eval-before-step` reproduces the historical evaluation
schedule: before updates 1, 50, 100, ..., 500. Validation reports target-token
weighted NLL, disables dropout while scoring, and restores the previous mode.
Metrics are recorded in TensorBoard under `validation/*` and in
`validation_metrics.jsonl` within the metrics directory.

The expected full-model topology is 297 adapter modules and 74,947,584 trainable
parameters. Historical loss and 500-question scores remain reference results;
a new complete training and generation evaluation is needed to measure this
implementation's results.
