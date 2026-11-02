# Matched Dolly SFT reproduction

These entrypoints preserve the frozen Dolly experiment's training/evaluation mathematics. CPU tests verify core AST identity and offline data replay. Formal GPU measurements remain attributed to the original frozen wrappers recorded in `frozen-ast-manifest.json`; this portable entrypoint has not run a formal three-seed GPU comparison.

Install/build AReno using the repository instructions in a CUDA environment. Data preparation uses the existing NumPy and Transformers dependencies; downloading additionally requires the optional ModelScope SDK. CPU tests require pytest. No runtime dependencies are added by these scripts. Run commands from the repository root, with AReno installed or `PYTHONPATH=.`. All paths below are relative examples; supply your own local model directory. Every train/eval/preparation command requires `--model` explicitly. Preparation uses its local tokenizer only.

## Current selected scale1 / all-matrix contract

The current selected condition is strict FP32-master GaLore, **rank128 / gap200 / scale1**, with all eligible matrices included. See `config.scale1-all.example.json` and [the current measured report](../../docs/reports/galore-sft-frontier.rst). Keep the historical parser default4 unchanged; pass `--scale 1` explicitly. The native model-owned precision experiment is excluded from the delivered optimizer.

Run each method and seed in a fresh process on one free assigned GPU. Formal training measurements retain their original frozen-wrapper source attribution; these portable commands are reproduction instructions.

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=4
unset CUBLAS_WORKSPACE_CONFIG
for seed in 0 1 2; do
  for optimizer in adamw adam8 galore; do
    CUDA_VISIBLE_DEVICES=0 ARENO_BENCH_SEED="$seed" \
    python scripts/galore_sft/train.py \
      --model assets/Qwen3-0.6B --seed "$seed" --optimizer "$optimizer" \
      --data-dir artifacts/data/dolly --context 512 --batch 4 --micro-batch 2 \
      --steps 1024 --eval-every 256 --monitor-examples 128 --checkpoint-every 0 \
      --rank 128 --gap 200 --scale 1 \
      --extra-config-json '{"optimizer":{"galore_exclude_vocab":false}}' \
      --output-dir "artifacts/galore-sft/native-scale1-all/seed$seed/$optimizer/train" \
      --checkpoint-dir "artifacts/checkpoints/native-scale1-all/seed$seed/$optimizer/final"

    CUDA_VISIBLE_DEVICES=0 ARENO_BENCH_SEED="$seed" \
    python scripts/galore_sft/eval.py \
      --model assets/Qwen3-0.6B --seed "$seed" --optimizer "$optimizer" \
      --data-dir artifacts/data/dolly --context 512 --batch 4 --micro-batch 2 \
      --rank 128 --gap 200 --scale 1 --test \
      --extra-config-json '{"optimizer":{"galore_exclude_vocab":false}}' \
      --output-dir "artifacts/galore-sft/native-scale1-all/seed$seed/$optimizer/eval" \
      --checkpoint-dir "artifacts/checkpoints/native-scale1-all/seed$seed/$optimizer/final"
  done
done
```

## Fixed data and offline replay

The only dataset source is ModelScope `AI-ModelScope/databricks-dolly-15k`, revision `40789acc12fc986e9145f87d159f19e2f1d75c6a`. `prepare_data.py` verifies both the JSONL and dataset-card hashes, fixed in code, before preprocessing. There is no automatic other-hub fallback. Supply a local Qwen3-0.6B model/tokenizer matching `input-manifest.example.json`; the pinned tokenizer files and all nine deterministic output hashes are in `data-manifest.json`.

Optional first download:

```bash
python scripts/galore_sft/prepare_data.py \
  --download --cache-dir artifacts/cache/modelscope \
  --raw-dir artifacts/raw/dolly --model assets/Qwen3-0.6B \
  --output-dir artifacts/data/dolly
```

With the pinned raw files already present, run fully offline into a fresh or empty directory:

```bash
python scripts/galore_sft/prepare_data.py \
  --raw-dir artifacts/raw/dolly --model assets/Qwen3-0.6B \
  --output-dir artifacts/data/dolly-replay
```

Expected rows: 15,011 raw, 16 normalized full-example duplicates removed, 14,995 retained; train 13,544, validation 708, test 743. NFKC/whitespace/casefold normalization is used for dedup/group keys; output text only loses outer whitespace. Fixed SHA256 group assignment splits 90/5/5 using normalized context (or instruction when context is absent). This is an exploratory split with a document proxy, not an official benchmark split or an article/near-duplicate separation guarantee. No model-generated examples are added. Examples are stored untruncated; tokenization happens at runtime. Retain `DATASET-LICENSE.md`/generated `LICENSE-ATTRIBUTION.md` with redistributed adaptations. The code bundle contains no dataset rows or model weights.

## Historical scale4 contract: 1024-update reproduction

This first example preserves the historical scale4 contract. It is not the current selected scale1 condition and does not claim a portable-entrypoint GPU result. Use a free GPU and the same device/runtime/software contract for every method. Seed is exported before Python starts and passed again to the CLI. This example runs sequentially on one assigned GPU. A multi-GPU scheduler must implement occupancy checks and one project lock per GPU outside these entrypoints. Keep native model-forward precision and all runtime flags equal across methods.

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=4
unset CUBLAS_WORKSPACE_CONFIG
for seed in 0 1 2; do
  for optimizer in adamw adam8 galore; do
    CUDA_VISIBLE_DEVICES=0 ARENO_BENCH_SEED="$seed" \
    python scripts/galore_sft/train.py \
      --model assets/Qwen3-0.6B --seed "$seed" --optimizer "$optimizer" \
      --data-dir artifacts/data/dolly --context 512 --batch 4 --micro-batch 2 \
      --steps 1024 --eval-every 256 --monitor-examples 128 --checkpoint-every 0 \
      --rank 128 --gap 200 --scale 4 \
      --output-dir "artifacts/galore-sft/native/seed$seed/$optimizer/train" \
      --checkpoint-dir "artifacts/checkpoints/seed$seed/$optimizer/final"

    CUDA_VISIBLE_DEVICES=0 ARENO_BENCH_SEED="$seed" \
    python scripts/galore_sft/eval.py \
      --model assets/Qwen3-0.6B --seed "$seed" --optimizer "$optimizer" \
      --data-dir artifacts/data/dolly --context 512 --batch 4 --micro-batch 2 \
      --rank 128 --gap 200 --scale 4 --test \
      --output-dir "artifacts/galore-sft/native/seed$seed/$optimizer/eval" \
      --checkpoint-dir "artifacts/checkpoints/seed$seed/$optimizer/final"
  done
done
```

AdamW: FP32 master optimizer. `adam8`: 8-bit Adam. `galore`: recycled projection, rank128, oversampling16, one iteration, final QR, compact masters, tile_rows16384, release gradients, fused writeback and transported moments, scale4, refresh gap200. All methods use lr=min_lr=1e-5, betas=(0.9,0.95), weight_decay=.01, clip_norm1, native attention, eager execution and activation checkpointing. `--retained` disables moment transport. `--extra-config-json` forwards explicitly supplied `optimizer`/`runtime` dictionaries and rejects other top-level keys; use an identical extra contract for baseline comparisons unless the named optimizer change is the registered intervention. For example, a separate explicit vocabulary-exclusion condition can pass:

```bash
--extra-config-json '{"optimizer":{"galore_exclude_vocab":true}}'
```

These example commands do not claim that a vocabulary-exclusion result has been measured. See `config.example.json` for the complete baseline/GaLore budget and flags.

## Historical selected scale2 / all-matrix contract

The earlier selected comparison used **scale2 with vocabulary matrices included**, keeping the parser default4 unchanged for historical compatibility. Pass `--scale 2` explicitly and use separate output/checkpoint directories. These are reproduction commands for the selected condition, not a claim that the portable wrappers have run its formal three-seed experiment. Formal measurements are produced by separately frozen original wrappers and must retain that lineage in the eventual report.

```bash
for seed in 0 1 2; do
  for optimizer in adamw adam8 galore; do
    CUDA_VISIBLE_DEVICES=0 ARENO_BENCH_SEED="$seed" \
    python scripts/galore_sft/train.py \
      --model assets/Qwen3-0.6B --seed "$seed" --optimizer "$optimizer" \
      --data-dir artifacts/data/dolly --context 512 --batch 4 --micro-batch 2 \
      --steps 1024 --eval-every 256 --monitor-examples 128 --checkpoint-every 0 \
      --rank 128 --gap 200 --scale 2 \
      --extra-config-json '{"optimizer":{"galore_exclude_vocab":false}}' \
      --output-dir "artifacts/galore-sft/native-scale2-all/seed$seed/$optimizer/train" \
      --checkpoint-dir "artifacts/checkpoints/native-scale2-all/seed$seed/$optimizer/final"

    CUDA_VISIBLE_DEVICES=0 ARENO_BENCH_SEED="$seed" \
    python scripts/galore_sft/eval.py \
      --model assets/Qwen3-0.6B --seed "$seed" --optimizer "$optimizer" \
      --data-dir artifacts/data/dolly --context 512 --batch 4 --micro-batch 2 \
      --rank 128 --gap 200 --scale 2 --test \
      --extra-config-json '{"optimizer":{"galore_exclude_vocab":false}}' \
      --output-dir "artifacts/galore-sft/native-scale2-all/seed$seed/$optimizer/eval" \
      --checkpoint-dir "artifacts/checkpoints/native-scale2-all/seed$seed/$optimizer/final"
  done
done
```

`config.scale2-all.example.json` documents this explicit variant. Vocabulary exclusion is false; matrices remain subject to the optimizer's normal rank/shape eligibility. Record new conditions separately from historical scale4 and any vocabulary-exclusion experiments. The baseline optimizers use the same initial model, budget, batches and runtime contract.

## Mathematics, logs and continuation

Prompt and response are encoded separately. Append EOS when needed and when it fits. Reserve up to half the context for response targets, keep head/tail prompt tokens, then truncate the response prefix to remaining space. Training batches follow absolute cursor `step * batch`, shuffling epoch `e` with `seed + 1000003*e`; continuation therefore preserves ordering. Use `--steps 3072 --resume-from artifacts/checkpoints/seed0/galore/final` to extend a completed 1024-update model to 4096 total updates, with a **new** child checkpoint/output path and the same model/optimizer/batch/context/seed/options. `--steps` is additional updates, not a target endpoint. Final training `next_step` counts completed updates; eval `restored_sdk_step` is its zero-based SDK cursor.

The public `Trainer` performs scoring and `sft_loss_fn` training, exactly one optimizer update per public call. Validation monitoring never selects a checkpoint or changes training. Independent eval reloads the full state and computes response-target-token-weighted negative log likelihood over all validation rows; `--test` additionally scores all test rows. At context512 the expected full-eval targets are 52,181 validation / 55,562 test. The profiler aggregates **sum** of per-microbatch CUDA time and **max** allocated/reserved peaks per update; first two updates are excluded from timing means. Wall time, scoring time, native SFT diagnostics and raw batch hashes remain in `steps.jsonl`. Memory/timing are framework measures; do not label them whole-process NVML peaks or clean standalone throughput without separate occupancy/profiling evidence.

Use a new empty output/checkpoint path for every run; the frozen-compatible train/eval wrappers do not guard against overwriting old log directories. `--checkpoint-every 0` avoids periodic intermediate snapshots. A stop-file request may save one recovery; do not duplicate a final at the same endpoint. Read old checkpoints without overwriting them in place (historical SHA-identical files may share hard links). Native bespoke CUDA backwards can be nondeterministic: this package does not promise bitwise resumed native trajectories. Any opt-in deterministic-backward arithmetic contract must be labeled and applied to all compared optimizers.

## CPU verification and shareable evidence

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  pytest -q tests/test_galore_sft_repro_cpu.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
ARENO_DOLLY_RAW_DIR=artifacts/raw/dolly \
ARENO_DOLLY_MODEL_DIR=assets/Qwen3-0.6B \
  pytest -q tests/test_galore_sft_repro_cpu.py
```

The first command skips only the full replay if local assets are unspecified. The second verifies all nine data/row-manifest/order files byte-for-byte against frozen hashes and all stable original preparation summary fields. Tests also cover AST identity, encoding, absolute-cursor ordering, weighted loss, profile aggregation, optimizer overrides, required model arguments and a three-update mocked CPU entrypoint smoke. The mocked smoke proves orchestration only; no CUDA training is exercised.

Retain relative artifacts under the layout in `EVIDENCE.md`. `metadata.json` records actual source/wrapper/native-binary/data hashes, runtime versions and actual CLI arguments. Also record your Git commit, original model file hashes, GPU occupancy and exact launch command using `input-manifest.example.json` and `run-manifest.example.json`. Do not relabel original frozen-wrapper GPU results as results produced by these portable wrappers. Checkpoints and full raw Dolly files are intentionally omitted from this small code distribution.
