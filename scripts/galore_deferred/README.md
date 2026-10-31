# Portable reproduction of the strict deferred recipe

This is a new portable entry point. It reuses `scripts/galore_sft/common.py` for encoding, deterministic batches, response-only validation NLL and public Trainer construction. It is **not** the byte-identical harness that produced the archived formal measurements. The archived source/configuration and results remain the authority for historical claims; results generated here have their own source and wrapper hashes.

Use a local copy of Qwen3-1.7B revision `4855588ea1a12789f2e965e5f52a9e4a24c94b2a`. The runner checks the model/tokenizer files against `model-manifest.json`, and checks only the data splits actually used against `../galore_sft/data-manifest.json`. Prepare the fixed Dolly splits following `../galore_sft/README.md`. The encoded training split must contain 13,544 rows; final validation must cover 708 examples and 52,181 response targets. `input-prefix.json` preserves three original seed0 batch indices, encoded-batch SHA256 values and target counts for comparison with the new `steps.jsonl`.

From the repository root, with the existing compatible native extension available:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" taskset -c 0-3 \
  python scripts/galore_deferred/run.py --method deferred --mode performance \
  --model /path/to/pinned/Qwen3-1.7B --data-dir /path/to/fixed-splits \
  --output-dir /path/to/new/performance-deferred-seed0 --seed 0

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" taskset -c 0-3 \
  python scripts/galore_deferred/run.py --method deferred --mode quality \
  --steps 4096 --model /path/to/pinned/Qwen3-1.7B \
  --data-dir /path/to/fixed-splits --output-dir /path/to/new/quality-deferred-seed0 \
  --seed 0
```

Choose a GPU and CPU affinity available on your machine. The caller owns device exclusion, locking, free-space admission and monitoring; this portable entry point does not arbitrate a shared host. Each method/seed must use a separate fresh process and a previously absent output directory. Run matched methods `adamw`, `adam8`, `strict` and `deferred`, and seeds 0, 1 and 2 as appropriate. `strict` is the same strict GaLore recipe with deferred lifetime disabled. `adam8` preserves the existing native low-bit control precision policy.

`--fused-scale-writeback` is a separate explicit opt-in, accepted only with `--method deferred`. Without this option the fusion flag is explicitly zero and the historical strict recipe stays unchanged. The option sets `ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK=1` before CUDA-capable imports; the worker checks the actual optimizer policy before training and records `optimizer_fused_scale_writeback_enabled` on every update. Use a distinct output directory for the opt-in result and its matched control. The bounded performance evidence belongs to the frozen `5d…` runtime at scale `1.0`. Integrating a later scalar-cast repair depends on its separate explicit-FP32-cast qualification and normal-float PTX identity bridge; executing this portable entry point does not establish that bridge. The runner provides no new 8-bit speed comparison or quality claim for the fused variant.

The fixed recipe is context512, batch4, one microbatch of4, rank128/gap200/scale1, lr1e-5, weight decay0.01, betas(0.9,0.95), global clip1, recycled projector/oversampling16/one iteration/final QR, statistical moment transport, all eligible matrices including vocabulary, lossless compact FP32 masters, tiled fused writeback and released gradients. No arbitrary configuration/private-policy overrides are accepted. The deferred flag and seed are set before CUDA-capable helper imports and inherited by spawned workers. Deferred support is restricted to the runtime's supported single-GPU dense Qwen3 path.

Performance mode performs exactly two warmup updates followed by 400 measured updates, counting the full score+train wall time and the refreshes at absolute updates200 and400. It reports allocated/reserved maxima and separates ordinary and refresh updates. Quality mode performs exactly1024 or4096 optimizer updates with no extra warmup, then one final validation pass. Both modes write every batch index/hash/target count, native update/clock/SFT-normalization guards and native `logp_abs_diff_mean`. Deferred mode additionally verifies once-only owners, the backward mutation barrier, released full gradients and empty post-step cache. Neither mode saves checkpoints or reads/hashes the test split. These runs therefore do not replace the separate save/resume acceptance checks.

Outputs: `metadata.json` binds actual imported Python source, native binary, wrapper, model assets, used data digests, software, device and affinity; `encoding.json` records tokenization coverage; `steps.jsonl` holds raw updates; `summary.json` holds measured totals/peaks and, in quality mode, final response NLL. A crash may leave partial raw output without a final summary; do not treat it as a completed run. Cold initialization and encoding are excluded from score+train timing. Optimizer-only timing cannot establish a speed win because deferred projection moves work into backward.

CPU contract smoke (mock public engine, no training or GPU initialization):

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_galore_deferred_repro_cpu.py -q
```

The smoke exercises exact budgets, warmup exclusion/refresh coverage, final-only validation using the actual shared NLL function, deterministic original input-index prefixes, and native-guard rejection. It does not establish GPU correctness or reproduce the archived learning/performance results.
