# KIVI KV cache: measurement report

This directory holds the harnesses behind the measurements of the opt-in KIVI
KV cache ([concept guide](../../docs/concepts/kivi-kv-cache.rst), issue
[#590](https://github.com/inclusionAI/AReno/issues/590)). Every number below
was produced by these scripts; the raw outputs (JSON, JSONL, TensorBoard
events and logs) are listed under [Raw logs](#raw-logs).

"BF16" is the default model-dtype cache (`kv_cache_dtype="auto"`). KIVI uses
group size G = 32 and residual window R = 128 unless a table says otherwise.

## Summary

Qwen3-0.6B on one RTX PRO 5000 72GB:

| | BF16 | KIVI-4 | KIVI-2 |
|---|---|---|---|
| Cache bytes at 8192 tokens, scales and residual ring included | 1 | 0.331x | 0.206x |
| Decode attention step, 128 rows x 16K tokens (CUDA graph) | 7.01 ms | 3.95 ms (1.78x) | 4.06 ms (1.73x) |
| GSM8K test accuracy, 1319 x 4 samples at T = 1 | 72.16% | 72.25% | 68.82% (72.25% with R = 512) |
| 8K-token prompts, 32 concurrent: accuracy / peak memory / tokens per s | 70.7% / 45.1 GiB / 854 | 70.5% / 19.5 GiB / 1206 | 64.7% / 14.7 GiB / 1175 |
| Most concurrent 8K-token rollouts measured on the 72 GB GPU | 48 (64 runs out of memory) | 128 in 57.6 GiB | 256 in 69.7 GiB |
| WikiText-2 perplexity change, positions up to 16K / 16K-32K | - | +0.3-0.8% / +1.0% | +6.9-12.3% / +24.5% |
| GSPO final-window reward (change vs BF16) | 0.806 (6 seeds) | 0.806, +0.000 (6 seeds) | 0.708, -0.097 (3 seeds); with R = 512: 0.769, -0.036 |
| Native `train/logp_abs_diff_mean` | 0.015 | 0.032 | 0.209; with R = 512: 0.117 |
| Trained policies, BF16-cache validation: loss change / GSM8K accuracy | - / 76.4% | +0.20% / 75.9% | +1.28% / 72.2%; with R = 512: +1.22% / 74.4% |

- **KIVI-4 meets every target of the issue.** It stores a third of the BF16
  cache, runs decode attention 1.4-1.8x faster with 32 or more sequences,
  keeps generation quality within noise at short and 8K-token contexts, lets
  2.7x more 8K-token rollouts share the GPU, and trains GSPO to the same
  final reward as BF16 over six seeds, with a validation loss 0.20% above
  BF16 and held-out accuracy within 0.5 points.
- **KIVI-2 trades quality for a fifth of the cache.** With the default
  128-token residual window it loses 3.3 GSM8K points (3.5-7 at 8K-token
  contexts), and its rollouts diverge enough from the training policy
  (native divergence 0.21) that GSPO stops improving after about 70 steps and
  ends 0.097 below BF16, missing the 0.05 reward target. A 512-token window
  recovers its GSM8K accuracy (72.25%), halves the divergence and keeps the
  RL reward within 0.036 of BF16, inside the target, though its held-out
  accuracy stays 2 points lower and its reward gap widens over the last 20
  steps.
- **Save/resume works for both caches**, and the resumed runs continue the
  uninterrupted ones.
- **Recommendation:** `kv_cache_dtype="kivi4"` for RL rollouts; `kivi2` only
  where memory is the binding constraint, with `kivi_residual_length=512`, and
  after validating on the target task.

## Acceptance criteria

| Issue #590 requirement | Evidence | Status |
|---|---|---|
| Opt-in KIVI storage: per-channel keys, per-token values, scales and a recent unquantized residual window | `RuntimeConfig.kv_cache_dtype = "kivi2" \| "kivi4"`, `kivi_group_size`, `kivi_residual_length`; [concept guide](../../docs/concepts/kivi-kv-cache.rst) | done; default unchanged |
| Incremental decode, window transitions and cache reset; explicit fallback for unsupported attention | fused flush + append per decode step; `clear_kv_cache` releases the cache; `RuntimeWarning` fallback (section 1) | done |
| Actual behavior-policy log-probabilities; cached attention and metadata lifetime validated | rollouts keep the KIVI log-probabilities (section 7); slot reuse, CUDA graph and clear tests (section 1) | done |
| Memory, throughput, generation quality and native training log-probability divergence vs BF16 | sections 2-7 | reported |
| Tests at 1024/4096/8192 cached tokens and residual boundaries; dequantization matches an independent implementation; reset leaks no storage; cached attention within BF16 `atol = rtol = 1e-2` | `tests/test_kivi_kv_cache_cuda.py` (section 1) | 61 tests pass |
| 8192-token case: cache bytes with scales and residual below BF16; total peak including dequantization workspace; decode throughput; quality | sections 2, 3 and 4-6 | cache 0.206x (KIVI-2) / 0.331x (KIVI-4) of BF16; allocator peak with workspace 0.21x / 0.34x at 128 rows; decode step 1.6-2.25x faster with 8+ rows |
| Fixed commit, model revision, seed, dtype, batch/context and software; 2 warmup + at least 10 measured steps; allocated/reserved peaks and step time | [Setup](#setup), section 3 (2 + 20 steps, allocator peaks) | done; discrete GPU, so no unified-memory accounting applies |
| Public `Trainer` end to end with updates and save/resume; native `train/logp_abs_diff_mean` | section 7 | done: `areno train` GSPO, step-50 checkpoints resumed for BF16 and KIVI-2; divergence 0.015 (BF16), 0.032 (KIVI-4), 0.209 (KIVI-2) |
| At least 3 seeds with fixed data and budget; final validation loss at most +2%; final-window reward no more than 0.05 below BF16, with reward curves | section 7 | KIVI-4 passes (6 seeds: reward +0.000, validation loss +0.20%); KIVI-2 passes with R = 512 (reward -0.036, loss +1.22%) but misses the reward target with the default R = 128 (reward -0.097, loss +1.28%) |
| Core tests and repository checks | `pytest tests/ -k cpu` with triton, flash-attn and CUDA hidden as in CI (only the 8 `test_real_gloo_*` tests fail, identically on upstream e17794c), `pre-commit run --all-files` | pass |

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell (sm_120), driver 580.126.20, CUDA 12.8; one GPU per run, two hosts with the same image |
| Software | Ubuntu 22.04.5, Python 3.12.11, torch 2.8.0+cu128, triton 3.4.0, flash-attn 2.8.3, transformers 5.22.0, datasets 4.0.0 |
| Code | AReno `main` at e17794c plus this change; each result directory records the commit it ran (`commit.txt`) |
| Model | `Qwen/Qwen3-0.6B` from ModelScope (revision `master`, created 2025-07-26), BF16 weights, `model.safetensors` sha256 `f47f7117…6874b`; `Qwen/Qwen3-1.7B` (same revision date) for two scaling checks |
| GSM8K | `gsm8k` `main` from ModelScope through `areno.cli.train`'s dataset resolver: 7473 train / 1319 test rows |
| WikiText-2 | `wikitext-2-raw-v1` test split from the raw parquet file of `Salesforce/wikitext` (sha256 `5f1bea06…6d91`), joined into one text |
| Seeds | rollout sampling seed `seed * 1_000_003 + step` per RL step (seeds 1-6 for BF16 and KIVI-4, 1-3 for the other arms); evaluation seed 1234 |

The measurements ran on development commits of this change, kept on branch
`kivi-590-dev` of the contributor's fork: 34c5ea8 (BF16 RL arm, generation and
throughput runs, greedy validation), c87ae99 (KIVI RL arms, BF16 seeds 4-6
and the resumed runs), 1c5929e (teacher-forced runs) and 953a1c5 (kernel
benchmark). Their `areno/` trees match this change apart from formatting, the
G ≥ 32 check and the allocation log line (`git diff -w <commit> -- areno/`).
The `commit.txt` files in the raw logs hold the hashes these commits had
before their messages were edited; `commit-map.txt` there pairs each with its
`kivi-590-dev` commit, whose tree is identical.

## 1. Correctness tests

`tests/test_kivi_kv_cache_cuda.py` checks the Triton kernels against an
independent PyTorch implementation of KIVI quantization written in the test
file:

- packed codes, scales and zero points written by prefill match the reference
  for G = 32, 64 and 128, and the packing layout round-trips;
- with 1024, 4096 and 8192 cached tokens, 2G + 7 decode steps run from three
  prompt lengths at different ring phases (one shorter than R), so steps
  before, at and after each group flush are covered; every step's attention
  output matches attention over the reference's dequantization of the same
  quantized tensors within BF16 `atol = rtol = 1e-2`, and the final cache
  state matches the reference;
- a prefill chunk that extends a cached sequence (flash and native backends)
  is bit-identical to the dense kernel run on independently dequantized
  inputs;
- a released rollout slot and its blocks are reused without stale data, CUDA
  graph replay is bit-identical to eager execution, and clearing the KV cache
  returns the allocator to its baseline;
- at 8192 tokens the cache, including scales, zero points and the residual
  ring, is below 0.25x (KIVI-2) and 0.375x (KIVI-4) of the BF16 cache.

`tests/test_kivi_kv_cache_cpu.py` covers configuration validation, the explicit
fallbacks, residual-window bookkeeping and prefill planning without a GPU.

```bash
pytest tests/test_kivi_kv_cache_cpu.py tests/test_kivi_kv_cache_cuda.py   # 61 passed on the GPU host
pytest tests/ -k cpu   # 8 test_real_gloo_* tests fail in this container, identically on upstream e17794c
```

## 2. Cache memory

Per token and KV head (head dim 128, G = 32) BF16 stores 512 bytes of K and V.
KIVI-2 stores 64 bytes of codes plus 32 bytes of FP16 scales and zero points
(18.75%); KIVI-4 stores 128 + 32 bytes (31.25%). With G = 64 the metadata
halves (KIVI-4: 28.1%). Each rollout slot also keeps R + G full-precision
tokens in its residual ring, a cost independent of the context length.

In the RL runs of section 7 (`max_running_prompts = 128` concurrent sequences,
prompts of up to 512 tokens plus 2048 new tokens, so 1281 paged blocks of 256
tokens including one scratch block, and 129 rollout slots), the engine
allocates:

| KV cache | allocated bytes (from the engine log) | relative to BF16 |
|---|---|---|
| BF16 | 37,609,488,384 (1281 × 256 tokens × 114,688 bytes) | 1.000 |
| KIVI-4 | 14,120,386,560 | 0.375 |
| KIVI-2 | 9,419,096,064 | 0.250 |

Here the rings (129 slots × 160 tokens, 2.2 GiB) are a quarter of the KIVI-2
pool; for longer sequences the ratio approaches 0.1875 / 0.3125, as the
8192-token rows below show.

At 8192 cached tokens, longer than the residual window, the decode benchmark
of section 3 measures the cache of one Qwen3-0.6B layer, with scales, zero
points and one residual ring per row, at 0.206x (KIVI-2) and 0.331x (KIVI-4)
of BF16; at 32768 tokens it is 0.192x and 0.317x. Including the
dequantization workspace of a decode step, the allocator peak for 128 rows is
890 MiB (KIVI-2) and 1418 MiB (KIVI-4) against 4226 MiB for BF16, that is
0.21x and 0.34x (table in section 3).

## 3. Decode attention kernel

`bench_decode_attention.py` times one decode step of one attention layer
(Qwen3-0.6B shape: 16 query heads, 8 KV heads, head dim 128) at a fixed number
of rows and cached tokens. The KIVI step includes everything the runtime
launches per layer: the residual-group flush, the ring append and the fused
dequantize + split-K attention. The BF16 step is `flash_attn_with_kvcache`
with the in-place append used by AReno's flash backend. Each configuration
warms up for 2 steps and is measured for 20.

Under CUDA graph replay, which is how AReno's decode loop runs, KIVI is
1.4-1.8x faster than flash-attn on the BF16 cache with 32 or more rows, and
up to 2.25x with 8 rows at 8192 tokens; both widths read 3-5x fewer bytes per
token. A single row leaves most of the GPU idle: it is 1.15-1.19x faster at
4-8K tokens, 0.8x at 16K and 1.56x at 32K. The extra memory of a KIVI step is the split-K
workspace, at most 17.3 MiB, and the allocator peaks below include it.
Without CUDA graphs the three launches per step (append, attention, combine)
dominate small batches.

#### Decode attention step, one layer, CUDA graph replay (`bench_final/decode_graph.json`)

NVIDIA RTX PRO 5000 72GB Blackwell, torch 2.8.0+cu128; 16 query / 8 KV heads, head dim 128, G=32, R=128; 2 warmup + 20 measured steps, median (p10-p90) in ms.

| context | rows | BF16 ms | KIVI-2 ms | KIVI-2 speedup | KIVI-4 ms | KIVI-4 speedup |
|---|---|---|---|---|---|---|
| 1024 | 1 | 0.012 (0.011-0.013) | 0.012 (0.011-0.013) | 1.00x | 0.012 (0.011-0.014) | 1.00x |
| 1024 | 8 | 0.028 (0.028-0.029) | 0.028 (0.028-0.030) | 1.00x | 0.028 (0.027-0.029) | 1.01x |
| 1024 | 32 | 0.126 (0.126-0.128) | 0.084 (0.083-0.086) | 1.50x | 0.084 (0.083-0.085) | 1.51x |
| 1024 | 128 | 0.464 (0.462-0.466) | 0.333 (0.331-0.334) | 1.39x | 0.338 (0.337-0.339) | 1.37x |
| 1024 | 256 | 0.902 (0.899-0.906) | 0.624 (0.622-0.625) | 1.45x | 0.623 (0.621-0.627) | 1.45x |
| 4096 | 1 | 0.021 (0.019-0.022) | 0.018 (0.017-0.019) | 1.15x | 0.018 (0.017-0.019) | 1.15x |
| 4096 | 8 | 0.120 (0.119-0.122) | 0.060 (0.059-0.064) | 2.00x | 0.059 (0.058-0.061) | 2.03x |
| 4096 | 32 | 0.460 (0.458-0.461) | 0.304 (0.304-0.306) | 1.51x | 0.308 (0.308-0.310) | 1.49x |
| 4096 | 128 | 1.765 (1.764-1.767) | 1.083 (1.082-1.088) | 1.63x | 1.066 (1.064-1.069) | 1.66x |
| 4096 | 256 | 3.492 (3.489-3.496) | 2.106 (2.104-2.115) | 1.66x | 2.069 (2.067-2.077) | 1.69x |
| 8192 | 1 | 0.026 (0.026-0.027) | 0.023 (0.021-0.024) | 1.16x | 0.022 (0.021-0.023) | 1.19x |
| 8192 | 8 | 0.230 (0.228-0.232) | 0.102 (0.101-0.105) | 2.25x | 0.103 (0.102-0.104) | 2.24x |
| 8192 | 32 | 0.908 (0.907-0.911) | 0.562 (0.561-0.564) | 1.62x | 0.554 (0.554-0.556) | 1.64x |
| 8192 | 128 | 3.516 (3.515-3.518) | 2.075 (2.074-2.078) | 1.69x | 2.040 (2.038-2.043) | 1.72x |
| 8192 | 256 | 6.961 (6.959-6.966) | 4.098 (4.095-4.101) | 1.70x | 3.999 (3.997-4.005) | 1.74x |
| 16384 | 1 | 0.037 (0.036-0.039) | 0.047 (0.046-0.048) | 0.78x | 0.045 (0.044-0.047) | 0.81x |
| 16384 | 8 | 0.449 (0.447-0.451) | 0.265 (0.265-0.268) | 1.69x | 0.281 (0.280-0.282) | 1.60x |
| 16384 | 32 | 1.804 (1.802-1.806) | 1.064 (1.062-1.066) | 1.70x | 1.042 (1.040-1.043) | 1.73x |
| 16384 | 128 | 7.008 (7.005-7.013) | 4.058 (4.056-4.060) | 1.73x | 3.945 (3.943-3.950) | 1.78x |
| 16384 | 256 | 13.873 (13.869-13.877) | 8.061 (8.058-8.065) | 1.72x | 7.828 (7.824-7.835) | 1.77x |
| 32768 | 1 | 0.123 (0.122-0.125) | 0.080 (0.078-0.083) | 1.55x | 0.078 (0.077-0.082) | 1.57x |
| 32768 | 8 | 0.882 (0.880-0.884) | 0.533 (0.532-0.536) | 1.65x | 0.521 (0.519-0.522) | 1.69x |
| 32768 | 32 | 3.592 (3.589-3.596) | 2.062 (2.061-2.066) | 1.74x | 2.011 (2.009-2.014) | 1.79x |
| 32768 | 128 | 14.000 (13.996-14.007) | 8.028 (8.025-8.032) | 1.74x | 7.793 (7.789-7.798) | 1.80x |
| 32768 | 256 | 27.717 (27.713-27.728) | 16.006 (16.002-16.014) | 1.73x | 15.532 (15.529-15.540) | 1.78x |

Memory of the same steps: cache bytes (codes, scales and residual rings for KIVI), the extra allocation of one step (split-K partials and temporaries) and the process-wide allocator peaks while stepping, in MiB.

| context | rows | cache BF16 / KIVI-2 / KIVI-4 | KIVI-2 / KIVI-4 cache ratio | step workspace BF16 / KIVI-2 / KIVI-4 | peak allocated BF16 / KIVI-2 / KIVI-4 | peak reserved BF16 / KIVI-2 / KIVI-4 |
|---|---|---|---|---|---|---|
| 8192 | 1 | 33.0 / 6.8 / 10.9 | 0.206 / 0.331 | 0.2 / 0.5 / 0.5 | 33 / 7 / 11 | 42 / 28 / 28 |
| 8192 | 8 | 264.0 / 54.5 / 87.5 | 0.206 / 0.331 | 0.2 / 4.1 / 4.1 | 264 / 60 / 95 | 270 / 102 / 134 |
| 8192 | 32 | 1056.0 / 218.0 / 350.0 | 0.206 / 0.331 | 0.1 / 16.4 / 16.4 | 1056 / 235 / 367 | 1062 / 246 / 378 |
| 8192 | 128 | 4224.0 / 872.0 / 1400.0 | 0.206 / 0.331 | 0.5 / 16.8 / 16.8 | 4226 / 890 / 1418 | 4230 / 894 / 1422 |
| 8192 | 256 | 8448.0 / 1744.0 / 2800.0 | 0.206 / 0.331 | 1.0 / 17.3 / 17.3 | 8451 / 1763 / 2819 | 8456 / 1768 / 2824 |
| 32768 | 1 | 129.0 / 24.8 / 40.9 | 0.192 / 0.317 | 0.2 / 0.5 / 0.5 | 129 / 25 / 41 | 138 / 46 / 62 |
| 32768 | 8 | 1032.0 / 198.5 / 327.5 | 0.192 / 0.317 | 0.2 / 4.1 / 4.1 | 1032 / 203 / 334 | 1038 / 250 / 378 |
| 32768 | 32 | 4128.0 / 794.0 / 1310.0 | 0.192 / 0.317 | 0.1 / 16.4 / 16.4 | 4128 / 811 / 1327 | 4134 / 822 / 1338 |
| 32768 | 128 | 16512.0 / 3176.0 / 5240.0 | 0.192 / 0.317 | 0.5 / 16.8 / 16.8 | 16514 / 3194 / 5258 | 16518 / 3198 / 5262 |
| 32768 | 256 | 33024.0 / 6352.0 / 10480.0 | 0.192 / 0.317 | 1.0 / 17.3 / 17.3 | 33027 / 6371 / 10499 | 33032 / 6376 / 10504 |

#### Decode attention step, one layer, eager launches (`bench_final/decode_eager.json`)

NVIDIA RTX PRO 5000 72GB Blackwell, torch 2.8.0+cu128; 16 query / 8 KV heads, head dim 128, G=32, R=128; 2 warmup + 20 measured steps, median (p10-p90) in ms.

| context | rows | BF16 ms | KIVI-2 ms | KIVI-2 speedup | KIVI-4 ms | KIVI-4 speedup |
|---|---|---|---|---|---|---|
| 1024 | 1 | 0.019 (0.018-0.021) | 0.084 (0.082-0.101) | 0.22x | 0.085 (0.082-0.093) | 0.22x |
| 1024 | 32 | 0.130 (0.130-0.132) | 0.142 (0.140-0.148) | 0.92x | 0.140 (0.138-0.147) | 0.93x |
| 1024 | 128 | 0.468 (0.465-0.471) | 0.376 (0.374-0.385) | 1.24x | 0.381 (0.379-0.387) | 1.23x |
| 8192 | 1 | 0.034 (0.033-0.034) | 0.083 (0.081-0.090) | 0.41x | 0.084 (0.082-0.094) | 0.40x |
| 8192 | 32 | 0.914 (0.913-0.916) | 0.618 (0.616-0.627) | 1.48x | 0.610 (0.607-0.614) | 1.50x |
| 8192 | 128 | 3.520 (3.519-3.523) | 2.118 (2.116-2.130) | 1.66x | 2.084 (2.082-2.092) | 1.69x |
| 32768 | 1 | 0.131 (0.129-0.132) | 0.138 (0.137-0.147) | 0.95x | 0.136 (0.135-0.142) | 0.96x |
| 32768 | 32 | 3.595 (3.594-3.597) | 2.119 (2.116-2.126) | 1.70x | 2.067 (2.064-2.076) | 1.74x |
| 32768 | 128 | 14.003 (14.002-14.010) | 8.074 (8.071-8.088) | 1.73x | 7.842 (7.836-7.894) | 1.79x |

## 4. Quality: teacher-forced long context

`teacher_forced_eval.py` feeds the same WikiText-2 windows through the model
token by token with each cache and compares next-token predictions with the
BF16 cache: perplexity, KL(BF16 ‖ KIVI) and top-1 agreement, per position
range. The first 256 tokens are prefilled.

KIVI-4 stays within 1% of BF16 perplexity in every position range up to
32K tokens (KL at most 0.007 nats per token, top-1 agreement at least 95.7%),
and within 0.7% with G = 64 or 128. KIVI-2 with R = 128 adds 6.9-12.3% below
16K tokens and 24.5% for positions 16K-32K; R = 256 removes about one point
of that. On Qwen3-1.7B the KIVI-2 penalty at 4K tokens drops to 1.5-4.8%.

#### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 16384 tokens, G=32, R=128 (`teacher_forced/len16384_rows4.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 15.355 | 16.659 (+8.49%) | 0.1278 | 83.1% | 15.425 (+0.46%) | 0.0048 | 96.8% |
| [1024, 2048) | 25.206 | 27.040 (+7.28%) | 0.1386 | 81.9% | 25.343 (+0.54%) | 0.0054 | 95.9% |
| [2048, 4096) | 23.770 | 25.566 (+7.56%) | 0.1460 | 80.8% | 23.837 (+0.28%) | 0.0055 | 96.3% |
| [4096, 8192) | 22.852 | 25.009 (+9.44%) | 0.1597 | 80.0% | 22.911 (+0.26%) | 0.0057 | 95.7% |
| [8192, 16384) | 16.143 | 18.035 (+11.72%) | 0.1787 | 80.3% | 16.187 (+0.27%) | 0.0059 | 96.0% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.197, KIVI-4 0.322.

#### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 16384 tokens, G=64, R=128 (`teacher_forced/len16384_rows4_kivi4_g64.json`)

| positions | BF16 PPL | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|
| [257, 1024) | 15.355 | 15.427 (+0.47%) | 0.0072 | 96.2% |
| [1024, 2048) | 25.206 | 25.305 (+0.39%) | 0.0073 | 95.9% |
| [2048, 4096) | 23.770 | 23.788 (+0.08%) | 0.0077 | 95.6% |
| [4096, 8192) | 22.852 | 22.936 (+0.37%) | 0.0078 | 94.9% |
| [8192, 16384) | 16.143 | 16.169 (+0.16%) | 0.0080 | 95.3% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-4 0.293.

#### Teacher-forced WikiText-2: Qwen3-0.6B, 2 x 32768 tokens, G=32, R=128 (`teacher_forced/len32768_rows2.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 12.629 | 14.185 (+12.32%) | 0.1529 | 83.4% | 12.717 (+0.70%) | 0.0058 | 97.1% |
| [1024, 2048) | 15.602 | 17.336 (+11.12%) | 0.1420 | 83.6% | 15.678 (+0.49%) | 0.0059 | 96.7% |
| [2048, 4096) | 18.654 | 19.940 (+6.90%) | 0.1279 | 82.3% | 18.781 (+0.68%) | 0.0057 | 96.4% |
| [4096, 8192) | 19.110 | 20.791 (+8.79%) | 0.1411 | 82.1% | 19.259 (+0.78%) | 0.0054 | 96.0% |
| [8192, 16384) | 12.203 | 13.660 (+11.94%) | 0.1733 | 81.3% | 12.295 (+0.75%) | 0.0059 | 96.3% |
| [16384, 32768) | 19.053 | 23.729 (+24.54%) | 0.3224 | 73.8% | 19.236 (+0.96%) | 0.0068 | 95.7% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.192, KIVI-4 0.317.

#### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 4096 tokens, G=32, R=128 (`teacher_forced/ablation/kivi24_g32_r128.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 20.310 | 22.606 (+11.30%) | 0.1456 | 80.9% | 20.460 (+0.74%) | 0.0051 | 96.6% |
| [1024, 2048) | 19.013 | 20.574 (+8.21%) | 0.1445 | 80.9% | 19.150 (+0.72%) | 0.0060 | 95.3% |
| [2048, 4096) | 20.548 | 22.296 (+8.51%) | 0.1407 | 81.8% | 20.674 (+0.61%) | 0.0053 | 96.2% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.227, KIVI-4 0.352.

#### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 4096 tokens, G=32, R=256 (`teacher_forced/ablation/kivi2_g32_r256.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 |
|---|---|---|---|---|
| [257, 1024) | 20.310 | 22.388 (+10.23%) | 0.1296 | 83.3% |
| [1024, 2048) | 19.013 | 20.357 (+7.07%) | 0.1365 | 81.8% |
| [2048, 4096) | 20.548 | 21.987 (+7.00%) | 0.1323 | 82.3% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.258.

#### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 4096 tokens, G=128, R=128 (`teacher_forced/ablation/kivi4_g128_r128.json`)

| positions | BF16 PPL | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|
| [257, 1024) | 20.310 | 20.411 (+0.50%) | 0.0105 | 94.6% |
| [1024, 2048) | 19.013 | 19.145 (+0.69%) | 0.0112 | 94.5% |
| [2048, 4096) | 20.548 | 20.592 (+0.21%) | 0.0106 | 94.4% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-4 0.328.

#### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 4096 tokens, G=64, R=128 (`teacher_forced/ablation/kivi4_g64_r128.json`)

| positions | BF16 PPL | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|
| [257, 1024) | 20.310 | 20.401 (+0.45%) | 0.0078 | 95.7% |
| [1024, 2048) | 19.013 | 19.140 (+0.67%) | 0.0075 | 95.3% |
| [2048, 4096) | 20.548 | 20.600 (+0.25%) | 0.0075 | 95.8% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-4 0.328.

#### Teacher-forced WikiText-2: Qwen3-1.7B, 4 x 4096 tokens, G=32, R=128 (`teacher_forced/ablation/qwen17_kivi24_g32_r128.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 16.940 | 17.754 (+4.81%) | 0.1157 | 85.9% | 17.046 (+0.63%) | 0.0045 | 97.2% |
| [1024, 2048) | 15.902 | 16.199 (+1.87%) | 0.1129 | 85.8% | 15.975 (+0.46%) | 0.0041 | 96.9% |
| [2048, 4096) | 16.201 | 16.446 (+1.51%) | 0.1003 | 87.3% | 16.329 (+0.79%) | 0.0039 | 97.2% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.227, KIVI-4 0.352.

## 5. Quality: generation

`rollout_eval.py` runs the public `Trainer` rollout on GSM8K test problems and
scores them with `examples/math/math_verify_reward.py`. Sampled runs use
4 samples per problem at temperature 1.0 (the RL setting) with seed 1234.

KIVI-4 matches BF16 with G = 32, 64 and 128 (72.25% against 72.16%, standard
error 0.62). KIVI-2 with R = 128 loses 3.3 points; R = 256 recovers about half
of that and R = 512 all of it, at the cost of a 544-token ring per slot. At
the same concurrency KIVI rollouts generate 15-25% more tokens per second.
Qwen3-1.7B shows the same pattern: KIVI-4 84.08% and KIVI-2 80.48% against
83.23% for BF16.

#### GSM8K test with Qwen3-0.6B, 1319 prompts x 4 samples, temperature 1.0, seed 1234, 128 concurrent sequences

| KV cache | G | R | accuracy +- stderr | generated tok/s | mean response tokens | run |
|---|---|---|---|---|---|---|
| BF16 | - | - | 72.16% +- 0.62 | 6766 | 1396 | `eval_sampled/auto` |
| KIVI-2 | 32 | 128 | 68.82% +- 0.64 | 8077 | 1424 | `eval_sampled/kivi2` |
| KIVI-2 | 32 | 256 | 70.24% +- 0.63 | 8044 | 1391 | `eval_sampled/kivi2_g32_r256` |
| KIVI-2 | 32 | 512 | 72.25% +- 0.62 | 7749 | 1356 | `eval_sampled/kivi2_g32_r512` |
| KIVI-4 | 32 | 128 | 72.25% +- 0.62 | 8208 | 1404 | `eval_sampled/kivi4` |
| KIVI-4 | 128 | 128 | 71.91% +- 0.62 | 8467 | 1385 | `eval_sampled/kivi4_g128_r128` |
| KIVI-4 | 64 | 128 | 72.04% +- 0.62 | 8048 | 1401 | `eval_sampled/kivi4_g64_r128` |

#### GSM8K test with Qwen3-1.7B, 1319 prompts x 4 samples, temperature 1.0, seed 1234, 128 concurrent sequences

| KV cache | G | R | accuracy +- stderr | generated tok/s | mean response tokens | run |
|---|---|---|---|---|---|---|
| BF16 | - | - | 83.23% +- 0.51 | 5939 | 1603 | `eval_sampled_17b/auto` |
| KIVI-2 | 32 | 128 | 80.48% +- 0.55 | 6959 | 1733 | `eval_sampled_17b/kivi2` |
| KIVI-4 | 32 | 128 | 84.08% +- 0.50 | 7013 | 1595 | `eval_sampled_17b/kivi4` |

## 6. Rollout throughput and concurrency

The same harness with greedy decoding at different `max_running_prompts`
(the number of sequences decoded concurrently). Peak device memory is the largest `nvidia-smi` reading sampled once per second
during the run (weights, KV pool, CUDA graphs and allocator cache). AReno sizes
the KV pool as `max_running_prompts × ceil((prompt + max_new_tokens) / 256)`
blocks, so the pool, not the model, decides how many rollouts fit.

With short GSM8K prompts BF16 peaks at 59.5 GiB with 192 concurrent
sequences, while KIVI-2 runs 512 in 42.7 GiB and KIVI-4 512 in 60.2 GiB,
raising throughput on the same GPU from 7716 to 10,268-10,594 tokens/s.

The long-context set prefixes each of 512 GSM8K test problems with a distinct
~8K-token WikiText-2 passage (`build_long_context_gsm8k.py`). BF16 fits 48
concurrent rollouts and runs out of memory at 64. At 32 concurrent rollouts
KIVI-4 cuts peak device memory from 45.1 to 19.5 GiB and generates tokens
1.41x faster end to end at the same accuracy, and it runs 128 concurrent
rollouts in 57.6 GiB. KIVI-2 with R = 128 loses 3.5-7 points at this
context; R = 512 narrows the gap to 1.4 points at 32 rollouts (1.3 GiB more
memory) and 3.1 points at 128. Two BF16 runs that differ only in concurrency
diverge after 138 tokens on average yet score the same here, while greedy
accuracy over 512 problems carries a standard error of about 2 points. Decode
at 8K tokens is bound by reading the cache, so tokens per second grows little
with concurrency; the gain comes from reading fewer bytes per token.

#### Greedy GSM8K rollout, first 1024 test prompts (up to 2048 new tokens): concurrency vs device memory

| KV cache | max running prompts | generated tok/s | peak device memory GiB | accuracy | mean prompt tokens | outcome |
|---|---|---|---|---|---|---|
| BF16 | 128 | 7228 | 41.9 | 74.8% | 95 | completed |
| BF16 | 192 | 7716 | 59.5 | 74.0% | 95 | completed |
| KIVI-4 | 512 | 10594 | 60.2 | 72.7% | 95 | completed |
| KIVI-2 | 256 | 9627 | 25.0 | 71.8% | 95 | completed |
| KIVI-2 | 512 | 10268 | 42.7 | 69.8% | 95 | completed |

#### Greedy rollout of 512 GSM8K test problems, each after a ~8K-token WikiText-2 passage: concurrency vs device memory

| KV cache | max running prompts | generated tok/s | peak device memory GiB | accuracy | mean prompt tokens | outcome |
|---|---|---|---|---|---|---|
| BF16 | 32 | 854 | 45.1 | 70.7% | 8095 | completed |
| BF16 | 48 | 889 | 64.4 | 70.7% | 8095 | completed |
| BF16 | 64 | - | 70.4 | - | - | CUDA out of memory |
| KIVI-4 | 32 | 1206 | 19.5 | 70.5% | 8095 | completed |
| KIVI-4 | 128 | 1345 | 57.6 | 71.1% | 8095 | completed |
| KIVI-2 | 32 | 1175 | 14.7 | 64.6% | 8095 | completed |
| KIVI-2 | 128 | 1286 | 38.4 | 67.2% | 8095 | completed |
| KIVI-2 | 256 | 1281 | 69.7 | 63.9% | 8095 | completed |
| KIVI-2 (R=512) | 32 | 1167 | 16.0 | 69.3% | 8095 | completed |
| KIVI-2 (R=512) | 128 | 1247 | 43.6 | 67.6% | 8095 | completed |

Greedy token agreement with the first run (`compare_rollouts.py`): mean position of the first differing response token, and the mean absolute log-probability difference of the shared prefix (same tokens after the same context, so it isolates the cache's effect on behavior log-probabilities).

| baseline vs candidate | first differing token (mean) | abs logprob diff on shared prefix | accuracy change |
|---|---|---|---|
| auto_mrp32 vs auto_mrp48 | 138 | 0.0078 | +0.0 pts |
| auto_mrp32 vs kivi2_mrp32 | 20 | 0.0405 | -6.1 pts |
| auto_mrp32 vs kivi2_r512_mrp32 | 20 | 0.0408 | -1.4 pts |
| auto_mrp32 vs kivi4_mrp32 | 71 | 0.0154 | -0.2 pts |

## 7. RL with the public Trainer

`train_seeded.py` runs `areno train` unchanged except for the KV-cache dtype
and a per-step rollout seed: GSPO on GSM8K train, Qwen3-0.6B, 32 prompts × 8
samples per step, up to 2048 new tokens at temperature 1.0, 128 concurrent
sequences, learning rate 1e-6 with cosine decay, 100 steps, checkpoints every
50 steps; seeds 1-6 for BF16 and KIVI-4, seeds 1-3 for KIVI-2 with R = 128
and with R = 512. Every run sees the same prompts at the same step; the seed
only changes sampling. Rollout log-probabilities are the ones the KIVI
attention produced while sampling; the trainer recomputes training
log-probabilities with the BF16 forward, so the native
`train/logp_abs_diff_mean` measures the KV-cache error in the behavior policy.
Rewards are the verifier's 0/1 scores. The BF16 repeat arm reruns the BF16
configuration with the same seeds (see [Raw logs](#raw-logs)) and shows the
run-to-run noise of this setup.

### 7.1 Reward and native log-probability divergence

KIVI-4 tracks BF16 within that noise: over six seeds each, both reach a
final-window reward of 0.8057 (the repeat arm differs by 0.0009), and its
native log-probability divergence is 0.032, twice the 0.015 that BF16
rollouts already show against the training forward. KIVI-2 with R = 128
starts 0.03 below BF16, the generation-quality gap of section 5, and keeps
pace until about step 70; then its reward falls back while the other arms keep
improving, ending 0.097 below BF16. Its divergence is 0.21, 14 times BF16's.
AReno's GSPO loss forms its ratio against the recomputed training
log-probabilities (`logprobs - logprobs.detach()`), so rollouts count as
on-policy and nothing reweights samples drawn from the shifted KIVI-2 policy.
With a 512-token residual window the divergence drops to 0.117, the reward
stays 0.003-0.022 below BF16 in every 10-step window of the first 80 steps,
and the final window ends 0.036 below BF16, inside the 0.05 target. The gap
widens to 0.03-0.04 over the last 20 steps, so longer runs should be checked
before relying on it.

#### GSPO on GSM8K train, 3-6 seeds per arm: reward and native log-probability divergence (`rl_final/summary_final.json`)

| KV cache | final-window reward per seed (last 20 steps) | mean +- std | change vs BF16 | train/logp_abs_diff_mean | median step s (rollout s) |
|---|---|---|---|---|---|
| BF16 | 0.8080, 0.8104, 0.8031, 0.8057, 0.8016, 0.8053 | 0.8057 +- 0.0032 | - | 0.0152 | 81.6 (47.1) |
| BF16 repeat | 0.8070, 0.8090, 0.8037 | 0.8066 +- 0.0027 | +0.0009 | 0.0152 | 82.1 (47.2) |
| KIVI-4 | 0.8115, 0.8033, 0.8098, 0.8088, 0.8027, 0.7979 | 0.8057 +- 0.0052 | +0.0000 | 0.0316 | 78.5 (42.5) |
| KIVI-2 | 0.6965, 0.6912, 0.7371 | 0.7083 +- 0.0251 | -0.0974 | 0.2086 | 87.8 (47.7) |
| KIVI-2, R = 512 | 0.7793, 0.7602, 0.7686 | 0.7693 +- 0.0096 | -0.0363 | 0.1167 | 67.6 (37.4) |

Mean reward over seeds in 10-step windows (the reward curve; per-step values are in the JSON):

| steps | BF16 | BF16 repeat | KIVI-4 | KIVI-2 | KIVI-2, R = 512 |
|---|---|---|---|---|---|
| 1-10 | 0.722 | 0.724 | 0.720 | 0.693 | 0.717 |
| 11-20 | 0.750 | 0.748 | 0.751 | 0.719 | 0.743 |
| 21-30 | 0.805 | 0.810 | 0.806 | 0.781 | 0.802 |
| 31-40 | 0.776 | 0.779 | 0.780 | 0.743 | 0.773 |
| 41-50 | 0.813 | 0.816 | 0.809 | 0.785 | 0.791 |
| 51-60 | 0.777 | 0.784 | 0.781 | 0.747 | 0.765 |
| 61-70 | 0.799 | 0.796 | 0.804 | 0.764 | 0.792 |
| 71-80 | 0.799 | 0.799 | 0.799 | 0.730 | 0.790 |
| 81-90 | 0.827 | 0.826 | 0.828 | 0.740 | 0.796 |
| 91-100 | 0.784 | 0.787 | 0.783 | 0.677 | 0.743 |

### 7.2 Save and resume

The step-50 checkpoints of seed 1 were loaded into fresh `areno train` runs
that skip the 1600 prompts already used and continue the seed schedule. The
first resumed step reproduces the uninterrupted run (BF16: reward 0.781 vs
0.773, mean response 1103 vs 1105 tokens, divergence 0.01528 vs 0.01531;
KIVI-2: 0.754 vs 0.746, 1291 vs 1302 tokens, 0.198 vs 0.201), and the resumed
BF16 run ends within 0.011 of the uninterrupted one. The checkpoint copied
between hosts for the BF16 resume matches its source file by file (sha256).

#### Save at step 50 and resume (seed 1): resumed steps 51-100 vs the uninterrupted run (`rl_final/summary_resume.json`)

| KV cache | run | reward steps 51-100 | reward steps 81-100 | train/logp_abs_diff_mean steps 51-100 |
|---|---|---|---|---|
| BF16 | uninterrupted | 0.7977 | 0.8080 | 0.0150 |
| BF16 | resumed from step 50 | 0.7916 | 0.7975 | 0.0148 |
| KIVI-2 | uninterrupted | 0.7206 | 0.6965 | 0.2299 |
| KIVI-2 | resumed from step 50 | 0.7473 | 0.7369 | 0.2154 |

### 7.3 Held-out validation

Every final policy was evaluated with the BF16 cache, so these numbers isolate
the effect of having trained with a KIVI cache. The validation loss is the
mean NLL of the GSM8K test reference solutions (8 windows of 4096 tokens),
the quantity behind the issue's "at most +2%" target: KIVI-4 +0.20%, KIVI-2
+1.28% and KIVI-2 with R = 512 +1.22% against BF16, while the BF16 repeat arm
differs by +0.12%. Greedy accuracy on the 1319 test problems rises from
73.16% (base model) to 76.37% after BF16 training. KIVI-4 policies reach
75.92%: with three seeds the gap was 1.5 points, and three more seeds
shrank it to 0.45, inside the seed spread. KIVI-2 policies reach 72.18%,
with one seed at 66.6%, and 74.37% with R = 512.

#### Held-out validation of the final policies with the BF16 KV cache: greedy GSM8K test accuracy (1319 problems) and teacher-forced NLL of GSM8K test solutions (8 x 4096 tokens)

| policy | accuracy per seed | accuracy mean +- std | change vs BF16 arm | NLL per seed | NLL mean | NLL change vs BF16 arm |
|---|---|---|---|---|---|---|
| trained with BF16 KV (seeds 1, 2, 3, 4, 5, 6) | 76.42%, 77.33%, 76.35%, 75.82%, 76.65%, 75.66% | 76.37% +- 0.60 | - | 1.1701, 1.1721, 1.1717, 1.1737, 1.1719, 1.1761 | 1.1726 | - |
| trained with BF16 KV, repeat runs (seeds 1, 2, 3) | 77.94%, 78.01%, 77.03% | 77.66% +- 0.55 | +1.29 pts | 1.1766, 1.1715, 1.1738 | 1.1740 | +0.12% |
| trained with KIVI-4 KV (seeds 1, 2, 3, 4, 5, 6) | 73.62%, 74.91%, 76.95%, 77.71%, 75.97%, 76.35% | 75.92% +- 1.47 | -0.45 pts | 1.1740, 1.1772, 1.1759, 1.1747, 1.1752, 1.1724 | 1.1749 | +0.20% |
| trained with KIVI-2 KV (seeds 1, 2, 3) | 72.63%, 66.57%, 77.33% | 72.18% +- 5.40 | -4.20 pts | 1.1854, 1.1856, 1.1921 | 1.1877 | +1.28% |
| trained with KIVI-2 KV, R = 512 (seeds 1, 2, 3) | 74.68%, 74.53%, 73.92% | 74.37% +- 0.40 | -2.00 pts | 1.1860, 1.1872, 1.1874 | 1.1869 | +1.22% |
| base model | 73.16% | | | 1.1693 | | |
| base model, second host | 73.16% | | | 1.1693 | | |
| BF16 seed 1, resumed at step 50 | 75.97% | | | 1.1708 | | |
| KIVI-2 seed 1, resumed at step 50 | 67.93% | | | 1.1785 | | |

## 8. Limitations

- Scope follows the issue: one GPU, the dense `qwen3` family and private
  per-request caches. Other model families, speculative decoding and NPU fall
  back to the BF16 cache with a `RuntimeWarning`; tensor-parallel rollout was
  not measured.
- Only decode attention reads quantized keys and values. Attention over fresh
  prompt tokens runs at full precision, so prefill cost is unchanged.
- AReno sizes the KV pool from `max_running_prompts` and the longest
  prompt + response, not from free memory. KIVI lowers the memory of a given
  configuration; using the headroom for more concurrent rollouts means raising
  `max_running_prompts`.
- The residual ring costs R + G model-dtype tokens per rollout slot regardless
  of context length (2.2 GiB for the 129 slots of the RL runs below, Qwen3-0.6B
  at R = 128). It dominates the cache for short sequences.
- With one row, few KV heads and 16K tokens the decode kernel is slower than
  flash-attn (table in section 3): the split-K grid cannot fill the GPU.
- KIVI-2 with R = 128 loses measurable quality on Qwen3-0.6B (sections 4-6);
  the loss is smaller on Qwen3-1.7B and with R = 512.
- The RL comparison covers 100 GSPO steps of one 0.6B model. AReno checkpoints
  store model weights only, so a resumed run starts fresh AdamW moments and a
  fresh learning-rate schedule; the BF16 resume control goes through the same
  path. Qwen3-0.6B declares tied embeddings, which `main` does not tie yet
  (#592); any effect on save/load applies to both arms equally.

## Reproducing

Assets come from ModelScope, except WikiText-2: ModelScope's `wikitext`
dataset ships a loading script, which `datasets` 4.x rejects ("Dataset scripts
are no longer supported"), so the raw parquet file of the `Salesforce/wikitext`
repository is read directly.

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
python - <<'PY'
import pyarrow.parquet as pq
from modelscope.msdatasets import MsDataset

for split in ("train", "test"):
    dataset = MsDataset.load("gsm8k", subset_name="main", split=split, trust_remote_code=True)
    dataset.to_hf_dataset().save_to_disk(f"data/gsm8k_main_{split}")
# wikitext-2-raw-v1/test-00000-of-00001.parquet from the Salesforce/wikitext dataset repository
table = pq.read_table("data/wikitext2_raw_test.parquet")
with open("data/wikitext2_test.txt", "w", encoding="utf-8") as handle:
    handle.write("".join(table.column("text").to_pylist()))
test = MsDataset.load("gsm8k", subset_name="main", split="test", trust_remote_code=True).to_hf_dataset()
with open("data/gsm8k_test_solutions.txt", "w", encoding="utf-8") as handle:
    for row in test:
        handle.write(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n")
PY
M=models/Qwen3-0.6B
D=data
```

Section 3, decode kernel:

```bash
python examples/kivi/bench_decode_attention.py --cuda-graph --warmup 2 --iters 20 \
    --rows 1 8 32 128 256 --context 1024 4096 8192 16384 32768 --output decode_graph.json
python examples/kivi/bench_decode_attention.py --warmup 2 --iters 20 \
    --rows 1 32 128 --context 1024 8192 32768 --output decode_eager.json
```

Section 4, teacher-forced quality (ablations add `--kivi-group-size`,
`--kivi-residual-length` and `--kv-cache-dtypes`; the 1.7B row uses
`--model models/Qwen3-1.7B --length 4096`):

```bash
python examples/kivi/teacher_forced_eval.py --model $M --text $D/wikitext2_test.txt \
    --length 16384 --rows 4 --prefill 256 --output tf_len16384.json
python examples/kivi/teacher_forced_eval.py --model $M --text $D/wikitext2_test.txt \
    --length 32768 --rows 2 --prefill 256 --output tf_len32768.json
```

Sections 5 and 6, generation quality and throughput:

```bash
for dtype in auto kivi2 kivi4; do
    python examples/kivi/rollout_eval.py --model $M --dataset $D/gsm8k_main_test --kv-cache-dtype $dtype \
        --n-samples 4 --temperature 1.0 --seed 1234 --max-running-prompts 128 --output gsm8k/$dtype
done
# Greedy, first 1024 problems, at a chosen concurrency; peak memory sampled with
# `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` once per second.
python examples/kivi/rollout_eval.py --model $M --dataset $D/gsm8k_main_test --kv-cache-dtype kivi2 \
    --limit 1024 --max-running-prompts 512 --output throughput/kivi2_mrp512
# Long context: 512 problems after ~8K-token passages.
python examples/kivi/build_long_context_gsm8k.py --tokenizer $M --gsm8k $D/gsm8k_main_test \
    --text $D/wikitext2_test.txt --output $D/gsm8k_test512_ctx8k
python examples/kivi/rollout_eval.py --model $M --dataset $D/gsm8k_test512_ctx8k --kv-cache-dtype kivi4 \
    --max-prompt-tokens 9216 --max-running-prompts 128 --output long_context/kivi4_mrp128
python examples/kivi/compare_rollouts.py long_context/auto_mrp32 long_context/kivi4_mrp32 \
    --output long_context/cmp_auto_mrp32_vs_kivi4_mrp32.json
```

Section 7, RL (seeds 1-6 for BF16 and KIVI-4, 1-3 for the KIVI-2 arms),
save/resume and held-out validation:

```bash
TRAIN_ARGS="--algo gspo --dataset-path $D/gsm8k_main_train --dataset-loader-fn examples/math/dataset_loader.py \
    --reward-fn-path examples/math/math_verify_reward.py --world-size 1 --tp-size 1 --batch-size 32 \
    --n-samples 8 --max-running-prompts 128 --max-prompt-tokens 512 --max-new-tokens 2048 --save-interval 50"
run() {  # NAME SEED [train_seeded.py options]
    name=$1 seed=$2; shift 2
    python examples/kivi/train_seeded.py --seed $seed "$@" -- $TRAIN_ARGS --ckpt $M --max-steps 100 \
        --metrics-log-dir rl/$name/tb --save-path rl/$name/ckpt
}
for seed in 1 2 3 4 5 6; do
    run s${seed}_auto $seed --kv-cache-dtype auto
    run s${seed}_kivi4 $seed --kv-cache-dtype kivi4
done
for seed in 1 2 3; do
    run s${seed}_kivi2 $seed --kv-cache-dtype kivi2
    run s${seed}_kivi2_r512 $seed --kv-cache-dtype kivi2 --kivi-residual-length 512
done
python examples/kivi/summarize_rl.py --window 20 --output rl/summary.json --plot rl/rewards.png \
    auto=$(echo rl/s{1..6}_auto | tr " " ,) kivi4=$(echo rl/s{1..6}_kivi4 | tr " " ,) \
    kivi2=$(echo rl/s{1..3}_kivi2 | tr " " ,) kivi2_r512=$(echo rl/s{1..3}_kivi2_r512 | tr " " ,)
# Resume seed 1 from its step-50 checkpoint: skip the 50 x 32 prompts already
# used and continue the per-step seed schedule for the remaining 50 steps.
python examples/kivi/train_seeded.py --seed 1 --kv-cache-dtype kivi2 --resume-step 50 --resume-prompts 1600 -- \
    $TRAIN_ARGS --ckpt rl/s1_kivi2/ckpt/step_000050 --max-steps 50 \
    --metrics-log-dir rl/s1_kivi2_resume50/tb --save-path rl/s1_kivi2_resume50/ckpt
# Validate a final policy with the BF16 cache.
python examples/kivi/rollout_eval.py --model rl/s1_kivi2/ckpt/step_000100 --dataset $D/gsm8k_main_test \
    --max-running-prompts 128 --output validation/s1_kivi2/gsm8k_greedy
python examples/kivi/teacher_forced_eval.py --model rl/s1_kivi2/ckpt/step_000100 \
    --text $D/gsm8k_test_solutions.txt --length 4096 --rows 8 --prefill 256 --kv-cache-dtypes kivi2 kivi4 \
    --output validation/s1_kivi2/tf_gsm8k_test.json
```

Every KIVI run logs the allocated cache once per engine, for example
`KIVI KV cache: dtype=kivi2 group_size=32 residual_length=128 blocks=1281 slots=129 bytes=9419096064`;
a run without that line used the BF16 cache.

## Raw logs

Branch [`kivi-590-artifacts`](https://github.com/EricChan2028/AReno/tree/kivi-590-artifacts)
of the contributor's fork holds every raw output behind this report: training
logs and TensorBoard events of all RL runs, evaluation summaries with one line
per completion, token-level dumps of the long-context runs, teacher-forced
and benchmark JSON, 1 Hz memory traces, the CI-equivalent CPU test logs, the
driver scripts of both hosts, `pip freeze`, `nvidia-smi` and the sha256 of
every data and model file. Checkpoints are not included. Its `README.md`
maps each directory to a section here.
