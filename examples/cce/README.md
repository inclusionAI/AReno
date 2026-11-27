# Cut Cross-Entropy (CCE) baseline

`F.cross_entropy(hidden @ weight.T, targets)` materializes a `[tokens, vocab]`
logits block. For 8k supervised tokens and a 152k vocabulary that is ~2.4 GB in
BF16 for every training step (plus the same amount again for backward).

`areno.accel.cut_cross_entropy` computes the identical loss from the hidden
states, the LM-head weight, and the target ids, streaming the vocabulary in
chunks: each chunk is projected, immediately reduced to a per-row
`(max, sum-of-exp)` pair, and thrown away. Only the small per-chunk statistics
and one `[rows, hidden]` target dot-product survive the forward pass, and
backward recomputes the chunks instead of retaining them. Peak activation memory
is `O(rows * vocab_chunk)` instead of `O(rows * vocab)`.

## Enabling it

Off by default. When off, training runs the existing path unchanged
(`tests/test_cut_cross_entropy.py` asserts the legacy path never calls into the
new module).

CLI:

```bash
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --reward-fn-path examples/math/math_verify_reward.py --algo gspo --tp-size 1 \
  --cut-cross-entropy
```

Python SDK:

```python
from areno.api.trainer_config import SFTTrainerConfig

config = SFTTrainerConfig(
    algo="sft",
    ckpt="Qwen/Qwen3-0.6B",
    dataset_path="gsm8k:main",
    backend="cuda",
    cut_cross_entropy=True,   # -> RuntimeConfig.cut_cross_entropy
)
```

The equivalent runtime knob (for callers that build `CudaConfig` directly) is
`runtime={"cut_cross_entropy": True}`.

### Scope of the first landing

| item | state |
| --- | --- |
| deferred LM-head path (CUDA training, `tp_size=1`) | supported |
| materialized-logits path (`tp_size>1`, NPU, MLX) | not applicable; the run warns once and keeps the existing path |
| `label_smoothing != 0` | `NotImplementedError` (never mis-scored) |
| LM head with a bias | `NotImplementedError` |
| logit soft-capping (Gemma-style) | supported via `logit_softcap` |
| `ignore_index` | supported (`reduction="none"` yields `0` for ignored rows like PyTorch) |
| all targets ignored | returns `0` for `sum`/`mean` instead of `nan` |
| fused CUDA/Triton kernel | not yet; the pure-PyTorch reference runs everywhere |

## Reproduce the CPU baseline

Measured on this workstation (Windows, CPU-only Python, torch 2.7.1+cpu,
`rows=2048 hidden=512 vocab=65536 chunk=4096`). Each row is a separate process
so peak RSS growth is meaningful:

```bash
python -m pytest tests/test_cut_cross_entropy.py -q

PYTHONPATH=. python examples/cce/bench.py --rows 2048 --hidden 512 --vocab 65536 \
  --chunk 4096 --iters 3 --mode materialized
PYTHONPATH=. python examples/cce/bench.py --rows 2048 --hidden 512 --vocab 65536 \
  --chunk 4096 --iters 3 --mode cce
```

FP32 forward+backward:

| path | loss | fwd+bwd median ms | peak RSS growth MB |
| --- | --- | --- | --- |
| materialized logits | 96.211494 | 3332 | 1285 |
| cut cross-entropy | 96.211487 | 5586 | 107 |

BF16 forward+backward (1 iteration, so timing is noisy):

| path | loss | fwd+bwd median ms | peak RSS growth MB |
| --- | --- | --- | --- |
| materialized logits | 95.500000 | 1555 | 494 |
| cut cross-entropy | 95.153748 | 6040 | 71 |

Reading the numbers:

- The loss agrees to `7e-6` in FP32. In BF16 the two paths differ by ~0.4%
  because the reference rounds the `[rows, vocab]` logits to BF16 while CCE
  accumulates in FP32; `tests/test_cut_cross_entropy.py` pins that to
  `rtol=1e-2`.
- Peak memory drops ~12x (FP32) / ~7x (BF16).
- On CPU the pure-PyTorch reference is **slower** (~1.7x) than one big matmul.
  This is expected: it trades a single BLAS call for a chunked Python loop with
  backward recompute. The payoff is memory, and on GPU the chunked reduction is
  where a fused kernel belongs.

## GPU baseline (TODO — no GPU on the machine that produced this report)

This issue is a first landing on a host without CUDA, so GPU numbers are not
recorded here yet. Fill this table with a CUDA host using the protocol below.

Protocol:

1. `torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()` before each
   run, and read `torch.cuda.max_memory_allocated() - base` after
   `loss.backward()`.
2. Compare `F.cross_entropy(hidden @ weight.T, targets)` against
   `cut_cross_entropy(..., vocab_chunk_size=<candidate>)`.
3. `tests/test_cut_cross_entropy.py::test_peak_memory_below_materialized_logits_cuda`
   already asserts `cce_peak < materialized_peak / 2`; run it first.

| model / shape | dtype | tp | path | step time ms | peak allocated GB | loss |
| --- | --- | --- | --- | --- | --- | --- |
| TODO | bf16 | 1 | materialized | TODO | TODO | TODO |
| TODO | bf16 | 1 | cut cross-entropy (chunk=4096) | TODO | TODO | TODO |

Also worth recording on GPU: the chunk-size sweep (1024 / 4096 / 8192) because
peak memory scales with `rows * vocab_chunk` and step time with
`vocab / vocab_chunk` matmuls.

## Follow-ups

- Fused (Triton) chunked projection + log-sum-exp for CUDA; the reference
  implementation keeps the public signature stable for it.
- Vocabulary-parallel support: the sharded head (`tp_size>1`) needs a
  cross-rank log-sum-exp plus a custom backward, so it currently raises
  `NotImplementedError` instead of silently scoring a local shard only.
