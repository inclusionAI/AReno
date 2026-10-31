GaLore complete-training capacity on Qwen3-1.7B
===============================================

Strict deferred GaLore supports twice the largest confirmed batch and context
sizes of native full AdamW in the registered grid on 40 GiB A100s. At both
candidate boundary shapes, it completes 203 updates on each of three cards,
including a production projection refresh; full AdamW has a retained CUDA
allocation failure on every card. Native 8-bit AdamW also completes both
shapes: this study establishes no capacity advantage over 8-bit.

Against 8-bit at these large packed inputs, allocated peak falls about 1.9%
and complete score-plus-train time increases 0.58--1.88% in same-card
comparisons. These are capacity measurements, not new quality results.
The historical 30.3% reduction in :doc:`the matched Dolly measurements
<galore-deferred-qwen3>` remains scoped to that context 512 / batch 4 workload
and historical runtime. It is not extrapolated to every input size.

Frozen configuration and workload
---------------------------------

* ModelScope ``Qwen/Qwen3-1.7B``, revision
  ``4855588ea1a12789f2e965e5f52a9e4a24c94b2a``:
  1,720,574,976 unique trainable parameters, 226 optimizer owners,
  113 rank-128 basis owners. Model assets are pinned by byte count/SHA256.
* Three NVIDIA A100-SXM4-40GB devices, each running an independent single-GPU
  job; these are device repeats, not distributed training. PyTorch 2.6.0+cu124.
* Final runtime SHA256
  ``337c7612282e8fc5254d890ac7659c88c2f38a8e8430d5b3cdc84f6dc96b28b8``
  and native extension SHA256
  ``f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226``.
  This directly measures the final fused runtime; it does not rerun the
  historical quality endpoints or change the precision policy.
* Native full AdamW, native 8-bit AdamW, and opt-in strict deferred GaLore
  share LR = min LR = 1e-5, betas (0.9, 0.95), decay 0.01, global clip 1.0, native
  eager attention and nonreentrant activation checkpointing. No compilation
  or offload; microbatch=batch.
* GaLore retains logical FP32 masters and FP32 projected moments/bases:
  rank 128, gap 200, scale 1.0, recycled projector, oversampling 16, one iteration,
  final QR, moment transport, compact masters, tile 16384, gradient release
  and fused writeback. Only this method enables deferred projection and the
  rounded-scale/writeback fusion. The optimizer implementation is unchanged.
* The fixed 13,544-record Dolly training fold produces 2,505,350 real tokens.
  Separately encoded prompt/response text, with the documented EOS boundary
  rule, is viewed as contiguous blocks of exactly the requested context.
  Each sequence has ``prompt_len=1``; every later token is an LM target.
  No random tokens or short examples stand in for full-length inputs.
* Seed 0 absolute-cursor epoch shuffling selects identical blocks across
  methods/devices. Each update uses one native SFT pack, gradient scale 1,
  and exactly ``batch * (context - 1)`` targets. Incomplete trailing blocks
  are dropped separately for each length.

This packed-text capacity workload differs from historical response-only
instruction SFT. No validation/test split is read or hashed and no checkpoint
is saved. Finite loss and native logprob consistency are execution guards,
not evidence of equal or improved model quality.

Grid, confirmation and capacity
-------------------------------

The batch axis fixes context 1024 and tests ``1,2,4,8,12,16,24,32``. The context
axis fixes batch 4 and tests ``512,1024,1536,2048,3072,4096,6144,8192``. Their
intersection is deduplicated: 15 cells per method, 45 screens. All cells are
attempted; feasibility is not inferred from monotonicity.

Each fresh screen runs two warmup plus 14 measured updates. The largest
screen pass per method/axis is confirmed in a fresh process on a rotated
card with two warmup plus 201 measured updates. GaLore includes actual
refresh 200 and ordinary updates 201/202. Only typed CUDA OOM permits the
next-smaller screen pass, with at most two attempts per axis; identical
method/shape confirmations are reused across axes.

Both candidate frontier shapes are then tested with all three methods on
all three cards, with method order rotated by card. Successful arms complete
203 updates. CUDA allocation failures retain traceback, stage, completed
updates and exit 42. Timeouts and ordinary errors never become OOM boundaries.

.. list-table:: Largest 203-update-confirmed sizes in the registered grid
   :header-rows: 1

   * - Method
     - Batch at context 1024
     - Context at batch 4
   * - Full AdamW
     - 8
     - 2048
   * - 8-bit AdamW
     - 16
     - 4096
   * - Strict deferred GaLore
     - 16
     - 4096

At both context 1024/batch 16 and context 4096/batch 4, GaLore and 8-bit complete
203 updates on all three cards. Full AdamW fails during training after one
completed update in all six same-cell/device comparisons. Initialization or
one update alone is insufficient to establish fit. The listed sizes are
maximum confirmed grid cells, not exact global limits or claims about
untested intermediate sizes.

The primary gate requires candidate success and full-AdamW typed OOM at the
same cell on all three cards, plus at least 1.25x confirmed-grid axis size.
Both axes pass with ratio 2.0. The separate stronger gate against 8-bit fails:
both ratios are 1.0 and 8-bit fits. Neither threshold nor grid is changed.

Same-card memory and time
-------------------------

The complete public score-plus-train call must fit. Allocated peaks come
from the CUDA-owning worker, spanning scoring and the whole training call,
including peak segments across native resets. Reserved peaks and sampled
NVML device usage are separate raw fields; samples are not an exact maximum.

Time includes public scoring, logprob assignment and training. Summaries
exclude two warmup updates and include refresh 200. Optimizer-only events
omit projection moved into backward and are not an end-to-end metric.

.. list-table:: Boundary measurements; ranges across three same-card comparisons
   :header-rows: 1

   * - Context / batch
     - 8-bit allocated GiB
     - GaLore allocated GiB
     - Memory reduction
     - Whole-call time overhead
   * - 1024 / 16
     - 34.2558
     - 33.6077
     - 1.8919%
     - 1.2773--1.8821%
   * - 4096 / 4
     - 34.2563--34.2578
     - 33.5994
     - 1.9175--1.9217%
     - 0.5778--0.6793%

Each timing comparison uses one 203-update arm per method/shape/card with
rotated order, rather than repeated ABBA timing. These ranges do not certify
a universal sub-percent speed difference. GaLore refresh takes 8.529--8.559
seconds at 1024/batch 16 and 12.275--12.302 seconds at 4096/batch 4: approximately
1.45--1.46x and 1.28x the corresponding ordinary-update means. All candidate
boundary arms pass refresh, lifecycle, finite-loss, clock and target guards.

How the advantage scales
------------------------

Even within the current frozen runtime and exact-length workload, the
context 1024 screens show the following change with batch size:

.. list-table:: Two warmup plus 14 measured updates; not refresh qualification
   :header-rows: 1

   * - Batch
     - 8-bit allocated GiB
     - GaLore allocated GiB
     - Reduction
   * - 1
     - 16.0706
     - 11.1206
     - 30.8018%
   * - 4
     - 16.9529
     - 14.7957
     - 12.7241%
   * - 8
     - 21.7274
     - 21.0695
     - 3.0278%
   * - 16
     - 34.2558
     - 33.6077
     - 1.8919%

These short screens do not independently certify production-refresh stability;
the large boundary shapes have the separate 203-update confirmations above.
Larger inputs increase shared activation/backward-temporary costs, and peaks
need not occur at the same point for each method. The absolute gap also
changes: this is not fixed saved bytes divided by a larger total. OOM stacks
locate failure in training/backward but do not establish a particular
LM-head/attention kernel as the cause; no kernel-level remedy is claimed.

Interrupted attempt and preserved evidence
------------------------------------------

The first controller completed all 45 screens and four 203-update confirmations.
Its 1800-second limit interrupted the 8-bit 4096/batch 4 arm after 186 updates;
the resulting controller stop interrupted GaLore after 184. Neither reached
refresh 200. Both remain invalid operational attempts, preserved and excluded
from success/OOM counts.

A separate repair changes only confirmation timeout 1800 to 3600 seconds,
reuses exactly 49 valid cases with explicit evidence roots, restarts both
incomplete confirmations fresh for 203 updates, and performs 18 boundary
comparisons. Runtime, precision, data, selection and gates stay fixed.
The repaired controller finishes without execution errors. Combined totals
are 69 valid cases: 47 successes and 22 typed CUDA OOMs, with 4,128 completed
update rows and 26,448 resource-monitor samples. The 370 rows from the two
invalid attempts remain separately preserved and excluded from those totals.

The tensor-free archive and portable verifier are in
:download:`galore-capacity-evidence <galore-capacity-evidence/README.md>`. From the
repository root::

    python docs/reports/galore-capacity-evidence/verify.py \
      docs/reports/galore-capacity-evidence --repo . --json-output /tmp/capacity-review.json

Verification reads archives without extracting or executing the original
machine-specific harness. It checks lossless packaging and recomputes the
schedule, source identity, per-step guards, paired input identities, peak/time
summaries and both capacity gates. Original native profiles, failures,
protocols and audits are retained. Weights, native binaries, checkpoints and
raw/tokenized training payloads are excluded. Portable checks therefore
cannot reconstruct token values from hashes alone; original full-stream and
per-update token replay checks are preserved as audit evidence, not claimed
as a fresh offline token replay. Source/model/data and packaging identities
are recorded separately from numerical and quality conclusions.
