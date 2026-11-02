# Strict deferred GaLore delivery integration

The new local branch `feat/galore-strict-deferred-delivery` starts at immutable
delivery commit `aae6eb27e0974863905b9cc5a6869af06f68a383`. The original delivery
checkout and all frozen experiments remain unchanged.

## Runtime change

Copied the reviewed `80a70ff0f855fd5ec742ee985a507bc8adcbfd0ee3dafc23cc51da9708981c79`
strict source from `runs/deferred_scale_confirm_20261030/repo`. All 221 runtime
Python files in the new checkout are byte-identical to that measured source,
using its original canonical path-and-content digest. `SOURCE_IDENTITY.json`
contains every individual runtime hash and frozen native extension identity.
The native extension was copied unchanged, not rebuilt or installed.

The integration changes three runtime files: `optim/deferred_projection.py`,
`optim/galore.py`, and `training.py`. The optimizer projects each native leaf
gradient in a post-accumulation hook, releases it, and keeps projected coordinates
plus owner norm scalars until the full-backward version barrier and global clip.
Persistent master/moment/basis state is published only after the barrier. The
FP32 master remains lossless, including when its storage uses the existing
BF16-plus-residual representation.

This path is private and defaults off. It requires
`ARENO_EXPERIMENT_DEFERRED_PROJECTION=1`, one nonreentrant microbatch, one GPU,
dense Qwen3, no compilation, no optimizer offload, and no active multimodal
learning-rate groups. It does not add or alter CLI/config fields. The measured
recipe also enables recycled projection/final QR, moment transport, compact
lossless masters, 16384-row tiled fused writeback and gradient release through
existing optimizer fields. Deferred and ordinary checkpoint backend policies
remain explicitly distinct and incompatible loads reject before mutation.

No BF16-owned approximate master policy or unsuccessful speed branch is added.

## Correctness verification

`tests/test_deferred_projection_cpu.py` carries the original preflight,
single-owner, no-early-publication, version-barrier and checkpoint-policy guards.
New portable independent-reference CPU/CUDA tests use the original independent
QR/Ritz projection, `torch.optim.AdamW` coordinate update and dense covariance
transport oracle. Each of 12 cases per device checks ten updates, refreshes at
0/3/6/9, original `atol=1e-6, rtol=1e-4`, and exact serialized continuation.
They test tall/wide/square FP32 and BF16-stored parameters, plus fallback vectors,
at actual and stress learning rates. They do not establish arbitrary-model or
future native CUDA trajectory equivalence.

The separate `check_default_off.py` creates 24 flag-unset/flag-zero trajectories
and 192 byte-level model/optimizer/RNG snapshots. Its outputs for the immutable
floor and the integrated checkout are byte-identical.

Fresh smoke/reference commands (source the root-write environment first):

```bash
source /home/lws/ling/final_round_20261031/env.sh
cd /home/lws/ling/final_round_20261031/delivery
CUDA_VISIBLE_DEVICES='' PYTHONPATH=repo taskset -c 44-47 python check_reference.py --device cpu --output CPU_INDEPENDENT_REFERENCE.json
taskset -c 40-43 python run_cuda_checks.py --name cuda-independent --seconds 180 -- python /home/lws/ling/final_round_20261031/delivery/check_reference.py --device cuda --output /home/lws/ling/final_round_20261031/delivery/CUDA_INDEPENDENT_REFERENCE.json
```

`run_cuda_checks.py` holds the shared GPU1 lock, resolves its UUID, excludes
foreign PIDs, pins inherited CPU affinity, checks 9 GiB admission/8 GiB runtime
root disk space, and terminates only its owned process group with birth-time
verification on timeout/error. Its logs and status records retain exact commands.

## Environment limitation retained

The first fresh complete CPU-selected run with CUDA hidden failed in unrelated
model imports because the installed FLA/Triton 3.2 stack initializes an autotune
GPU driver at import time (`0 active drivers`). Its original `cpu-full.log` is
retained. A separate complete CPU-selected run under the GPU1 lock with a visible
driver is recorded as `cpu-full-visible-driver.*`; it is not relabeled as the
CUDA-hidden result. No dependency or native library was altered to repair it.

Historical quality, memory, and speed results belong to their frozen protocols.
Exact source identity makes their runtime attribution explicit; these fresh
engineering checks are not new training endpoints or performance measurements.
