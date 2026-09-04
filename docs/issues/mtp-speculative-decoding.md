# [RFC] Integrate MTP / speculative decoding into the AReno rollout engine

## Summary

Measurements on AReno's own engine show rollout decode is deeply memory-bound: on a single H200, per-step decode latency is nearly flat from 1 to 64 concurrent sequences (8.2 ms → 8.8 ms, flash backend, Llama-3.1-8B-Instruct, temperature 1.0). Verifying multiple draft tokens per sequence is therefore almost free in that regime, which puts the speculative-decoding speedup ceiling for RL rollout at roughly **1.3–2.2x** at typical concurrency and close to the full accept length on the rollout long tail. We propose a two-phase integration: wire up the already-reserved MTP training loss first, then prototype a chain-draft verify path in the rollout engine.

## Background

- AReno is a train-infer integrated RL framework. Rollout runs on AReno's own lightweight engine (`areno/engine`); we deliberately do not depend on a heavyweight serving framework for rollout.
- Rollout decode is strictly one token per step (`areno/engine/inference.py`, `_infer_decode_next_token_tensor`), with per-bucket CUDA graphs, MoE routing replay, and linear-attention (`recurrent_slots`) state threaded through the step.
- The engine config already parses `num_nextn_predict_layers` and `mtp_loss_scaling_factor` from bailing / bailing_v3 HF checkpoints (`areno/engine/config.py:211-212`), but nothing consumes them: MTP head weights are currently dropped at load time, and the head degrades silently during RL fine-tuning.
- SpecForge (sibling project) trains draft models (EAGLE3, DFlash, …) but is not a serving runtime; it can serve as a tensor-semantics reference and a source of pretrained draft weights only.

## Measurements

Setup: 1× H200, Llama-3.1-8B-Instruct (BF16, TP=1), AReno engine via the `Trainer` SDK, temperature 1.0, 256 forced new tokens per sequence (`ignore_eos=True`), batch = concurrent sequences.

Method: no speculative decoding implemented yet. A verify step for B sequences with k draft tokens costs approximately one decode step at batch k·B, so the flatness of the per-step latency curve L(B) bounds the achievable gain: ceiling ≈ accept_length / (L(k·B) / L(B)).

Flash backend (AReno default):

| Concurrency B | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step latency (ms) | 8.2 | 7.8 | 7.0 | 7.0 | 7.7 | 7.9 | 8.8 | 10.7 | 16.6 |
| Per-seq speed (tok/s) | 123 | 129 | 143 | 143 | 130 | 126 | 114 | 93 | 60 |
| Total throughput (tok/s) | 123 | 258 | 573 | 1,143 | 2,083 | 4,040 | 7,270 | 11,955 | 15,421 |

Derived verify-cost ratios and ceilings (k=4 draft tokens, accept length τ=2.5 assumed from EAGLE3/MTP literature):

| Base concurrency | ≤16 | 32 | 64 | long tail (few active rows) |
| --- | --- | --- | --- | --- |
| Verify cost L(4B)/L(B) | ≈1.15 | 1.35 | 1.89 | <1 |
| Speedup ceiling | ≈2.2x | ≈1.85x | ≈1.3x | ≈ full τ |

The RL rollout wall clock is dominated by the long tail (a few long samples holding the batch), exactly where the gain is largest.

Control: the native attention backend does not show a flat curve (11 ms → 117 ms from B=8 to 256), so the benefit case assumes the flash backend.

Correctness note: speculative decoding with rejection sampling preserves the sampling distribution, and per-token logprobs for accepted tokens are available from the verify step's target logits, so on-policy training is unaffected in principle.

## Proposal

**Phase 0 — MTP training loss (small, independent).** Load MTP layer weights for bailing-family checkpoints and add the auxiliary loss scaled by `mtp_loss_scaling_factor`. This keeps the MTP head aligned with the policy during RL so downstream serving deployments keep their speculative-decoding acceptance rate. No rollout engine changes.

**Phase 1 — measure real accept length.** Done for bailing's native MTP head (see the Phase 1 section below); the Llama-3.1-8B + EAGLE3 pair remains available locally for a full-attention comparison.

**Phase 2 — rollout verify prototype.** Chain drafting (k=2–4, no tree attention) for full-attention model families (llama, qwen3) plus bailing's native MTP head, flash backend only. Main engine work items:

- a multi-token verify step beside the single-token decode CUDA graph (new graph shapes or eager verify first);
- KV rollback on rejection (truncate `cache_seqlens`);
- routing replay alignment: the verify step's routing is the target's;
- logprob extraction for accepted tokens from verify logits.

## Out of scope (for now)

- KDA / hybrid linear-attention models (bailing_v3): recurrent state is updated destructively, so rejection requires per-step state snapshots — deferred.
- Online draft-model co-training (SpecForge-style) — revisit after Phase 2.
- Taking SpecForge or any serving framework as a runtime dependency.

## Risks / caveats

- Numbers above are 8B dense, TP=1, single GPU; MoE and TP>1 shift the verify-cost ratio and need re-measurement (Ling-flash-2.0 weights are not cached locally yet).
- Accept length at temperature 1.0 is lower than greedy; Phase 1 exists to de-risk this.
- External draft heads drift as the policy updates; bailing's jointly-trained MTP head (Phase 0) avoids this, external EAGLE3 drafts do not.

## Phase 1 measurement: MTP head acceptance rate (on-policy)

Setup: `inclusionAI/Ling-3.0-tiny-base` on 1x H200, 32 gsm8k prompts rolled out with AReno's engine at temperature 1.0 (256 new tokens each, 8192 scored positions). The accept probability under rejection sampling is exact in closed form, alpha = sum_x min(p_target(x), q_draft(x)), so it was computed by teacher-forcing the sampled sequences and comparing the trunk's distribution for token t+2 (logits at t+1) with the MTP head's prediction of t+2 (logits at t). No speculative decoding implementation was needed.

| Metric | Value |
| --- | --- |
| alpha (depth 1, T=1 sampling) | 0.896 (std 0.17) |
| greedy top-1 agreement (T=0 bound) | 0.909 |
| alpha (depth 2, given depth-1 accept, approx.) | 0.417 |
| expected tau, k=1 | 1.90 |
| expected tau, k=2 | 2.27 |

Reading: the pretrained MTP head is a strong draft — a single MTP layer already yields ~1.9 accepted tokens per verify step at temperature 1.0, and the gap to greedy is small (0.896 vs 0.909), so the RL-temperature penalty feared in the risks section is minor for this head. Chaining the single layer to depth 2 degrades sharply (0.42) because the layer was trained only for t+2, so k=2 is the practical ceiling for bailing's native head (tau ~2.3); deeper drafting needs a multi-layer MTP checkpoint or an EAGLE-style draft. Combined with the Phase 0 verify-cost curve, this puts the expected rollout speedup at ~1.6x (k=1) to ~1.9x (k=2) for B<=16 and near-full tau on the long tail.

## Phase 0 implementation status

Implemented and reviewed (10 findings from an adversarial review, all addressed or triaged):

- MTP layers for bailing_v3 (`BailingMTPLayer`), gated on `TrainMeta.mtp_enabled`; checkpoint load/save/policy-plan via a generic `ExtraLayerListSpec`; auxiliary loss in the training step with packed-row boundary masking.
- **Opt-in only**: the loss activates solely via `runtime.mtp_loss_scale`. The checkpoint's `mtp_loss_scaling_factor` is deliberately not inherited — the auxiliary NLL trains toward every response token, which corrupts preference objectives (DPO packs rejected rows under the same response mask).
- Review fixes: `eh_proj` TP gradient all-reduce tagging (silent replica divergence at TP>1), the `ignore_eos` cancel-token fallback no longer becomes a live truncation stop id, `num_nextn_predict_layers > 1` is rejected, resuming from pre-MTP saves skips missing MTP tensors with a warning, and a warning fires when the opt-in is set on a family without MTP support (bailing v2 parses the config fields but implements no layers).
- Deferred with TODO(agent): rollout/reference partitions still build and policy-sync MTP weights they cannot execute (Phase 2 will consume them there); the MTP logits hold a second fp32 vocab shard through backward (fix pattern: chunked projection from hidden, as `packed_next_token_logprobs_from_hidden`).
- Validation: CPU tests (`tests/test_bailing_v3_mtp_cpu.py`), GPU integration on H200 with real kernels (forward/backward gradient flow, disabled-path bit-exactness, checkpoint roundtrip), and real weights on `inclusionAI/Ling-3.0-tiny-base` (16GB, nextn=1 — the `-base` variants keep the MTP head that the instruct releases strip): the pretrained head scores t+2 at NLL 0.779 vs the trunk's t+1 NLL 0.766 (uniform baseline 11.97), and a full SFT train step with `runtime.mtp_loss_scale=0.1` reports and decreases the `mtp_loss` metric through the Trainer stats path (2.17 → 0.72 after one optimizer step).

## Phase 2 implementation status: rollout speculative decoding

Implemented for bailing_v3 (Ling-3.0-tiny-base) on the flash backend, chain drafting with the checkpoint's native MTP layer, opt-in via `runtime.speculative_draft_tokens = k`.

Design (all inside AReno's own engine, no new kernels):

- `InferMeta.tokens_per_seq`: a decode forward feeds each active row `k + 1` consecutive tokens (the sampled token plus its drafts), packed sequence-major. The flash backend regroups the flat token axis and calls `flash_attn_with_kvcache` with a causal query block; the native backend rejects `tokens_per_seq > 1`.
- KDA linear attention: the existing recurrent kernel already supports SGLang's target-verify mode (`intermediate_states_buffer` + `disable_state_update`). During verify every KDA layer writes the state after each fed token into one buffer stacked over layers (and the causal-conv windows likewise); `commit_speculative_state(committed, infer_meta)` writes back the state of the last accepted token for all layers in four kernels. The recurrent and conv caches of all KDA layers are now one stacked tensor each, with per-layer views.
- MTP layer as draft model: `enable_mtp_draft()` gives it its own paged KV cache and fused MoE inference weights; `mtp_draft_forward` runs it on the next-token embeddings fused with the trunk hidden states. Drafts are chained EAGLE-style: the layer's own hidden output feeds the next draft position (this differs from the "second pass" approximation used in the Phase 1 measurement; the on-policy accept length below is the real one).
- Sampling (`areno/engine/runtime/speculative.py`): `sampling_probs` reproduces the single-token sampler's processing (temperature, top-k/top-p, EOS and suppression masks) as a distribution; `verify_drafts` runs chain rejection sampling with the exact residual, so the sampled sequence distribution is unchanged; reported logprobs are the raw target log-probabilities as in plain decoding. Draws use Gumbel-max instead of `torch.multinomial`.
- Engine loop: prefill also runs the MTP layer over the prompt (`PrefillPayload.next_input_ids`) so the draft has KV for every position; the loop carries per-row draft tokens and their full-vocab distributions, writes a variable number of tokens per row per step (stop token and length cap applied inside the step), advances `cache_seqlens` by the committed count, and reports `spec_verify_rows` so the mean accept length is `decode_scheduled_tokens / spec_verify_rows`. Verify and draft forwards have CUDA graphs per bucket (`tokens_per_seq = k + 1` and `1`).
- The model-side contract is `areno/models/base.py::SpeculativeDraftModel`; other families do not implement it yet.

Validation:

- CPU: `tests/test_speculative_cpu.py` (sampler equivalence, rejection sampling preserves the target distribution to 1.5% over 40k trials, greedy exactness, mask and layout helpers) and `tests/test_speculative_rollout_cpu.py` (a deterministic fake model driving the real loop: accepted/rejected drafts, stop token inside a step, length cap, continuous batching admission, k=1).
- GPU (Ling-3.0-tiny-base, 1x H200): one verify forward over 3 fed tokens vs. three sequential decodes agrees to KL <= 8e-3 per position, below the 2e-2 noise floor between the prefill path and sequential decode; recurrent, conv and KV state after full and partial commits match; the MTP draft forward in prefill mode matches the Phase 0 training-path MTP logits (argmax agreement 100%). End to end, greedy speculative and greedy plain rollout agree on 12/16 gsm8k prompts and the rest diverge at bf16 near-ties late in the sequence; eager and CUDA-graph speculative runs are bit-identical.

Performance (decode loop measured inside the worker; the client-side wall clock also contains ~10 s of per-session weight fuse/offload overhead that is unrelated and dominates short runs):

| Concurrency B, k=2 | plain step | speculative step | accept length (tokens / row / step) | decode-phase speedup |
| --- | --- | --- | --- | --- |
| 1 | 6.5 ms | 9.3 ms | 2.41 | 1.67x |
| 4 | 7.3 ms | 10.7 ms | 2.27 | 1.41x |
| 8 | 8.6 ms | 12.0 ms | 2.37 | 1.66x |
| 16 | 9.7 ms | 13.7 ms | 2.37 | 1.60x |
| 32 | 10.4 ms | 14.9 ms | 2.36 | 1.52x |
| 64 | 12.7 ms | 19.6 ms | 2.36 | 1.36x |

(gsm8k prompts, 256 forced new tokens per row, temperature 1.0; decode phase = sum of decode-step time inside the worker.) k=1 at B=16 gives 1.29x (accept length 1.87), so k=2 is the better default for this head. A load test with EOS enabled (64 prompts x 2 samples, 512 max tokens, 32 running) produced 65k tokens with finite logprobs and consistent lengths for both plain and speculative rollout. The remaining gap to the ~1.9x zero-overhead ceiling is the fp32 LM head read three times per step, the two MTP-layer forwards, and about 1 ms of host-side gap per step from the per-step `.item()` syncs.

Two defects found by the sweep are fixed: a single active row made the verify conv output a strided view that the recurrent kernel read as contiguous (garbage at B=1), and every verify CUDA graph bucket held its own KDA intermediate-state buffer (about 16 GB across buckets at 64 running rows); the buffer is now allocated once and shared.

Not done: native backend support, TP > 1 measurement, gating MTP layer construction on the worker role (reference workers still build them), and an async scheduler to hide the host syncs.

## Related fix found during the study

`_cancel_stop_token` in `areno/engine/inference.py` crashed with `TypeError: int(())` whenever `ignore_eos=True` (the CUDA backend passes `eos_token_id=()`), meaning the CUDA `ignore_eos` path had never worked. Fixed, with CPU regression tests in `tests/test_cancel_stop_token_cpu.py`.
