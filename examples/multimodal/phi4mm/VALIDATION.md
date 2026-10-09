# PR2 vision validation

Validated on 2026-10-08/09 with the official BF16 Phi-4 Multimodal snapshot,
PyTorch 2.9.1+cu128, Transformers 4.48.2, PEFT 0.13.2, and RTX 3090 GPUs.
The reference uses the checkpoint's `Phi4MMAttention` and `SiglipAttention`
eager implementations; AReno uses native decoder attention and SDPA vision
attention. FP32 logits projection remains enabled in AReno.

## Checkpoint coverage

| Classification | Keys |
| --- | ---: |
| Total official checkpoint | 2,047 |
| Loaded language | 194 |
| Loaded vision encoder/projector | 454 |
| Loaded vision LoRA | 256 |
| Deliberately skipped speech LoRA | 256 |
| Deliberately skipped audio tower | 887 |
| Unknown / missing required | 0 / 0 |

## Official numerical comparison

The processor's tokens, crop tensors, image sizes, and attention masks match
exactly through both AReno dataset and serving encoders. Prompt lengths are
558, 430, 971, and 12 tokens respectively. AReno uses 256-token prefill chunks.

| Fixture | Both implementations generate | TP1 minimum decode-logits cosine | TP2 minimum decode-logits cosine |
| --- | --- | ---: | ---: |
| Square image | `red<\|end\|>` | 0.999988409 | 0.999979847 |
| Portrait image | `blue<\|end\|>` | 0.999987027 | 0.999987822 |
| Two images, unequal shapes | `red blue<\|end\|>` | 0.999978080 | 0.999984348 |
| Text only | `Four.<\|end\|>` | 0.999993581 | 0.999992184 |

All generated tokens, including EOS, match. Across fixtures, TP1's minimum
intermediate-tensor cosine is 0.997182987 (final norm); the minimum cosine
between chunked-prefill and full-prefill logits is 0.999987270. All recorded
tensors are finite. These are numerical BF16 comparisons, not bitwise equality
with the official implementation.

Initial comparisons used module-wise GPU weight staging because other jobs
occupied GPU memory. The final TP1 run loaded each complete model on GPU and
reproduced the same metrics. TP2 used module-wise staging with live NCCL
collectives. The raw tensor artifacts are kept outside the repository;
the adjacent scripts regenerate them.

The fixtures are deterministic synthetic images and short outputs. They do not
establish broad visual task accuracy, long-answer token identity, or performance.
The LongRoPE boundary is additionally exercised with a reduced-size GPU model,
not a full-checkpoint 131k-context workload.

## Worker and checkpoint lifecycle

The real worker pipeline passed two rounds of four mixed image/text requests
with two active slots and CUDA-graph decode enabled. Reversing request order
in the second round preserved exact agreement with the reference. A separate
GPU test captures and replays the model while changing vision/text slot flags,
and verifies agreement with eager execution and slot reset behavior.

A fresh-process reload of the saved full checkpoint produced bitwise-identical
single-image and multi-image full-prefill/cached logits (`max_abs=0`) and token
sequences. All 2,047 keys were retained, including unused audio/speech weights.
Tokenizer, processor, generation config, and the checked official Python assets
retained identical SHA-256 hashes.

## Fixes established by validation

- Convert structured image messages to Phi's string template and numbered
  image markers in dataset and serving paths. Reuse the same image-token
  resolver so unknown-token fallback is not mistaken for an image token.
- Read cached keys from previous prefill chunks. Before this fix, the red-image
  fixture generated `submissions` instead of `red` despite correct full forward.
- Carry full prompt lengths through prefill metadata so every chunk chooses
  the same LongRoPE factors, including chunks preceding the context boundary.
  Legacy callers without full-length metadata retain the explicit boundary guard.

## Test scope and environment limitation

Final relevant CPU regression suite: 186 passed, 4 CUDA-dependent skips in the
sandbox. The separate GPU suite passed 79 tests across Phi4MM adapter/vision
and native attention, including the final LongRoPE change. Ruff lint/format,
Python compilation, and `git diff --check` also passed.

An additional cross-family sweep previously had 124 passes and one unrelated
Gemma4 import failure: this reference environment's Transformers 4.48.2 does
not export `transformers.video_utils`. The failing test and Gemma4 implementation
were unchanged. This report does not claim the entire repository suite passes
under the older official-reference environment.

No full-checkpoint training job, audio inference, TP4/TP8 numerical run, or
performance benchmark is claimed by this PR2 validation.

## Reference provenance

| File | SHA-256 |
| --- | --- |
| `config.json` | `49e1c05f93d43d7f17715b779a2576235b019f587285d7d914e5b05156253f62` |
| `modeling_phi4mm.py` | `e2b44eb7a66d6cc54524cee1ff9ba92d0658d435ea8900329ea0dbdb85c6439d` |
| `processing_phi4mm.py` | `84914d3e12256b4e2186e040c9830c11408468b6774f42afe85e6f8de2626d50` |
| `vision_siglip_navit.py` | `7d5c053341ee9c099126fe675d5dcdc0ed5c0246f92fffdec78a1ab2f804e28d` |
