---
name: areno-model-adaptation
description: Use this skill when adapting or debugging areno model plugins, checkpoint load/save paths, rollout behavior, fused kernels, or model-specific compatibility with areno runtime and trainer interfaces.
---

# areno Model Adaptation

Use this skill when adapting a new model plugin for `areno.models`.
The goal is to keep model-specific code in this repository while preserving
the areno core runtime, trainer, and checkpoint interfaces.

Current project direction:

- Runtime-critical paths should use ARENO-owned model code and kernels. Do not
  reintroduce TransformerEngine, SGLang kernels, or FLA as runtime dependencies
  from model plugins.
- Third-party implementations are references for tensor semantics, cache
  layout, and numerics only. Production execution should go through
  `areno.engine.layers`, `areno.engine.runtime`, `areno.engine.checkpoints`, or ARENO CUDA
  extension entry points.
- The production validation target is CUDA/Linux. Do not add non-CUDA kernel
  requirements unless the user explicitly changes that scope.

## Start By Asking

Before touching code or starting a remote run, actively ask the user for any
missing operational details. Do not guess these values from prior sessions.

Required inputs:

- Model identity: model name, expected architecture, and target checkpoint
  layout.
- GPU access: the exact user-approved way to enter and interact with the GPU
  environment.
- Checkpoint acquisition: the exact command, mounted path, or already-available
  local path for the model checkpoint.
- Dataset and reward/preferences: dataset path, reward file for online RL,
  preference fields for DPO, SFT schema, algorithm (`sft`, `dpo`, `gspo`,
  `grpo`, or PPO), and any sampling/training args the user cares about.
- Validation target: normally checkpoint load, rollout, train, save, reload,
  and at least two training steps on the requested algorithm.

If one of these is missing and cannot be discovered from the current repo, ask
a concise question and wait. Once the user has specified an access path or
workflow, use that path consistently.

For remote code changes, edit locally, push the relevant repo, then pull on the
remote host. Commit and push the repo that actually changed.

## Repository Layout

Each model plugin is a directory named after the model family:

```text
areno.models/
  __init__.py
  model_name/
    __init__.py
    model.py
    checkpoint.py
```

`model.py` owns:

- The `torch.nn.Module` implementation.
- The `ModelAdapter` subclass.
- HF config matching and conversion into `areno.config.ModelConfig`.
- Runtime lifecycle hooks such as `prepare_infer_weights`,
  `clear_infer_weights`, `onload_train_weights`, `offload_train_weights`,
  `reset_kv_cache`, and any model-local cache setup.

`checkpoint.py` owns:

- HF safetensors load and save mapping.
- Tensor-parallel shard and merge behavior.
- Checkpoint specs using shared checkpoint primitives wherever possible.

Keep these files separate. Do not hide checkpoint naming or tensor layout logic
inside `model.py`.

## Dependency Rules

Allowed dependencies from plugins:

- `areno.config`
- `areno.models.base`
- `areno.engine.layers`
- `areno.engine.runtime`
- `areno.parallel`
- `areno.engine.checkpoints`
- model-local files in the same plugin directory

Avoid importing:

- `areno.engine`
- `areno.models.registry`
- trainer APIs
- backend APIs
- `transformer_engine`
- `sglang`
- `fla`
- third-party CUDA/Triton kernels from model code

The only registry touch point should be the top-level package registration in
`areno.models/__init__.py`.

## Reference Implementations

Always inspect nearby plugins first:

- Dense or mostly dense models: `qwen3`, `gemma4`, `llama`
- MoE or custom kernel models: inspect the closest current plugin and core
  kernel primitives
- Hybrid or linear-attention models: the closest current plugin plus external
  references below

For models already supported elsewhere, compare against:

- SGLang model and kernel implementation as a semantic reference only,
  especially inference cache layout, fused decode kernels, and
  attention/linear-attention backends.
- Megatron model implementation, especially train-time tensor shapes,
  sequence packing, tensor-parallel splits, and checkpoint naming.
- FlashAttention and FLA implementations as references for tiling, state update
  equations, and backward math, not as imports.

Do not copy large code blindly. Use these references to confirm semantics:
projection order, q/k/v shape, gate/beta/alpha conventions, rotary layout,
normalization, cache indexing, packed sequence handling, and save/load names.

## Adaptation Plan

For a new model, work in this order:

1. Parse config and construct the module with correct tensor-parallel local
   shapes.
2. Implement checkpoint load only, then verify load reaches 100% on every rank.
3. Run a base rollout with printed completions before touching training.
4. Add train forward with the simplest correct ARENO/runtime path, then compare
   rollout logprobs against train logprobs on the same tokens.
5. Add fused kernels only when backward and dtype behavior are implemented and
   numerically checked.
6. Add save support and verify save/reload before running long training.
7. Optimize decode and train performance only after correctness is stable.

Do not tune RL hyperparameters to hide model-adaptation bugs. Garbled
completions, impossible logprob drift, or reload-only failures are model or
checkpoint issues until proven otherwise.

## Kernel Requirements

Prefer fused ARENO production kernels over Python reference paths:

- Use areno attention backends for normal full attention.
- Use ARENO-owned kernels for fused activation, normalization, routing, and model
  specific fast paths. Do not add new model code that imports third-party kernel
  packages directly.
- FlashAttention may be used through an existing areno-owned attention
  abstraction when the core runtime owns that dependency. Model plugins should
  not call FlashAttention APIs directly.
- Kernel API names should describe math semantics, not model families. Prefer
  names like `areno_rmsnorm_silu_gate` or `areno_optional_scale_rmsnorm` over
  model-branded names. Model classes may stay model-specific, but the exported
  kernel surface should be reusable and neutral.
- Do not silently fall back to PyTorch reference code for required train or
  inference paths. If a CUDA kernel is required and missing, fail loudly and add
  the ARENO kernel rather than hiding the gap.
- Linear-attention and grouped-MoE fast paths should be implemented as ARENO
  kernels before they become required runtime dependencies.
- For gated-delta or other recurrent linear-attention models, use SGLang/FLA
  only as semantic references. The runtime path should call ARENO-owned kernels
  with explicit train backward and decode state-update implementations, not
  import `fla` or `sglang` from model code.
- Do not add TransformerEngine fallback paths. If TE code is useful, translate
  the semantics into local ARENO layers/kernels and verify against a small
  PyTorch reference.
- For gated MLPs, align with Megatron's GLU/SwiGLU semantics: use one fused
  gate/up column projection, apply the activation-and-multiply in a fused
  kernel when it supports backward, then use a row-parallel down projection.
- Training paths must preserve gradients. Use fused ARENO activation kernels only
  when their backward path is implemented and numerically checked.
- Training layer loops must honor runtime activation recompute. Wrap decoder
  layer calls with `areno.engine.runtime.recompute.checkpoint_layer(...)` and gate
  only on `TrainMeta.activation_checkpointing`; rollout, prefill, decode, and
  scoring paths must keep their normal forward behavior when the flag is off.
- Search model files for `torch.nn.functional`, direct `torch.sigmoid`, and
  third-party kernel imports during review. Any remaining occurrence should be
  intentional, documented as a temporary implementation gap, and prioritized if
  it appears in a hot train or rollout path.
- Keep train and inference semantics aligned. If train and rollout use
  different kernels, verify the same scale, q/k normalization, window/cache
  behavior, and dtype assumptions.
- If a new kernel cannot be implemented immediately, mark the reference path as
  a temporary correctness path and keep it outside hot decode/train loops when
  possible. Do not bury slow or unsupported behavior behind silent fallback.

Autotune and first-step Triton compilation can be expensive. Diagnose with
`py-spy` before changing kernels. A worker stack inside Triton compile/autotune
is not by itself a correctness bug.

## Attention And Decode

Decode is usually one token per active sequence, so the kernel and cache path
must be optimized for tiny query length and growing KV/state length:

- Prefill and decode may use different kernels, but rotary, norm, scale,
  sliding-window, and cache-index semantics must match exactly.
- For paged KV cache models, verify block table width, cache seqlens, and graph
  replay bucket behavior under long contexts and dynamic active counts.
- Decode graph capture must not invoke TorchDynamo compilation or RNG-state
  queries inside CUDA graph capture. Warm up compiled callables before capture.
- For serving workloads, validate cancellation and subsequent requests do not
  reuse stale cache state. A cancelled or failed request must not make the next
  request resume from old `step/cache_tokens`.
- When adding dynamic batching or request admission, avoid reallocating KV
  caches per batch. Reuse allocated cache capacity or explicitly release old
  cache state before resizing.
- If active counts fall outside captured graph buckets, pad to a supported
  bucket rather than failing or skipping the graph, unless correctness requires
  eager fallback.

## Checkpoint Conventions

`checkpoint.py` should be declarative and consistent:

- Use shared `areno.engine.checkpoints` primitives such as safetensors indexes,
  tensor stores, column/row sharding helpers, and HF shard writers.
- Encode model-specific names in small mapping helpers or specs.
- Keep load and save inverse to each other.
- Save HF-compatible safetensors when the adapter exposes `save_weights`.
- Preserve source config/tokenizer files when saving if the core helper
  supports it.
- If an adapter intentionally loads only a checkpoint subset, set checkpoint
  progress totals to the expected loaded key set. A progress bar ending below
  100% should be treated as a bug or a misleading total.

For tensor-parallel weights:

- Column-parallel tensors are loaded from the rank-local column shard and saved
  by gathering/merging the full column tensor.
- Row-parallel tensors are loaded from the rank-local row shard and saved by
  gathering/merging the full row tensor.
- Replicated tensors are copied identically on each rank and saved once.
- Special fused or merged projections must document the source HF names and the
  split/merge order.
- If a save path materializes lazy gathered tensors and then consumes them to
  build another tensor, synchronize pending CPU copies before the second
  operation. Otherwise reload can produce coherent shapes with zero or stale
  values, usually showing up as gibberish after save/reload.

Checkpoint round-trip rules:

- Load progress should reach 100% for the keys the adapter claims to load.
  If only a subset is intentionally loaded, the progress total must reflect
  exactly that subset.
- `load -> save -> load` without training should preserve logits closely. If
  it does not, fix checkpoint layout before debugging optimizer or reward code.
- After every new or changed `save_weights` implementation, compare the saved
  checkpoint against the source checkpoint with the skill-local diff script:

  ```bash
  python skills/areno-model-adaptation/scripts/compare_ckpt_diff.py \
    /path/to/source_hf_ckpt \
    /path/to/saved_ckpt \
    --device cuda \
    --top-k 50
  ```

  Use `--pattern '<fnmatch>'` to narrow the check to suspected projections,
  norms, experts, or language-model-only tensors. Treat unexpected
  `missing_in_other`, `extra_in_other`, `shape_mismatch`, zero-like tensors,
  or large same-name numeric diffs as save bugs until the tensor layout is
  explained. Continue save correction from the largest-diff keys rather than
  guessing from sampled completions alone.
- After a trained save, compare the saved checkpoint to the source checkpoint
  on representative tensors. Small training deltas are expected; whole tensors
  with zero std, transposed shape-compatible values, or missing projection
  blocks are save bugs.
- Preserve non-weight assets from the source checkpoint. Tokenizer and config
  drift can look like model corruption.
- For multimodal checkpoints where areno uses only the language model,
  document the intentionally omitted vision/projector tensors and ensure the
  saved config still reloads through the same text adapter path.

Model-specific checkpoint notes from current plugins:

- Llama-style dense adapters currently build `Qwen3ForCausalLM`, so they inherit
  Qwen3's training layer loop, sequence-parallel handling, and activation
  recompute support. Do not add a duplicate Llama forward unless the
  architecture genuinely diverges.
- Gemma-family adapters may need double-wide MLP variants and per-layer input
  projections; verify row/column split direction against HF checkpoint shapes.
- MiniCPM-like recurrent/linear-attention models often carry convolution, gate,
  decay, and norm parameters whose dtypes differ from standard dense layers.
  Kernel wrappers should accept the checkpoint dtype or explicitly convert once
  during load.
- MoE adapters must keep routing, token counting, grouped linear, and expert
  permutation paths free of CPU synchronization in hot training loops.
- Qwen3-MoE uses ARENO `topk_softmax` for softmax routing plus ARENO MoE
  permute/grouped-linear/unpermute kernels; do not route it through SGL kernels
  or PyTorch fallback paths.

High-risk checkpoint areas:

- Interleaved q/gate projections, especially per-head interleaving.
- Grouped-query attention where q heads and kv heads shard differently.
- Norm weights when the runtime stores `1 + weight` but HF stores residual
  scale weights.
- Row-parallel output projections and vocab/lm-head tying.
- Linear-attention state tensors such as convolution weights, decay logs,
  dt bias, and gate projections.

## Runtime Lifecycle

Implement lifecycle hooks even when they are no-ops. The trainer and backend may
switch roles between rollout, logprob scoring, reward/value scoring, and train.

Guidelines:

- `prepare_infer_weights` should allocate or transform inference-only state.
- `clear_infer_weights` should release inference-only state.
- `onload_train_weights` and `offload_train_weights` should be correct for
  multi-role training without aliasing model weights across actor/ref/critic or
  reward roles.
- Cache setup must be deterministic and rank-local.
- Do not add public offload/onload APIs to the trainer just to make a model
  work; route through existing backend/model lifecycle hooks.

Role and lifecycle correctness:

- Actor, reference, critic, and reward models must not alias trainable storage
  unless the backend explicitly owns that sharing. Updating critic weights must
  not mutate actor or reference weights.
- Reference and reward roles should stay inference-only unless explicitly
  trained by the algorithm.
- Clear inference caches before switching back to train if the model keeps
  cache tensors on modules.
- Any derived inference weights must be rebuilt after train weights change.

## Training Algorithm Validation

Validate model adaptation against the algorithms currently exposed by
areno:

- `sft`: offline teacher-forced next-token loss. Dataset rows may be
  `messages`, `prompt/response`, `question/answer`, `instruction/output`, or
  plain `text`. This path should not require reward functions or rollout-only
  config.
- `dpo`: offline preference optimization. Rows should contain
  `prompt + chosen/rejected` or full chosen/rejected conversations. The trainer
  uses a frozen reference role and requires chosen/rejected rows to stay
  adjacent inside microbatches.
- `gspo`/`grpo`: online rollout plus Python reward function. Validate reward
  grouping, response masks, rollout logprobs, and train logprobs.
- `ppo`: online rollout with optional ref/reward/critic roles. Validate role
  lifecycle, non-aliasing, value training, and reference KL.

Do not pass reward-specific config into SFT or DPO dataclasses. Keep config
types narrow: offline trainers should not own rollout sampling fields unless
they actually use them.

For all algorithms:

- Check first-step metrics before running long jobs. For DPO, an initial loss
  near `log(2)` is expected when actor and reference start equal; exploding
  loss or persistently negative margins can indicate too large LR, too long
  responses, or reversed preference fields.
- Keep default full-parameter LR conservative. Current trainer defaults are
  `lr=1e-6` and `min_lr=1e-7`; override explicitly for LoRA or known-safe
  recipes.
- Long responses make sequence-summed DPO margins large. Reduce
  `max_new_tokens` or `dpo_beta` before assuming model-code corruption.
- Verify metrics use response-only masks; prompt and padding positions should
  not contribute to SFT/DPO/RL losses.

## Remote Validation Loop

When the user asks to use a remote GPU environment:

1. Edit code locally.
2. Commit and push the model plugin repo.
3. Pull the same commit on the remote checkout.
4. Run the exact user-requested dataset, reward, model path, and algorithm.
5. Monitor with the user-requested interaction method. Use only the access
   path, sessions, panes, hosts, or terminals the user authorized.
6. If the job stalls, inspect workers with `py-spy dump -p <pid>` before
   changing code.
7. Iterate until checkpoint load, rollout, train, save, reload, and two train
   steps complete.

Keep remote runs reproducible:

- Print the exact git commit used by `areno.models` before each validation
  run.
- Print the exact git commit used by the outer areno repo when trainer,
  backend, or serving code changed.
- Keep one short command or script per run so the test can be repeated after a
  push/pull.
- Log completions during bring-up. Disable completion logging only after the
  model is known to generate coherent text.
- When a run is interrupted after saving, immediately reload the saved
  checkpoint and run rollout before deleting logs.

Useful remote checks:

```bash
nvidia-smi
ps -o pid,ppid,stat,etime,cmd -p <pid-list>
py-spy dump -p <worker-pid>
git -C /path/to/areno.models rev-parse --short HEAD
```

## Validation Signals

Minimum acceptance for a new plugin:

- `config_from_hf` identifies the model and builds the right adapter.
- `load_weights` completes on the requested tensor-parallel size.
- A base-checkpoint rollout produces coherent text before any training. Print
  sample completions during bring-up and treat multilingual word salad,
  repeated filler tokens, or template fragments as a correctness bug.
- Rollout reaches decode progress and completes without cache shape errors.
- Training runs backward and optimizer step without illegal memory access.
- `save_weights` writes a reloadable HF-style checkpoint.
- A saved checkpoint reloads and produces coherent completions. Compare key
  tensors against the source checkpoint when reload is gibberish; shape matches
  are not enough, and full-attention q/gate projections are a common failure
  point.
- The requested algorithm runs for at least two steps.

If base rollout is gibberish, debug model adaptation before tuning RL
hyperparameters. Check prompt/chat-template encoding first, then inspect
`lm_head`/embedding tying, EOS ids, checkpoint progress totals, packed
projection layouts, rotary layout, and cache-state initialization. For
Qwen3.5/MiniCPM-style attention gates, verify whether q and gate rows are
interleaved per head rather than stored as one contiguous q block followed by
one contiguous gate block; load and save must be inverse to the HF layout.

Performance sanity:

- Decode should use the intended fused kernel path.
- Training should not fall back to slow Python reference kernels unless the user
  explicitly accepts it.
- A faster fused op is not acceptable in train if it drops gradients. Check that
  gate/up projection weights receive nonzero gradients after backward.
- Graph breaks should be understood. Fix them when they cause capture failures
  or material slowdowns, but do not remove required autotune/fused kernels just
  to make logs quieter.
- Packed or prefill paths that need Python sequence-boundary loops should be
  isolated from Dynamo tracing instead of emitting repeated `Tensor.item()`
  graph-break warnings from model code.

Numerical sanity:

- Reward can be noisy step to step, but a coherent base model should not turn
  into repeated boilerplate, template fragments, or very low-entropy output
  after reload.
- Rollout logprob should not monotonically collapse toward zero unless the
  model is becoming nearly deterministic for a real reason. Inspect sampled
  completions and EOS behavior when this happens.
- A large train/rollout logprob absolute diff usually means train and inference
  kernels disagree on scale, masking, normalization, windowing, or packed
  sequence boundaries.
- A zero reward with coherent text can be reward/data mismatch; a zero reward
  with gibberish is model/checkpoint correctness.
- The first fused-kernel/autotune step may be slow. Later steps should be
  checked separately before replacing kernels.

## Coding Standards

- Keep changes scoped to `areno.models` unless core support is truly missing.
- Do not break existing `grpo` or `gspo` behavior while adding a model.
- Follow existing plugin style and naming.
- Use explicit shape variables and assertions for non-obvious tensor layouts.
- Add short comments only for layout, kernel, or checkpoint details that are
  easy to get wrong.
- Prefer structured checkpoint helpers over ad hoc string manipulation.
- Keep imports lazy for optional heavy kernel dependencies when possible.
- Do not leave temporary debug code, local paths, or one-off dataset paths in
  committed model code.
