---
name: areno-model-adaptation
description: Add or debug an AReno model family, including config conversion, module construction, checkpoint load/save, text or multimodal inference, training backward, tensor parallelism, CUDA graph decode, and checkpoint round trips. Use only when model-specific implementation or compatibility changes are required.
---

# Adapt an AReno Model

Read `AGENTS.md`, `CODEMAP.md`, the nearest registered adapter, and the upstream reference implementation. External frameworks define semantics only unless AReno explicitly owns the dependency.

Develop in a dedicated local branch and update a remote GPU checkout only by
pulling that committed branch. Read
[references/remote-validation.md](references/remote-validation.md). Obtain model
assets from ModelScope as described in
[references/modelscope-assets.md](references/modelscope-assets.md).

## Inventory first

```bash
python .agents/skills/areno-model-adaptation/scripts/inspect_checkpoint.py <checkpoint> \
  [--pattern 'model.layers.*'] [--limit 200]
```

Record architecture/config fields, tokenizer or processor class, tensor names/shapes/dtypes, fused projection order, tied weights, and reference outputs. Never infer layout from class names.

## Gated phases

1. **Config and construction:** implement model-family matching and `ModelConfig` translation. Verify local TP shapes.
2. **Load:** implement checkpoint mapping and prove claimed key coverage. Read [references/checkpoint-contract.md](references/checkpoint-contract.md).
3. **Inference:** compare bounded reference logits or staged activations, then decode coherent text. Multimodal paths must verify processor order, vision tower, projector/merger, token replacement, and position IDs.
4. **Training:** run forward/backward and verify finite nonzero gradients. Then run an end-to-end training job for at least two consecutive successful steps with the adapted model. Verify finite losses, metrics, and gradients on both steps. For rollout algorithms, also compare rollout and fixed-token train logprobs.
5. **Save:** implement inverse mapping, save, reload, and compare assets/tensors using the scripts in this directory.
6. **Parallel/runtime:** validate requested TP, packed/sequence-parallel positions, lifecycle hooks, and CUDA graph decode.
7. **Performance:** optimize only after prior gates pass. Read [references/kernel-policy.md](references/kernel-policy.md).

If `areno/accel` changes, install remotely with `pip install -e . --no-deps --no-build-isolation`; otherwise pull the branch without reinstalling.

## Completion evidence

Provide config mapping, checkpoint coverage, reference comparison, coherent decode, evidence from at least two successful end-to-end training steps, save/reload comparison, requested topology, and CUDA graph evidence. A successful weight load or a single backward pass is not completion.
