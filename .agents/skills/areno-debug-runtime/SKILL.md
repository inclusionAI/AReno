---
name: areno-debug-runtime
description: Diagnose failed, hung, slow, OOM, NaN, illegal-memory-access, NCCL, compilation, rollout, or training runs in AReno. Use when runtime evidence must identify the first causal stage. Do not use for routine capacity planning without a failure.
---

# Debug AReno Runtime

Do not modify parameters or code until the lifecycle stage and first causal error are identified.

When diagnosis requires a code fix, implement and commit it locally, then pull
the branch on the remote reproduction host. Do not hot-patch the remote source.

## Primitives

```bash
python .agents/skills/areno-debug-runtime/scripts/summarize_traceback.py run.log
python .agents/skills/areno-debug-runtime/scripts/process_snapshot.py --pid <pid>
python .agents/skills/areno-debug-runtime/scripts/inspect_core.py <core-file> --executable <python>
```

Use `py-spy dump -p <pid>` for a Python-side stall. Use Nsight Systems only after a bounded steady-state workload exists. Read [references/error-taxonomy.md](references/error-taxonomy.md) and [references/nan-triage.md](references/nan-triage.md) as applicable.

## Workflow

1. Preserve exact command, commit, config, earliest logs, and worker exit data.
2. Classify config/load, prefill, decode, reward, scoring, train forward, backward, optimizer, save, or distributed teardown.
3. For multi-rank output, group signatures and prioritize the earliest distinct exception over secondary NCCL watchdog failures.
4. Distinguish compilation/autotune work from deadlock using elapsed time and stacks.
5. Reproduce with the same semantic workload at the smallest topology that still fails.
6. Correct the owning layer. Required kernels must fail loudly; do not introduce silent fallback.
7. Re-run the reproduction and the original bounded path.

Report evidence, root cause, changed ownership boundary, and verification. A process exit without the first error is incomplete diagnosis.
