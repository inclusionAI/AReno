---
name: areno-debug-runtime
description: Diagnose failed, hung, slow, OOM, NaN, illegal-memory-access, NCCL, compilation, rollout, or training runs in AReno. Use when runtime evidence must identify the first causal stage. Do not use for routine capacity planning without a failure.
---

# Debug AReno Runtime

Do not modify parameters or code until the lifecycle stage and first causal error are identified.

When diagnosis requires a code fix, implement and commit it locally, then pull
the branch on the remote reproduction host. Do not hot-patch the remote source.

## Collect failure evidence

The primary workflow produces a self-contained diagnostic bundle (JSON + Markdown)
via `areno.cli.debug`.  Use this first, before any interactive diagnosis:

```bash
python .agents/skills/areno-debug-runtime/scripts/collect_evidence.py [--output-dir <dir>] [--traceback-file <path>] [<command...>]
```

Options:
- `--output-dir` — where to write the bundle (default `./areno-debug`).
- `--traceback-file` — post-mortem traceback from a file.
- `--no-gpu` / `--no-env` — skip GPU or environment collection.
- `--no-redact` — disable sensitive-value redaction.
- `--json` — output the JSON bundle instead of Markdown.

Minimal examples:

```bash
# Collect with the current environment (no error context)
python .agents/skills/areno-debug-runtime/scripts/collect_evidence.py

# Post-mortem from a saved traceback file
python .agents/skills/areno-debug-runtime/scripts/collect_evidence.py --traceback-file crash.log

# Collect with command context and JSON output
python .agents/skills/areno-debug-runtime/scripts/collect_evidence.py --output-dir ./evidence/ --json -- areno train --ckpt ./model --algo gspo

# Skip GPU collection on non-CUDA hosts
python .agents/skills/areno-debug-runtime/scripts/collect_evidence.py --no-gpu
```

## Primitives

```bash
python .agents/skills/areno-debug-runtime/scripts/summarize_traceback.py run.log
python .agents/skills/areno-debug-runtime/scripts/process_snapshot.py --pid <pid>
python .agents/skills/areno-debug-runtime/scripts/inspect_core.py <core-file> --executable <python>
```

Use `py-spy dump -p <pid>` for a Python-side stall. Use Nsight Systems only after a bounded steady-state workload exists. Read [references/error-taxonomy.md](references/error-taxonomy.md) and [references/nan-triage.md](references/nan-triage.md) as applicable.

## Workflow

1. Preserve exact command, commit, config, earliest logs, and worker exit data.
2. Run `collect_evidence.py` to produce a structured bundle. Collection failures must not hide the original error.
3. Classify config/load, prefill, decode, reward, scoring, train forward, backward, optimizer, save, or distributed teardown.
4. For multi-rank output, group signatures and prioritize the earliest distinct exception over secondary NCCL watchdog failures.
5. Distinguish compilation/autotune work from deadlock using elapsed time and stacks.
6. Reproduce with the same semantic workload at the smallest topology that still fails.
7. Correct the owning layer. Required kernels must fail loudly; do not introduce silent fallback.
8. Re-run the reproduction and the original bounded path.

Report evidence, root cause, changed ownership boundary, and verification. A process exit without the first error is incomplete diagnosis.