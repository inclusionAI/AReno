---
name: areno-build-singleturn-game-demo
description: Autonomously research, implement, train, evaluate, visualize, and submit a genuinely new AReno RLVR mini-game whose mechanically generated samples are independent state-to-single-output tasks. Use when the user requests a complete new game demo with real reward improvement, not for multi-turn agent workflows, SFT-only examples, or design-only proposals.
---

# Build an AReno Single-Turn Game Demo

Deliver a working, reproducible AReno mini-game rather than a proposal. Continue
through research, implementation, data generation, baseline evaluation, RLVR
training, held-out evaluation, WebUI delivery, and PR submission until the
evidence meets the requested acceptance criteria or a genuine external blocker
remains.

## Non-negotiable task shape

Mechanically generable sequential games must be transformed into independent
single-turn examples:

```text
complete state -> exactly one generation -> deterministic reward
```

- Do not use a multi-turn conversation, agent loop, tool-feedback loop, or a
  sequence of assistant/tool messages for training or evaluation.
- Generate intermediate states directly when the source game is sequential.
- Each prompt contains the complete public state and output contract, but no
  oracle answer, private reward field, or reversible instance identifier.
- Each reward call scores only that prompt and its one completion.
- A WebUI may show successive states for human play; this does not change the
  independent single-turn training contract.

## Before acting

1. Read repository `AGENTS.md`, `CODEMAP.md`, current CLI help, and the
   repository-local training, serving, correctness, and capacity skills that
   match the requested work.
2. Inspect current `examples/agentic/`, `examples/sft/`, `examples/vl/`, and
   any other example directories completely enough to inventory every demo's
   game/task, state representation, output, reward target, and training paradigm.
3. Read [research-and-selection.md](references/research-and-selection.md) before
   choosing a game.
4. Read [implementation-contract.md](references/implementation-contract.md)
   before writing source or data.
5. Read [experiment-runbook.md](references/experiment-runbook.md) before running
   dataset inspection, baseline, training, evaluation, or checkpoint commands.
6. Read [webui-and-reporting.md](references/webui-and-reporting.md) before
   implementing the UI, writing README results, or opening the PR.

Use latest `origin/main` in a dedicated branch/worktree and preserve unrelated
user changes. Verify every interface against checked-out source rather than
memory. Do not modify public config or CLI surfaces unless the request explicitly
requires it and repository policy permits it.

## Authorization boundary

This skill does not itself authorize paid compute, remote mutations, public
services, pushes, or PRs. Treat an invocation that explicitly requests autonomous
training, serving, pushing, and PR creation as authorization for those in-scope
actions. Otherwise obtain only the missing authorization when it becomes
necessary. Never ask the user to perform routine steps the agent can safely do.

## Required execution loop

1. Inventory existing demos and record overlap risks.
2. Research at least five viable games using accessible Chinese sources and
   cross-check selected rules with two independent sources.
3. Select a game only if it has a deterministic oracle, scalable mechanical
   generation, strong random-versus-reasoning separation, and a clear future UI.
4. Implement the decoupled game/oracle, public state view, generator, loader,
   prompt contract, parser, reward, evaluation, and focused tests.
5. Generate leak-free train/validation/test data and run legality/oracle self-checks.
6. Inspect normalized data before training. Run a bounded smoke workload.
7. Evaluate the unmodified base checkpoint on fixed held-out splits.
8. Run real single-turn RLVR training. Diagnose and iterate when learning is weak;
   do not change the held-out set or hide failed runs.
9. Evaluate checkpoints under matched conditions and stop only after the target
   improvement is achieved and reward reaches a measured plateau.
10. Implement and probe a polished playable WebUI, then start requested model and
    UI services if authorized.
11. Complete documentation, remove generated artifacts from version control,
    review the diff, commit, push, and open a focused PR.

## Completion gate

Do not claim success unless actual logs and held-out evaluation establish all of
the following:

- mean reward improves by at least `0.15` absolute or `30%` relative;
- core success/accuracy clearly improves, not merely formatting validity;
- at least two evaluation seeds agree in direction;
- train/validation/test remain isolated and baseline/post-training conditions match;
- focused tests, smoke validation, real training, checkpoint save/reload as
  applicable, serving probe, and WebUI probe have actually run;
- README contains commands, all formal evaluation runs, failed experiments,
  known limitations, source links, demo-difference evidence, and WebUI design.

If available compute cannot meet the gate, report the exact blocker and evidence;
never fabricate a curve, checkpoint, test result, or service status.
