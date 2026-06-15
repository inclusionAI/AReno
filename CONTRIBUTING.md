# How to contribute to AReno

Everyone is welcome to contribute, and we value every contribution. Writing code
is not the only way to help: answering questions, helping others, reporting bugs,
and improving the documentation are all immensely valuable.

It also helps if you spread the word — reference AReno in blog posts about the
projects it made possible, shout it out when it helps you, or simply ⭐️ the
[repository](https://github.com/inclusionAI/asystem-areno) to say thank you.

## AI usage policy

We encourage using AI tools to help with contributions — they are a great way to
write better code, catch issues early, and navigate the codebase. The repository
ships an [`AGENTS.md`](AGENTS.md) (and a `CLAUDE.md` pointing to it) describing how
agents should work here; read it before starting an AI-assisted change.

That said, **we will not review fully AI-generated PRs from first-time
contributors.** Reviewing agent-generated code is especially costly when the
contributor cannot engage in the discussion or vouch for the correctness of the
change. Every PR we spend time on should reflect a genuine understanding of what
is being proposed — a human submitter must understand and defend each line.

## Ways to contribute

There are several ways to contribute to AReno:

- Fix outstanding issues with the existing code.
- Submit issues for bugs or desired new features.
- Implement a new post-training algorithm, model adapter, or hardware backend.
- Contribute to the examples or the documentation.

> All contributions are equally valuable to the community. 🥰

AReno builds a CUDA extension, so contributing requires an existing
**Linux + NVIDIA GPU + CUDA + PyTorch ≥ 2.6** environment. See
[README.md](README.md) for full installation notes.

## Submitting a bug-related issue or feature request

### Did you find a bug?

Before reporting, please **make sure the bug was not already reported** (search
the issue tracker). Your issue should be about a bug in AReno itself, not your own
code. Include:

- Your **OS**, **Python**, **CUDA**, **PyTorch**, and **AReno** versions, plus GPU model.
- A short, self-contained snippet that reproduces the bug.
- The *full* traceback if an exception is raised.
- The exact command you ran (`areno train ...` / `areno serve ...`) and the relevant flags.
- Any other context (logs, screenshots) you think may help.

### Do you want a new feature?

Open an issue and describe:

1. The *motivation* — what problem does it solve, or what need does it address?
2. The feature in as much detail as possible, ideally with a code snippet showing how it would be used.
3. A link to the paper, if the feature comes from one.

## Do you want to implement a new algorithm?

New post-training methods appear frequently. Good candidates for AReno satisfy at
least one of:

- **Simplicity** — comparable results to prior methods with less complexity.
- **Efficiency** — a meaningful improvement in single-node training throughput or memory.

Methods that add significant complexity or compute for only incremental gains are
unlikely to be merged into the stable surface.

AReno is built to be extended **without forking the core**. Before writing code,
open an issue with a short description of the method, a link to the paper, and a
link to a reference implementation if one exists. Then:

- **Algorithms** are registered, not branched. Add an `AlgorithmSpec` via
  `register_algorithm(...)` in `areno/api/algorithms.py`, with a loss function in
  `areno/api/loss_fns/`. New or unstable algorithms should land in
  `areno/experimental/` first and graduate to `areno/api/` once they have proven
  out — this keeps the stable API surface protected while lowering the bar to
  experiment. See the registered `gspo` / `grpo` / `ppo` specs for the pattern.
- **Model families** are adapters under `areno/models/<family>/`, registered
  through `areno/models/registry.py` — no core changes needed.
- **Reward functions** are plain Python files exposing
  `reward_fn(example, completions) -> list[float]`, injected via `--reward-fn-path`
  (see `examples/math/math_verify_reward.py`).

## Do you want to add documentation?

We're always looking for improvements that make the documentation clearer and more
accurate — typos, dead links, and missing or confusing content. Let us know, or
open a PR directly.

## Submitting a pull request (PR)

Before writing code, search existing PRs and issues to make sure nobody is already
working on the same thing. If unsure, open an issue first to get feedback.

Follow these steps:

1. Fork the [repository](https://github.com/inclusionAI/asystem-areno) and clone
   your fork, adding the base repo as a remote:

   ```bash
   git clone git@github.com:<your-handle>/asystem-areno.git
   cd asystem-areno
   git remote add upstream https://github.com/inclusionAI/asystem-areno.git
   # Pull from upstream, but never push to it — push only to your fork (origin)
   git remote set-url --push upstream no-pushing
   ```

2. Create a branch for your changes — **do not** work on `main`:

   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git checkout -b a-descriptive-name-for-my-changes
   ```

3. Set up a development install (in a fresh conda or virtual environment with CUDA
   + PyTorch already present):

   ```bash
   pip install psutil
   pip install -e . --no-build-isolation
   pip install flash-attn flash-linear-attention pytest
   ```

4. Develop on your branch. Keep changes surgical — touch only what the task
   requires and match the surrounding style.

5. Make sure the tests pass. The CPU suite runs without a GPU and is the fast
   feedback loop:

   ```bash
   pytest tests/ -k cpu
   ```

   Add CPU tests under `tests/` for new algorithm, loss, or config behavior. Tests
   that require a GPU should be skipped cleanly when no GPU is available.

6. Commit with a clear message, keep your branch synced, and push:

   ```bash
   git fetch upstream
   git rebase upstream/main
   git push -u origin a-descriptive-name-for-my-changes
   ```

7. Open the pull request from your fork. It's fine if maintainers request changes —
   it happens to everyone. Push updates to your branch and they'll appear in the PR.

### Checklist

1. The PR title summarizes its contribution.
2. If the PR addresses an issue, mention the issue number in the description so the two are linked.
3. Prefix the title with `[WIP]` or mark the PR as a draft if it's a work in progress.
4. Existing tests pass.
5. New behavior is covered by tests. No testing = no merge.
6. The description notes the test commands you ran and any hardware limitations.

### Default values guidelines

1. **Use defaults when appropriate.** Provide defaults unless a value varies
   significantly by use case — models and datasets should not have defaults, but
   things like `learning_rate` should.
2. **Prioritize proven defaults.** Align defaults with the original paper or
   method; alternatives need strong evidence.
3. **Ensure safety and predictability.** Avoid defaults that lead to surprising
   memory use or poor behavior in edge cases.
4. **Balance consistency and flexibility.** Keep defaults consistent across
   similar functions, but not at the expense of points 2 and 3.
5. **Opt-in for new features.** Do not enable new features (e.g. a novel loss) by
   default; users should explicitly opt in.

### Deprecation and backward compatibility

Public API (`areno/api/`), exported symbols, and CLI options are what users build
on. When changing them, prefer additive changes: add fields with defaults,
deprecate before removing, and avoid silent type changes. When deprecating, emit a
`FutureWarning` with migration guidance and a target removal version:

```python
warnings.warn(
    "`Trainer.foo` is deprecated and will be removed in 0.x.0. Use `Trainer.bar` instead.",
    FutureWarning,
    stacklevel=2,
)
```

Experimental code under `areno/experimental/` carries no backward-compatibility
guarantee and may change between releases.
