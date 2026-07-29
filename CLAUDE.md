# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AReno is a self-contained, single-node LLM RL post-training toolkit (train, serve, agentic RL) built on PyTorch ≥ 2.6 + CUDA. Full setup, commands, conventions, and architectural detail live in one place — read it before working here:

@AGENTS.md

For task-to-file and call-path pointers, read the code navigation map:

@CODEMAP.md

## Big picture (before you touch files)

- **Layering (`cli → api → engine → accel`) only flows downward; never call up.** The SDK process (`areno.api.Trainer`) holds no weights — it dispatches through `Backend` (`areno/api/backend/base.py`) → `ArenoBackend` → `ArenoEngine` (`areno/engine/api.py`), which spawns one TP/DP worker process per rank (`ArenoWorker`, `areno/engine/worker.py`) where model weights and GPU compute actually live. IPC is the `TPCluster` `Command`/`Op` protocol in `areno/engine/protocol.py`.
- **Everything registers, nothing branches in a factory.** Algorithms (`register_algorithm(AlgorithmSpec(...))` in `areno/api/algorithms.py`), model families (`register_adapter` in `areno/models/registry.py`), and backends (`@register_backend` in `areno/api/backend/base.py`) are discovered by name — add a new one by registering, not by editing a dispatch site. Unstable algorithms go in `areno/experimental/` and graduate to `api/`.
- **Reward functions are `reward_fn(record: RewardRecord) -> float`** (`areno/api/rewards.py`), `reward_fn(record) -> float` — the `reward_fn(row, completions) -> list[float]` form in some older README/docs prose is stale. Verify the live contract before touching reward code.
- **Repository-local skills** under `.agents/skills/` encode repeatable workflows (run-training, run-serving, tune-capacity, validate-correctness, debug-runtime, profile-performance, build-agentic-workflow, model-adaptation, add-algorithm, develop-kernel). Load only the skill whose description matches the task; each points to its own scripts and references.

## Dev commands not covered in AGENTS.md

AGENTS.md covers install, train, serve, and the CPU test suite. Day-to-day
quality tooling (configured in `pyproject.toml` + `.pre-commit-config.yaml`):

```bash
ruff check .            # lint (E, F, W, I, UP; line-length 120, E501 ignored)
ruff format .           # format
pyright                 # type check, basic mode, includes areno/ (excludes accel/csrc)
pre-commit run -a       # the full pre-commit gate (ruff + format + whitespace + large-file + conventional-commits)
pytest tests/test_foo_cpu.py::test_bar   # run a single CPU test
areno check             # machine readiness: CUDA, nvcc, CUDA_HOME, optional deps, accel extension
areno env --json        # full environment report for bug reports
```

Commits must follow Conventional Commits
(`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`,
`chore`, `revert`); the `commit-msg` hook enforces this.