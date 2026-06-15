<!-- Go-to brief for AI coding agents working on AReno. -->

# AGENTS.md -- AReno Agent Operations Guide

## Quick reference

**Tech stack**: Python 3.10+ | CUDA | PyTorch ≥ 2.6 | FlashAttention | flash-linear-attention | Transformers | safetensors

```bash
# Install (requires an existing Linux + NVIDIA GPU + CUDA + PyTorch >= 2.6 env)
pip install psutil                         # required with --no-build-isolation
pip install -e . --no-build-isolation         # --no-build-isolation: build against your installed torch
pip install flash-attn flash-linear-attention
ARENO_BUILD_EXT=0 pip install -e . --no-build-isolation   # skip CUDA build (metadata-only / dry run)

# Train -- swap algorithm with --algo: sft | dpo | gspo | grpo | ppo
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --reward-fn-path examples/math/math_verify_reward.py --algo gspo --tp-size 4

# Serve an OpenAI-compatible endpoint
areno serve --model-path /path/to/model --tp-size 1 --world-size 1 --port 8000

# Tests -- CPU suite runs without a GPU; the fast feedback loop
pytest tests/ -k cpu

# Check GPU before training/serving
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

**Hard rules** -- never violate:

- No wildcard imports (`from x import *`).
- No hardcoded secrets, absolute paths, or internal endpoints.
- No reintroducing TransformerEngine, SGLang kernels, or FLA as runtime dependencies from model plugins -- third-party code is a reference for tensor semantics only.
- Full training/serving needs CUDA hardware; integration paths require a GPU -- explain skips explicitly.
- Never fabricate symbols, error messages, API responses, or stack traces. If you did not read or run it, say so.
- Do not claim tests or builds pass unless you actually ran the command in this session.

**Always do**:

- Read relevant files before modifying code; follow existing patterns in the same module.
- Before using a symbol (function, class, type, constant), confirm it exists -- read its definition or `grep -r "name" areno/`; check `pyproject.toml` for a dependency. If you skip this, prefix the code with `# UNVERIFIED:`.
- Use `from areno.api import Trainer` -- the public SDK surface. `from areno import Trainer` does **not** work.
- Add a CPU test under `tests/` for new algorithm / loss / config behavior.
- Register new capabilities (algorithms, model adapters) instead of editing a factory.
- Ask for decisions with short, structured options rather than broad open-ended questions.

**Ask first** before:

- Modifying config dataclasses in `areno/api/trainer_config.py` or `areno/api/config.py`.
- Adding new dependencies (`pyproject.toml`).
- Altering CLI option surfaces in `areno/cli/train.py` / `serve.py`.
- Deleting or renaming public API in `areno/api/`.
- Running GPU training or serving.

When you cannot verify a claim, say "I haven't verified this" or "I don't know" -- both beat a confident guess. When unsure, leave a `TODO(agent)` comment and note the constraint in your response.

______________________________________________________________________

## Working principles

Behavioral guardrails (derived from Karpathy's notes on LLM coding pitfalls).
These bias toward caution over speed; use judgment on trivial tasks.

- **Think before coding.** Do not assume silently. State assumptions; if a
  request is ambiguous, present the interpretations instead of picking one. If a
  simpler approach exists, say so. If something is unclear, stop and ask.
- **Simplicity first.** Write the minimum code that solves the problem — nothing
  speculative. No abstractions for single-use code, no unrequested
  configurability, no error handling for impossible cases. areno favors readable,
  registry-driven extension over premature generality; if 200 lines could be 50,
  rewrite it.
- **Surgical changes.** Touch only what the task requires. Do not reformat or
  "improve" adjacent code, and match the surrounding style even if you'd write it
  differently. Remove only the imports/symbols *your* change orphaned; if you spot
  unrelated dead code, mention it rather than delete it.
- **Goal-driven execution.** Turn a task into a verifiable goal — e.g. "fix the
  bug" → "write a CPU test that reproduces it, then make it pass". For multi-step
  work, state a brief plan with a `verify:` check per step, then loop until the
  checks pass.

______________________________________________________________________

## Repository map

```
areno/                     Core Python package (layered cli -> api -> engine -> accel)
|-- cli/                   CLI entrypoints (train.py, serve.py, main.py)
|-- api/                   SDK: Trainer, algorithm registry, loss_fns/, trainers/, backend/
|-- engine/                Tensor-parallel workers, runtime, rollout, training, checkpoints
|-- accel/                 Fused CUDA kernels (csrc/*.cu + Python wrappers)
|-- models/                Per-family adapters (llama, qwen3, qwen3_5, bailing, gemma4, minicpmv46)
+-- experimental/          Incubation area for new algorithms

examples/                  Runnable reward functions and dataset loaders
skills/                    Claude Code skills
tests/                     CPU test suite (*_cpu.py)
docs/                      Architecture notes, CLI/SDK guides
```

______________________________________________________________________

## Code style & patterns

- **Composition over inheritance** -- keep hierarchies shallow; prefer delegation.

| Type             | Pattern         | Example                          |
| ---------------- | --------------- | -------------------------------- |
| Config dataclass | `XxxConfig`     | `TrainerConfig`, `PPOTrainerConfig` |
| Algorithm spec   | `AlgorithmSpec` | registered in `areno/api/algorithms.py` |
| Loss function    | `xxx_loss_fn`   | `gspo_loss_fn`, `grpo_loss_fn`   |
| Reward function  | `reward_fn`     | `examples/math/math_verify_reward.py` |
| Model adapter    | per family      | `areno/models/<family>/`         |

**Performance**: avoid GPU-CPU sync in hot paths (`.item()`, `.tolist()`, `print(tensor)`); batch ops over Python loops on tensor elements; be explicit about `dtype`/`device`.

**Typing & imports**: explicit type hints; no wildcard imports; keep heavy optional deps imported inside functions.

**Kernels**: when editing CUDA in `areno/accel/csrc/`, keep the `.cu` source and its Python wrapper in sync and rebuild with `pip install -e . --no-build-isolation`.

______________________________________________________________________

## Core concepts

**Trainer** (`areno/api/trainer.py`): the high-level entry. A typical RL loop is `init()` -> `rollout_batch()` -> score -> `train()` -> repeat -> `close()`. The CLI builds the same `Trainer` under the hood (`areno/cli/train.py`).

**Algorithm registry** (`areno/api/algorithms.py`): each algorithm is an `AlgorithmSpec` (trainer class, default loss, `requires_rollout`). `--algo` selects one; discover them with `from areno.api import list_algorithms`.

**Loss functions** (`areno/api/loss_fns/`): `sft`, `dpo`, `gspo`, `grpo`, `ppo`, passed into `Trainer.train`.

**Rewards**: plain Python files exposing `reward_fn(example, completions) -> list[float]`, injected via `--reward-fn-path`.

**Models** (`areno/models/`): per-family adapters with HF-compatible config and weights; each `<family>/` defines the model and checkpoint logic, registered through `areno/models/registry.py` (`register_adapter`, with `load_model_weights` / `save_model_weights` for weight I/O).

**Engine** (`areno/engine/`): tensor-parallel workers, rollout/train runtime, CUDA graphs, and checkpoint I/O behind the backend boundary.

______________________________________________________________________

## API & config rules

*Applies to: `areno/api/**`*

- Each algorithm gets the narrowest config dataclass it needs (offline trainers omit rollout/reward fields by construction).
- Public configs, exported symbols, and CLI options are public API -- add fields with defaults, deprecate before removing, avoid type changes.
- The SDK entry is `from areno.api import Trainer`; keep the top-level package free of kernel-heavy imports.

______________________________________________________________________

## Extension rules

*Applies to: `areno/api/algorithms.py`, `areno/models/`, `areno/experimental/`*

- **Algorithms register, not branch.** Add via `register_algorithm(AlgorithmSpec(...))`; do not edit a factory.
- New / unstable algorithms enter `areno/experimental/` first and graduate to `api/` once stable.
- A new model family is a directory under `areno/models/<family>/` registered through `areno/models/registry.py` -- no core changes needed.
- Runtime-critical paths use areno-owned code in `areno/engine` and `areno/accel`, not third-party runtime deps.
