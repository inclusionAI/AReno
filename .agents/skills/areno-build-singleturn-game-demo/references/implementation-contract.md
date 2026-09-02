# Implementation contract

Read this reference before implementing the game, dataset, reward, or tests.

## Repository-first design

Inspect current dataset loader and generator contracts, reward API, trainer, CLI,
evaluation path, and the two or three closest high-quality demos. Follow current
public API and local style. Do not copy an obsolete example or introduce an
agentic API merely because the source game is sequential.

Create a self-contained example in the most appropriate current `examples/`
location. File names may follow current conventions, but responsibilities must be
obvious. The demo normally needs equivalents of:

```text
README.md
dataset_generator.py
dataset_loader.py
game.py and/or solver.py
reward.py
single-turn inference entry
eval.py when common evaluation is insufficient
web_ui.py after training evidence is complete
```

## Game and public-view boundary

- Represent internal state as stable JSON-compatible data.
- Define a finite, structured, independently verifiable action or full solution.
- Decouple game rules, transition or oracle logic, reward, prompt rendering, and UI.
- Provide a pure function or equivalent method that converts internal state to a
  public view.
- Never expose oracle answers, reward-private labels, seeds, or hidden metadata in
  the public view.
- Do not make ANSI codes, terminal coordinates, rendered prompt text, or CLI output
  the canonical state.

## Dataset generator

The generator must:

- expose CLI parameters for seed, sample counts, difficulty range, and output path;
- use a local reproducible RNG rather than global randomness;
- create distinct train, validation, and test splits;
- use disjoint seeds or non-overlapping IDs and prevent duplicate or equivalent
  instances across splits;
- generate complete independent states, never conversation trajectories;
- store oracle answer, difficulty, and needed private metadata outside the visible
  prompt;
- support both fast smoke data and formal data;
- validate legality, solver correctness, output reproducibility, split isolation,
  and any canonical-equivalence rule before reporting success.

Do not commit formal generated datasets or checkpoints unless the repository
explicitly tracks small fixtures. Tiny deterministic test fixtures are acceptable.

## Prompt and output

Use a compact, self-contained prompt:

```text
Input: complete public game state
Output: exactly one constrained action or complete structured solution
```

- State the minimum rules needed to solve the instance.
- Require one generation and a minimal stable format.
- Do not request chain-of-thought.
- Do not include oracle values, hidden fields, or reversible ID-to-answer mappings.
- Do not use assistant or tool history, environment feedback, or future observations.
- Keep training and evaluation rendering identical unless a documented experiment
  deliberately compares them.

## Deterministic anti-exploit reward

The reward must parse safely and return a score rather than raising on malformed
output. Define signals for:

- fully correct or optimal output: maximum score;
- legal but suboptimal output where the game permits it;
- partially correct progress when a meaningful verifier-derived dense signal exists;
- illegal action;
- malformed output.

Highest reward must mean the task is genuinely solved. Defend against multiple
answers, prompt copying, extra prose, overly long output, duplicate actions,
injection-like suffixes, NaN or overflow values, and any alternative syntax that
could bypass validation. Do not call another LLM and do not use private labels to
grant unearned reward.

Add focused CPU-safe tests for:

- game transitions and edge states;
- solver or oracle correctness and determinism;
- generator repeatability and split leakage detection;
- loader normalization and prompt privacy;
- correct, wrong, partial, illegal, malformed, oversized, multi-answer, and
  adversarial completions;
- public-view privacy and JSON serialization;
- evaluation aggregation and difficulty buckets.
