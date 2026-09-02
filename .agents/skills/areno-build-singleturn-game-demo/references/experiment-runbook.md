# RLVR experiment runbook

Read this reference before data inspection, model inference, training,
evaluation, or checkpoint handling.

## Workflow defaults

- Do not hardcode a model family or checkpoint in the skill. Use the checkpoint
  explicitly requested by the invoking user. If none is specified, select a
  currently supported AReno checkpoint that fits the available compute and task,
  then record the exact choice and rationale before baseline evaluation.
- Training paradigm: RLVR only. Never replace it with SFT and never fabricate an
  agentic loop. Select the current supported single-turn rollout RL algorithm
  from checked-out AReno API and document the choice.
- Optimizer: always use AReno's current Adam4bit option (expected CLI spelling
  `--adam-4bit`; verify it from current help before use).
- Never reinstall AReno. Use the installed package, environment, and repository
  mechanisms already available.
- Never use eager decode. Preserve the normal CUDA-graph decode path and fix or
  tune the actual issue instead of disabling graphs.
- Save experiment checkpoints and logs below a task-specific directory under
  `/new/`, which is a mounted path. Never place them in the repository.
- Retain only the latest useful checkpoint during training. Prefer the current
  supported keep-latest option; otherwise delete only verified old checkpoints
  inside the task-specific checkpoint directory immediately before a new save.

## Environment and data gate

Before building the training command:

1. Record commit, branch, `areno env --json`, `areno check`, GPU topology and
   memory, installed AReno and model paths, and model hub selection.
2. Read current `areno train --help` and repository training or capacity skills.
3. Inspect raw and normalized train, validation, and test samples using repository
   dataset inspection tools. Do not train until inspection reports success and
   confirms one prompt or messages input per row with no oracle leakage.
4. Run generator self-checks and compare canonical hashes across splits.

Use ModelScope for remote AReno assets when repository policy requires it. Do
not silently switch hubs.

## Baseline protocol

Evaluate the untouched base checkpoint on fixed held-out validation and test
data. Record:

- mean reward;
- task success or accuracy;
- legal or parseable output rate;
- results by difficulty bucket;
- dataset size and split seed;
- model or checkpoint, commit, temperature, max tokens, and all inference settings;
- random-policy lower bound and oracle upper bound when applicable.

Use at least two formal evaluation seeds. Save machine-readable output outside
the repository and a compact checked-in result summary without private data.

## Training and capacity loop

Start with a small real smoke workload that exercises rollout, reward, backward,
optimizer step, and checkpoint save. Then run enough real RLVR steps to observe
a learning curve.

Capacity tuning order:

1. If rollout OOMs, reduce `max_running_prompts` before changing semantic context
   or generation length.
2. If training OOMs, reduce `mini_bs` before semantic token limits.
3. Respect `batch_size * n_samples` total demand and keep concurrency separate.
4. Do not use eager decode as an OOM workaround.

When reward grows slowly, diagnose prompt clarity, parsing, reward density,
difficulty mix, sampling, batch or group size, and training budget. Consider
learning rate `1e-5` with minimum learning rate `1e-6` after evidence indicates
the original schedule is too weak. Do not tune on held-out test data.

Evaluate intermediate checkpoints under the exact baseline protocol. Stop
training only after:

- the acceptance improvement is met; and
- held-out mean reward improves by less than `0.02` across two consecutive
  scheduled evaluations, or another predeclared statistically comparable
  plateau rule.

Do not stop merely because training reward is high. Do not continue indefinitely
after held-out reward has clearly plateaued.

## Post-training comparison

Use the same held-out rows, parser, reward, temperature, decoding settings, and
difficulty buckets as baseline. Success requires:

- at least `0.15` absolute mean-reward gain or `30%` relative gain;
- clear core correctness gain;
- consistent direction on at least two eval seeds;
- evidence the gain is not only formatting compliance.

Report every formal run, including failed configurations. Check for answer
leakage, position bias, generator shortcuts, memorized seed mappings, duplicated
states, and format-only learning.

If the target remains unmet after reasonable prompt, reward, curriculum, and
optimizer iteration, replace the game with a more trainable researched candidate
rather than misrepresenting the result.
