# Train with the pi coding agent

Use the [pi coding agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
as AReno's rollout harness on software tasks from ModelScope BigCodeBench.
The generator prepares JSONL records; the dataset loader normalizes them; the
runner executes pi and grades its changes in the existing training environment.

```text
ModelScope → generate_dataset.py → JSONL → dataset_loader.py
                                              ↓
                                         run_agent.py
                                              ↓
pi CLI → example secondary proxy → AReno rollout proxy → current policy
                ↓
       exact per-turn trajectories + verifier reward → AReno trainer
```

All integration code lives in this example. Pi's read, bash, edit and write tools
run unchanged. The secondary proxy adapts pi's streaming Chat Completions requests
to AReno's non-streaming endpoint, preserving exact input tokens, generated
tokens, log-probabilities and routing replay IDs. It emits buffered SSE after
each completed model call, rather than incremental token streaming.

## Install

Use an existing AReno training environment with Python 3.10+ and Node.js 22+.
Run these commands from the repository root. Both `node` and `pi` must be on the
training process's `PATH`:

```bash
node --version
python -m pip install -r examples/agentic/pi/requirements.txt
npm install -g @mariozechner/pi-coding-agent@0.83.0
pi --version
```

Set `ARENO_PI_EXECUTABLE` if pi uses a different executable or wrapper. A custom
pi path does not provide Node.js; its runtime must still be available.

## Generate the dataset

The generator downloads [bigcode/bigcodebench from ModelScope](https://modelscope.cn/datasets/bigcode/bigcodebench)
at revision `a4da68573cf2ead10e049a580ba0016d9eb5f281`, split `v0.1.4`.
It scans all 1,140 upstream tasks, preserves their grading tests and selects tasks
for one shared environment:

| Profile | Selection | Candidates on Python 3.12 |
| --- | --- | ---: |
| `stdlib` (default) | Standard-library tasks after portability screening | 197 |
| `extended` | Standard library plus shared third-party dependencies | 865 |
| `smoke` | Ten reviewed file, CSV, ZIP, hashing and SQLite tasks | 10 |

These counts precede executable reference checks and depend on the environment.
The extended set includes the standard-library set. These are software utility
tasks with file/database side effects and edge cases, not whole-repository issues.

```bash
python examples/agentic/pi/generate_dataset.py \
  --output /tmp/pi-engineering.jsonl \
  --check-reference
```

`--check-reference` executes the upstream reference solution and unfinished stub
against the same tests in separate temporary workspaces. Bulk selection retains
only tasks whose reference passes and whose stub fails. Generated records contain
no reference solution. Without this flag, generation screens source and dependency
metadata without executing dataset code.

Use `--limit 50` for 50 accepted tasks, `--profile smoke` for the ten-task subset,
or repeat `--task-id BigCodeBench/ID` for explicit selection. Omit the limit to
process every candidate. Explicit selections and the smoke profile fail if any
requested task is rejected. Failed generation preserves an existing output file.

For the larger shared-dependency profile, install its packages once in the same
environment used by AReno and pi:

```bash
python -m pip install -r examples/agentic/pi/requirements-extended.txt
python examples/agentic/pi/generate_dataset.py \
  --profile extended \
  --output /tmp/pi-engineering.jsonl \
  --check-reference
```

The shared packages cover numerical, data, plotting and image-processing tasks.
There is no package installation during rollout. Grading uses Matplotlib's
noninteractive `Agg` backend.

Every generation writes `OUTPUT.report.json` (override with `--report`), recording
the pinned revision, selected IDs, exclusion reasons, reference-check status,
Python version and installed dependency versions. Source checksums are stored
with each task. Selection order is numeric task ID; the generator does not fall
back to Hugging Face. Portability screening excludes unsupported dependencies,
external process/network/GUI requirements and shared path literals; it is a
conservative heuristic, not a sandbox.

## Train

```bash
export MPLBACKEND=Agg
areno train \
  --ckpt /path/to/tool-capable-checkpoint \
  --dataset-path /tmp/pi-engineering.jsonl \
  --dataset-loader-fn examples/agentic/pi/dataset_loader.py \
  --agent-fn examples/agentic/pi/run_agent.py \
  --reward-fn-path examples/agentic/pi/reward.py \
  --algo grpo --world-size 1 --tp-size 1 \
  --batch-size 4 --mini-bs 4 --n-samples 4 \
  --max-running-prompts 4 \
  --max-new-tokens 4096 --max-context-len 32768 \
  --max-steps 20 --save-path /tmp/pi-output
```

Use `--algo gspo` for GSPO. The training session owns the current policy and
rollout endpoint; no separate serving process is needed. AReno's
`--max-new-tokens` controls generation length. Synchronous rollout batches finish
before weights change, and concurrency follows `--max-running-prompts`.

## Dataset contract and rewards

Each JSONL record contains:

- `prompt`: the task instruction.
- `files`: relative workspace paths mapped to initial text contents.
- `verify`: trusted Python verification code, kept outside the initial workspace.
- Optional `source`: dataset revision, task ID, checksum and required packages.
- Optional `max_turns`, `timeout` and `verify_timeout`: per-attempt budgets.

`dataset_loader.py` validates the record and also accepts `problem_statement`
as a prompt fallback. Other datasets can use the same runner by emitting this
contract.

Pi edits `solution.py` and can run its own tests. After it exits, the controller
executes the edited implementation and upstream grading tests in a separate
Python process. Reward is 1 only when pi finishes within budget and the nonempty
test suite passes without skipped tests; otherwise it is 0. All model turns of
an attempt share its task reward. Failed attempts with usable traces remain
negative training examples. Transport/metadata failures and attempts without
model calls are marked invalid and logged. Tool results remain prompt context,
not generated actions. `pi_result` retains process status and log tails.

Each attempt receives its own temporary workspace, pi configuration, secondary
proxy and bearer key. Sessions, extensions, skills, prompt templates, automatic
compaction and retries are disabled. Timeouts terminate process groups, including
tool children; proxy shutdown drains in-flight model requests.

Temporary workspaces are not an OS security sandbox. Pi and dataset verification
execute code with the current user's permissions. Use trusted tasks in a disposable
container or VM. This workflow runs in that existing environment and does not
build per-task images. Training on these tasks is not a held-out benchmark evaluation.

## CPU checks

```bash
python -m pytest -q examples/agentic/pi/test_generate_dataset_cpu.py \
  examples/agentic/pi/test_pi_cpu.py
```

The offline tests cover dataset selection, provenance, reference filtering,
verification, the secondary proxy, exact training tokens/masks, task rewards and
process cleanup. They use a fake policy and fixture process, without downloading
a model or running GPU training.
