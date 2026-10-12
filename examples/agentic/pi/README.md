# Train with the pi coding agent

Use the [pi coding agent](https://pi.dev) as AReno's rollout harness for repository
repair tasks from [ModelScope SWE-smith](https://modelscope.cn/datasets/SWE-bench/SWE-smith).
Pi inspects a broken sqlparse repository, edits package code and runs pytest.
All cases use one preinstalled Python environment, with separate working
directories and no per-case environment installation or image builds.

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
to AReno's endpoint, preserving exact input/output tokens, log-probabilities and
routing replay IDs. It buffers SSE until each model call completes and restores
sampled thinking for replay in subsequent tool turns when present.

## Install

Use an existing AReno CUDA training environment. Install pi using its official
installer; accept the prompts to install its required Node.js runtime and pi:

```bash
curl -fsSL https://pi.dev/install.sh | sh
export PATH="$HOME/.pi/agent/bin:$PATH"
python -m pip install -r examples/agentic/pi/requirements.txt
apt-get update && apt-get install -y fd-find ripgrep
command -v fd || ln -s /usr/bin/fdfind /usr/local/bin/fd
pi --version
fd --version
rg --version
```

Prepare one shared task environment, separate from AReno's CUDA packages:

```bash
python3.12 -m venv /tmp/pi-task-env
/tmp/pi-task-env/bin/pip install -r examples/agentic/pi/requirements-task.txt
export ARENO_PI_PYTHON=/tmp/pi-task-env/bin/python
```

`ARENO_PI_EXECUTABLE` can point to a different pi executable or wrapper.
`ARENO_PI_PYTHON` selects the preinstalled Python used by both pi's bash tools
and the controller's grader.

## Generate the dataset

```bash
python examples/agentic/pi/generate_dataset.py \
  --python "$ARENO_PI_PYTHON" --output /tmp/pi-swesmith.jsonl
```

The default selects four single-file sqlparse repairs, with 1–8 failing tests,
using seed 0. It pins SWE-smith revision
`552363dc45a3bdce7ff4f3d7cdaf6945117f66ca` and sqlparse commit
`e57923b3aa823c524c807953cecc48cf6eec2cb2`. Only the repository's dataset shard is
downloaded from ModelScope. There is no Hugging Face fallback.

Pi receives repository source, tests, documentation and the issue description.
The generator applies each upstream bug-injection patch and checks that every
`FAIL_TO_PASS` test fails, every `PASS_TO_PASS` test passes, and the pristine
reference passes all tests. It excludes incompatible cases and records them in
`OUTPUT.report.json`, along with selected IDs, source checksum, dependency
versions and per-test outcomes. Records omit gold repairs and reference code.
Selection happens before policy evaluation and does not filter on reward.
`--limit` and `--seed` allow larger or different selections from this same dataset.

## Train

The default example uses Ling 3.0 Tiny with QLoRA, 4-bit Adam, LoRA rank 32,
GRPO, four tasks and four samples per task, with 16 concurrent attempts:

```bash
export ARENO_PI_RESULT_DIR=/tmp/pi-swesmith-attempts
areno train \
  --ckpt inclusionAI/Ling-3.0-tiny --model-hub modelscope \
  --dataset-path /tmp/pi-swesmith.jsonl \
  --dataset-loader-fn examples/agentic/pi/dataset_loader.py \
  --agent-fn examples/agentic/pi/run_agent.py \
  --reward-fn-path examples/agentic/pi/reward.py \
  --algo grpo --world-size 1 --tp-size 1 \
  --qlora --lora-rank 32 --lora-alpha 64 --adam-4bit \
  --batch-size 4 --mini-bs 1 --n-samples 4 --max-running-prompts 16 \
  --max-context-len 32768 --max-new-tokens 4096 --disable-thinking \
  --lr 1e-5 --max-steps 1 --drop-rollout-state \
  --metrics-log-dir /tmp/pi-swesmith-metrics --save-path /tmp/pi-swesmith-output
```

The training session owns the policy and rollout endpoint; no separate serving
process is needed. Generation length follows AReno's CLI settings. Thinking is
disabled in this example to limit long reasoning turns. Each attempt has a
32-call budget, a 1,800-second execution budget and a 120-second grading budget.
Pi's HTTP timeout follows the execution budget, including queued model calls.
Synchronous rollout completes before weights change.

## Rewards and task records

Each JSONL record contains `prompt`, initial `files`, optional base64
`binary_files`, trusted Python `verify` code and source provenance. The generator
also supplies `max_turns`, `timeout` and `verify_timeout`.
`dataset_loader.py` validates this contract.

The controller grades only submitted `sqlparse/` code in a fresh directory,
using immutable upstream tests. Editing visible tests or pytest configuration
cannot weaken grading. Reward is 1 only when pi finishes within budget and all
expected tests pass without skips; otherwise it is 0. Every model turn receives
its attempt's reward. Failed or timed-out attempts with usable traces remain
negative training examples. Transport/metadata failures and attempts without
model calls are invalid, logged and excluded from training.

`ARENO_PI_RESULT_DIR` retains per-attempt rewards, verifier logs and exact model
traces. Pi's JSON log includes model and tool events during execution. Each
attempt has independent configuration and proxy credentials; sessions,
extensions, skills, automatic compaction and retries are disabled. Cleanup
terminates tool process groups and drains in-flight model requests.

Calculate success over attempts, not model turns. In the 16-attempt Ling
rollout with these settings, 2 repairs passed and 14 attempts reached the time
budget: mean reward **0.125**, with no invalid attempts. This is a bounded smoke
result, not a held-out benchmark score or a difficulty estimate free of timeouts.

Temporary directories are not an OS security sandbox. Run trusted tasks in an
existing disposable container or VM: pi and dataset verification execute with
the current user's permissions.

## CPU checks

```bash
ARENO_CHECK_TOKEN_IDS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q examples/agentic/pi/test_generate_dataset_cpu.py \
  examples/agentic/pi/test_pi_cpu.py
```

These offline checks cover reproducible repository bugs, immutable regression
grading, HTTP/SSE replay, exact trajectory metadata, per-attempt rewards,
timeouts and cleanup, without downloading a model or invoking real pi.
