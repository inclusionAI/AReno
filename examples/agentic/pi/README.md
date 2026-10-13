# Train with the pi coding agent

Use the [pi coding agent](https://pi.dev) as AReno's rollout harness for **100 real
upstream issues** from [ModelScope SWE-bench](https://modelscope.cn/datasets/AI-ModelScope/SWE-bench).
Pi inspects each repository at its historical pre-fix commit, edits package code
and runs upstream regression tests. Cases share one preinstalled Python environment;
each attempt gets a separate working directory, without per-case installations or images.

```text
ModelScope → generate_swebench.py → JSONL → dataset_loader.py
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

Prepare one shared task environment, separate from AReno's CUDA packages.
Python 3.9 runs the historical repository tests; AReno keeps its existing Python:

```bash
python -m pip install uv
uv venv --python 3.9 /tmp/pi-task-env
uv pip install --python /tmp/pi-task-env/bin/python \
  -r examples/agentic/pi/requirements-swebench.txt
export ARENO_PI_PYTHON=/tmp/pi-task-env/bin/python
```

`ARENO_PI_EXECUTABLE` can point to a different pi executable or wrapper.
`ARENO_PI_PYTHON` selects the preinstalled Python used by both pi's bash tools
and the controller's grader.

## Generate the dataset

```bash
python examples/agentic/pi/generate_swebench.py \
  --python "$ARENO_PI_PYTHON" --output /tmp/pi-swebench-100.jsonl
```

`swebench_100.json` pins 100 distinct GitHub issue URLs, corresponding repair PRs,
base commits and regression IDs. The source is ModelScope SWE-bench revision
`8fe7c991907c740e8400dd51d8b08af51271acf8`; downloaded shards are checked against
pinned SHA-256 hashes. The generator clones the source repositories once into
`--repo-cache`, then checks out each case using Git archives.

Every original failing regression must fail on the base code and pass after the
original repair. Up to 20 deterministic upstream passing regressions must also
pass in both states. This is a focused regression suite, not each project's full
test suite. The generator records per-test outcomes and the shared environment
in `OUTPUT.report.json`. It publishes the JSONL only if all selected cases pass
validation. `--limit` can select a smaller prefix of the frozen manifest.

Tasks contain the original issue description, pre-fix source, upstream tests and
verified failure output in `REPRODUCTION.md`. Gold repair patches are used only
by the generator and are excluded from task records. Repository snapshots are
compressed in each record and unpacked only for active attempts. Selection is
based on reproducibility and small repairs, before policy evaluation.

The older `generate_dataset.py` generates **synthetic SWE-smith bug injections**;
those tasks are distinct from this real-issue dataset.

## Train

The default example uses Ling 3.0 Tiny with QLoRA, 4-bit Adam, LoRA rank 64,
GRPO, 100 real issues, two prompts per batch and four samples per task, with eight
concurrent attempts:

```bash
export ARENO_PI_RESULT_DIR=/tmp/pi-swebench-attempts
areno train \
  --ckpt inclusionAI/Ling-3.0-tiny --model-hub modelscope \
  --dataset-path /tmp/pi-swebench-100.jsonl \
  --dataset-loader-fn examples/agentic/pi/dataset_loader.py \
  --agent-fn examples/agentic/pi/run_agent.py \
  --reward-fn-path examples/agentic/pi/reward.py \
  --algo grpo --world-size 1 --tp-size 1 \
  --qlora --lora-rank 64 --lora-alpha 128 --adam-4bit \
  --batch-size 2 --mini-bs 8 --n-samples 4 --max-running-prompts 8 \
  --max-context-len 65536 --max-new-tokens 4096 --disable-thinking \
  --lr 1e-5 --min-lr 1e-5 --epochs 200 --max-steps 200 --drop-rollout-state \
  --metrics-log-dir /tmp/pi-swebench-metrics
```

The training session owns the policy and rollout endpoint; no separate serving
process is needed. Generation length follows AReno's CLI settings. Thinking is
disabled in this example to limit long reasoning turns. Each attempt has a
32-call budget, a 3,600-second execution budget and a 120-second grading budget.
Pi's HTTP timeout follows the execution budget, including queued model calls.
Synchronous rollout completes before weights change. No checkpoint path is set
in this run; only metrics and attempt records are retained.

## Rewards and task records

Each JSONL record contains `prompt`, initial `files`, optional base64
`binary_files` or a compressed `workspace_archive`, trusted Python `verify` code
and source provenance. The generator
also supplies `max_turns`, `timeout` and `verify_timeout`.
`dataset_loader.py` validates this contract.

The controller grades only submitted package code in a fresh directory,
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

Calculate success over attempts, not model turns. Report timeouts and invalid
attempts separately. These are real upstream issues; training results on this selected training set
are not a held-out SWE-bench benchmark score.

Temporary directories are not an OS security sandbox. Run trusted tasks in an
existing disposable container or VM: pi and dataset verification execute with
the current user's permissions.

## CPU checks

```bash
ARENO_CHECK_TOKEN_IDS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q examples/agentic/pi/test_generate_dataset_cpu.py \
  examples/agentic/pi/test_swebench_cpu.py examples/agentic/pi/test_pi_cpu.py
```

These offline checks cover reproducible repository bugs, immutable regression
grading, HTTP/SSE replay, exact trajectory metadata, per-attempt rewards,
timeouts and cleanup, without downloading a model or invoking real pi.
