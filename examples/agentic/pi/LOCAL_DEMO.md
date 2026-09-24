# Software utility tasks with pi, without task-image builds

Run pi in the existing AReno environment. Each attempt gets a temporary workspace;
there is no nested Docker daemon, task image build, repository checkout, or
per-issue dependency installation. The existing example secondary proxy still
captures exact rollout tokens and logprobs for training.

The generator downloads [bigcode/bigcodebench from ModelScope](https://modelscope.cn/datasets/bigcode/bigcodebench)
at revision `a4da68573cf2ead10e049a580ba0016d9eb5f281`, split `v0.1.4`.
It selects ten software utility implementation tasks using Python's standard
library and their original upstream unit tests:

| Task ID | Feature |
| --- | --- |
| 7 | Aggregate product sales from CSV |
| 19 | Archive a directory's files as ZIP |
| 24 | Salted PBKDF2 password hashing |
| 25 | JSON serialization, compression and Base64 encoding |
| 118 | Back up JSON files from a directory |
| 127 | Move files using content hashes |
| 539 | Create and populate an SQLite table |
| 992 | Register paths in SQLite without duplicate entries |
| 1130 | Generate a recursive SHA256 file manifest |
| 1134 | Write transformed files with content-checksum prefixes |

These are library-feature tasks with file/database side effects and edge cases,
not whole-repository SWE-bench issues. They are a small training/integration
subset of an evaluation dataset; training on them is not a held-out benchmark
evaluation.

## Install and generate

Use your existing AReno training environment with Python 3.10+ and Node.js 22+.
Run the following from the AReno repository root, inside the existing container
or VM:

```bash
python -m pip install 'modelscope>=1.40.1' pyarrow
npm install -g @mariozechner/pi-coding-agent@0.83.0

python examples/agentic/pi/generate_local_dataset.py \
  --output /tmp/pi-local-engineering.jsonl \
  --check-reference
```

`--check-reference` runs the upstream reference solution and unfinished stub
against the same tests in separate temporary workspaces. Generation stops if
the reference fails or the stub succeeds. This checks your current Python
environment before GPU training. The downloaded reference is used only for
this optional check; generated records contain no reference solution.

`--limit 3` selects the first three tasks for a smaller run. To select different
tasks from the same snapshot, repeat `--task-id BigCodeBench/ID`. The converter
rejects non-standard-library imports; use `--check-reference` to check the
selected tasks' actual runtime requirements. It downloads only the requested
Parquet version and README, records the source checksum, and never falls back
to Hugging Face.

## Train

```bash
areno train \
  --ckpt /path/to/tool-capable-checkpoint \
  --dataset-path /tmp/pi-local-engineering.jsonl \
  --dataset-loader-fn examples/agentic/pi/dataset_loader.py \
  --agent-fn examples/agentic/pi/run_agent.py \
  --reward-fn-path examples/agentic/pi/reward.py \
  --algo grpo --world-size 1 --tp-size 1 \
  --batch-size 1 --mini-bs 1 --n-samples 2 \
  --max-running-prompts 2 \
  --max-new-tokens 4096 --max-context-len 32768 \
  --max-steps 1 --save-path /tmp/pi-local-output
```

Use `--algo gspo` for GSPO. Change the step and batch settings for longer runs.
If pi is installed outside `PATH`, set `ARENO_PI_EXECUTABLE` to its executable.
This demo uses `run_agent.py`, not `run_swe_agent.py`; no Docker proxy, SWE-bench
installation or `/opt/areno-pi` copy is needed.

Pi reads `solution.py` with the original signature, constants and specification,
implements the feature, and can write/run its own tests. Upstream grading tests
remain in the controller's dataset record and are not written into the initial
workspace. After pi finishes, a separate Python process executes the edited
implementation and upstream tests. Reward is 1 only if pi finishes within budget
and the nonempty test suite passes without skipped tests; otherwise it is 0.
Every model turn in the attempt shares that reward. The original local runner's
handling of invalid proxy traces and process timeouts is unchanged.

Attempts use separate working directories, not an OS security sandbox. Pi and
dataset tests execute code with the permissions of the current container/VM.
Use the same disposable training environment as the existing local pi example.

## CPU checks

```bash
python -m pytest -q examples/agentic/pi/test_local_dataset_cpu.py \
  examples/agentic/pi/test_pi_cpu.py
```

The offline checks cover conversion, exclusion of reference solutions from
generated records, dependency validation, executable verifier behavior, and
the existing proxy/reward path. `--check-reference` additionally checks the
real downloaded tasks. Neither check performs GPU training.
