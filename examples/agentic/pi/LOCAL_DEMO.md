# Software utility tasks with pi, without task-image builds

Run pi in the existing AReno environment. Each attempt gets a temporary workspace;
there is no nested Docker daemon, task image build, repository checkout, or
per-issue dependency installation. The existing example secondary proxy still
captures exact rollout tokens and logprobs for training.

The generator downloads [bigcode/bigcodebench from ModelScope](https://modelscope.cn/datasets/bigcode/bigcodebench)
at revision `a4da68573cf2ead10e049a580ba0016d9eb5f281`, split `v0.1.4`
(1,140 tasks). It scans the full snapshot, preserves upstream unit tests, and
selects tasks for one shared environment:

| Profile | Selection | Candidates on Python 3.12 |
| --- | --- | --- |
| `stdlib` (default) | Standard-library tasks after portability screening | 197 |
| `extended` | Standard library plus the shared packages below | 865 |
| `smoke` | Original ten reviewed CSV, ZIP, hashing, file and SQLite tasks | 10 |

These counts are **candidates before executing reference checks**, not promises
that every task works on every Python/library version. The extended set includes
the standard-library set; the counts are not additive. Selection order is numeric
task ID, making limits reproducible.

These are library-feature tasks with file/database side effects and edge cases,
not whole-repository SWE-bench issues. They are a training/integration
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
against the same tests in separate temporary workspaces. Bulk selection keeps
only tasks whose reference passes and whose stub fails. Failed or timed-out
checks are excluded with diagnostics in `/tmp/pi-local-engineering.jsonl.report.json`.
The downloaded reference is used only for this optional check; generated records
contain no reference solution. Without the flag, generation only screens source
and dependency metadata and does not execute dataset code.

`--limit 50` stops after 50 accepted tasks, continuing past rejected tasks.
Omit it to process every candidate. `--profile smoke` reproduces the original
ten-task selection. To select specific tasks, repeat `--task-id BigCodeBench/ID`.
Explicit selections and the smoke profile fail if any requested task is rejected;
they never silently shrink. If generation fails or no tasks qualify, an existing
output dataset is preserved.

## Larger shared-dependency set

Install the optional packages **once in the same environment used for AReno and
pi**, then generate the extended set:

```bash
python -m pip install -r examples/agentic/pi/requirements-local.txt

python examples/agentic/pi/generate_local_dataset.py \
  --profile extended \
  --output /tmp/pi-local-engineering.jsonl \
  --check-reference
```

The shared packages are NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn,
Pillow, Faker, python-dateutil and pytz. They add data processing, numerical,
plotting and image-processing tasks. Package names and installed versions are
recorded in the report; each generated row also lists its required packages.
There is no automatic package installation during rollout. Grading uses
Matplotlib's noninteractive `Agg` backend.

The automatic screening checks both import statements and upstream library
metadata. It conservatively excludes unsupported packages, process/network/GUI
modules, external process operations, and absolute/shared or parent-relative
path literals. It can exclude harmless mocked paths or URL-processing tasks;
it is a portability heuristic, not a sandbox. The original ten individually
reviewed tasks retain their harmless nonexistent-path test cases.

Every run writes `OUTPUT.report.json` (override with `--report`). It records the
source revision, Python version, selected IDs, exclusion reasons, reference-check
status, package versions, and tasks not considered because a limit was reached.
Inspect it before training, particularly after changing the environment. The
generator downloads only the pinned Parquet version and README, records source
checksums, and never falls back to Hugging Face.

## Train

```bash
export MPLBACKEND=Agg
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

The offline checks cover full-snapshot selection, dependency profiles, limits,
reports, reference-failure filtering, timeouts, exclusion of reference solutions
from generated records, executable verifier behavior, and the existing
proxy/reward path. `--check-reference` additionally checks the real downloaded
tasks. Neither check performs GPU training.
