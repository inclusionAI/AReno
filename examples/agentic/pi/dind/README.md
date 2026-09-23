# SWE-bench tasks with pi in Docker-in-Docker

This runs an independent `dockerd` **inside the outer AReno training container**.
It does not mount the host Docker socket or launch sibling containers on the host
Docker daemon. The outer container requires a Linux x86-64 Docker host, privileged
DinD support, and the usual GPU runtime for training.

```text
Host Docker
└── privileged AReno container
    ├── AReno policy / rollout proxy
    ├── example secondary proxy (exact trajectories)
    └── private dockerd (Unix socket only)
        ├── per-attempt pi container + repository at base commit
        └── fresh SWE-bench grading container + predicted patch + trusted tests
```

The task containers receive no Docker socket, host bind mounts, training keys or
GPUs. Pi receives a per-attempt secondary-proxy key. Its internal Docker network
has no external routing; it can reach the proxy on the bridge gateway. Images and
repository dependencies are prepared before pi starts. The privileged outer
container is a trusted controller, not a security boundary against a kernel escape;
run it on a disposable worker without host credentials.

## Build the outer environment

From the AReno repository root, select an existing Ubuntu/Debian-based AReno
training image with Python 3.10+ and the current example's core APIs:

```bash
docker build -f examples/agentic/pi/dind/Dockerfile \
  --build-arg ARENO_IMAGE=your-areno-training-image:tag \
  -t areno-pi-dind .

docker run --rm -it --privileged --gpus all \
  --shm-size=8g \
  -v /path/to/local/checkpoint:/models/policy:ro \
  areno-pi-dind
```

Do not add a `/var/run/docker.sock` mount or publish a Docker TCP port. The
entrypoint starts the nested daemon, waits for readiness, and stops it on exit.
Its `areno.pi.dind=true` label is checked by the rollout adapter. Nested image
storage is ephemeral unless you explicitly attach a dedicated volume at
`/var/lib/areno-docker`; do not reuse the host daemon's data directory.

## Generate actual repository tasks from ModelScope

Inside the outer container:

```bash
python /opt/areno-pi/generate_dataset.py \
  --output /tmp/swe-dev.jsonl --split dev --limit 3
```

The default is a pinned ModelScope snapshot of
[princeton-nlp/SWE-bench_Lite](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Lite),
using its **dev** split for integration experiments. These are real repository
issues, not invented function exercises. The Lite test split remains held out by
default. Lite does not provide a training split. For the full SWE-bench training
split, select its ModelScope mirror explicitly:

```bash
python /opt/areno-pi/generate_dataset.py \
  --dataset princeton-nlp/SWE-bench \
  --revision e571863f65e426a6fa843f2a098eb21ccb5385a2 \
  --split train --output /tmp/swe-train.jsonl --limit 100
```

The generator uses ModelScope `snapshot_download`, reads only the selected
split's Parquet files, and records the dataset revision and source checksum.
It never falls back to Hugging Face. It copies the issue and repository metadata,
keeps test patches/FAIL_TO_PASS/PASS_TO_PASS in controller-only fields, and omits
the gold solution patch and hints entirely. No reference solution is staged in
the agent container. Changing `--limit` changes the number of tasks, not their
difficulty. The JSONL is generated locally rather than committed to this repo.

## Train

```bash
areno train \
  --ckpt /models/policy \
  --dataset-path /tmp/swe-dev.jsonl \
  --dataset-loader-fn /opt/areno-pi/swe_dataset_loader.py \
  --agent-fn /opt/areno-pi/run_swe_agent.py \
  --reward-fn-path /opt/areno-pi/reward.py \
  --algo grpo --world-size 1 --tp-size 1 \
  --batch-size 1 --mini-bs 1 --n-samples 2 \
  --max-running-prompts 2 \
  --max-new-tokens 4096 --max-context-len 32768 \
  --max-steps 1 --save-path /tmp/swe-output
```

Use `/tmp/swe-train.jsonl` for the training dataset generated above. Training on dev
or test data is not a held-out benchmark evaluation. GSPO uses the same adapter.

The adapter builds task environments with the official SWE-bench image builder and
installs pi 0.83.0 in a separate `/opt/pi` prefix. Images are cached on the nested
daemon; the first attempt may require a large image download/build. Build errors
or unsupported repository versions fail explicitly. Image preparation
requires network access from the controller. This example pins `swebench==4.1.0`
because its image naming and grading API are part of the integration contract.

Pi edits `/testbed` with 64 model turns and a 30-minute deadline by default. It may
run the repository's existing tests, but receives no injected benchmark tests.
After pi exits, the controller collects `git diff` (including new files), deletes
the agent container and uses official SWE-bench grading in a fresh instance
container. Reward is 1 only when the official report marks the issue resolved,
including FAIL_TO_PASS and PASS_TO_PASS checks. Incomplete grading is excluded,
not counted as an ordinary incorrect answer. Empty patches, failed pi attempts
and budget exhaustion receive 0 when a usable trajectory exists.

Per-row `timeout`, `verify_timeout` and `max_turns` configure budgets.
`ARENO_PI_TASK_MEMORY` (default `8g`) and `ARENO_PI_TASK_CPUS` (default `2`) limit
agent containers; these limits do not change official grader resource settings.
Set `ARENO_PI_SWE_NAMESPACE=swebench` to use published instance images when
available; by default images are built locally because train/dev instances may
not have published images. Containers are removed after
success, failure, timeout or cancellation. `pi_result` retains the predicted patch,
reward and tail of grader output. The existing secondary proxy still preserves
upstream token IDs, logprobs and trajectory grouping without modifying AReno core.

## Validation

```bash
python -m pytest -q examples/agentic/pi/test_generate_dataset_cpu.py \
  examples/agentic/pi/test_swe_dind_cpu.py examples/agentic/pi/test_pi_cpu.py
```

CPU checks cover conversion and controller contracts, not a real Docker/GPU run.
To validate a worker, first generate three dev tasks and run the one-step command
above. A working nested daemon, image registry access and compatible GPU checkpoint
are required for that integration check.
