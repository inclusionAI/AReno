# SWE-bench tasks with pi in Docker-in-Docker

This runs an independent `dockerd` **inside the outer AReno training container**.
It does not mount the host Docker socket or launch sibling containers on the host
Docker daemon. The outer container requires a Linux AMD64 or ARM64 Docker host, privileged
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

### Nested Docker storage

The entrypoint defaults to the classic `vfs` storage driver and disables
`features.containerd-snapshotter` using its dedicated
`/opt/areno-pi/dind/daemon.json`. This avoids containerd/overlayfs mount failures
when the outer container's writable layer cannot support nested overlay mounts.
Setting `--storage-driver` alone is insufficient when the containerd image store
is active. VFS copies layers, so builds take more disk space and time.

Confirm the running daemon with:

```bash
docker info --format '{{.Driver}}'
```

The default must report `vfs`. To use `overlay2` on a worker where it has been
validated, attach dedicated compatible storage at `/var/lib/areno-docker` and
pass `-e ARENO_PI_DOCKER_STORAGE_DRIVER=overlay2` when starting the outer
container. The outer container still requires `--privileged`.

For an existing image, rebuild it or copy both updated startup files from the
repository inside the outer container:

```bash
cp examples/agentic/pi/dind/daemon.json /opt/areno-pi/dind/daemon.json
cp examples/agentic/pi/dind/entrypoint.sh /usr/local/bin/areno-pi-dind
chmod +x /usr/local/bin/areno-pi-dind
```

Then restart the outer container so its entrypoint starts a new nested daemon.
For containers launched with `--rm`, rebuild and launch a new container instead;
their writable layer is deleted on exit. Changing storage backends does not
migrate cached images; expect to rebuild them. Do not delete the old data root
as a troubleshooting step.

See Docker's [VFS driver documentation](https://docs.docker.com/engine/storage/drivers/vfs-driver/)
and [daemon feature settings](https://docs.docker.com/reference/cli/dockerd/#feature-options).

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
default. Lite does not provide a training split. For more integration tasks,
the full SWE-bench **dev** split contains 225 issues:

```bash
python /opt/areno-pi/generate_dataset.py \
  --dataset princeton-nlp/SWE-bench \
  --revision e571863f65e426a6fa843f2a098eb21ccb5385a2 \
  --split dev --output /tmp/swe-dev.jsonl
```

The original SWE-bench **train** split at this revision is not runnable with this
grader: all 19,008 rows have an empty `version` and empty `FAIL_TO_PASS` and
`PASS_TO_PASS` lists. The generator rejects such rows rather than inventing an
environment version or producing invalid rewards. Preparing these raw training
issues requires selecting supported environment recipes and running the trusted
tests before and after the reference fix to establish grading targets. That
preparation is not performed by this converter. For a separate training corpus,
select a ModelScope dataset with validated SWE-bench grading metadata and
environment versions supported by the pinned harness.

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

These dev tasks are for integration experiments; training on dev or test data
is not a held-out benchmark evaluation. GSPO uses the same adapter.

The adapter builds task environments with the official SWE-bench image builder and
installs pi 0.83.0 in a separate `/opt/pi` prefix. Images are cached on the nested
daemon; the first attempt may require a large image download/build. Build errors
or unsupported repository versions fail explicitly. Image preparation
requires network access from the controller. This example pins `swebench==4.1.0`
because its image naming and grading API are part of the integration contract.

If Docker Hub is unreachable from the nested daemon, set an image proxy before
starting training:

```bash
export ARENO_PI_DOCKER_PROXY=v4.gh-proxy.org/docker
```

The adapter writes this prefix directly into the Ubuntu and Node `FROM`
instructions, including the harness-generated base Dockerfile:

```dockerfile
FROM --platform=linux/amd64 v4.gh-proxy.org/docker/ubuntu:22.04
```

```dockerfile
FROM v4.gh-proxy.org/docker/node:22-bookworm-slim AS pi
```

The Ubuntu version follows the task's harness recipe. Pulling and retagging an
image alone is insufficient: a builder can still resolve an unqualified `FROM`
through Docker Hub. Cached task environments remain reusable.

Set the variable **inside the outer AReno container** before starting training.
If its command uses `/opt/areno-pi/run_swe_agent.py`, pulling the repository does
not update that image-baked copy. Rebuild the outer image, or use the updated
checkout directly:

```bash
export ARENO_PI_DOCKER_PROXY=v4.gh-proxy.org/docker
# Replace --agent-fn in the training command with this path:
# --agent-fn /workspace/areno/examples/agentic/pi/run_swe_agent.py
```

The agent loads its sibling helper and grader scripts from the same directory.
Alternatively, refresh the existing copy from the updated repository root:

```bash
cp examples/agentic/pi/*.py /opt/areno-pi/
```

This prefix routes Ubuntu and Node image downloads only; apt, conda, npm and
repository downloads still need network access. It is an image-name prefix,
not a Docker daemon `registry-mirrors` URL.

### Task image architecture

The adapter reads `Architecture` from the private Docker daemon and uses its
native architecture for the SWE-bench environment, pi image, task container,
and grading subprocess. AMD64 selects `linux/amd64`; ARM64 selects
`linux/arm64/v8`, including the harness's ARM64 Miniconda installer. Image cache
keys include the architecture. No cross-architecture emulation is configured.

The startup message prints the selected task/grader platform. Check the worker
with `docker info --format '{{.Architecture}}'`. An ARM64 worker reporting
`exec /bin/sh: no such file or directory` while running an AMD64 image needs
native ARM64 images or separately configured emulation. Update all example
Python files, including `swe_images.py` and `swe_grade.py`, in `/opt/areno-pi`
and start a new training process.

Environment recipes and historical dependencies must also support the selected
architecture; native image selection does not make AMD64-only packages work on
ARM64. Use an AMD64 worker for issues whose dependencies require it. Published
images selected with `ARENO_PI_SWE_NAMESPACE` must exist for that architecture.

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
