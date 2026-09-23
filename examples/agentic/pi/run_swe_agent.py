"""Run pi and SWE-bench grading in separate containers on a private DinD daemon."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import math
import os
import sys
import tarfile
import tempfile
import threading
import uuid
from pathlib import Path

from areno.api.agentic import AgentTrajectory

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))
from pi_proxy import PiProxy  # noqa: E402
from run_agent import configure_pi, log_tail, run_process  # noqa: E402
from swe_images import docker_image, proxy_test_spec  # noqa: E402

_BUILD_LOCK = threading.Lock()


def check_dind(client):
    if "areno.pi.dind=true" not in client.info().get("Labels", []):
        raise RuntimeError("Use the example DinD entrypoint; refusing an unlabelled Docker daemon")


def agent_image(client, spec):
    import docker

    spec = proxy_test_spec(spec)
    base_image = spec.instance_image_key
    node_image = docker_image("node:22-bookworm-slim")
    tag = "areno-pi-swe:" + hashlib.sha256((base_image + node_image + "pi-0.83.0-v2").encode()).hexdigest()[:24]
    with _BUILD_LOCK:
        try:
            client.images.get(tag)
        except docker.errors.ImageNotFound:
            if not spec.is_remote_image:
                from swebench.harness.docker_build import build_instance_images

                _, failed = build_instance_images(client, [spec], max_workers=1)
                if failed:
                    raise RuntimeError(f"Could not build SWE-bench environment: {spec.instance_id}")
            # No dataset tests/gold patches are added to the agent image.
            dockerfile = (
                f"FROM {node_image} AS pi\n"
                "RUN npm install -g @mariozechner/pi-coding-agent@0.83.0\n"
                f"FROM {base_image}\n"
                "COPY --from=pi /usr/local /opt/pi\n"
                "ENV PATH=/opt/pi/bin:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
            )
            client.images.build(fileobj=io.BytesIO(dockerfile.encode()), tag=tag, rm=True, platform="linux/amd64")
    return tag


def put_directory(container, directory: Path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        archive.add(directory, arcname="pi-config")
    container.put_archive("/tmp", stream.getvalue())


def checked_exec(container, command):
    result = container.exec_run(command, workdir="/testbed")
    if result.exit_code:
        raise RuntimeError(f"task container command failed: {result.output[-2000:].decode(errors='replace')}")
    return result.output


def extract_patch(container, base_commit):
    # Stage intent-to-add so newly created files also appear in the prediction.
    checked_exec(container, ["git", "add", "-N", "."])
    return checked_exec(container, ["git", "diff", "--binary", base_commit, "--"]).decode()


async def acquire_resource(factory, cleanup, *args, **kwargs):
    """Do not abandon a Docker create request if its awaiting task is cancelled."""
    pending = asyncio.create_task(asyncio.to_thread(factory, *args, **kwargs))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        resource = await pending
        await asyncio.to_thread(cleanup, resource)
        raise


def positive_seconds(record, key, default):
    value = float(record.get(key, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


async def run_agent(ctx, batch):
    import docker
    from swebench.harness.test_spec.test_spec import make_test_spec

    client = docker.from_env(timeout=120)
    try:
        await asyncio.to_thread(check_dind, client)
    except BaseException:
        client.close()
        raise
    semaphore = asyncio.Semaphore(ctx.max_running_prompts)
    namespace = os.environ.get("ARENO_PI_SWE_NAMESPACE") or None

    async def run_sample(original):
        async with semaphore:
            item = copy.deepcopy(original)
            instance = item.record["swebench"]
            timeout = positive_seconds(item.record, "timeout", 1800)
            verify_timeout = positive_seconds(item.record, "verify_timeout", 1800)
            spec = make_test_spec(instance, namespace=namespace, arch="x86_64")
            image = await asyncio.to_thread(agent_image, client, spec)
            run_id = "areno-pi-" + uuid.uuid4().hex
            network = await acquire_resource(
                client.networks.create, lambda value: value.remove(), run_id, driver="bridge", internal=True
            )
            container = None
            try:
                await asyncio.to_thread(network.reload)
                gateway = network.attrs["IPAM"]["Config"][0]["Gateway"]
                with tempfile.TemporaryDirectory(prefix="areno-pi-swe-") as directory:
                    root = Path(directory)
                    async with PiProxy(
                        ctx,
                        max_turns=int(item.record.get("max_turns", 64)),
                        timeout=timeout,
                        bind_host="0.0.0.0",
                        connect_host=gateway,
                    ) as proxy:
                        config = root / "pi-config"
                        configure_pi(config, proxy)
                        container = await acquire_resource(
                            client.containers.create,
                            lambda value: value.remove(force=True),
                            image,
                            command=["sleep", "infinity"],
                            name=run_id,
                            network=network.name,
                            working_dir="/testbed",
                            environment={"PI_CODING_AGENT_DIR": "/tmp/pi-config"},
                            cap_drop=["ALL"],
                            security_opt=["no-new-privileges:true"],
                            mem_limit=os.environ.get("ARENO_PI_TASK_MEMORY", "8g"),
                            nano_cpus=int(float(os.environ.get("ARENO_PI_TASK_CPUS", "2")) * 1e9),
                            pids_limit=256,
                            detach=True,
                        )
                        await asyncio.to_thread(container.start)
                        await asyncio.to_thread(put_directory, container, config)
                        await asyncio.to_thread(
                            checked_exec, container, ["git", "reset", "--hard", instance["base_commit"]]
                        )
                        await asyncio.to_thread(checked_exec, container, ["git", "clean", "-fd"])
                        command = [
                            "pi",
                            "--print",
                            "--no-session",
                            "--no-extensions",
                            "--no-skills",
                            "--no-prompt-templates",
                            "--no-themes",
                            "--provider",
                            "areno",
                            "--model",
                            "policy",
                            "Task:\n" + item.prompt,
                        ]
                        timed_out = False
                        try:
                            result = await asyncio.wait_for(asyncio.to_thread(container.exec_run, command), timeout)
                            code, log = result.exit_code, result.output[-8000:].decode(errors="replace")
                        except asyncio.TimeoutError:
                            timed_out, code, log = True, None, "pi task timed out"
                            await asyncio.to_thread(container.kill)
                    details = {"returncode": code, "timed_out": timed_out, "log": log, "reward": 0.0}
                    item.record["pi_result"] = details
                    if proxy.errors or not proxy.trace:
                        details["proxy_errors"] = proxy.errors
                        return AgentTrajectory(invalid_items=[item])
                    if code == 0 and not timed_out and not proxy.limit_reached:
                        patch = await asyncio.to_thread(extract_patch, container, instance["base_commit"])
                        # Destroy the editable agent container before starting trusted grading.
                        await asyncio.to_thread(container.remove, force=True)
                        container = None
                        prediction = {
                            "instance_id": instance["instance_id"],
                            "model_name_or_path": "areno-pi",
                            "model_patch": patch,
                        }
                        details["model_patch"] = patch
                        if patch.strip():
                            job = root / "grade.json"
                            output = root / "result.json"
                            job.write_text(
                                json.dumps(
                                    {
                                        "instance": instance,
                                        "namespace": namespace,
                                        "prediction": prediction,
                                        "run_id": run_id,
                                        "timeout": int(verify_timeout),
                                    }
                                )
                            )
                            grade_code, grade_timeout = await run_process(
                                [sys.executable, str(_EXAMPLE_DIR / "swe_grade.py"), str(job), str(output)],
                                cwd=root,
                                env=dict(os.environ),
                                timeout=verify_timeout + 120,
                                log_path=root / "grade.log",
                            )
                            details["verify_log"] = log_tail(root / "grade.log")
                            if grade_code or grade_timeout or not output.exists():
                                details["grading_error"] = "grader failed or timed out"
                                return AgentTrajectory(invalid_items=[item])
                            report = json.loads(output.read_text())
                            if not report.get("completed"):
                                details["grading_error"] = "SWE-bench evaluation did not complete"
                                return AgentTrajectory(invalid_items=[item])
                            details["reward"] = float(report["resolved"])
                    return AgentTrajectory(turns=proxy.turns(item))
            finally:
                if container is not None:
                    await asyncio.to_thread(container.remove, force=True)
                # Also remove a grader container after timeout/cancellation of its controller.
                try:
                    grading = await asyncio.to_thread(client.containers.get, spec.get_instance_container_name(run_id))
                except docker.errors.NotFound:
                    pass
                else:
                    await asyncio.to_thread(grading.remove, force=True)
                await asyncio.to_thread(network.remove)

    tasks = [asyncio.create_task(run_sample(item)) for item in batch.iter_samples()]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        client.close()
    return AgentTrajectory(
        turns=[turn for result in results for turn in result.turns],
        invalid_items=[item for result in results for item in result.invalid_items],
    )
