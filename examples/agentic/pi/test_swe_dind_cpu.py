"""Controller tests with a Docker double; no Docker daemon or GPU required."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swe_agent as runner


def test_refuses_host_daemon():
    with pytest.raises(RuntimeError, match="unlabelled"):
        runner.check_dind(SimpleNamespace(info=lambda: {"Labels": []}))


@pytest.mark.parametrize(
    "reported,expected", [("x86_64", "x86_64"), ("amd64", "x86_64"), ("aarch64", "arm64"), ("arm64", "arm64")]
)
def test_uses_docker_daemon_architecture(reported, expected):
    client = SimpleNamespace(info=lambda: {"Labels": ["areno.pi.dind=true"], "Architecture": reported})
    assert runner.check_dind(client) == expected


def test_unknown_daemon_architecture_is_not_guessed():
    with pytest.raises(RuntimeError, match="Unsupported Docker daemon architecture"):
        runner.check_dind(SimpleNamespace(info=lambda: {"Labels": ["areno.pi.dind=true"], "Architecture": "unknown"}))


def image_spec(version="22.04", arch="x86_64"):
    from swebench.harness.test_spec.test_spec import TestSpec

    return TestSpec(
        instance_id="django__django-123",
        repo="django/django",
        version="4.0",
        repo_script_list=[],
        eval_script_list=[],
        env_script_list=[],
        arch=arch,
        FAIL_TO_PASS=["test_fix"],
        PASS_TO_PASS=[],
        language="py",
        docker_specs={"ubuntu_version": version},
        namespace=None,
    )


@pytest.mark.parametrize("version", ["20.04", "22.04"])
@pytest.mark.parametrize(
    "arch,platform,conda_arch", [("x86_64", "linux/amd64", "x86_64"), ("arm64", "linux/arm64/v8", "aarch64")]
)
def test_harness_build_uses_proxy_in_actual_dockerfile(monkeypatch, tmp_path, version, arch, platform, conda_arch):
    import docker
    import swebench.harness.docker_build as builds

    monkeypatch.setenv("ARENO_PI_DOCKER_PROXY", "v4.gh-proxy.org/docker/")
    monkeypatch.setattr(builds, "BASE_IMAGE_BUILD_DIR", tmp_path)
    client = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    client.api.build.return_value = iter([{"stream": "built"}])
    original = image_spec(version, arch)
    spec = runner.proxy_test_spec(original)
    # Exercise the real harness conversion and Dockerfile writer, not a mocked builder.
    builds.build_base_images(client, [spec])
    build_path = Path(client.api.build.call_args.kwargs["path"])
    text = (build_path / "Dockerfile").read_text()
    assert f"FROM --platform={platform} v4.gh-proxy.org/docker/ubuntu:{version}" in text
    assert f"FROM --platform={platform} ubuntu:{version}" not in text
    assert f"Linux-{conda_arch}.sh" in text
    assert client.api.build.call_args.kwargs["platform"] == platform
    assert "v4.gh-proxy.org" not in original.base_dockerfile
    assert spec.base_image_key == original.base_image_key
    assert spec.env_image_key == original.env_image_key
    assert spec.instance_image_key == original.instance_image_key


def test_no_proxy_preserves_normal_docker_build(monkeypatch):
    monkeypatch.delenv("ARENO_PI_DOCKER_PROXY", raising=False)
    spec = image_spec()
    assert "FROM --platform=linux/amd64 ubuntu:22.04" in runner.proxy_test_spec(spec).base_dockerfile
    assert runner.docker_image("node:22-bookworm-slim") == "node:22-bookworm-slim"


@pytest.mark.parametrize("prefix", ["https://v4.gh-proxy.org/docker", "proxy.invalid/ docker"])
def test_invalid_proxy_prefix_rejected(monkeypatch, prefix):
    monkeypatch.setenv("ARENO_PI_DOCKER_PROXY", prefix)
    with pytest.raises(ValueError, match="image prefix"):
        runner.proxy_test_spec(image_spec())


@pytest.mark.parametrize("arch,platform", [("x86_64", "linux/amd64"), ("arm64", "linux/arm64/v8")])
def test_agent_build_routes_ubuntu_and_node_directly(monkeypatch, arch, platform):
    import docker
    import swebench.harness.docker_build as builds

    monkeypatch.setenv("ARENO_PI_DOCKER_PROXY", "v4.gh-proxy.org/docker")
    client = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    spec = image_spec(arch=arch)

    def build_instances(client, specs, **kwargs):
        assert "v4.gh-proxy.org/docker/ubuntu:22.04" in specs[0].base_dockerfile
        return specs, []

    monkeypatch.setattr(builds, "build_instance_images", build_instances)
    runner.agent_image(client, spec)
    dockerfile = client.images.build.call_args.kwargs["fileobj"].getvalue()
    assert b"FROM v4.gh-proxy.org/docker/node:22-bookworm-slim AS pi" in dockerfile
    assert b"FROM node:" not in dockerfile
    assert f"FROM {spec.instance_image_key}".encode() in dockerfile
    assert client.images.build.call_args.kwargs["platform"] == platform


@pytest.mark.parametrize("arch", ["x86_64", "arm64"])
def test_grader_uses_the_task_architecture(monkeypatch, tmp_path, arch):
    import docker
    import swe_grade
    import swebench.harness.run_evaluation as evaluation
    import swebench.harness.test_spec.test_spec as specs

    job = tmp_path / "job.json"
    output = tmp_path / "output.json"
    job.write_text(
        json.dumps({"instance": {}, "namespace": None, "arch": arch, "prediction": {}, "run_id": "test", "timeout": 10})
    )
    make_spec = Mock(return_value=image_spec(arch=arch))
    monkeypatch.setattr(specs, "make_test_spec", make_spec)
    run = Mock(return_value={"completed": True, "resolved": True})
    monkeypatch.setattr(evaluation, "run_instance", run)
    monkeypatch.setattr(docker, "from_env", MagicMock())
    monkeypatch.setattr(sys, "argv", ["swe_grade.py", str(job), str(output)])
    swe_grade.main()
    make_spec.assert_called_once_with({}, namespace=None, arch=arch)
    assert run.call_args.args[0].arch == arch
    assert run.call_args.args[0].platform == runner.docker_platform(arch)


def test_cached_agent_image_does_not_rebuild(monkeypatch):
    monkeypatch.setenv("ARENO_PI_DOCKER_PROXY", "v4.gh-proxy.org/docker")
    client = Mock()
    runner.agent_image(client, image_spec())
    client.images.build.assert_not_called()


def test_base_build_failure_includes_log_without_cache_miss_traceback(monkeypatch, tmp_path):
    import traceback

    import docker
    import swebench.harness.docker_build as builds

    monkeypatch.setenv("ARENO_PI_DOCKER_PROXY", "v4.gh-proxy.org/docker")
    monkeypatch.setattr(builds, "BASE_IMAGE_BUILD_DIR", tmp_path)
    client = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound("cache miss")
    client.api.build.return_value = iter(
        [{"stream": "fixture diagnostic from build stderr\n"}, {"errorDetail": {"message": "fixture build failed"}}]
    )

    def build_instances(client, specs, **kwargs):
        builds.build_base_images(client, specs)

    monkeypatch.setattr(builds, "build_instance_images", build_instances)
    with pytest.raises(RuntimeError, match="fixture diagnostic from build stderr") as error:
        runner.agent_image(client, image_spec())
    formatted = "".join(traceback.format_exception(error.value))
    assert "ImageNotFound" not in formatted
    assert "build_image.log" in formatted


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_rejects_unbounded_timeout(value):
    with pytest.raises(ValueError, match="finite and positive"):
        runner.positive_seconds({"timeout": value}, "timeout", 10)


@pytest.fixture
def docker_double(monkeypatch, request):
    import docker
    import swebench.harness.test_spec.test_spec as specs

    architecture = getattr(request, "param", "x86_64")
    arch = "arm64" if architecture == "aarch64" else "x86_64"
    state = {"containers": [], "networks": [], "closed": False, "arch": arch}

    class Container:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.removed = False
            self.commands = []

        def start(self):
            pass

        def put_archive(self, path, data):
            assert path == "/tmp"
            assert b"PRIVATE_TEST" not in data

        def exec_run(self, command, **kwargs):
            self.commands.append(command)
            if command[0] == "pi":
                assert all("PRIVATE_TEST" not in arg for arg in command)
            output = b"diff --git a/x b/x\n" if command[:2] == ["git", "diff"] else b"ok"
            return SimpleNamespace(exit_code=0, output=output)

        def remove(self, force):
            self.removed = True

    def create(image, **kwargs):
        assert image == "pi-image"
        assert kwargs["platform"] == runner.docker_platform(arch)
        assert "volumes" not in kwargs and "mounts" not in kwargs
        assert kwargs["cap_drop"] == ["ALL"]
        assert kwargs["security_opt"] == ["no-new-privileges:true"]
        assert kwargs["pids_limit"] == 256
        c = Container(kwargs)
        state["containers"].append(c)
        return c

    def get(name):
        raise docker.errors.NotFound("no grading container")

    def network_create(name, **kwargs):
        assert kwargs["internal"] is True
        n = SimpleNamespace(name=name, attrs={"IPAM": {"Config": [{"Gateway": "172.30.0.1"}]}}, removed=False)
        n.reload = lambda: None
        n.remove = lambda: setattr(n, "removed", True)
        state["networks"].append(n)
        return n

    client = SimpleNamespace(
        info=lambda: {"Labels": ["areno.pi.dind=true"], "Architecture": architecture},
        containers=SimpleNamespace(create=create, get=get),
        networks=SimpleNamespace(create=network_create),
        close=lambda: state.update(closed=True),
    )
    monkeypatch.setattr(docker, "from_env", lambda **kwargs: client)
    monkeypatch.setattr(runner, "agent_image", lambda *args: "pi-image")

    def make_spec(*args, **kwargs):
        assert kwargs["arch"] == arch
        return SimpleNamespace(
            instance_image_key="swe-image", get_instance_container_name=lambda run_id: "grade-" + run_id
        )

    monkeypatch.setattr(specs, "make_test_spec", make_spec)

    class Proxy:
        def __init__(self, *args, **kwargs):
            assert kwargs["bind_host"] == "0.0.0.0"
            assert kwargs["connect_host"] == "172.30.0.1"
            self.base_url = "http://172.30.0.1:1234/v1"
            self.api_key = "sample-key"
            self.errors, self.trace, self.limit_reached = [], ["trace"], False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def turns(self, item):
            return [SimpleNamespace(item=item)]

    monkeypatch.setattr(runner, "PiProxy", Proxy)
    return state


@pytest.mark.parametrize(
    "report,expected,invalid",
    [
        ({"completed": True, "resolved": True}, 1, False),
        ({"completed": True, "resolved": False}, 0, False),
        ({"completed": False, "resolved": False}, 0, True),
    ],
)
@pytest.mark.parametrize("docker_double", ["x86_64", "aarch64"], indirect=True)
def test_clean_grading_reward_and_cleanup(monkeypatch, docker_double, report, expected, invalid):
    async def grade(command, **kwargs):
        assert docker_double["containers"][0].removed
        job = json.loads(Path(command[2]).read_text())
        assert job["arch"] == docker_double["arch"]
        assert job["instance"]["test_patch"] == "PRIVATE_TEST"
        assert job["prediction"]["model_patch"].startswith("diff --git")
        Path(command[3]).write_text(json.dumps(report))
        kwargs["log_path"].write_text("graded")
        return 0, False

    monkeypatch.setattr(runner, "run_process", grade)
    original = SimpleNamespace(
        prompt="fix issue",
        record={"swebench": {"base_commit": "base", "instance_id": "test-1", "test_patch": "PRIVATE_TEST"}},
    )
    batch = SimpleNamespace(iter_samples=lambda: iter([original]))
    result = asyncio.run(runner.run_agent(SimpleNamespace(max_running_prompts=1), batch))
    item = result.invalid_items[0] if invalid else result.turns[0].item
    assert item.record["pi_result"]["reward"] == expected
    assert "pi_result" not in original.record
    assert all(c.removed for c in docker_double["containers"])
    assert all(n.removed for n in docker_double["networks"])
    assert docker_double["closed"]


def test_cancellation_cleans_containers_and_network(monkeypatch, docker_double):
    entered = asyncio.Event()

    async def grade(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "run_process", grade)
    original = SimpleNamespace(prompt="fix", record={"swebench": {"base_commit": "base", "instance_id": "test-1"}})

    async def check():
        task = asyncio.create_task(
            runner.run_agent(
                SimpleNamespace(max_running_prompts=1), SimpleNamespace(iter_samples=lambda: iter([original]))
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(check())
    assert all(c.removed for c in docker_double["containers"])
    assert all(n.removed for n in docker_double["networks"])
    assert docker_double["closed"]


def test_cancelled_docker_create_is_not_orphaned():
    import threading

    entered, release = threading.Event(), threading.Event()
    resource = SimpleNamespace(removed=False)

    def create():
        entered.set()
        release.wait(5)
        return resource

    async def check():
        task = asyncio.create_task(runner.acquire_resource(create, lambda obj: setattr(obj, "removed", True)))
        await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(check())
    assert resource.removed
