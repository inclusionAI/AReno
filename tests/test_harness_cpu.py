"""Harness registry and Harbor lifecycle contracts, without models or GPUs."""

import asyncio
import base64
import binascii
import json
import shlex
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.trial.config import AgentConfig

from areno.agent.harness_base import TaskVerification, WorkspaceDataset, WorkspaceFiles, workspace_path
from areno.api import harness
from areno.api.agentic import AgentBatch
from areno.api.harness import HarnessAttempt


class FileTaskDataset(WorkspaceDataset):
    def __init__(self, *, reserved_roots: tuple[str, ...] = (".pi",)) -> None:
        super().__init__(reserved_roots=reserved_roots)

    def normalize(self, record: dict) -> dict:
        record = dict(record)
        prompt = record.get("prompt") or record.get("problem_statement")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("task requires prompt or problem_statement")
        for name in ("files", "binary_files"):
            files = record.get(name, {} if name == "binary_files" else None)
            if not isinstance(files, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in files.items()
            ):
                raise ValueError(
                    f"task {name} must map relative paths to {'base64 strings' if name == 'binary_files' else 'text'}"
                )
            for path, content in files.items():
                workspace_path(path, self.reserved_roots)
                if name == "binary_files":
                    try:
                        base64.b64decode(content, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ValueError(f"invalid base64 file: {path}") from exc
        if record["files"].keys() & record.get("binary_files", {}).keys():
            raise ValueError("text and binary task file paths overlap")
        if not isinstance(record.get("verify"), str) or not record["verify"].strip():
            raise ValueError("task requires trusted Python verify code")
        record["prompt"] = prompt
        return record

    def workspace_files(self, record: dict) -> WorkspaceFiles:
        return {
            **record["files"],
            **{
                name: base64.b64decode(content, validate=True)
                for name, content in record.get("binary_files", {}).items()
            },
        }

    def verification(self, attempt: HarnessAttempt) -> TaskVerification:
        handle = attempt.environment
        assert handle is not None
        code = "import sys; sys.path.insert(0, sys.argv[1]);\n" + attempt.item.record["verify"]
        path = str(PurePosixPath(handle.control_dir) / "verify.py")
        return TaskVerification(command=[handle.python, "-I", path, handle.workspace], files={"verify.py": code})


@pytest.fixture(autouse=True)
def registry_state(monkeypatch):
    for name in ("_HARNESSES", "_ENVIRONMENTS", "_DATASETS"):
        monkeypatch.setattr(harness, name, dict(getattr(harness, name)))
    harness.register_dataset("file-task-fixture", FileTaskDataset)


def test_public_api_keeps_harbor_and_torch_imports_lazy():
    script = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'harbor', 'torch'}:
        raise AssertionError('eager dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from areno.api import HarnessRunner, register_harness, list_environments, list_datasets
assert not any(name.startswith('harbor') for name in sys.modules)
builtins.__import__ = original
assert {'docker', 'local'} <= set(list_environments())
assert list_datasets() == []
class CustomDataset:
    def normalize(self, row): return dict(row, prompt=row['problem_statement'])
from areno.api import register_dataset
register_dataset('custom', CustomDataset)
rows = HarnessRunner(harness='pi', dataset='custom').dataset_loader('unused', default_loader=lambda _: [
    {'problem_statement': 'fix', 'files': {}, 'verify': 'assert True'}])
assert rows[0]['prompt'] == 'fix'
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[1])


@pytest.mark.parametrize(
    "register,list_names",
    [
        (harness.register_harness, harness.list_harnesses),
        (harness.register_environment, harness.list_environments),
        (harness.register_dataset, harness.list_datasets),
    ],
)
def test_registry_names_duplicates_and_selection(register, list_names):
    def factory(**kwargs):
        return kwargs

    register(" Test-Adapter ", factory)
    assert "test-adapter" in list_names()
    with pytest.raises(ValueError, match="already registered"):
        register("test-adapter", factory)
    register("test-adapter", factory, replace=True)
    with pytest.raises(ValueError, match="nonempty"):
        register(" ", factory)
    with pytest.raises(ValueError, match="callable"):
        register("not-callable", None)


def test_unknown_dataset_reports_registered_names():
    with pytest.raises(ValueError, match="unknown adapter.*registered:"):
        harness.HarnessRunner(harness="pi", dataset="missing").dataset_loader("unused", default_loader=lambda _: [])


def test_dataset_selection_must_be_explicit():
    with pytest.raises(TypeError, match="dataset"):
        harness.HarnessRunner(harness="external-agent")


@pytest.mark.parametrize("expected,score", [("value with spaces; $(touch injected)", 0.375), ("wrong", 0.0)])
def test_reusable_bases_support_other_record_schemas_agents_and_continuous_scores(tmp_path, expected, score):
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.models.agent.context import AgentContext

    from areno.agent.harness_base import TaskVerification, WorkspaceDataset
    from areno.agent.harness_env import local_environment

    class CustomDataset(WorkspaceDataset):
        def workspace_files(self, record):
            return {"payload.bin": record["custom_binary_payload"]}

        def verification(self, attempt):
            handle = attempt.environment
            script = 'test "$(cat answer.txt)" = "$EXPECT" || exit 7\nprintf \'{"score":0.375}\''
            return TaskVerification(
                command=["bash", str(Path(handle.control_dir) / "grade.sh")],
                files={"grade.sh": script},
                env={"EXPECT": attempt.item.record["expected_value"]},
            )

        def score(self, result):
            return json.loads(result.stdout)["score"] if result.return_code == 0 else 0.0

    class CustomAgent(BaseInstalledAgent):
        @staticmethod
        def name():
            return "custom-command"

        async def install(self, environment):
            pass

        async def run(self, instruction, environment, context):
            command = [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path('answer.txt').write_text(sys.argv[1]); print('done')",
                instruction,
            ]
            output = await self.exec_as_agent(environment, shlex.join(command), cwd=str(tmp_path / "workspace"))
            context.metadata = {**context.metadata, "returncode": output.return_code}

    async def check():
        record = {"custom_binary_payload": b"\xff\x80", "expected_value": expected}
        item = next(AgentBatch([record], ["custom"], [[1]], 1).iter_samples())
        attempt = harness.HarnessAttempt(item, tmp_path, 1, 1)
        handle = local_environment(attempt)
        attempt.environment = handle
        adapter = CustomDataset()
        assert adapter.normalize(record) == record and adapter.normalize(record) is not record
        await handle.environment.start(False)
        try:
            await adapter.prepare(attempt)
            assert (Path(handle.workspace) / "payload.bin").read_bytes() == b"\xff\x80"
            agent = CustomAgent(logs_dir=tmp_path)
            context = AgentContext(metadata={"custom_metadata": 1})
            await agent.run("value with spaces; $(touch injected)", handle.environment, context)
            assert context.metadata["returncode"] == 0 and context.metadata["custom_metadata"] == 1
            assert not (Path(handle.workspace) / "injected").exists()
            result = await adapter.verify(attempt)
            assert result["reward"] == score
            assert result["verify_returncode"] == (0 if score else 7)
            assert not result["verify_timed_out"]
            assert not (Path(handle.workspace) / "grade.sh").exists()
        finally:
            await handle.environment.stop()

    asyncio.run(check())


def test_generic_dataset_reserves_only_configured_roots():
    record = {"prompt": "fix", "files": {".pi/file": "data"}, "verify": "assert True"}
    assert FileTaskDataset(reserved_roots=()).normalize(record)["files"] == record["files"]
    with pytest.raises(ValueError, match="invalid task file path"):
        FileTaskDataset(reserved_roots=(".pi",)).normalize(record)


def test_custom_callable_generator_and_adapter_need_no_file_task_fields():
    class CustomDataset:
        def normalize(self, record):
            return {"prompt": record["question"], "custom_payload": record["payload"]}

    calls = []

    def load_rows(path, *, split, count):
        calls.append((path, split, count))
        for index in range(count):
            yield {"question": f"{split} task {index}", "payload": {"source": path}}

    harness.register_dataset("custom-records", CustomDataset)
    workflow = harness.HarnessRunner(
        harness="any", dataset="custom-records", loader_fn=load_rows, loader_kwargs={"split": "train", "count": 2}
    )
    rows = workflow.dataset_loader(
        "custom-source",
        split="validation",
        load_dataset=object(),
        load_from_disk=object(),
    )
    assert calls == [("custom-source", "validation", 2)]
    assert [row["prompt"] for row in rows] == ["validation task 0", "validation task 1"]
    assert all("files" not in row and "verify" not in row for row in rows)


@pytest.mark.parametrize("function", [None, "read_rows"])
def test_custom_file_loader_receives_helpers_and_user_options(tmp_path, function):
    path = tmp_path / "custom_loader.py"
    symbol = function or "load_training_dataset"
    path.write_text(
        "from dataclasses import dataclass\n@dataclass\nclass Row:\n    prompt: str\n"
        f"def {symbol}(path, *, default_loader, load_dataset, prefix):\n"
        "    assert load_dataset == 'user helper'\n"
        "    for row in default_loader(path):\n"
        "        yield dict(row, prompt=Row(prefix + row['prompt']).prompt)\n"
    )
    specification = str(path) if function is None else f"{path}:{function}"
    workflow = harness.HarnessRunner(
        harness="pi", dataset="file-task-fixture", loader_fn=specification, loader_kwargs={"prefix": "custom: "}
    )
    rows = workflow.dataset_loader(
        "input",
        default_loader=lambda _: [{"prompt": "fix", "files": {}, "verify": "assert True"}],
        load_dataset="user helper",
        load_from_disk=object(),
    )
    assert rows[0]["prompt"] == "custom: fix"


def test_default_loader_receives_user_options_without_framework_helpers():
    calls = []

    def default(path, *, split):
        calls.append((path, split))
        return [{"prompt": "fix", "files": {}, "verify": "assert True"}]

    workflow = harness.HarnessRunner(harness="pi", dataset="file-task-fixture", loader_kwargs={"split": "train"})
    assert workflow.dataset_loader("input", default_loader=default, split="test", load_dataset=object())
    assert calls == [("input", "test")]


def test_custom_loader_errors_are_not_retried_or_hidden():
    calls = []

    def broken(path):
        calls.append(path)
        raise TypeError("loader implementation failed")

    workflow = harness.HarnessRunner(harness="pi", dataset="file-task-fixture", loader_fn=broken)
    with pytest.raises(TypeError, match="loader implementation failed"):
        workflow.dataset_loader("input", default_loader=lambda _: [])
    assert calls == ["input"]
    with pytest.raises(TypeError, match="unexpected keyword"):
        workflow.dataset_loader("input", default_loader=lambda _: [], misspelled_option=1)


def test_invalid_custom_loader_reports_the_source(tmp_path):
    path = tmp_path / "not_callable.py"
    path.write_text("load_training_dataset = 42\n")
    workflow = harness.HarnessRunner(harness="pi", dataset="file-task-fixture", loader_fn=path)
    with pytest.raises(ValueError, match="must define callable load_training_dataset"):
        workflow.dataset_loader("input", default_loader=lambda _: [])
    workflow.loader_fn = str(path) + ":"
    with pytest.raises(ValueError, match="function name"):
        workflow.dataset_loader("input", default_loader=lambda _: [])
    workflow.loader_fn = None
    with pytest.raises(ValueError, match="loader_fn or default_loader"):
        workflow.dataset_loader("input")


@pytest.mark.parametrize(
    "update",
    [
        {"files": {"../outside": "bad"}},
        {"files": {"/absolute": "bad"}},
        {"files": {".pi/settings.json": "bad"}},
        {"binary_files": {"x": "!not-base64!"}},
        {"binary_files": {"x": "eA=="}, "files": {"x": "text"}},
        {"verify": ""},
    ],
)
def test_dataset_rejects_invalid_files_before_execution(update):
    record = {"prompt": "fix", "files": {}, "verify": "assert True", **update}
    with pytest.raises(ValueError):
        FileTaskDataset(reserved_roots=(".pi",)).normalize(record)


@contextmanager
def policy_endpoint(concurrency=2):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            assert self.headers["Authorization"] == "Bearer upstream-key"
            json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response = {
                "id": "sample",
                "created": 1,
                "model": "policy",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "areno": {
                    "input_tokens": [10, 11],
                    "response_tokens": [12],
                    "response_logprobs": [-0.25],
                    "routing_replay_id": "exact-route",
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield SimpleNamespace(
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            api_key="upstream-key",
            max_running_prompts=concurrency,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ContractAgent(BaseAgent):
    active = 0
    maximum = 0

    def __init__(self, *args, fail_stage=None, delay=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_stage, self.delay = fail_stage, delay

    @staticmethod
    def name():
        return "contract-agent"

    def version(self):
        return "test"

    async def setup(self, environment):
        if self.fail_stage == "setup":
            raise RuntimeError("setup failed")

    async def run(self, instruction, environment, context):
        type(self).active += 1
        type(self).maximum = max(type(self).maximum, type(self).active)
        try:
            request = Request(
                self.extra_env["OPENAI_BASE_URL"] + "/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": instruction}]}).encode(),
                headers={"Authorization": "Bearer " + self.extra_env["OPENAI_API_KEY"]},
            )

            def complete():
                with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
                    return json.load(response)["choices"][0]["message"]["content"]

            answer = await asyncio.to_thread(complete)
            script = "from pathlib import Path; Path('solution.py').write_text('answer = ' + " + repr(answer) + ")"
            await environment.exec(shlex.join([sys.executable, "-c", script]))
            if self.fail_stage == "run":
                raise NonZeroAgentExitCodeError("agent exited after saving code")
            await asyncio.sleep(self.delay)
            context.metadata = {"ran": True}
        finally:
            type(self).active -= 1


@pytest.fixture
def adapters():
    ContractAgent.active = ContractAgent.maximum = 0
    harness.register_harness(
        "test-agent",
        lambda attempt, **kwargs: AgentConfig(
            import_path=__name__ + ":ContractAgent",
            kwargs=kwargs,
        ),
    )
    harness.register_dataset("test-dataset", FileTaskDataset)


def batch(verify="from solution import answer; assert answer == 42", **kwargs):
    record = {"prompt": "fix", "files": {"solution.py": "answer = 0"}, "verify": verify, **kwargs}
    return AgentBatch([record], ["fix"], [[1]], 1)


def runner(**kwargs):
    return harness.HarnessRunner(harness="test-agent", environment="local", dataset="test-dataset", **kwargs)


def test_native_factories_trial_lifecycle_and_exact_metadata(adapters, tmp_path):
    source = batch().records + batch("assert False").records
    with policy_endpoint(concurrency=1) as ctx:
        result = asyncio.run(runner(result_dir=tmp_path)(ctx, AgentBatch(source, ["fix"] * 2, [[1]] * 2, 2)))
    assert not result.invalid_items and len(result.turns) == 4
    assert ContractAgent.maximum == 1 and ContractAgent.active == 0
    assert [turn.item.record["harness_result"]["reward"] for turn in result.turns] == [1, 1, 0, 0]
    assert all("harness_result" not in record for record in source)
    assert len({id(turn.item.record) for turn in result.turns}) == 4
    assert all(
        turn.input_tokens == [10, 11]
        and turn.response_tokens == [12]
        and turn.response_logprobs == [-0.25]
        and turn.routing_replay_id == "exact-route"
        for turn in result.turns
    )
    paths = list(tmp_path.glob("*/result.json"))
    assert len(paths) == 4
    for path in paths:
        native = json.loads(path.read_text())
        assert native["finished_at"] and native["agent_info"]["name"] == "contract-agent"
    configs = [json.loads(path.read_text()) for path in tmp_path.glob("*/config.json")]
    assert len({native["trial_name"] for native in configs}) == 4
    for native in configs:
        assert native["verifier"]["disable"] is True


@pytest.mark.parametrize("stage,valid", [("setup", False), ("run", True)])
def test_native_trial_failure_classification_and_cleanup(adapters, stage, valid):
    with policy_endpoint() as ctx:
        result = asyncio.run(runner(harness_options={"fail_stage": stage})(ctx, batch()))
    assert bool(result.turns) == valid
    assert bool(result.invalid_items) != valid
    if valid:
        assert result.turns[0].item.record["harness_result"]["reward"] == 0
    assert ContractAgent.active == 0


def test_rollout_waits_for_all_environment_cleanup(adapters, monkeypatch):
    from areno.agent.harness_env import LocalEnvironment

    stopped = []
    stop = LocalEnvironment.stop

    async def delayed_stop(self, delete=True):
        await asyncio.sleep(0.02)
        await stop(self, delete=delete)
        assert not self._processes
        stopped.append(delete)

    monkeypatch.setattr(LocalEnvironment, "stop", delayed_stop)
    with policy_endpoint(concurrency=2) as ctx:
        result = asyncio.run(runner()(ctx, AgentBatch(batch().records, ["fix"], [[1]], 4)))
    assert len(result.turns) == 4 and not result.invalid_items
    assert stopped == [True] * 4
    assert ContractAgent.active == 0


@pytest.mark.parametrize("required,reward", [(True, 0.0), (False, 1.0)])
def test_native_trial_timeout_allows_dataset_owned_partial_reward(adapters, required, reward):
    with policy_endpoint() as ctx:
        result = asyncio.run(
            runner(require_agent_success=required, harness_options={"delay": 10})(
                ctx, batch(verify="assert True", timeout=0.1)
            )
        )
    assert not result.invalid_items and len(result.turns) == 1
    outcome = result.turns[0].item.record["harness_result"]
    assert outcome["timed_out"] and outcome["reward"] == reward
    assert ContractAgent.active == 0


def test_native_trial_cancel_stops_agent_and_releases_resources(adapters):
    async def check(ctx):
        task = asyncio.create_task(runner(harness_options={"delay": 10})(ctx, batch()))
        for _ in range(200):
            if ContractAgent.active:
                break
            await asyncio.sleep(0.01)
        assert ContractAgent.active == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ContractAgent.active == 0

    with policy_endpoint() as ctx:
        asyncio.run(check(ctx))


@pytest.mark.parametrize(
    "update", [{"timeout": float("nan")}, {"verify_timeout": -1}, {"max_turns": 1.5}, {"max_turns": 0}]
)
def test_bad_budgets_rejected_before_trial(adapters, update):
    with policy_endpoint() as ctx, pytest.raises(ValueError):
        asyncio.run(runner()(ctx, batch(**update)))


def test_native_harbor_pi_and_environment_options_are_reused(tmp_path):
    from harbor.agents.installed.pi import Pi
    from harbor.models.environment_type import EnvironmentType

    from areno.agent.harbor_agent import PreinstalledAgent
    from areno.agent.harness_trial import native_agent_config, native_environment_spec

    agent = PreinstalledAgent(
        logs_dir=tmp_path,
        native_config=native_agent_config("pi", kwargs={"thinking": "off", "max_turns": 20}).model_dump(),
    )
    assert type(agent.delegate) is Pi
    assert agent.capabilities.atif and agent.capabilities.resume
    assert agent.delegate.options.max_turns == 20
    spec = native_environment_spec(
        "docker", image="prebuilt-pi", task_env={"CUSTOM": "yes"}, kwargs={"keep_containers": False}
    )
    assert spec.config.type == EnvironmentType.DOCKER and spec.config.env["CUSTOM"] == "yes"
    assert spec.task_options["docker_image"] == "prebuilt-pi"
    custom = native_environment_spec(
        "areno.agent.harness_env:LocalEnvironment",
        env={"SOURCE": "custom"},
        task_env={"TASK": "value"},
    )
    assert custom.config.type is None
    assert custom.config.import_path == "areno.agent.harness_env:LocalEnvironment"
    assert custom.config.env == {"SOURCE": "custom", "TASK": "value"}


@pytest.mark.parametrize("cancel", [False, True])
def test_local_process_groups_reaped_on_timeout_and_cancellation(tmp_path, cancel):
    from areno.agent.harness_env import local_environment

    async def check():
        attempt = harness.HarnessAttempt(next(batch().iter_samples()), tmp_path, 1, 1)
        handle = local_environment(attempt)
        await handle.environment.start(False)
        pid_path = tmp_path / "pid"
        script = "import os,time,pathlib,subprocess; p=subprocess.Popen(['sleep','60']); pathlib.Path('pid').write_text(str(p.pid)); time.sleep(60)"
        task = asyncio.create_task(
            handle.environment.exec(
                shlex.join([sys.executable, "-c", script]), cwd=str(tmp_path), timeout_sec=10 if cancel else 0.3
            )
        )
        try:
            if cancel:
                for _ in range(100):
                    if pid_path.exists():
                        break
                    await asyncio.sleep(0.01)
                assert pid_path.exists()
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
                await task
            # A killed grandchild can briefly be a zombie; it must never be running.
            pid = int(pid_path.read_text())
            status = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
            ).stdout.strip()
            assert not status or status.startswith("Z")
            assert not handle.environment._processes
        finally:
            await handle.environment.stop()

    asyncio.run(check())
