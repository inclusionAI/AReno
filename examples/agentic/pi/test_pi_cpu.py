"""CPU contract tests; no pi installation, model download or GPU required."""

import asyncio
import json
import os
import signal
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import load_training_dataset
from pi_proxy import PiProxy
from reward import reward_fn
from run_agent import run_agent, run_process, write_files

from areno.api.agentic import AgentBatch, RolloutSession
from areno.api.models import SamplingParams


def response_for(body):
    second = body["messages"][-1]["role"] == "tool"
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "write", "arguments": '{"path":"solution.py","content":"answer = 42"}'},
    }
    message = (
        {"role": "assistant", "content": "done"}
        if second
        else {"role": "assistant", "content": None, "tool_calls": [call]}
    )
    return {
        "id": "chatcmpl-test",
        "created": 1,
        "model": "policy",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop" if second else "tool_calls"}],
        "usage": {"prompt_tokens": 3 if second else 2, "completion_tokens": 2, "total_tokens": 5 if second else 4},
        "areno": {
            "input_tokens": [201, 202, 203] if second else [101, 102],
            "response_tokens": [33, 34] if second else [31, 32],
            "response_logprobs": [-0.1, -0.2],
        },
    }


@contextmanager
def upstream(transform=lambda response: response):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            assert self.headers["Authorization"] == "Bearer upstream-key"
            data = json.dumps(transform(response_for(body))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield (
            SimpleNamespace(
                base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                api_key="upstream-key",
                max_running_prompts=2,
            ),
            requests,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def post(proxy, body, *, key=None):
    request = Request(
        proxy.base_url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key or proxy.api_key}"},
    )
    with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
        return response.read().decode()


def test_proxy_stream_tool_calls_and_exact_training_rows():
    async def check(ctx):
        async with PiProxy(ctx) as proxy:
            body = {
                "model": "policy",
                "messages": [{"role": "user", "content": "fix it"}],
                "stream": True,
                "max_completion_tokens": 99999,
                "stream_options": {"include_usage": True},
            }
            wire = await asyncio.to_thread(post, proxy, body)
            chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: {")]
            assert wire.endswith("data: [DONE]\n\n")
            assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
            assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
            assert chunks[2]["usage"]["completion_tokens"] == 2
            assert '"areno"' not in wire
            body["messages"] = [
                {"role": "user", "content": "new compacted context"},
                {"role": "tool", "tool_call_id": "call_1", "content": "written"},
            ]
            await asyncio.to_thread(post, proxy, dict(body, stream=False))
        item = next(AgentBatch([{}], ["p"], [[0]], 1).iter_samples())
        turns = proxy.turns(item)
        assert turns[1].messages == body["messages"]
        assert turns[0].response_logprobs == [-0.1, -0.2]
        tokenizer = SimpleNamespace(decode=lambda tokens: "done")  # No encode: re-tokenization must not happen.
        trainer = SimpleNamespace(get_tokenizer=lambda: tokenizer, dp_size=lambda: 1)
        session = RolloutSession(trainer, sampling_params=SamplingParams())
        samples = [session._sample_from_trajectory_turn(turn) for turn in turns]
        rows = session._train_rows_from_samples(samples)
        assert rows.token_rows == [[101, 102, 31, 32], [201, 202, 203, 33, 34]]
        assert rows.loss_masks == [[False, False, True, True], [False, False, False, True, True]]
        assert rows.rollout_logprobs[1] == [0, 0, 0, -0.1, -0.2]
        assert not proxy.errors

    with upstream() as (ctx, requests):
        asyncio.run(check(ctx))
    assert all(body["stream"] is False for body in requests)
    assert all("max_completion_tokens" not in body and "stream_options" not in body for body in requests)


def test_auth_limits_and_sample_isolation():
    async def check(ctx):
        async with PiProxy(ctx, max_turns=1) as first, PiProxy(ctx) as second:
            body = {"messages": [{"role": "user", "content": "first"}]}
            with pytest.raises(HTTPError) as exc:
                await asyncio.to_thread(post, first, body, key=second.api_key)
            assert exc.value.code == 401
            await asyncio.gather(
                asyncio.to_thread(post, first, body),
                asyncio.to_thread(post, second, {"messages": [{"role": "user", "content": "second"}]}),
            )
            with pytest.raises(HTTPError) as exc:
                await asyncio.to_thread(post, first, body)
            assert exc.value.code == 429
        assert first.limit_reached and not second.limit_reached
        assert not first.errors
        assert first.trace[0][0]["messages"][0]["content"] == "first"
        assert second.trace[0][0]["messages"][0]["content"] == "second"

    with upstream() as (ctx, requests):
        asyncio.run(check(ctx))
    assert len(requests) == 2


def test_missing_metadata_fails_instead_of_retokenizing():
    async def check(ctx):
        async with PiProxy(ctx) as proxy:
            with pytest.raises(HTTPError) as exc:
                await asyncio.to_thread(post, proxy, {"messages": [{"role": "user", "content": "test"}]})
            assert exc.value.code == 400
        assert not proxy.trace
        assert "tokens/logprobs" in proxy.errors[0]

    with upstream(lambda response: {k: v for k, v in response.items() if k != "areno"}) as (ctx, _):
        asyncio.run(check(ctx))


@pytest.fixture
def fake_pi(tmp_path, monkeypatch):
    executable = tmp_path / "pi-fixture"
    executable.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, pathlib, sys, urllib.request
config = json.loads((pathlib.Path(os.environ["PI_CODING_AGENT_DIR"]) / "models.json").read_text())
provider = config["providers"]["areno"]
assert "--" not in sys.argv
assert sys.argv[-1].startswith("Task:\n")
messages = [{"role": "user", "content": sys.argv[-1]}]
for turn in range(2):
    request = urllib.request.Request(provider["baseUrl"] + "/chat/completions",
        data=json.dumps({"model":"policy", "messages":messages, "stream":True, "tools":[{"type":"function", "function":{"name":"write", "parameters":{"type":"object", "properties":{"path":{"type":"string"}, "content":{"type":"string"}}}}}]}).encode(),
        headers={"Content-Type":"application/json", "Authorization":"Bearer " + provider["apiKey"]})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request) as response:
        wire = response.read().decode()
    chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: {")]
    delta = chunks[0]["choices"][0]["delta"]
    if delta.get("tool_calls"):
        call = delta["tool_calls"][0]
        args = json.loads(call["function"]["arguments"])
        pathlib.Path(args["path"]).write_text(args["content"])
        messages += [{"role":"assistant", "content":None, "tool_calls":[call]},
                     {"role":"tool", "tool_call_id":call["id"], "content":"written"}]
"""
    )
    executable.chmod(0o755)
    monkeypatch.setenv("ARENO_PI_EXECUTABLE", str(executable))
    return executable


def test_runner_verifier_reward_and_isolated_records(fake_pi):
    records = [
        {
            "prompt": "fix",
            "files": {"solution.py": "answer = 0"},
            "verify": "from solution import answer; assert answer == 42",
        },
        {
            "prompt": "fix",
            "files": {"solution.py": "answer = 0"},
            "verify": "from solution import answer; assert answer == 999",
        },
    ]
    batch = AgentBatch(records, ["fix", "fix"], [[1], [1]], 2)
    with upstream() as (ctx, requests):
        trajectory = asyncio.run(run_agent(ctx, batch))
    assert len(requests) == 8
    assert len(trajectory.turns) == 8
    assert trajectory.invalid_items == []
    assert all("pi_result" not in record for record in records)
    items = [trajectory.turns[i].item for i in range(0, 8, 2)]
    assert len({id(item.record) for item in items}) == 4
    assert [reward_fn(SimpleNamespace(source_record=item.record)) for item in items] == [1, 1, 0, 0]
    assert [item.sample_index for item in items] == [0, 1, 0, 1]
    assert all(trajectory.turns[i].item is trajectory.turns[i + 1].item for i in range(0, 8, 2))


@pytest.mark.parametrize("cancel", [False, True])
def test_process_timeout_and_cancellation_reap_process(tmp_path, cancel):
    pid_path = tmp_path / "pid"

    async def check():
        task = asyncio.create_task(
            run_process(
                [
                    sys.executable,
                    "-c",
                    "import os,time,pathlib; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(60)",
                ],
                cwd=tmp_path,
                env=dict(os.environ),
                timeout=10 if cancel else 0.3,
                log_path=tmp_path / "log",
            )
        )
        if cancel:
            for _ in range(100):
                if pid_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_path.exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task == (None, True)

    asyncio.run(check())
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), signal.SIGCONT)


def test_loader_and_workspace_reject_invalid_inputs(tmp_path):
    rows = load_training_dataset(
        "unused", default_loader=lambda _: [{"problem_statement": "fix", "files": {}, "verify": "assert True"}]
    )
    assert rows[0]["prompt"] == "fix"
    with pytest.raises(ValueError, match="verify"):
        load_training_dataset("unused", default_loader=lambda _: [{"prompt": "fix", "files": {}}])
    for path in ("../outside", "/absolute", ".pi/settings.json"):
        with pytest.raises(ValueError):
            write_files(tmp_path, {path: "bad"})


class FakePolicy:
    """CPU policy double behind the unmodified AReno rollout HTTP server."""

    def __init__(self):
        self.events = []
        self.tokenizer = SimpleNamespace(
            chat_template="test-template",
            apply_chat_template=lambda messages, **kwargs: list(json.dumps(messages).encode()),
            decode=lambda tokens, **kwargs: (
                '{"name":"write","arguments":{"path":"solution.py","content":"answer = 42"}}'
                if tokens == [31]
                else "done"
            ),
        )

    def get_tokenizer(self):
        return self.tokenizer

    def dp_size(self):
        return 1

    async def begin_rollout_session_async(self):
        self.events.append("begin")

    async def end_rollout_session_async(self):
        self.events.append("end")

    async def rollout_token_batch_async(self, prompts, n_samples, params):
        self.events.append("rollout")
        results = []
        for prompt in prompts:
            messages = json.loads(bytes(prompt))
            second = messages[-1]["role"] == "tool"
            results.append(
                SimpleNamespace(sequences=[SimpleNamespace(resp_tokens=[33 if second else 31], resp_logprobs=[-0.25])])
            )
        return results


def test_existing_rollout_proxy_roundtrip(fake_pi):
    policy = FakePolicy()

    async def check():
        async with RolloutSession(policy, sampling_params=SamplingParams(), max_running_prompts=2) as ctx:
            row = {"prompt": "fix", "files": {}, "verify": "from solution import answer; assert answer == 42"}
            result = await run_agent(ctx, AgentBatch([row], ["fix"], [[1]], 2))
            assert not result.invalid_items
            assert len(result.turns) == 4
            assert result.turns[0].parsed_tool_calls[0]["function"]["name"] == "write"
            assert result.turns[0].item.record["pi_result"]["reward"] == 1
            samples = [ctx._sample_from_trajectory_turn(turn) for turn in result.turns]
            rows = ctx._train_rows_from_samples(samples)
            assert [row[-1] for row in rows.token_rows] == [31, 33, 31, 33]
            assert all(row[-1] == -0.25 for row in rows.rollout_logprobs)

    asyncio.run(check())
    assert policy.events == ["begin", "rollout", "rollout", "rollout", "rollout", "end"]
