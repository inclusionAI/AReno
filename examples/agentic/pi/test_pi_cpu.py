"""CPU contract tests; no pi installation, model download or GPU required."""

import asyncio
import json
import sys
import threading
import time
from contextlib import contextmanager
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import load_training_dataset
from run_agent import create_runner

from areno.agent.harness_base import write_workspace_files
from areno.agent.harness_proxy import ModelProxy
from areno.api.agentic import AgentBatch, RolloutSession
from areno.api.models import SamplingParams


def test_cli_preflight_accepts_example_file_hooks():
    from areno.cli.train import _validate_python_callable

    root = Path(__file__).resolve().parent
    for filename, symbol, arity in [
        ("dataset_loader.py", "load_training_dataset", 1),
        ("reward.py", "reward_fn", 1),
        ("run_agent.py", "run_agent", 2),
    ]:
        _validate_python_callable(
            root / filename, symbol, option_name="--test-hook", expected=f"{symbol}(...)", positional_args=arity
        )


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


def post(proxy, body, *, key=None, timeout=5):
    request = Request(
        proxy.base_url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key or proxy.api_key}"},
    )
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return response.read().decode()


def test_proxy_stream_tool_calls_and_exact_training_rows():
    async def check(ctx):
        async with ModelProxy(ctx) as proxy:
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


def test_stream_keepalive_survives_long_generation_without_changing_metadata(monkeypatch):
    def slow(response):
        time.sleep(0.3)
        return response

    async def check(ctx):
        async with ModelProxy(ctx) as proxy:
            body = {"messages": [{"role": "user", "content": "write a long page"}], "stream": True}
            wire = await asyncio.to_thread(post, proxy, body, timeout=0.1)
        assert wire.count(": keep-alive\n\n") >= 3
        chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: {")]
        assert chunks[0]["choices"][0]["delta"] == response_for(body)["choices"][0]["message"] | {
            "tool_calls": [dict(response_for(body)["choices"][0]["message"]["tool_calls"][0], index=0)]
        }
        assert len(proxy.trace) == 1 and proxy.trace[0][1] == response_for(body)
        assert not proxy.errors

    monkeypatch.setattr(ModelProxy, "_keepalive_interval", 0.02)
    with upstream(slow) as (ctx, _):
        asyncio.run(check(ctx))


def test_stream_errors_remain_sse_and_never_become_training_turns():
    async def check(ctx):
        async with ModelProxy(ctx) as proxy:
            wire = await asyncio.to_thread(
                post, proxy, {"messages": [{"role": "user", "content": "test"}], "stream": True}
            )
        assert wire.startswith(": keep-alive\n\n")
        assert "HTTP/1." not in wire
        error = next(json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: {"))
        assert "tokens/logprobs" in error["error"]["message"]
        assert proxy.errors and not proxy.trace

    with upstream(lambda response: {k: v for k, v in response.items() if k != "areno"}) as (ctx, _):
        asyncio.run(check(ctx))


@pytest.mark.parametrize("stream", [True, False])
def test_auth_limits_and_sample_isolation(stream):
    async def check(ctx):
        async with ModelProxy(ctx, max_turns=1) as first, ModelProxy(ctx) as second:
            body = {"messages": [{"role": "user", "content": "first"}], "stream": stream}
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


def test_tool_reasoning_is_streamed_and_replayed_with_exact_metadata():
    class Tokenizer:
        def decode(self, tokens):
            if tokens == [101, 102]:
                return "<role>ASSISTANT</role>\n<think>"
            if tokens == [31, 32]:
                return "Inspect the implementation before changing it.</think><tool_call>write</tool_call>"
            return "done"

    async def check(ctx):
        ctx.get_tokenizer = lambda: Tokenizer()
        async with ModelProxy(ctx) as proxy:
            body = {"messages": [{"role": "user", "content": "fix it"}], "stream": True}
            wire = await asyncio.to_thread(post, proxy, body)
            chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: {")]
            assistant = dict(chunks[0]["choices"][0]["delta"])
            assert assistant["reasoning_content"] == "Inspect the implementation before changing it."
            assert assistant["content"] is None
            assert proxy.trace[0][1]["areno"] == response_for(body)["areno"]
            body["messages"] += [assistant, {"role": "tool", "tool_call_id": "call_1", "content": "written"}]
            await asyncio.to_thread(post, proxy, body)
        item = next(AgentBatch([{}], ["p"], [[0]], 1).iter_samples())
        turn = proxy.turns(item)[0]
        assert turn.input_tokens == [101, 102]
        assert turn.response_tokens == [31, 32]
        assert turn.response_logprobs == [-0.1, -0.2]

    with upstream() as (ctx, requests):
        asyncio.run(check(ctx))
    assert requests[1]["messages"][1]["reasoning_content"] == "Inspect the implementation before changing it."


def test_missing_metadata_fails_instead_of_retokenizing():
    async def check(ctx):
        async with ModelProxy(ctx) as proxy:
            with pytest.raises(HTTPError) as exc:
                await asyncio.to_thread(post, proxy, {"messages": [{"role": "user", "content": "test"}]})
            assert exc.value.code == 400
        assert not proxy.trace
        assert "tokens/logprobs" in proxy.errors[0]

    with upstream(lambda response: {k: v for k, v in response.items() if k != "areno"}) as (ctx, _):
        asyncio.run(check(ctx))


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError])
@pytest.mark.parametrize("stream", [True, False])
def test_client_disconnect_preserves_completed_turn(error, stream, monkeypatch):
    def disconnected(*args):
        raise error("pi stopped after its task budget")

    async def check(ctx):
        async with ModelProxy(ctx) as proxy:
            monkeypatch.setattr(proxy, "_stream" if stream else "_json", disconnected)
            body = {"messages": [{"role": "user", "content": "fix it"}], "stream": stream}
            if stream:
                wire = await asyncio.to_thread(post, proxy, body)
                assert wire.startswith(": keep-alive\n\n") and "data:" not in wire
            else:
                with pytest.raises(RemoteDisconnected):
                    await asyncio.to_thread(post, proxy, body)
        assert not proxy.errors
        assert len(proxy.trace) == 1
        assert proxy.trace[0][1]["areno"]["response_logprobs"] == [-0.1, -0.2]

    with upstream() as (ctx, _):
        asyncio.run(check(ctx))


def test_loader_and_workspace_reject_invalid_inputs(tmp_path):
    rows = load_training_dataset(
        "unused", default_loader=lambda _: [{"problem_statement": "fix", "files": {}, "verify": "assert True"}]
    )
    assert rows[0]["prompt"] == "fix"
    with pytest.raises(ValueError, match="prompt"):
        load_training_dataset("unused", default_loader=lambda _: [{"files": {}}])
    for path in ("../outside", "/absolute", ".pi/settings.json"):
        with pytest.raises(ValueError):
            write_workspace_files(tmp_path, {path: "bad"}, reserved_roots=(".pi",))


def test_example_uses_native_pi_and_explicit_docker_image(monkeypatch):
    monkeypatch.setenv("ARENO_HARBOR_IMAGE", "my-preinstalled-pi")
    monkeypatch.setenv("ARENO_HARBOR_ENVIRONMENT_OPTIONS", '{"task_env":{"CUSTOM":"yes"}}')
    workflow = create_runner()
    assert workflow.harness == "pi" and workflow.environment == "docker"
    assert workflow.environment_options["image"] == "my-preinstalled-pi"
    assert workflow.environment_options["task_env"]["CUSTOM"] == "yes"
    assert workflow.preinstalled and not workflow.require_agent_success
    assert workflow.harness_options["kwargs"]["model_api"] == "openai-completions"


def test_native_pi_exports_atif_and_usage_from_session(tmp_path):
    from harbor.agents.installed.pi import Pi
    from harbor.models.agent.context import AgentContext
    from harbor.models.trajectories import Trajectory

    agent = Pi(logs_dir=tmp_path, model_name="openai/policy", thinking="off")
    sessions = tmp_path / "pi/sessions"
    sessions.mkdir(parents=True)
    events = [
        {"type": "session", "id": "test-session"},
        {"type": "message", "id": "user", "message": {"role": "user", "content": "fix"}},
        {
            "type": "message",
            "id": "assistant",
            "parentId": "user",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "model": "policy",
                "provider": "openai",
                "usage": {"input": 3, "output": 2, "totalTokens": 5},
            },
        },
    ]
    (sessions / "session.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    context = AgentContext()
    agent.populate_context_post_run(context)
    trajectory = Trajectory.model_validate_json((tmp_path / "trajectory.json").read_text())
    assert trajectory.session_id == "test-session" and trajectory.steps[-1].message == "done"
    assert context.n_input_tokens == 3 and context.n_output_tokens == 2
