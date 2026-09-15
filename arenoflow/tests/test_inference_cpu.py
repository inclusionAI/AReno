"""Inference proxy uses the recorded deployment and never persists test inputs."""

import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from arenoflow.inference import test_deployment as run_inference


def deployment():
    return {"kind": "deployment", "status": "ready", "endpoint": "https://example.modal.run/v1"}


def test_request_uses_recorded_endpoint_and_ephemeral_key(monkeypatch):
    requests = []

    def send(request, timeout):
        requests.append(request)
        assert timeout == 120
        return io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 3}}).encode()
        )

    monkeypatch.setattr("arenoflow.inference.urllib.request.build_opener", lambda *_: SimpleNamespace(open=send))
    job = deployment()
    result = run_inference(
        job,
        {
            "api_key": "private-test-key",
            "prompt": "test",
            "system": "Be concise",
            "base_url": "https://ignored.example",
            "max_tokens": 32,
            "temperature": 0,
        },
    )
    request = requests[0]
    assert request.full_url == "https://example.modal.run/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer private-test-key"
    payload = json.loads(request.data)
    assert payload["messages"] == [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "test"}]
    assert payload["max_tokens"] == 32 and payload["temperature"] == 0 and payload["stream"] is False
    assert result["response"]["choices"][0]["message"]["content"] == "hello"
    assert "private-test-key" not in json.dumps(job) + json.dumps(result)


@pytest.mark.parametrize("changes", [{"status": "starting"}, {"status": "succeeded"}, {"kind": "training"}])
def test_only_ready_deployments_accept_inference(changes):
    with pytest.raises(ValueError, match="ready"):
        run_inference({**deployment(), **changes}, {"api_key": "key", "prompt": "hello"})


@pytest.mark.parametrize(
    "body",
    [
        {"api_key": "key\r\nInjected: yes", "prompt": "hello"},
        {"api_key": "key", "prompt": ""},
        {"api_key": "key", "prompt": "hello", "max_tokens": 0},
        {"api_key": "key", "prompt": "hello", "temperature": float("nan")},
    ],
)
def test_invalid_requests_are_rejected_before_network(body):
    with pytest.raises(ValueError):
        run_inference(deployment(), body)


def test_authentication_error_does_not_echo_upstream_body(monkeypatch):
    def send(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://example.modal.run", 401, "private-test-key", {}, io.BytesIO(b"private-test-key")
        )

    monkeypatch.setattr("arenoflow.inference.urllib.request.build_opener", lambda *_: SimpleNamespace(open=send))
    with pytest.raises(ValueError, match="authentication failed") as error:
        run_inference(deployment(), {"api_key": "private-test-key", "prompt": "hello"})
    assert "private-test-key" not in str(error.value)
