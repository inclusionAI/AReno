"""Exercise the real HTTP boundary without Modal credentials or GPU work."""

import base64
import hashlib
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from areno.dashboard.flow.assets import referenced_uploads, save_upload
from areno.dashboard.flow.server import Application, handler_for


@pytest.fixture
def server(tmp_path):
    app = Application(tmp_path)
    instance = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance, app
    instance.shutdown()
    instance.server_close()
    thread.join()


def call(server, method, path, body=None, headers=None):
    instance, app = server
    conn = http.client.HTTPConnection("127.0.0.1", instance.server_port)
    conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, json.loads(data)


def test_csrf_and_same_origin_required(server):
    status, bootstrap = call(server, "GET", "/api/bootstrap")
    assert status == 200 and "csrf" in bootstrap
    assert "token_secret" not in json.dumps(bootstrap)
    assert call(server, "POST", "/api/connect", {})[0] == 403
    headers = {
        "X-Arenoflow-CSRF": bootstrap["csrf"],
        "Content-Type": "application/json",
        "Origin": "https://evil.example",
    }
    assert call(server, "POST", "/api/connect", {}, headers)[0] == 403
    assert call(server, "GET", "/api/bootstrap", headers={"Host": "evil.example"})[0] == 403


def test_preview_and_billing_without_credentials(server):
    instance, app = server
    headers = {"X-Arenoflow-CSRF": app.csrf, "Content-Type": "application/json"}
    request = {
        "model": {"adapter": "qwen3", "checkpoint": "Qwen/Qwen3-0.6B"},
        "stages": [{"algo": "sft", "params": {"dataset_path": "mydata"}}],
    }
    status, output = call(server, "POST", "/api/preview", request, headers)
    assert status == 200 and output["commands"][0].startswith("areno train")
    assert call(server, "POST", "/api/jobs", request, headers)[0] == 400
    assert call(server, "GET", "/api/billing")[0] == 400
    assert call(server, "GET", "/api/missing")[0] == 404


def test_dataset_upload_and_path_traversal(tmp_path):
    content = b'{"prompt":"hi","response":"hello"}\n'
    result = save_upload(tmp_path, "../../mydata.jsonl", base64.b64encode(content).decode())
    assert result["path"].endswith(hashlib.sha256(content).hexdigest() + ".jsonl")
    files = referenced_uploads({"args": [result["path"]]}, tmp_path)
    assert files[0][0].read_bytes() == content
    with pytest.raises(ValueError):
        referenced_uploads({"args": ["/artifacts/uploads/../secret"]}, tmp_path)
    with pytest.raises(ValueError):
        save_upload(tmp_path, "data.exe", base64.b64encode(content).decode())
