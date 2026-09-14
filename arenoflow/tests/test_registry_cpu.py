"""Registry discovery uses public metadata, never cloud compute or fixed tags."""

import io
import json
import urllib.error

import pytest

from arenoflow import provider


class Response(io.BytesIO):
    def __init__(self, payload=None, headers=None):
        super().__init__(json.dumps(payload or {}).encode())
        self.headers = headers or {}


def test_latest_tag_is_numeric_and_follows_pagination(monkeypatch):
    urls = []

    def open_url(request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        urls.append(url)
        if "/token?" in url:
            return Response({"token": "public-registry-token"})
        if "last=" in url:
            return Response({"tags": ["v0.0.10", "v1.0.0-rc1"]})
        return Response(
            {"tags": ["latest", "v0.0.9"]}, {"Link": '</v2/inclusionai/areno/tags/list?n=100&last=v0.0.9>; rel="next"'}
        )

    monkeypatch.setattr(provider.urllib.request, "urlopen", open_url)
    result = provider.latest_image("ghcr.io/inclusionai/areno:latest")
    assert result["reference"] == "ghcr.io/inclusionai/areno:v0.0.10"
    assert len(urls) == 3
    assert result["checked_at"] > 0


def test_missing_latest_resolves_published_release_digest(monkeypatch):
    digest = "sha256:" + "a" * 64

    def open_url(request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        if "/token?" in url:
            return Response({"token": "public-token"})
        if url.endswith("/manifests/latest") or url.endswith("/manifests/missing"):
            raise urllib.error.HTTPError(url, 404, "Not found", {}, None)
        if "/tags/list" in url:
            return Response({"tags": ["v0.0.7", "v0.0.8"]})
        assert url.endswith("/manifests/v0.0.8")
        return Response(headers={"Docker-Content-Digest": digest})

    monkeypatch.setattr(provider.urllib.request, "urlopen", open_url)
    assert provider.resolve_image("ghcr.io/inclusionai/areno:latest") == "ghcr.io/inclusionai/areno@" + digest
    with pytest.raises(urllib.error.HTTPError):
        provider.resolve_image("ghcr.io/inclusionai/areno:missing")


def test_discovery_cache_expires_without_modal_credentials(tmp_path, monkeypatch):
    from arenoflow.server import Application

    calls = []

    def latest(reference):
        calls.append(reference)
        return {"reference": "ghcr.io/inclusionai/areno:v1.2.3", "tag": "v1.2.3", "checked_at": 100}

    monkeypatch.setattr("arenoflow.server.latest_image", latest)
    monkeypatch.setattr("arenoflow.server.time.time", lambda: 100)
    app = Application(tmp_path)
    assert app.get("/api/images/latest", {})["tag"] == "v1.2.3"
    app.get("/api/images/latest", {})
    assert len(calls) == 1
    monkeypatch.setattr("arenoflow.server.time.time", lambda: 161)
    app.get("/api/images/latest", {})
    assert len(calls) == 2
