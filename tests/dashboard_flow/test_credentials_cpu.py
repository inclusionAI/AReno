"""Persistence and reconnect tests use fake Modal clients only."""

import json
import stat
from unittest.mock import Mock

import pytest

from areno.dashboard.flow.catalog import catalog
from areno.dashboard.flow.controller import Controller
from areno.dashboard.flow.credentials import Credentials
from areno.dashboard.flow.server import Application
from areno.dashboard.flow.store import Store


def controller(directory):
    instance = Controller(Store(directory), catalog(), provider_factory=lambda *_: Mock())
    instance.env_credentials = ("", "")
    return instance


def test_saved_credentials_are_private_and_never_returned(tmp_path):
    app = Application(tmp_path, controller=controller(tmp_path))
    app.post("/api/connect", {"token_id": "test-id", "token_secret": "test-secret"})
    assert stat.S_IMODE(app.credentials.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert app.credentials.load() == ("test-id", "test-secret")
    bootstrap = app.get("/api/bootstrap", {})
    assert bootstrap["credentials_saved"] and bootstrap["connected"]
    assert "test-secret" not in json.dumps(bootstrap)
    assert "test-id" not in json.dumps(bootstrap)


def test_saved_credentials_reconnect_after_restart(tmp_path):
    Credentials(tmp_path).save("test-id", "test-secret")
    backend = controller(tmp_path)
    app = Application(tmp_path, controller=backend)
    app.reconnect_thread.join(timeout=3)
    assert backend.provider is not None
    assert backend.credentials == ("test-id", "test-secret")
    assert app.get("/api/bootstrap", {})["reconnecting"] is False


def test_failed_replacement_keeps_previous_credentials(tmp_path):
    app = Application(tmp_path, controller=controller(tmp_path))
    app.post("/api/connect", {"token_id": "test-id", "token_secret": "test-secret"})
    provider = Mock()
    provider.check.side_effect = ValueError("invalid credentials")
    app.controller.provider_factory = lambda *_: provider
    with pytest.raises(ValueError):
        app.post("/api/connect", {"token_id": "bad-id", "token_secret": "bad-secret"})
    assert app.credentials.load() == ("test-id", "test-secret")


def test_session_only_and_forgetting_credentials(tmp_path):
    app = Application(tmp_path, controller=controller(tmp_path))
    app.post("/api/connect", {"token_id": "test-id", "token_secret": "test-secret", "remember": False})
    assert not app.credentials.saved
    app.post("/api/connect", {"token_id": "test-id", "token_secret": "test-secret"})
    app.post("/api/forget-credentials", {})
    assert not app.credentials.saved
    assert app.controller.provider is not None


def test_loose_permissions_refused(tmp_path):
    credentials = Credentials(tmp_path)
    credentials.save("test-id", "test-secret")
    credentials.path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        credentials.load()


def test_reconnect_retries_transient_failure_without_leaking_tokens(tmp_path):
    app = Application(tmp_path, controller=controller(tmp_path))
    app.credentials.save("test-id", "test-secret")
    broken = Mock()
    broken.check.side_effect = ValueError("temporary failure test-secret")
    app.controller.provider_factory = Mock(side_effect=[broken, Mock()])
    app.reconnect_stop.wait = Mock(return_value=False)
    app._auto_reconnect()
    assert app.controller.provider is not None
    assert app.controller.provider_factory.call_count == 2
    assert app.connection_error is None
