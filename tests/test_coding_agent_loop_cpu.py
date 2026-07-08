from __future__ import annotations

import asyncio

from areno.agentic.coding import agent_loop


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise TimeoutError("temporary timeout")
        return {"ok": True, "kwargs": kwargs}


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def test_chat_completion_retry_recovers_after_transient_failures(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(agent_loop.asyncio, "sleep", fake_sleep)
    client = _FakeClient()

    response = asyncio.run(agent_loop.create_chat_completion_with_retry(client, model="policy", messages=[]))

    assert response["ok"] is True
    assert client.chat.completions.calls == 3
    assert sleeps == [1.0, 2.0]
