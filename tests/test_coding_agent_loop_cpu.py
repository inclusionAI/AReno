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


class _NoToolCompletions:
    async def create(self, **kwargs):
        del kwargs
        message = type("Message", (), {"content": "What value should I use?", "tool_calls": None})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _NoToolClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _NoToolCompletions()})()


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


def test_assistant_message_omits_empty_tool_calls():
    message = type("Message", (), {"content": "Need one more value.", "tool_calls": None})()
    choice = type("Choice", (), {"message": message})()
    response = type("Response", (), {"choices": [choice]})()

    assistant = agent_loop._assistant_message_from_response(response)

    assert assistant == {"role": "assistant", "content": "Need one more value."}


def test_no_tool_assistant_can_delegate_to_interaction_hook(tmp_path):
    phases = []
    messages = [{"role": "user", "content": "run"}]
    task = {"instance_id": "local", "repo": str(tmp_path)}
    item = type("Item", (), {"record": task, "prompt": "run"})()
    workspace = type("Workspace", (), {"task": task})()

    async def interaction_hook(current_messages, phase):
        phases.append(phase)
        if phase == "assistant_no_tool":
            current_messages.append({"role": "user", "content": "User runtime hint:\n128"})
            return True
        return True

    asyncio.run(
        agent_loop.run_conversation_turns(
            client=_NoToolClient(),
            item=item,
            workspace=workspace,
            model="policy",
            messages=messages,
            max_turns=1,
            record_trajectory=False,
            interaction_hook=interaction_hook,
        )
    )

    assert phases == ["before_turn", "assistant_no_tool"]
    assert messages[-1] == {"role": "user", "content": "User runtime hint:\n128"}
