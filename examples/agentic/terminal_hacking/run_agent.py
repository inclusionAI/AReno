"""Single-turn state-to-candidates rollout for terminal hacking."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    candidate_filter_request_messages,
    candidate_filter_session,
    candidate_tool,
    extra_body_for_base_url,
    tool_choice_for_base_url,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def run_agent(ctx, batch):
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Terminal hacking requires openai and httpx") from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=None,
    )
    base_url = ctx.get_base_url()
    client = AsyncOpenAI(base_url=base_url, api_key=ctx.api_key, http_client=http_client, max_retries=0)
    try:
        turns = await asyncio.gather(*(_run_filter(item, client, base_url=base_url) for item in items))
        return AgentTrajectory(turns=list(turns))
    finally:
        await client.close()


async def _run_filter(item, client, *, base_url: str = "") -> AgentTrajectoryTurn:
    session = candidate_filter_session(item.record)
    state = session.public_state()
    active = state["active_candidates"]
    tool = candidate_tool(active)
    request_messages = candidate_filter_request_messages(item.prompt, state)
    tool_choice = tool_choice_for_base_url(base_url)
    request_kwargs = {
        "model": "policy",
        "messages": request_messages,
        "tools": [tool],
        "tool_choice": tool_choice,
        "stream": False,
    }
    extra_body = extra_body_for_base_url(base_url)
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body
    response = await client.chat.completions.create(**request_kwargs)
    assistant = _assistant_message(response)
    if _parse_candidates(assistant, active) is None:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        logger.warning(
            "Terminal-hacking model returned no executable submit_candidates call: "
            "finish_reason=%r completion_tokens=%r content=%r tool_calls=%r",
            getattr(choice, "finish_reason", None),
            getattr(usage, "completion_tokens", None),
            str(assistant.get("content") or "")[-600:],
            assistant.get("tool_calls"),
        )
    return AgentTrajectoryTurn(
        item=item,
        messages=request_messages,
        response=response,
        tools=[tool],
        tool_choice=tool_choice,
    )


def _assistant_message(response) -> dict:
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])
        ],
    }


def _parse_candidates(assistant: dict, active_candidates: list[str]) -> list[str] | None:
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "submit_candidates":
        return None
    try:
        arguments = json.loads(calls[0]["function"].get("arguments") or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict) or set(arguments) != {"candidates"}:
        return None
    candidates = arguments["candidates"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(candidate, str) for candidate in candidates)
        or len(candidates) != len(set(candidates))
    ):
        return None
    active = set(active_candidates)
    if any(candidate not in active for candidate in candidates):
        return None
    return candidates
