"""Agent entrypoint for one-episode elevator-dispatch tool-call rollouts.

The policy returns a full dispatch episode as a single ``dispatch`` tool call.
The agent itself stays a thin model caller -- exactly like the Tic-Tac-Toe and
2048 examples -- and AReno replays the actions deterministically in the reward
function.
"""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are an elevator dispatcher. Pick a sequence of single-letter actions to "
    "deliver every passenger while minimizing wait. Call the dispatch tool with a "
    "string of letters: U move up, D move down, O open the door to let passengers "
    "off and on, C close the door. The door must be open to exchange passengers and "
    "closed to move. Keep capacity; invalid actions are penalized."
)

DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "dispatch",
        "description": "Submit a full elevator-dispatch episode as an ordered action sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "string",
                    "description": "Ordered actions, one letter each from U/D/O/C, e.g. 'OCUUOC'.",
                    "pattern": "^[UDOC]+$",
                }
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each building."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The elevator agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("elevator agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        tool_choice = {"type": "function", "function": {"name": "dispatch"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[DISPATCH_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[DISPATCH_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()
