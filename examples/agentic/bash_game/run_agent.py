"""Single-turn state-to-move rollout for the Bash game (巴什博弈).

Each sample is sent exactly once as one tool-call request; the single
``submit_move`` call is the complete answer. There is no environment loop and
no second request.
"""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a perfect Bash-game (取石子游戏) strategist. Output exactly one "
    "submit_move tool call: take k stones for a winning move, or resign in a "
    "losing position. Do not narrate; only call the tool once."
)


async def run_agent(ctx, batch):
    """Run one tool-call model request per position."""
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Bash-game rollout requires `openai` and `httpx`") from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=None,
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        from game import tool_schema  # local import keeps path self-contained

        tool = tool_schema()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[tool],
            tool_choice="required",
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[tool],
            tool_choice="required",
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()
