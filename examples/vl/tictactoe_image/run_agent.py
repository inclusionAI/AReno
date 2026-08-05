"""Agent entrypoint for Qwen3.5-VL tic-tac-toe image rollouts."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a careful vision-language Tic-Tac-Toe player. You play X. "
    "Inspect the board image and choose exactly one legal empty square by calling the choose_square tool. "
    "Digits on the board are empty square labels, not marks. "
    "Win immediately if possible; otherwise block any immediate O win."
)

CHOOSE_SQUARE_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_square",
        "description": "Choose the next Tic-Tac-Toe square for X.",
        "parameters": {
            "type": "object",
            "properties": {
                "square": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "The square number to place X in.",
                }
            },
            "required": ["square"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one image chat-completion request for each board."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The VL tic-tac-toe agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("VL Tic-Tac-Toe agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        image_base64 = _record_image_base64(item.record)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": item.prompt},
                ],
            },
        ]
        tool_choice = {"type": "function", "function": {"name": "choose_square"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[CHOOSE_SQUARE_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[CHOOSE_SQUARE_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()


def _record_image_base64(record: dict) -> str:
    image_base64 = record.get("image_base64")
    if image_base64:
        return str(image_base64)
    image_path = record.get("image_path")
    if not image_path:
        raise ValueError("VL tic-tac-toe agent rows require image_base64 or image_path")
    with Path(str(image_path)).expanduser().open("rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")
