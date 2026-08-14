"""Audiovisual event temporal-grounding rollout agent."""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You localize a named audiovisual event in a 10-second video. Use synchronized visible and audible evidence, "
    "not unrelated background activity. Return exactly one report_event_range tool call. Times are seconds from "
    "the clip start and must satisfy 0 <= start_seconds < end_seconds <= 10."
)

REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "report_event_range",
        "description": "Report the temporal range of the requested event in the audiovisual clip.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Event start time in seconds from the clip beginning.",
                },
                "end_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Event end time in seconds from the clip beginning.",
                },
            },
            "required": ["start_seconds", "end_seconds"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("This example requires the openai and httpx packages included with AReno.") from exc

    items = list(batch.iter_samples())
    max_connections = max(1, ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(1800.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    semaphore = asyncio.Semaphore(max_connections)

    async def run_one(item):
        record = item.record
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": str(record["video_path"])}},
                    {"type": "audio_url", "audio_url": {"url": str(record["audio_path"])}},
                    {"type": "text", "text": item.prompt},
                ],
            },
        ]
        tool_choice = {"type": "function", "function": {"name": "report_event_range"}}
        async with semaphore:
            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                tools=[REPORT_TOOL],
                tool_choice=tool_choice,
                stream=False,
            )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[REPORT_TOOL],
            tool_choice=tool_choice,
        )

    try:
        logger.info("AVE temporal-grounding requests=%d concurrency=%d", len(items), max_connections)
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()
