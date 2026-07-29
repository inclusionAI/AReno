"""Agent entrypoint for calendar scheduling tool-call rollouts."""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Your task is to schedule a meeting by calling the provided tools. "
    "First, call query_availability for each required participant to learn their available times. "
    "Then, convert their local availability to UTC and find a common slot that fits the meeting duration. "
    "Call propose_slot with the meeting_id and the UTC time range. "
    "Finally, call confirm_slot to finalize the booking. "
    "Use exactly one tool call per turn. Do not write free text."
)

QUERY_AVAILABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_availability",
        "description": "Query the available time slots for a participant in their local timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "participant": {
                    "type": "string",
                    "description": "The name of the participant to query.",
                },
            },
            "required": ["participant"],
            "additionalProperties": False,
        },
    },
}

PROPOSE_SLOT_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_slot",
        "description": "Propose a meeting time in UTC hours.",
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "The ID of the meeting to schedule.",
                },
                "utc_start_hour": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 23,
                    "description": "Start hour of the meeting in UTC (0-23).",
                },
                "utc_end_hour": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 24,
                    "description": "End hour of the meeting in UTC (1-24).",
                },
            },
            "required": ["meeting_id", "utc_start_hour", "utc_end_hour"],
            "additionalProperties": False,
        },
    },
}

CONFIRM_SLOT_TOOL = {
    "type": "function",
    "function": {
        "name": "confirm_slot",
        "description": "Confirm a proposed meeting slot to finalize the booking.",
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "The ID of the meeting to confirm.",
                },
                "utc_start_hour": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 23,
                    "description": "Start hour of the meeting in UTC (0-23).",
                },
                "utc_end_hour": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 24,
                    "description": "End hour of the meeting in UTC (1-24).",
                },
            },
            "required": ["meeting_id", "utc_start_hour", "utc_end_hour"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [QUERY_AVAILABILITY_TOOL, PROPOSE_SLOT_TOOL, CONFIRM_SLOT_TOOL]


async def run_agent(ctx, batch):
    """Run one tool-call model request for each calendar scenario."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The calendar agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info(
        "Calendar agent start requests=%d max_running_prompts=%d",
        len(items),
        ctx.max_running_prompts,
    )
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        ),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0
    )

    async def run_one(item):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        tool_choice = "auto"
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=TOOLS,
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(
            turns=list(await asyncio.gather(*(run_one(item) for item in items)))
        )
    finally:
        await client.close()