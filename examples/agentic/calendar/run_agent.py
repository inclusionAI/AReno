"""Agent entrypoint for multi-turn calendar scheduling tool-call rollouts.

Implements a four-turn agentic flow similar to the shopping example:
  Turn 1: query_availability for each required participant
  Turn 2: (model sees availability results) propose_slot
  Turn 3: (model sees proposal validation) confirm_slot

Each tool call is actually executed and the result is fed back into the
conversation so the model can make informed decisions step by step.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    execute_confirm_slot,
    execute_propose_slot,
    execute_query_availability,
    record_to_state,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Your task is to schedule a meeting by calling the provided tools. "
    "First, call query_availability for each required participant to learn their available times in UTC. "
    "Then, find a UTC time range that overlaps all participants' availability and fits the meeting duration. "
    "Call propose_slot with the meeting_id and the UTC time range. "
    "After seeing the proposal result, call confirm_slot to finalize the booking. "
    "Use exactly one tool call per turn. Do not write free text."
)

# Turn-by-turn prompts guide the model through the scheduling flow.
TURN_PROMPTS = {
    "query_availability": "Turn 1: call query_availability for each required participant to learn their UTC availability.",
    "propose_slot": "Turn 2: based on the availability results above, call propose_slot with a UTC time range that works for all participants.",
    "confirm_slot": "Turn 3: based on the proposal result above, call confirm_slot to finalize the booking.",
}

QUERY_AVAILABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_availability",
        "description": "Query the available time slots for a participant, returned in UTC.",
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

TOOL_BY_NAME = {
    "query_availability": QUERY_AVAILABILITY_TOOL,
    "propose_slot": PROPOSE_SLOT_TOOL,
    "confirm_slot": CONFIRM_SLOT_TOOL,
}


async def run_agent(ctx, batch):
    """Run multi-turn tool-call rollouts for each calendar scenario.

    Flow per scenario:
      1. For each required participant: call query_availability, execute, feed result back.
      2. Call propose_slot, execute, feed result back.
      3. Call confirm_slot, execute, feed result back.
    """

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
    # Share one httpx connection pool sized to the max rollout concurrency.
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
        turns = []
        # Build initial messages: system prompt + user scenario.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        # Reconstruct the calendar state from the source record.
        state = record_to_state(item.record)
        meeting_id = item.record.get("target_meeting_id", "")
        meeting = state.meeting_by_id(meeting_id)
        if meeting is None:
            logger.warning("meeting %s not found in record", meeting_id)
            return turns

        # --- Turn 1: query_availability for each required participant ---
        for participant_name in meeting.required_participants:
            assistant_msg, turn = await _call_model(
                item, client, messages, "query_availability"
            )
            turns.append(turn)
            tool_result = _run_tool(assistant_msg, state)
            messages.extend(_tool_messages(assistant_msg, tool_result))

        # --- Turn 2: propose_slot ---
        assistant_msg, turn = await _call_model(
            item, client, messages, "propose_slot"
        )
        turns.append(turn)
        tool_result = _run_tool(assistant_msg, state)
        messages.extend(_tool_messages(assistant_msg, tool_result))

        # --- Turn 3: confirm_slot ---
        assistant_msg, turn = await _call_model(
            item, client, messages, "confirm_slot"
        )
        turns.append(turn)
        tool_result = _run_tool(assistant_msg, state)
        messages.extend(_tool_messages(assistant_msg, tool_result))

        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def _call_model(item, client, messages: list[dict], tool_name: str):
    """Call the model with a single-tool constraint and return (assistant_message, turn).

    The turn prompt guides the model to call the expected tool at this step.
    """
    turn_messages = [*messages, {"role": "user", "content": TURN_PROMPTS[tool_name]}]
    tools = [TOOL_BY_NAME[tool_name]]
    tool_choice = {"type": "function", "function": {"name": tool_name}}
    response = await client.chat.completions.create(
        model="policy",
        messages=turn_messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=False,
    )
    message = response.choices[0].message
    # Extract only the matching tool call (first one).
    tool_calls = [call for call in (message.tool_calls or []) if call.function.name == tool_name][:1]
    assistant_message = {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    }
    # If the model didn't produce the expected tool call, insert a placeholder
    # so the downstream execution can report an error.
    if not assistant_message["tool_calls"]:
        assistant_message["tool_calls"] = [
            {
                "id": f"missing_{tool_name}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": "{}",
                },
            }
        ]
    return assistant_message, AgentTrajectoryTurn(
        item=item,
        messages=turn_messages,
        response=response,
        tools=tools,
        tool_choice=tool_choice,
    )


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    """Build the assistant + tool-result messages to append to the conversation."""
    messages = [assistant_message]
    for call in assistant_message.get("tool_calls") or []:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )
    return messages


def _run_tool(assistant_message: dict, state) -> dict:
    """Execute the tool call and return the real result.

    This is the key difference from single-step agents: the tool is actually
    executed, and the result is fed back to the model so it can make
    informed decisions in subsequent turns.
    """
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return {"error": "missing tool call"}
    call = calls[0]
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid JSON arguments"}

    if name == "query_availability":
        participant = str(args.get("participant", ""))
        return execute_query_availability(state, participant)
    if name == "propose_slot":
        meeting_id = str(args.get("meeting_id", ""))
        utc_start = int(args.get("utc_start_hour", -1))
        utc_end = int(args.get("utc_end_hour", -1))
        return execute_propose_slot(state, meeting_id, utc_start, utc_end)
    if name == "confirm_slot":
        meeting_id = str(args.get("meeting_id", ""))
        utc_start = int(args.get("utc_start_hour", -1))
        utc_end = int(args.get("utc_end_hour", -1))
        return execute_confirm_slot(state, meeting_id, utc_start, utc_end)
    return {"error": f"unknown tool: {name}"}