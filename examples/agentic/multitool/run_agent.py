"""Agent entrypoint for multi-tool tool-call rollouts.

Each task requires two or more tool calls executed in the correct order.
The agent runs one model turn per required tool, guided by per-turn prompts.
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
    calculate,
    list_contacts_by_city,
    lookup_contact,
    lookup_parcel,
    read_note,
    search_notes,
    unit_convert,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a multi-tool task agent. Use the available tools in the correct order "
    "to complete each task. Do not answer in plain text."
)

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

CONTACTS_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_contact",
        "description": "Look up a contact by partial name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full or partial contact name.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

NOTES_TOOL = {
    "type": "function",
    "function": {
        "name": "read_note",
        "description": "Read a note by its key.",
        "parameters": {
            "type": "object",
            "properties": {
                "note_key": {
                    "type": "string",
                    "description": "The note key, e.g. 'meeting', 'budget', 'shipping'.",
                },
            },
            "required": ["note_key"],
            "additionalProperties": False,
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a safe arithmetic expression with +, -, *, /, and parentheses.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '3 * 15' or '(10 + 5) / 3'.",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

UNIT_CONVERT_TOOL = {
    "type": "function",
    "function": {
        "name": "unit_convert",
        "description": "Convert a value between supported length (m, cm, mm, km) or weight (g, kg, mg) units.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numeric value to convert."},
                "from_unit": {"type": "string", "description": "Source unit, e.g. 'cm'."},
                "to_unit": {"type": "string", "description": "Target unit, e.g. 'm'."},
            },
            "required": ["value", "from_unit", "to_unit"],
            "additionalProperties": False,
        },
    },
}

PARCEL_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_parcel",
        "description": "Look up parcel tracking information by tracking id.",
        "parameters": {
            "type": "object",
            "properties": {
                "tracking_id": {
                    "type": "string",
                    "description": "The parcel tracking id, e.g. 'P001'.",
                },
            },
            "required": ["tracking_id"],
            "additionalProperties": False,
        },
    },
}

SEARCH_NOTES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_notes",
        "description": "Search all notes by keyword (case-insensitive). Returns matching note keys and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "The keyword to search for in note content."},
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
    },
}

LIST_CONTACTS_BY_CITY_TOOL = {
    "type": "function",
    "function": {
        "name": "list_contacts_by_city",
        "description": "List all contacts in a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name, e.g. 'Shanghai'."},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [CONTACTS_TOOL, NOTES_TOOL, CALCULATOR_TOOL, UNIT_CONVERT_TOOL, PARCEL_TOOL, SEARCH_NOTES_TOOL, LIST_CONTACTS_BY_CITY_TOOL]
TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}

# Per-turn guidance for each tool name.
TURN_PROMPTS = {
    "lookup_contact": "Call lookup_contact with the name from the task.",
    "read_note": "Call read_note with the note key from the task.",
    "calculate": "Call calculate with the expression from the task.",
    "unit_convert": "Call unit_convert with the value and units from the task.",
    "lookup_parcel": "Call lookup_parcel with the tracking id from the task.",
    "search_notes": "Call search_notes with the keyword from the task.",
    "list_contacts_by_city": "Call list_contacts_by_city with the city from the task.",
}


async def run_agent(ctx, batch):
    """Run tool-call turns for each multi-tool task."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The multi-tool agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Multi-tool agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        turns = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        required_tools = list(item.record.get("required_tools", []))
        for tool_name in required_tools:
            assistant_message, turn = await _call_model(item, client, messages, tool_name)
            turns.append(turn)
            messages.extend(_tool_messages(assistant_message, _run_tool(assistant_message)))
        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        await client.close()


async def _call_model(item, client, messages: list[dict], tool_name: str):
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
    if not assistant_message["tool_calls"]:
        assistant_message["tool_calls"] = [
            {
                "id": f"missing_{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
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


def _run_tool(assistant_message: dict) -> dict:
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return {"error": "missing tool call"}
    call = calls[0]
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid JSON arguments"}
    if name == "lookup_contact":
        result = lookup_contact(str(args.get("name", "")))
        return result if result else {"error": "contact not found"}
    if name == "read_note":
        result = read_note(str(args.get("note_key", "")))
        return result if result else {"error": "note not found"}
    if name == "calculate":
        return calculate(str(args.get("expression", "")))
    if name == "unit_convert":
        return unit_convert(
            float(args.get("value", 0)),
            str(args.get("from_unit", "")),
            str(args.get("to_unit", "")),
        )
    if name == "lookup_parcel":
        result = lookup_parcel(str(args.get("tracking_id", "")))
        return result if result else {"error": "parcel not found"}
    if name == "search_notes":
        results = search_notes(str(args.get("keyword", "")))
        return {"results": results} if results else {"error": "no notes found"}
    if name == "list_contacts_by_city":
        results = list_contacts_by_city(str(args.get("city", "")))
        return {"contacts": results} if results else {"error": "no contacts in that city"}
    return {"error": f"unknown tool: {name}"}