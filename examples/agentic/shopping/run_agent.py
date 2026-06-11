"""Agent entrypoint for multi-turn shopping kit tool-call rollouts."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import check_kit, inspect_items, search_catalog_many  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a shopping kit planner. Use tools to search the catalog, inspect candidates, "
    "check the full kit against constraints, then submit item ids. Do not answer in plain text."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search compact catalog results for one or more categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["jacket", "shoes", "bottle"]},
                    },
                    "max_price": {"type": "integer"},
                },
                "required": ["categories"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_items",
            "description": "Inspect full details for several item ids.",
            "parameters": {
                "type": "object",
                "properties": {"item_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["item_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_kit",
            "description": "Check whether a proposed kit satisfies budget and feature constraints.",
            "parameters": {
                "type": "object",
                "properties": {"item_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["item_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_bundle",
            "description": "Submit the final kit item ids.",
            "parameters": {
                "type": "object",
                "properties": {"item_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["item_ids"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}

TURN_PROMPTS = {
    "search_catalog": "Turn 1: call search_catalog only. Search for the required categories from the task.",
    "inspect_items": "Turn 2: call inspect_items only. Inspect promising item ids from the search results.",
    "check_kit": "Turn 3: call check_kit only. Check one complete proposed kit against the task constraints.",
    "submit_bundle": "Turn 4: call submit_bundle only. Submit the final item ids.",
}


async def run_agent(ctx, batch) -> None:
    """Run four tool-call turns for each shopping task."""

    try:
        from openai import AsyncOpenAI
        import httpx
    except ImportError as exc:
        raise RuntimeError("The shopping agentic example requires `openai` and `httpx`. Install them with `pip install openai`.") from exc

    items = list(batch.iter_samples())
    logger.info("Shopping agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=300.0,
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item) -> None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        search = await _call_model(client, messages, "search_catalog")
        messages.extend(_tool_messages(search, _run_tool(search, item.record)))
        inspect = await _call_model(client, messages, "inspect_items")
        messages.extend(_tool_messages(inspect, _run_tool(inspect, item.record)))
        check = await _call_model(client, messages, "check_kit")
        messages.extend(_tool_messages(check, _run_tool(check, item.record)))
        submit = await _call_model(client, messages, "submit_bundle")
        messages.extend(_tool_messages(submit, _run_tool(submit, item.record)))

    try:
        await asyncio.gather(*(run_one(item) for item in items))
    finally:
        await client.close()


async def _call_model(client, messages: list[dict], tool_name: str):
    turn_messages = [*messages, {"role": "user", "content": TURN_PROMPTS[tool_name]}]
    response = await client.chat.completions.create(
        model="policy",
        messages=turn_messages,
        tools=[TOOL_BY_NAME[tool_name]],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        stream=False,
    )
    message = response.choices[0].message
    tool_calls = [call for call in (message.tool_calls or []) if call.function.name == tool_name][:1]
    return {
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


def _run_tool(assistant_message: dict, record: dict) -> dict:
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return {"error": "missing tool call"}
    call = calls[0]
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid JSON arguments"}
    if name == "search_catalog":
        categories = [str(category) for category in args.get("categories", [])]
        return {"results_by_category": search_catalog_many(categories, max_price=args.get("max_price"))}
    if name == "inspect_items":
        item_ids = [str(item_id) for item_id in args.get("item_ids", [])]
        return {"items": inspect_items(item_ids)}
    if name == "check_kit":
        item_ids = [str(item_id) for item_id in args.get("item_ids", [])]
        return {"kit": check_kit(record, item_ids)}
    if name == "submit_bundle":
        return {"submitted": [str(item_id) for item_id in args.get("item_ids", [])]}
    return {"error": f"unknown tool: {name}"}
