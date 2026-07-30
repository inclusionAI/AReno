"""Bounded multi-turn agent loop for logic-circuit diagnosis."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    ALL_TOOLS,
    MAX_PROBES,
    evaluate,
    verify_diagnosis,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a rigorous digital circuit diagnostician. A combinational logic circuit has "
    "exactly ONE stuck-at fault on a single internal gate (AND/OR/NOT). Use set_input_vector "
    "to observe the faulty circuit's output (free), inspect_node to probe internal gate values "
    "(each costs 1 probe), and submit_diagnosis to give your final answer. "
    "Reason step by step: apply input vectors, compare against expected behavior of the healthy "
    "circuit, probe suspicious nodes only when needed, then submit your diagnosis."
)


async def run_agent(ctx, batch):
    """Run bounded concurrent diagnosis episodes and preserve exact model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Logic diagnosis requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    try:
        grouped = await asyncio.gather(*(_run_episode(item, client) for item in items))
        return AgentTrajectory(turns=[turn for episode in grouped for turn in episode])
    finally:
        await client.close()


async def _run_episode(item, client) -> list[AgentTrajectoryTurn]:
    nodes = item.record.get("nodes", [])
    fault = item.record.get("fault", {})
    max_probes = int(item.record.get("max_probes", MAX_PROBES))
    max_turns = max_probes + 5  # allow some free set_input_vector calls + submit

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item.prompt}]
    turns: list[AgentTrajectoryTurn] = []

    state: dict = {
        "input_vector": None,
        "probes_used": 0,
        "diagnosis_submitted": False,
    }

    for turn_index in range(1, max_turns + 1):
        turn_prompt = {
            "role": "user",
            "content": f"Turn {turn_index}: Choose one tool: set_input_vector, inspect_node, or submit_diagnosis.",
        }
        turn_messages = [*messages, turn_prompt]

        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=ALL_TOOLS,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=ALL_TOOLS,
            )
        )

        assistant_msg = _assistant_message(response)
        tool_result = _execute_tool(assistant_msg, nodes, fault, state)

        if tool_result is None:
            logger.warning("Logic diagnosis model returned no executable tool call on turn %d", turn_index)
            break

        messages.extend(_tool_messages(assistant_msg, tool_result))

        if state.get("diagnosis_submitted"):
            finish_prompt = {
                "role": "user",
                "content": "The episode is over. Briefly summarize your diagnosis reasoning without calling a tool.",
            }
            finish_response = await client.chat.completions.create(
                model="policy",
                messages=[*messages, finish_prompt],
                stream=False,
            )
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=[*messages, finish_prompt],
                    response=finish_response,
                )
            )
            break

        if state["probes_used"] >= max_probes:
            messages.append(
                {"role": "user", "content": f"Maximum {max_probes} probes used. You must call submit_diagnosis now."}
            )

    if not state.get("diagnosis_submitted"):
        logger.warning("Logic diagnosis episode ended without submit_diagnosis")

    return turns


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


def _execute_tool(
    assistant_message: dict,
    nodes: list[dict],
    fault: dict,
    state: dict,
) -> dict | None:
    """Execute the parsed tool call, update state, return a result dict."""
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return None

    call = calls[0]
    name = call.get("function", {}).get("name", "")

    try:
        args_raw = call.get("function", {}).get("arguments", "{}")
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
    except (json.JSONDecodeError, TypeError):
        return {"error": "invalid JSON tool arguments"}

    if not isinstance(args, dict):
        return {"error": "tool arguments must be a JSON object"}

    if name == "set_input_vector":
        return _tool_set_input_vector(args, nodes, fault, state)

    if name == "inspect_node":
        return _tool_inspect_node(args, nodes, fault, state)

    if name == "submit_diagnosis":
        return _tool_submit_diagnosis(args, nodes, fault, state)

    return {"error": f"unknown tool: {name}"}


def _tool_set_input_vector(args: dict, nodes: list[dict], fault: dict, state: dict) -> dict:
    inputs_raw = args.get("inputs")
    if not isinstance(inputs_raw, list):
        return {"error": "inputs must be a list of booleans"}

    n_in = sum(1 for n in nodes if n["type"] == "input")
    inputs = [bool(v) for v in inputs_raw[:n_in]]
    if len(inputs) < n_in:
        inputs.extend([False] * (n_in - len(inputs)))

    state["input_vector"] = inputs
    values = evaluate(nodes, inputs, fault)

    output_node = next((n for n in nodes if n["type"] == "output"), None)
    output_value = values[output_node["id"]] if output_node else None

    return {
        "input_vector": inputs,
        "output_value": output_value,
    }


def _tool_inspect_node(args: dict, nodes: list[dict], fault: dict, state: dict) -> dict:
    node_id = args.get("node_id")
    if not isinstance(node_id, int):
        return {"error": "node_id must be an integer"}

    if state.get("input_vector") is None:
        return {"error": "must call set_input_vector before inspect_node"}

    node = next((n for n in nodes if n["id"] == node_id), None)
    if node is None:
        return {"error": f"node {node_id} not found"}
    if node["type"] == "input":
        return {"error": f"node {node_id} is a primary input — cannot probe inputs"}
    if node["type"] == "output":
        return {"error": f"node {node_id} is the output node — use set_input_vector to observe it"}

    state["probes_used"] += 1
    values = evaluate(nodes, state["input_vector"], fault)

    return {
        "node_id": node_id,
        "node_type": node["type"],
        "probed_value": values[node_id],
        "probes_used": state["probes_used"],
        "probes_remaining": MAX_PROBES - state["probes_used"],
    }


def _tool_submit_diagnosis(args: dict, nodes: list[dict], fault: dict, state: dict) -> dict:
    node_id = args.get("node_id")
    fault_type = args.get("fault_type")

    if not isinstance(node_id, int) or fault_type not in ("stuck_at_0", "stuck_at_1"):
        return {"error": "submit_diagnosis requires node_id (int) and fault_type (stuck_at_0 or stuck_at_1)"}

    state["diagnosis_submitted"] = True
    correct = verify_diagnosis(nodes, fault, node_id, fault_type)

    return {
        "diagnosis": {"node_id": node_id, "fault_type": fault_type},
        "correct": correct,
        "probes_used": state["probes_used"],
    }


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    messages: list[dict] = [assistant_message]
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