"""Agent entrypoint for circuit-diagnosis multi-turn tool-call rollouts (issue #193).

The agent interacts with a faulty logic circuit through two tools:
- ``probe``: Set input values and inspect a wire's output.
- ``submit``: Submit the guessed faulty gate ID.

Unlike Tic-Tac-Toe (one tool call), circuit diagnosis requires multi-turn
interaction: the agent probes several wires across multiple turns, observes
the results, then submits its diagnosis. This mirrors the coding agent loop
in ``areno/agent/agent_loop.py``.

The agent sees the circuit structure but not which gate is faulty.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_TURNS = 10
MODEL_QUERY_RETRIES = 5
MODEL_QUERY_BACKOFF_S = 1.0

SYSTEM_PROMPT = (
    "You are a digital circuit diagnosis expert. "
    "A logic circuit has one faulty gate (stuck-at-0 or stuck-at-1). "
    "Use the 'probe' tool to set inputs and inspect wire outputs. "
    "Compare observed outputs against expected logic to narrow down the fault. "
    "When you have identified the faulty gate, use the 'submit' tool. "
    "You have at most 10 turns. Call exactly one tool per turn."
)

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "probe",
        "description": "Set circuit inputs and inspect a wire's output value from the faulty circuit.",
        "parameters": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "description": "Boolean values for each input gate, in order.",
                },
                "wire_id": {
                    "type": "integer",
                    "description": "The wire index to inspect (0 to num_gates-1).",
                },
            },
            "required": ["inputs", "wire_id"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit the gate ID you believe is faulty. This ends the diagnosis.",
        "parameters": {
            "type": "object",
            "properties": {
                "gate_id": {
                    "type": "integer",
                    "description": "The gate index that you believe is faulty.",
                }
            },
            "required": ["gate_id"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [PROBE_TOOL, SUBMIT_TOOL]


def _build_faulty_circuit(source_record: dict[str, Any]) -> circuit.FaultyCircuit:
    """Reconstruct a FaultyCircuit from a dataset record."""

    gates = []
    for g in source_record.get("gates", []):
        gates.append(
            circuit.Gate(
                gate_id=g["gate_id"],
                gate_type=circuit.GateType(g["gate_type"]),
                inputs=tuple(g.get("inputs", [])),
            )
        )
    circ = circuit.Circuit(
        gates=gates,
        num_inputs=source_record.get("num_inputs", 3),
        num_outputs=1,
    )
    return circuit.FaultyCircuit(
        reference=circ,
        faulty_gate_id=source_record["faulty_gate_id"],
        fault_type=source_record.get("fault_type", "stuck_at_0"),
    )


def _execute_probe(faulty: circuit.FaultyCircuit, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a probe tool call against the faulty circuit."""

    inputs_raw = arguments.get("inputs", [])
    wire_id = arguments.get("wire_id", -1)
    # Coerce inputs to booleans.
    inputs = [bool(i) for i in inputs_raw] if isinstance(inputs_raw, list) else []
    try:
        value = faulty.get_faulty_wire_value(inputs, int(wire_id))
        return {"wire_id": int(wire_id), "value": bool(value)}
    except (ValueError, IndexError) as exc:
        return {"error": str(exc)}


async def run_agent(ctx, batch):
    """Run multi-turn circuit-diagnosis tool-call rollouts.

    Each rollout is a multi-turn conversation:
    1. The model receives the circuit description.
    2. The model calls `probe` to inspect wire values — we execute the probe
       on the faulty circuit and return the result as a tool message.
    3. The model calls `submit` with its diagnosis — the conversation ends.
    4. All turns are recorded as AgentTrajectoryTurn for training.
    """

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The circuit-diagnosis agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info(
        "Circuit diagnosis agent start requests=%d max_running_prompts=%d",
        len(items),
        ctx.max_running_prompts,
    )
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        source = item.source_record if hasattr(item, "source_record") else item.record
        faulty = _build_faulty_circuit(source)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        turns: list[AgentTrajectoryTurn] = []

        for turn_idx in range(MAX_TURNS):
            response = await _create_chat_completion_with_retry(
                client,
                model="policy",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=False,
            )
            # Record this turn for training.
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=list(messages),
                    response=response,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            )

            # Extract the assistant message from the response.
            assistant_message = _assistant_message_from_response(response)
            messages.append(assistant_message)

            # Check if the model made a tool call.
            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                # No tool call — nudge the model to use a tool.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not include a tool call. "
                            "Call 'probe' to inspect a wire, or 'submit' to give your diagnosis."
                        ),
                    }
                )
                continue

            # Execute the first tool call.
            call = tool_calls[0]
            call_name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if call_name == "submit":
                # Submit ends the conversation.
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "submit",
                    "content": json.dumps({"received": True, "gate_id": arguments.get("gate_id")}),
                }
                messages.append(tool_message)
                break
            elif call_name == "probe":
                # Execute the probe on the faulty circuit.
                result = _execute_probe(faulty, arguments)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "probe",
                    "content": json.dumps(result, ensure_ascii=False),
                }
                messages.append(tool_message)
            else:
                # Unknown tool.
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call_name,
                    "content": json.dumps({"error": f"unknown tool: {call_name}"}),
                }
                messages.append(tool_message)

        return turns

    try:
        all_turns = await asyncio.gather(*(run_one(item) for item in items))
        # Flatten: each item produces a list of turns; AgentTrajectory expects a flat list.
        flat_turns: list[AgentTrajectoryTurn] = []
        for turns in all_turns:
            flat_turns.extend(turns)
        return AgentTrajectory(turns=flat_turns)
    finally:
        await client.close()


async def _create_chat_completion_with_retry(client: Any, **kwargs: Any) -> Any:
    """Query an OpenAI-compatible chat endpoint with bounded exponential backoff."""

    last_error: Exception | None = None
    for attempt in range(MODEL_QUERY_RETRIES):
        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int) and status_code < 500 and status_code != 429:
                raise
            if attempt < MODEL_QUERY_RETRIES - 1:
                await asyncio.sleep(MODEL_QUERY_BACKOFF_S * (2**attempt))
    raise last_error  # type: ignore[misc]


def _assistant_message_from_response(response: Any) -> dict[str, Any]:
    """Extract the assistant message dict from an OpenAI chat completion response."""

    choice = response.choices[0]
    message = choice.message
    assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in message.tool_calls
        ]
    return assistant_message
