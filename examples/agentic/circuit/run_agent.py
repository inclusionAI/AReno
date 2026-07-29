"""Agent entrypoint for circuit-diagnosis multi-turn tool-call rollouts (issue #193).

The agent interacts with a faulty logic circuit through two tools:
- ``probe``: Set input values and inspect a wire's output.
- ``submit``: Submit the guessed faulty gate ID.

Multi-turn conversation (max 10 turns):
1. Model receives circuit description.
2. Model calls probe → executed on faulty circuit → result returned as tool message.
3. Model calls submit → conversation ends.
4. All turns recorded as AgentTrajectoryTurn for training.

Only the first tool call in a response is executed.  If the model returns
multiple tool calls, the extras are ignored and not recorded in tool_calls
for reward purposes.
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
    """Execute a probe tool call against the faulty circuit with strict validation.

    Returns a dict with either ``{"wire_id": int, "value": bool}`` on success
    or ``{"error": str}`` on invalid input.  Never raises.
    """

    inputs_raw = arguments.get("inputs")
    wire_id_raw = arguments.get("wire_id", -1)

    # Validate inputs: must be a list of true booleans.
    if not isinstance(inputs_raw, list):
        return {"error": "inputs must be a list of booleans"}
    for item in inputs_raw:
        if not isinstance(item, bool):
            return {"error": f"inputs contains non-boolean value: {item!r}"}
    if len(inputs_raw) != faulty.reference.num_inputs:
        return {"error": f"inputs length {len(inputs_raw)} does not match num_inputs {faulty.reference.num_inputs}"}

    # Validate wire_id: must be an integer in range.
    if isinstance(wire_id_raw, bool) or not isinstance(wire_id_raw, int):
        if isinstance(wire_id_raw, str):
            try:
                wire_id = int(wire_id_raw)
            except ValueError:
                return {"error": f"wire_id must be an integer, got {wire_id_raw!r}"}
        else:
            return {"error": f"wire_id must be an integer, got {wire_id_raw!r}"}
    else:
        wire_id = wire_id_raw

    if wire_id < 0 or wire_id >= faulty.reference.num_gates:
        return {"error": f"wire_id {wire_id} out of range [0, {faulty.reference.num_gates})"}

    try:
        value = faulty.get_faulty_wire_value(inputs_raw, wire_id)
        return {"wire_id": wire_id, "value": bool(value)}
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

    Only the first tool call in each model response is executed.  Extra
    tool calls are ignored and not recorded for reward purposes.
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

        for _turn_idx in range(MAX_TURNS):
            response = await _create_chat_completion_with_retry(
                client,
                model="policy",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=False,
            )
            # Extract the assistant message from the response.
            assistant_message = _assistant_message_from_response(response)
            raw_tool_calls = assistant_message.get("tool_calls") or []

            # If the model returned multiple tool calls, only keep the first.
            # This ensures parsed_tool_calls (used by reward) does not contain
            # unexecuted submit calls.
            if len(raw_tool_calls) > 1:
                assistant_message["tool_calls"] = [raw_tool_calls[0]]
                _trim_response_tool_calls(response, 1)

            # Record this turn for training (response already trimmed).
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=list(messages),
                    response=response,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            )

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

            # Only execute the first tool call.  Extra calls are ignored.
            call = tool_calls[0]
            call_name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if call_name == "submit":
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "submit",
                    "content": json.dumps({"received": True, "gate_id": arguments.get("gate_id")}),
                }
                messages.append(tool_message)
                break
            elif call_name == "probe":
                result = _execute_probe(faulty, arguments)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "probe",
                    "content": json.dumps(result, ensure_ascii=False),
                }
                messages.append(tool_message)
            else:
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


def _trim_response_tool_calls(response: Any, keep: int = 1) -> None:
    """Trim the response's tool_calls list to only the first ``keep`` entries.

    This ensures ``AgentTrajectoryTurn.__post_init__`` (which calls
    ``_chat_response_message_tool_calls``) does not record unexecuted
    tool calls in ``parsed_tool_calls``.
    """

    try:
        message = response.choices[0].message
        if message.tool_calls and len(message.tool_calls) > keep:
            message.tool_calls = message.tool_calls[:keep]
    except (AttributeError, IndexError, TypeError):
        pass
