"""Agent entrypoint for circuit-diagnosis multi-turn tool-call rollouts (issue #193).

The agent interacts with a faulty logic circuit through two tools:
- ``probe``: Set input values and inspect a wire's output.
- ``submit``: Submit the guessed faulty gate ID.

Multi-turn conversation (max 10 turns):
1. Model receives circuit description.
2. Model calls probe → executed on faulty circuit → result returned as tool message.
3. Model calls submit → conversation ends.
4. All turns recorded as AgentTrajectoryTurn for training.

Only the first tool call in a response is executed. The exact model output is
preserved; every additional call receives a structured not-executed result and
is ignored by the trace-aware reward verifier.
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

    Returns a structured success or error result and never raises.
    """

    inputs_raw = arguments.get("inputs")
    wire_id_raw = arguments.get("wire_id", -1)

    # Validate inputs: must be a list of true booleans.
    if not isinstance(inputs_raw, list):
        return _tool_error("invalid_inputs_type", "inputs must be a list of booleans")
    for item in inputs_raw:
        if not isinstance(item, bool):
            return _tool_error("invalid_input_value", f"inputs contains non-boolean value: {item!r}")
    if len(inputs_raw) != faulty.reference.num_inputs:
        return _tool_error(
            "invalid_input_width",
            f"inputs length {len(inputs_raw)} does not match num_inputs {faulty.reference.num_inputs}",
        )

    # Validate wire_id: must be an integer in range.
    if isinstance(wire_id_raw, bool) or not isinstance(wire_id_raw, int):
        return _tool_error("invalid_wire_id_type", f"wire_id must be an integer, got {wire_id_raw!r}")
    wire_id = wire_id_raw

    if wire_id < 0 or wire_id >= faulty.reference.num_gates:
        return _tool_error(
            "wire_id_out_of_range",
            f"wire_id {wire_id} out of range [0, {faulty.reference.num_gates})",
        )

    try:
        value = faulty.get_faulty_wire_value(inputs_raw, wire_id)
        return {"ok": True, "wire_id": wire_id, "inputs": inputs_raw, "value": bool(value)}
    except (ValueError, IndexError) as exc:
        return _tool_error("probe_failed", str(exc))


def _execute_submit(faulty: circuit.FaultyCircuit, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate a final diagnosis without disclosing whether it is correct."""

    gate_id = arguments.get("gate_id")
    if isinstance(gate_id, bool) or not isinstance(gate_id, int):
        return _tool_error("invalid_gate_id_type", f"gate_id must be an integer, got {gate_id!r}")
    if gate_id < faulty.reference.num_inputs or gate_id >= faulty.reference.num_gates:
        return _tool_error(
            "gate_id_out_of_range",
            f"gate_id {gate_id} must identify a non-INPUT gate in "
            f"[{faulty.reference.num_inputs}, {faulty.reference.num_gates})",
        )
    return {"ok": True, "accepted": True, "gate_id": gate_id}


def _tool_error(code: str, message: str) -> dict[str, Any]:
    """Return the stable error envelope used by every circuit tool."""

    return {"ok": False, "error": {"code": code, "message": message}}


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Parse one tool argument object without silently accepting malformed JSON."""

    if isinstance(raw_arguments, dict):
        return raw_arguments, None
    if not isinstance(raw_arguments, str):
        return None, _tool_error("invalid_arguments_type", "tool arguments must be a JSON object")
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return None, _tool_error("invalid_json", f"tool arguments are not valid JSON: {exc.msg}")
    if not isinstance(parsed, dict):
        return None, _tool_error("invalid_arguments_type", "tool arguments must decode to a JSON object")
    return parsed, None


async def run_agent(ctx, batch):
    """Run multi-turn circuit-diagnosis tool-call rollouts.

    Each rollout is a multi-turn conversation:
    1. The model receives the circuit description.
    2. The model calls `probe` to inspect wire values — we execute the probe
       on the faulty circuit and return the result as a tool message.
    3. The model calls `submit` with its diagnosis — the conversation ends.
    4. All turns are recorded as AgentTrajectoryTurn for training.

    Only the first tool call in each model response is executed. Extra calls
    remain in the raw trajectory and receive explicit not-executed results.
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
        seen_probes: set[tuple[tuple[bool, ...], int]] = set()

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

            # Record the exact model output. Execution below is bounded, but
            # the trajectory never rewrites or drops emitted tool calls.
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
            arguments, argument_error = _parse_tool_arguments(call["function"].get("arguments"))

            if argument_error is not None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call_name,
                        "content": json.dumps(argument_error, ensure_ascii=False),
                    }
                )
                _append_unexecuted_tool_results(messages, tool_calls[1:])
                continue
            assert arguments is not None

            should_end = False
            if call_name == "submit":
                result = _execute_submit(faulty, arguments)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "submit",
                    "content": json.dumps(result, ensure_ascii=False),
                }
                messages.append(tool_message)
                if result["ok"]:
                    should_end = True
            elif call_name == "probe":
                result = _execute_probe(faulty, arguments)
                if result["ok"]:
                    probe_key = (tuple(result["inputs"]), result["wire_id"])
                    if probe_key in seen_probes:
                        result = _tool_error(
                            "duplicate_probe",
                            "this input-vector and wire combination has already been probed",
                        )
                    else:
                        seen_probes.add(probe_key)
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
                    "content": json.dumps(_tool_error("unknown_tool", f"unknown tool: {call_name}")),
                }
                messages.append(tool_message)

            _append_unexecuted_tool_results(messages, tool_calls[1:])
            if should_end:
                break

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


def _append_unexecuted_tool_results(messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> None:
    """Pair every additional model-emitted tool call with an error result."""

    for call in tool_calls:
        name = call["function"]["name"]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps(
                    _tool_error(
                        "additional_tool_call_not_executed",
                        "only the first tool call in a turn is executed",
                    )
                ),
            }
        )
