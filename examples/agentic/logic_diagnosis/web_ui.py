"""Cartoon web UI for the logic-circuit diagnosis game.

Run from the repository root:

    python examples/agentic/logic_diagnosis/web_ui.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    MAX_PROBES,
    SET_INPUT_VECTOR_TOOL,
    INSPECT_NODE_TOOL,
    SUBMIT_DIAGNOSIS_TOOL,
    evaluate,
    generate_circuit,
    inject_fault,
    verify_diagnosis,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768

# ---------------------------------------------------------------------------
# Backend: layout helpers
# ---------------------------------------------------------------------------


def _node_depth_by_id(nid: int, id_map: dict[int, dict[str, Any]]) -> int:
    node = id_map.get(nid)
    if node is None or node["type"] == "input":
        return 0
    inputs = node.get("inputs", [])
    if not inputs:
        return 1
    return 1 + max(_node_depth_by_id(inp, id_map) for inp in inputs)


def _node_depth(nid: int, id_map: dict[int, dict[str, Any]]) -> int:
    return _node_depth_by_id(nid, id_map)


def _compute_layout(
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign (x, y) pixel positions for a layered DAG layout."""
    id_map = {n["id"]: n for n in nodes}

    # Group by depth
    layers: dict[int, list[int]] = {}
    for node in nodes:
        d = _node_depth(node["id"], id_map)
        layers.setdefault(d, []).append(node["id"])

    max_depth = max(layers.keys()) if layers else 0
    node_w, node_h = 78, 52
    layer_gap_y = 100
    padding_x = 40

    positions: dict[int, tuple[float, float]] = {}
    for depth, nids in sorted(layers.items()):
        y = padding_x + (max_depth - depth) * layer_gap_y
        count = len(nids)
        total_w = count * node_w + (count - 1) * 24
        start_x = max(padding_x, (620 - total_w) / 2)
        for idx, nid in enumerate(nids):
            x = start_x + idx * (node_w + 24)
            positions[nid] = (x, y)

    total_h = padding_x + max_depth * layer_gap_y + node_h + 20
    svg_w, svg_h = 620, max(total_h, 200)

    result = []
    for node in nodes:
        x, y = positions[node["id"]]
        result.append(
            {
                "id": node["id"],
                "type": node["type"],
                "x": x,
                "y": y,
                "w": node_w,
                "h": node_h,
                "inputs": node.get("inputs", []),
                "input_positions": [positions[inp] for inp in node.get("inputs", []) if inp in positions],
                "depth": _node_depth(node["id"], id_map),
            }
        )
    return result, svg_w, svg_h


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class DiagnosisServer(ThreadingHTTPServer):
    """Stateful HTTP server for one local logic diagnosis game."""

    def __init__(self, server_address, request_handler, *, seed: int | None = None, args=None):
        super().__init__(server_address, request_handler)
        self.rng = random.Random(seed)
        self.args = args
        self.openai_client = None
        self.nodes: list[dict[str, Any]] = []
        self.fault: dict[str, Any] = {}
        self.input_vector: list[bool] | None = None
        self.probes_used: int = 0
        self.diagnosis_submitted: bool = False
        self.correct: bool | None = None
        self.max_probes: int = MAX_PROBES
        self.n_inputs: int = 0
        self.n_gates: int = 0
        self.events: list[str] = []
        self._new_game()

    def _new_game(self) -> None:
        for _attempt in range(200):
            n_in = self.rng.randint(3, 5)
            n_g = self.rng.randint(4, 10)
            c_seed = self.rng.randint(0, 2**31 - 1)
            self.nodes = generate_circuit(n_in, n_g, seed=c_seed)
            gate_count = sum(1 for n in self.nodes if n["type"] in ("and", "or", "not"))
            if gate_count < 2:
                continue
            f_seed = self.rng.randint(0, 2**31 - 1)
            self.fault = inject_fault(self.nodes, seed=f_seed)
            self.n_inputs = sum(1 for n in self.nodes if n["type"] == "input")
            self.n_gates = gate_count
            break
        self.input_vector = None
        self.probes_used = 0
        self.diagnosis_submitted = False
        self.correct = None
        self.max_probes = min(MAX_PROBES, max(1, self.n_gates))
        self.events = [
            f"New circuit: {self.n_inputs} inputs, {self.n_gates} gates. Find the faulty gate!",
            f"You have {self.max_probes} probes. Click a gate to probe, then diagnose.",
        ]
        if hasattr(self, "_messages"):
            del self._messages


class DiagnosisHandler(BaseHTTPRequestHandler):
    server: DiagnosisServer

    def do_GET(self) -> None:
        route = _route_path(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            self._send_json(_make_payload(self.server))
        elif route == "new":
            self.server._new_game()
            self._send_json(_make_payload(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route_path(self.path)
        body = self._read_json()
        if route == "new":
            self.server._new_game()
            self._send_json(_make_payload(self.server))
        elif route == "set-input":
            inputs = body.get("inputs") if isinstance(body, dict) else None
            self._handle_set_input(inputs)
        elif route == "probe":
            node_id = body.get("node_id") if isinstance(body, dict) else None
            self._handle_probe(node_id)
        elif route == "submit":
            node_id = body.get("node_id") if isinstance(body, dict) else None
            fault_type = body.get("fault_type") if isinstance(body, dict) else None
            self._handle_submit(node_id, fault_type)
        elif route == "agent":
            self._handle_agent()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("logic-diag-web: " + fmt % args + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_set_input(self, inputs: Any) -> None:
        server = self.server
        if not isinstance(inputs, list):
            self._send_json({"error": "inputs must be a list of booleans"}, HTTPStatus.BAD_REQUEST)
            return
        bits = [bool(v) for v in inputs[: server.n_inputs]]
        if len(bits) < server.n_inputs:
            bits.extend([False] * (server.n_inputs - len(bits)))
        server.input_vector = bits
        values = evaluate(server.nodes, bits, server.fault)
        output_id = next(n["id"] for n in server.nodes if n["type"] == "output")
        output_val = values[output_id]
        server.events.insert(0, f"Input {_bits_str(bits)} → output = {int(output_val)}")
        server.events = server.events[:12]
        self._send_json(_make_payload(server))

    def _handle_probe(self, node_id: Any) -> None:
        server = self.server
        if not isinstance(node_id, int):
            self._send_json({"error": "node_id must be an integer"}, HTTPStatus.BAD_REQUEST)
            return
        node = next((n for n in server.nodes if n["id"] == node_id), None)
        if node is None:
            self._send_json({"error": f"node {node_id} not found"}, HTTPStatus.BAD_REQUEST)
            return
        if node["type"] in ("input", "output"):
            self._send_json({"error": f"cannot probe {node['type']} node {node_id}"}, HTTPStatus.BAD_REQUEST)
            return
        if server.input_vector is None:
            self._send_json({"error": "set input vector first"}, HTTPStatus.BAD_REQUEST)
            return
        if server.probes_used >= server.max_probes:
            self._send_json({"error": f"out of probes ({server.max_probes} max)"}, HTTPStatus.BAD_REQUEST)
            return

        server.probes_used += 1
        values = evaluate(server.nodes, server.input_vector, server.fault)
        probed_val = values[node_id]
        server._last_probed_value = probed_val
        server.events.insert(
            0,
            f"Probed {_node_label_text(server.nodes, node_id)} → value = {int(probed_val)} "
            f"({server.probes_used}/{server.max_probes} probes)",
        )
        server.events = server.events[:12]
        self._send_json(_make_payload(server, probed_value=probed_val, probed_node=node_id))

    def _handle_submit(self, node_id: Any, fault_type: Any) -> None:
        server = self.server
        if server.diagnosis_submitted:
            self._send_json({"error": "diagnosis already submitted"}, HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(node_id, int) or fault_type not in ("stuck_at_0", "stuck_at_1"):
            self._send_json({"error": "invalid diagnosis"}, HTTPStatus.BAD_REQUEST)
            return
        server.diagnosis_submitted = True
        server.correct = verify_diagnosis(server.nodes, server.fault, node_id, fault_type)
        if server.correct:
            server.events.insert(
                0, f"✓ Correct! Fault was {_node_label_text(server.nodes, node_id)} ({fault_type})."
            )
        else:
            actual = server.fault
            actual_type = "stuck_at_0" if actual["stuck_value"] == 0 else "stuck_at_1"
            server.events.insert(
                0,
                f"✗ Wrong. Actual fault: {_node_label_text(server.nodes, actual['node'])} ({actual_type}).",
            )
        server.events = server.events[:12]
        self._send_json(_make_payload(server))

    def _handle_agent(self) -> None:
        """Let the LLM agent make one move, with full conversation history."""
        server = self.server
        if server.diagnosis_submitted:
            self._send_json(_make_payload(server))
            return
        if server.openai_client is None:
            server.openai_client = _make_openai_client(server.args)

        # Build conversation from scratch on first call, then reuse
        if not hasattr(server, "_messages"):
            from game import make_prompt as game_make_prompt
            record = {"nodes": server.nodes, "n_inputs": server.n_inputs, "n_gates": server.n_gates, "max_probes": server.max_probes}
            server._messages = [
                {"role": "system", "content": SYSTEM_PROMPT_AGENT},
                {"role": "user", "content": game_make_prompt(record)},
            ]

        # Add turn prompt
        if server.input_vector is None:
            tools = [SET_INPUT_VECTOR_TOOL]
            tc = {"type": "function", "function": {"name": "set_input_vector"}}
            turn_msg = {"role": "user", "content": "First, call set_input_vector with your chosen input bits."}
        elif server.probes_used >= server.max_probes:
            tools = [SUBMIT_DIAGNOSIS_TOOL]
            tc = {"type": "function", "function": {"name": "submit_diagnosis"}}
            turn_msg = {"role": "user", "content": "Out of probes. Call submit_diagnosis with your best guess."}
        else:
            tools = [INSPECT_NODE_TOOL, SUBMIT_DIAGNOSIS_TOOL]
            tc = None
            turn_msg = {"role": "user", "content": "Call inspect_node to probe a gate, or submit_diagnosis if confident."}

        server._messages.append(turn_msg)
        try:
            kwargs = {"model": server.args.model, "messages": server._messages, "tools": tools, "stream": False}
            if tc:
                kwargs["tool_choice"] = tc
            response = server.openai_client.chat.completions.create(**kwargs)
        except Exception as exc:
            server.events.insert(0, f"LLM error: {exc}")
            server.events = server.events[:12]
            self._send_json(_make_payload(server))
            return

        msg = response.choices[0].message
        calls = list(msg.tool_calls or [])
        if not calls:
            server.events.insert(0, "LLM returned no tool call")
            server.events = server.events[:12]
            self._send_json(_make_payload(server))
            return

        call = calls[0]
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        # Record assistant message and tool result in history
        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{
                "id": call.id, "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }],
        }
        server._messages.append(assistant_msg)

        if name == "set_input_vector":
            bits_raw = args.get("inputs", [])
            if isinstance(bits_raw, list):
                self._handle_set_input(bits_raw)
                # Record tool result
                out = next((n for n in server.nodes if n["type"] == "output"), None)
                from game import evaluate
                vals = evaluate(server.nodes, server.input_vector, server.fault)
                out_val = vals[out["id"]] if out else None
                server._messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": name,
                    "content": json.dumps({"input_vector": bits_raw, "output_value": out_val}),
                })
        elif name == "inspect_node":
            self._handle_probe(args.get("node_id"))
            server._messages.append({
                "role": "tool", "tool_call_id": call.id, "name": name,
                "content": json.dumps({"node_id": args.get("node_id"), "probed_value": server._last_probed_value}),
            })
        elif name == "submit_diagnosis":
            self._handle_submit(args.get("node_id"), args.get("fault_type"))
            server._messages.append({
                "role": "tool", "tool_call_id": call.id, "name": name,
                "content": json.dumps({"received": True}),
            })
        else:
            server.events.insert(0, f"LLM called unknown tool: {name}")
            server.events = server.events[:12]
            self._send_json(_make_payload(server))


def _make_openai_client(args):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("LLM mode requires `openai`. Install it with `pip install openai`.") from exc
    return OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)


SYSTEM_PROMPT_AGENT = (
    "You are a digital circuit diagnostician. Call ONE tool per turn. "
    "Use set_input_vector to observe outputs (free), inspect_node to probe "
    "a gate (costs 1 probe), and submit_diagnosis when confident."
)


def _route_path(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    for name in ("state", "new", "set-input", "probe", "submit", "agent"):
        if path.endswith(f"/api/{name}"):
            return name
    if "/api/" in path:
        return "missing"
    return "index"


def _bits_str(bits: list[bool]) -> str:
    return "".join("1" if b else "0" for b in bits)


def _node_label_text(nodes: list[dict[str, Any]], nid: int) -> str:
    for n in nodes:
        if n["id"] == nid:
            t = n["type"]
            if t == "input":
                return f"IN{nid}"
            if t == "output":
                return "OUT"
            return f"{t.upper()}{nid}"
    return f"?{nid}"


def _make_payload(
    server: DiagnosisServer,
    *,
    probed_value: bool | None = None,
    probed_node: int | None = None,
) -> dict[str, Any]:
    layout, svg_w, svg_h = _compute_layout(server.nodes)

    # Build nodes with layout positions and probe state
    probed_values: dict[int, bool] = {}
    if server.input_vector is not None and server.probes_used > 0:
        values = evaluate(server.nodes, server.input_vector, server.fault)
        for n in server.nodes:
            if n["type"] in ("and", "or", "not"):
                probed_values[n["id"]] = values[n["id"]]

    nodes_payload = []
    for node, pos in zip(server.nodes, layout):
        nid = node["id"]
        entry = {
            "id": nid,
            "type": node["type"],
            "label": _node_label_text(server.nodes, nid),
            "inputs": node.get("inputs", []),
            "depth": pos["depth"],
            "x": pos["x"],
            "y": pos["y"],
            "w": pos["w"],
            "h": pos["h"],
            "input_positions": pos["input_positions"],
        }
        nodes_payload.append(entry)

    # Fault info (only revealed after submission)
    fault_revealed = None
    if server.diagnosis_submitted:
        fault_revealed = {"node": server.fault["node"], "stuck_value": server.fault["stuck_value"]}

    output_value = None
    if server.input_vector is not None:
        values = evaluate(server.nodes, server.input_vector, server.fault)
        output_id = next(n["id"] for n in server.nodes if n["type"] == "output")
        output_value = values[output_id]

    return {
        "nodes": nodes_payload,
        "svg_w": svg_w,
        "svg_h": svg_h,
        "n_inputs": server.n_inputs,
        "n_gates": server.n_gates,
        "input_vector": server.input_vector,
        "output_value": output_value,
        "probes_used": server.probes_used,
        "max_probes": server.max_probes,
        "diagnosis_submitted": server.diagnosis_submitted,
        "correct": server.correct,
        "fault_revealed": fault_revealed,
        "events": server.events,
        "probed_value": probed_value,
        "probed_node": probed_node,
    }


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Logic Circuit Diagnosis</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#1a1a2e;background:#e3d5c8}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#e3d5c8,#d4c4b0 40%,#c9d5c0);display:grid;place-items:center}
.app{width:min(1020px,98vw);display:grid;grid-template-columns:minmax(420px,580px) 1fr;gap:20px;align-items:start}
.panel{background:#fffaf3;border:4px solid #2d2d2d;border-radius:22px;box-shadow:7px 7px 0 #2d2d2d;padding:16px}
h1{font-size:32px;line-height:1;margin:0 0 4px;color:#c1542c;text-shadow:2px 2px 0 #f0d060}
.subtitle{font-weight:800;color:#555;margin-bottom:10px;font-size:14px}
/* Circuit SVG area */
.circuit-wrap{position:relative;background:#ece6dc;border:3px solid #2d2d2d;border-radius:16px;overflow:hidden;margin-bottom:10px}
.circuit-wrap svg{display:block}
.node-rect{stroke:#2d2d2d;stroke-width:3;rx:12}
.node-rect.input{fill:#9bd5a0}.node-rect.output{fill:#7db8d8}
.node-rect.and{fill:#ffd166}.node-rect.or{fill:#ffb347}.node-rect.not{fill:#e8a0d0}
.node-rect.gate{cursor:pointer;transition:transform .1s}.node-rect.gate:hover{filter:brightness(1.08)}
.node-rect.faulty{stroke:#c44;stroke-width:4}
.node-rect.probed{filter:drop-shadow(0 0 6px rgba(0,120,255,.5))}
.node-text{font-family:Inter,ui-rounded,system-ui,sans-serif;font-size:12px;font-weight:900;fill:#1a1a2e;text-anchor:middle;dominant-baseline:central;pointer-events:none}
.node-val{font-family:Inter,ui-rounded,system-ui,sans-serif;font-size:11px;font-weight:800;text-anchor:middle;dominant-baseline:central;pointer-events:none}
.conn-line{stroke:#7a7a8a;stroke-width:2.5;fill:none;stroke-linecap:round}
.conn-line.input-line{stroke:#9bd5a0;stroke-width:3}
/* Controls */
.input-row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.input-row .label{font-weight:800;font-size:13px;color:#555;margin-right:4px}
.input-switch{display:flex;flex-direction:column;align-items:center;gap:2px}
.input-switch button{width:40px;height:38px;border:3px solid #2d2d2d;border-radius:10px;font-size:18px;font-weight:900;cursor:pointer;box-shadow:3px 3px 0 #2d2d2d;transition:.1s;line-height:1}
.input-switch button.on{background:#9be564;color:#1a1a2e}
.input-switch button.off{background:#eee;color:#aaa}
.input-switch span{font-size:10px;font-weight:800;color:#777}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{border:3px solid #2d2d2d;border-radius:14px;background:#ffd166;box-shadow:4px 4px 0 #2d2d2d;color:#1a1a2e;font-weight:900;padding:9px 14px;cursor:pointer;font-size:13px;transition:.1s}
button:hover:not(:disabled){transform:translateY(-1px)}
button:disabled{filter:grayscale(.6);opacity:.5;cursor:not-allowed}
button.primary{background:#7bc87b;color:#fff}
button.danger{background:#e07070;color:#fff}
.pill{display:inline-block;background:#fff;border:3px solid #2d2d2d;border-radius:999px;padding:5px 11px;font-weight:900;font-size:12px}
.diag-bar{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
.diag-bar.hidden{display:none}
.diag-bar .pick{font-weight:800;font-size:13px}
/* Right panel */
.rules{font-weight:700;line-height:1.5;font-size:13px}.rules li{margin:4px 0}
.events{display:grid;gap:5px;margin-top:8px;max-height:360px;overflow-y:auto}
.event{background:#fff;border:2.5px solid #2d2d2d;border-radius:11px;padding:7px 9px;font-weight:700;font-size:12px}
@media(max-width:820px){.app{grid-template-columns:1fr}}
</style>
</head>
<body>
<main class="app">
  <section class="panel">
    <h1>Logic Diagnosis</h1>
    <div class="subtitle" id="subtitle"></div>
    <div class="circuit-wrap" id="circuitWrap"></div>
    <div class="input-row">
      <span class="label">Inputs:</span>
      <span id="inputSwitches"></span>
    </div>
    <div class="btn-row">
      <button class="primary" id="setInputBtn">Set Input Vector</button>
      <span class="pill" id="outputPill" style="display:none"></span>
      <span class="pill" id="probePill">Probes: 0/0</span>
    </div>
    <div class="diag-bar hidden" id="diagBar">
      <span class="pick" id="diagTarget"></span>
      <button class="danger" onclick="doSubmit('stuck_at_0')">Stuck-at-0</button>
      <button class="danger" onclick="doSubmit('stuck_at_1')">Stuck-at-1</button>
      <button onclick="cancelDiag()">✕ Cancel</button>
    </div>
    <div class="btn-row" style="margin-top:6px">
      <button id="agentBtn">Agent Move</button>
      <button id="autoBtn">Auto Play</button>
      <button id="newBtn">New Circuit</button>
    </div>
  </section>
  <aside class="panel">
    <h1 style="font-size:22px;color:#5a8a6a">How to Play</h1>
    <ul class="rules">
      <li>Toggle each input to <b>0</b> or <b>1</b>, then click <b>Set Input Vector</b>.</li>
      <li>Click a <b style="background:#ffd166;padding:1px 5px;border-radius:3px">gate</b> in the diagram to probe its value (costs 1 probe).</li>
      <li>Probed values appear on the gate. When confident, click a gate to select it, then choose <b>Stuck-at-0</b> or <b>Stuck-at-1</b>.</li>
      <li>Goal: find the faulty gate using as few probes as possible!</li>
    </ul>
    <div id="events" class="events"></div>
  </aside>
</main>
<script>
let state = null, selectedGate = null;

async function api(path, body){
  const opts = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {};
  const res = await fetch(new URL(path,window.location.href).toString(),opts);
  return await res.json();
}

async function refresh(){ state = await api("api/state"); render(); }

function render(){
  if(!state) return;

  // Build SVG
  const wrap = document.getElementById("circuitWrap");
  let svg = `<svg width="${state.svg_w}" height="${state.svg_h}" viewBox="0 0 ${state.svg_w} ${state.svg_h}">`;
  // Connection lines
  for(const n of state.nodes){
    for(const inp of n.input_positions){
      const x1 = inp[0] + 39, y1 = inp[1] + 52;
      const x2 = n.x + 39, y2 = n.y;
      svg += `<line class="conn-line" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
    }
  }
  // Nodes
  for(const n of state.nodes){
    let cls = `node-rect ${n.type}`;
    if(n.type !== "input" && n.type !== "output") cls += " gate";
    if(state.fault_revealed && state.fault_revealed.node === n.id) cls += " faulty";
    if(state.probed_node === n.id || (state.input_vector && state.probes_used > 0)) cls += " probed";
    const onClick = (n.type==="and"||n.type==="or"||n.type==="not")
      ? `onclick="onGateClick(${n.id})"` : "";
    svg += `<rect class="${cls}" x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" ${onClick}/>`;
    svg += `<text class="node-text" x="${n.x+39}" y="${n.y+20}">${n.label}</text>`;
    // Show probed/input value
    let valText = "";
    if(n.type === "input" && state.input_vector){
      valText = state.input_vector[n.id] ? "1" : "0";
    }
    if(state.probes_used > 0 && state.input_vector && n.type !== "input" && n.type !== "output"){
      // Show probe result if this node was probed
      if(state.probed_node === n.id && state.probed_value !== undefined && state.probed_value !== null){
        valText = state.probed_value ? "1" : "0";
      }
    }
    if(valText){
      const color = valText === "1" ? "#2a7a2a" : "#a03030";
      svg += `<text class="node-val" fill="${color}" x="${n.x+39}" y="${n.y+40}">${valText}</text>`;
    }
  }
  svg += '</svg>';
  wrap.innerHTML = svg;

  // Input toggles
  let sw = "";
  for(let i=0;i<state.n_inputs;i++){
    const val = state.input_vector ? state.input_vector[i] : false;
    sw += `<div class="input-switch"><button class="${val?'on':'off'}" onclick="toggleInput(${i})">${val?'1':'0'}</button><span>IN${i}</span></div>`;
  }
  document.getElementById("inputSwitches").innerHTML = sw;

  // Status pills
  document.getElementById("probePill").textContent = `Probes: ${state.probes_used}/${state.max_probes}`;
  const op = document.getElementById("outputPill");
  if(state.output_value !== undefined && state.output_value !== null){
    op.style.display = "inline-block";
    op.textContent = `Output: ${state.output_value?1:0}`;
    op.style.background = state.output_value ? "#9be564" : "#f08888";
  } else {
    op.style.display = "none";
  }

  // Diagnosis bar
  if(selectedGate !== null && !state.diagnosis_submitted){
    document.getElementById("diagBar").classList.remove("hidden");
    const label = state.nodes.find(n=>n.id===selectedGate)?.label || `?${selectedGate}`;
    document.getElementById("diagTarget").textContent = `Fault at ${label}?`;
  } else {
    document.getElementById("diagBar").classList.add("hidden");
  }

  // Disable set-input if game over
  document.getElementById("setInputBtn").disabled = state.diagnosis_submitted;

  // Events
  document.getElementById("events").innerHTML = (state.events||[]).map(e=>`<div class="event">${esc(e)}</div>`).join("");

  // Subtitle
  const sub = document.getElementById("subtitle");
  if(state.diagnosis_submitted){
    sub.innerHTML = state.correct
      ? '<span style="color:#3a3">✓ Correct!</span>'
      : '<span style="color:#c44">✗ Wrong</span>';
  } else if(state.output_value !== null && state.output_value !== undefined){
    sub.textContent = `Find the faulty gate (${state.n_inputs} inputs, ${state.n_gates} gates, ${state.max_probes} probes max)`;
  } else {
    sub.textContent = `Find the faulty gate — toggle inputs and click Set Input Vector to begin.`;
  }
}

let probeResults = {};
function onGateClick(nodeId){
  if(state.diagnosis_submitted) return;
  if(!state.input_vector){
    alert("Set input vector first — toggle the input switches, then click 'Set Input Vector'.");
    return;
  }
  // If already selected, deselect; otherwise select for diagnosis
  if(selectedGate === nodeId){
    selectedGate = null;
    render();
    return;
  }
  // Probe if we haven't probed this one yet and have probes left
  if(state.probes_used < state.max_probes){
    api("api/probe",{node_id:nodeId}).then(r=>{state=r;render();});
  }
  selectedGate = nodeId;
  render();
}

function toggleInput(idx){
  if(state.diagnosis_submitted) return;
  if(!state.input_vector) state.input_vector = new Array(state.n_inputs).fill(false);
  state.input_vector[idx] = !state.input_vector[idx];
  render();
}

document.getElementById("setInputBtn").onclick = async ()=>{
  const bits = state.input_vector || new Array(state.n_inputs).fill(false);
  state = await api("api/set-input",{inputs:bits});
  selectedGate = null;
  render();
};

function cancelDiag(){ selectedGate = null; render(); }

async function doSubmit(faultType){
  if(!selectedGate) return;
  state = await api("api/submit",{node_id:selectedGate,fault_type:faultType});
  selectedGate = null;
  render();
}

async function agentMove(){
  if(state.diagnosis_submitted) return;
  document.getElementById("agentBtn").disabled = true;
  document.getElementById("autoBtn").disabled = true;
  document.getElementById("agentBtn").textContent = "Thinking...";
  state = await api("api/agent", {});
  selectedGate = null;
  document.getElementById("agentBtn").disabled = state.diagnosis_submitted;
  document.getElementById("agentBtn").textContent = state.diagnosis_submitted ? "Game Over" : "Agent Move";
  render();
  return state;
}

document.getElementById("agentBtn").onclick = () => agentMove();

let autoRunning = false;
document.getElementById("autoBtn").onclick = async ()=>{
  if(autoRunning) return;
  autoRunning = true;
  document.getElementById("autoBtn").textContent = "Running...";
  while(!state || !state.diagnosis_submitted){
    await agentMove();
    await new Promise(r => setTimeout(r, 300));
  }
  autoRunning = false;
  document.getElementById("autoBtn").textContent = "Auto Play";
};

document.getElementById("newBtn").onclick = async ()=>{
  state = await api("api/new");
  selectedGate = null;
  probeResults = {};
  render();
};

function esc(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));}

refresh();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the logic-circuit diagnosis cartoon web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent", action="store_true", help="Use an LLM agent via OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    server = DiagnosisServer((args.host, args.port), DiagnosisHandler, seed=args.seed, args=args)
    url = f"http://{args.host}:{args.port}"
    print(f"Logic Diagnosis web UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()