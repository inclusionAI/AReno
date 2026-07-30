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
    evaluate,
    generate_circuit,
    inject_fault,
    verify_diagnosis,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768


def _node_depth_by_id(nid: int, id_map: dict[int, dict[str, Any]]) -> int:
    node = id_map.get(nid)
    if node is None or node["type"] == "input":
        return 0
    inputs = node.get("inputs", [])
    if not inputs:
        return 1
    return 1 + max(_node_depth_by_id(inp, id_map) for inp in inputs)


class DiagnosisServer(ThreadingHTTPServer):
    """Stateful HTTP server for one local logic diagnosis game."""

    def __init__(self, server_address, request_handler, *, seed: int | None = None):
        super().__init__(server_address, request_handler)
        self.rng = random.Random(seed)
        self.nodes: list[dict[str, Any]] = []
        self.fault: dict[str, Any] = {}
        self.input_vector: list[bool] | None = None
        self.probes_used: int = 0
        self.diagnosis_submitted: bool = False
        self.correct: bool | None = None
        self.max_probes: int = MAX_PROBES
        self.n_inputs: int = 0
        self.events: list[str] = []
        self._new_game()

    def _new_game(self) -> None:
        for attempt in range(200):
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
            f"You have {self.max_probes} probes. Click a gate node to probe it.",
        ]


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
        server.events.insert(
            0,
            f"Set inputs to {_bits_str(bits)} → output = {int(output_val)}",
        )
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
        remaining = server.max_probes - server.probes_used
        server.events.insert(
            0,
            f"Probed {node['type'].upper()}{node_id} → value = {int(probed_val)} "
            f"({server.probes_used}/{server.max_probes} probes used)",
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
                0,
                f"✓ Correct! Fault was at {_node_label(server.nodes, node_id)} ({fault_type}). "
                f"Used {server.probes_used} probe(s).",
            )
        else:
            actual = server.fault
            actual_type = "stuck_at_0" if actual["stuck_value"] == 0 else "stuck_at_1"
            server.events.insert(
                0,
                f"✗ Wrong. Fault was at {_node_label(server.nodes, actual['node'])} ({actual_type}), "
                f"not {_node_label(server.nodes, node_id)} ({fault_type}).",
            )
        server.events = server.events[:12]
        self._send_json(_make_payload(server))


def _route_path(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    for name in ("state", "new", "set-input", "probe", "submit"):
        if path.endswith(f"/api/{name}"):
            return name
    if "/api/" in path:
        return "missing"
    return "index"


def _bits_str(bits: list[bool]) -> str:
    return "".join("1" if b else "0" for b in bits)


def _node_label(nodes: list[dict[str, Any]], nid: int) -> str:
    for n in nodes:
        if n["id"] == nid:
            t = n["type"]
            if t == "input":
                return f"IN{nid}"
            if t == "output":
                return "OUT"
            return f"{t.upper()}{nid}"
    return f"?{nid}"


def _node_depth(nid: int, id_map: dict[int, dict[str, Any]]) -> int:
    return _node_depth_by_id(nid, id_map)


def _make_payload(
    server: DiagnosisServer, *, probed_value: bool | None = None, probed_node: int | None = None
) -> dict[str, Any]:
    id_map = {n["id"]: n for n in server.nodes}

    # Build layered node list for frontend rendering
    nodes_payload = []
    for node in server.nodes:
        nid = node["id"]
        depth = _node_depth(nid, id_map)
        nodes_payload.append(
            {
                "id": nid,
                "type": node["type"],
                "label": _node_label(server.nodes, nid),
                "inputs": node.get("inputs", []),
                "depth": depth,
            }
        )

    # Fault info (only revealed after submission)
    fault_revealed = None
    if server.diagnosis_submitted:
        fault_revealed = {
            "node": server.fault["node"],
            "stuck_value": server.fault["stuck_value"],
        }

    return {
        "nodes": nodes_payload,
        "n_inputs": server.n_inputs,
        "n_gates": server.n_gates,
        "input_vector": server.input_vector,
        "probes_used": server.probes_used,
        "max_probes": server.max_probes,
        "diagnosis_submitted": server.diagnosis_submitted,
        "correct": server.correct,
        "fault_revealed": fault_revealed,
        "events": server.events,
        "probed_value": probed_value,
        "probed_node": probed_node,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Logic Circuit Diagnosis</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#1a1a2e;background:#e8e8f0}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#d4d4e4,#a8b8d8 40%,#88a8c8);display:grid;place-items:center}
.app{width:min(980px,96vw);display:grid;grid-template-columns:minmax(400px,550px) 1fr;gap:22px;align-items:start}
.panel{background:#f8f8fc;border:4px solid #1a1a2e;border-radius:22px;box-shadow:7px 7px 0 #1a1a2e;padding:16px}
h1{font-size:34px;line-height:1;margin:0 0 6px;color:#c44;text-shadow:2px 2px 0 #ffd977}
.subtitle{font-weight:900;color:#444;margin-bottom:12px}
.circuit{position:relative;background:#eef;border:3px solid #1a1a2e;border-radius:16px;padding:14px;min-height:240px;display:flex;flex-direction:column;align-items:center;gap:8px}
.layer{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.node{position:relative;border:3px solid #1a1a2e;border-radius:14px;padding:8px 14px;font-weight:1000;font-size:13px;text-align:center;cursor:default;box-shadow:3px 3px 0 #1a1a2e;transition:.12s transform;min-width:64px}
.node.input{background:#b8e6b8}.node.and{background:#ffd166}.node.or{background:#ffb347}.node.not{background:#e8a0d0}.node.output{background:#87ceeb}
.node.gate:not(.output):not(.input){cursor:pointer}.node.gate:not(.output):not(.input):hover{transform:translateY(-2px)}
.node.probed{animation:flash .5s ease}@keyframes flash{0%{background:#fff}100%{}}
.node.faulty{border-color:#c44;border-width:4px;box-shadow:3px 3px 0 #c44}
.input-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}
.input-switch{display:flex;flex-direction:column;align-items:center;gap:3px}
.input-switch button{width:44px;height:44px;border:3px solid #1a1a2e;border-radius:12px;font-size:20px;font-weight:900;cursor:pointer;box-shadow:3px 3px 0 #1a1a2e;transition:.1s}
.input-switch button.on{background:#9be564;color:#1a1a2e}.input-switch button.off{background:#e8e8e8;color:#999}
.input-switch span{font-size:11px;font-weight:800;color:#555}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
button{border:3px solid #1a1a2e;border-radius:14px;background:#ffd166;box-shadow:4px 4px 0 #1a1a2e;color:#1a1a2e;font-weight:1000;padding:10px 14px;cursor:pointer;font-size:14px}
button:hover{transform:translateY(-1px)}button:disabled{filter:grayscale(.7);opacity:.5;cursor:not-allowed}
button.danger{background:#f08888}
.pill{display:inline-block;background:#fff;border:3px solid #1a1a2e;border-radius:999px;padding:6px 12px;font-weight:1000;font-size:13px}
.status{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
.fault-choices{display:flex;gap:8px;margin-top:8px}.fault-choices button{font-size:12px;padding:8px 12px}
.rules{font-weight:800;line-height:1.5}.rules li{margin:4px 0}
.events{display:grid;gap:6px;margin-top:10px;max-height:340px;overflow-y:auto}.event{background:#fff;border:3px solid #1a1a2e;border-radius:12px;padding:8px 10px;font-weight:800;font-size:13px}
.selected-diagnosis{background:#ffe0e0;border:3px dashed #c44;border-radius:12px;padding:10px;margin-top:8px;display:none;font-weight:800;font-size:14px;align-items:center;gap:8px}
.selected-diagnosis.on{display:flex}
@media(max-width:800px){.app{grid-template-columns:1fr}}
</style>
</head>
<body>
<main class="app">
  <section class="panel">
    <h1>Logic Diagnosis</h1>
    <div class="subtitle" id="subtitle">Find the faulty gate in this circuit.</div>
    <div id="circuit" class="circuit"></div>
    <div class="input-row" id="inputRow"></div>
    <button id="setInputBtn">Set Input Vector</button>
    <div class="status">
      <span class="pill" id="probeCounter"></span>
      <span class="pill" id="outputPill" style="display:none"></span>
    </div>
    <div id="selectedDiagnosis" class="selected-diagnosis">
      <span id="diagTarget"></span>
      <div class="fault-choices" id="faultChoices"></div>
      <button class="danger" id="cancelDiag">✕</button>
    </div>
    <div class="actions">
      <button id="submitBtn" disabled>Submit Diagnosis</button>
      <button id="newBtn">New Circuit</button>
    </div>
  </section>
  <aside class="panel">
    <h1 style="font-size:24px;color:#227">How to Play</h1>
    <ul class="rules">
      <li>Toggle inputs above and click <b>Set Input Vector</b> to see the circuit output.</li>
      <li>Click any <b style="background:#ffd166;padding:1px 6px;border-radius:4px">AND</b>,
        <b style="background:#ffb347;padding:1px 6px;border-radius:4px">OR</b>, or
        <b style="background:#e8a0d0;padding:1px 6px;border-radius:4px">NOT</b> gate to probe its value (costs 1 probe).</li>
      <li>When confident, click a gate to select it, choose the fault type, and <b>Submit Diagnosis</b>.</li>
      <li>Fewer probes = better diagnosis! Try to deduce the fault by reasoning about expected vs actual outputs.</li>
    </ul>
    <div id="events" class="events"></div>
  </aside>
</main>
<script>
let state = null, selectedGate = null;

async function api(path, body){
  const opts = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {};
  const res = await fetch(new URL(path, window.location.href).toString(), opts);
  return await res.json();
}
async function refresh(){
  state = await api("api/state");
  render();
}
function render(){
  if(!state) return;

  // Circuit
  const circuit = document.getElementById("circuit");
  const maxDepth = Math.max(...state.nodes.map(n => n.depth), 0);
  let html = "";
  for(let d = maxDepth; d >= 0; d--){
    const layer = state.nodes.filter(n => n.depth === d);
    html += '<div class="layer">';
    for(const n of layer){
      let cls = `node ${n.type}`;
      if(n.type !== "input" && n.type !== "output") cls += " gate";
      if(state.probed_node === n.id) cls += " probed";
      if(state.fault_revealed && state.fault_revealed.node === n.id) cls += " faulty";
      const onclick = (n.type === "and" || n.type === "or" || n.type === "not")
        ? `onclick="onGateClick(${n.id})"` : "";
      const faultBadge = (state.fault_revealed && state.fault_revealed.node === n.id)
        ? ` ✦FAULT` : "";
      html += `<div class="${cls}" ${onclick}>${n.label}${faultBadge}</div>`;
    }
    html += '</div>';
  }
  circuit.innerHTML = html;

  // Input toggles
  const inputRow = document.getElementById("inputRow");
  inputRow.innerHTML = "";
  for(let i = 0; i < state.n_inputs; i++){
    const val = state.input_vector ? state.input_vector[i] : false;
    const cls = val ? "on" : "off";
    inputRow.innerHTML += `<div class="input-switch">
      <button class="${cls}" onclick="toggleInput(${i})">${val ? "1" : "0"}</button>
      <span>IN${i}</span></div>`;
  }

  // Probes
  document.getElementById("probeCounter").textContent = `Probes: ${state.probes_used}/${state.max_probes}`;

  // Output
  const op = document.getElementById("outputPill");
  if(state.input_vector !== null && !state.diagnosis_submitted){
    op.style.display = "inline-block";
    op.textContent = `Output: computing...`;
  } else {
    op.style.display = "none";
  }

  // Submit button
  document.getElementById("submitBtn").disabled = !selectedGate || state.diagnosis_submitted;
  document.getElementById("setInputBtn").disabled = state.diagnosis_submitted;

  // Diagnosis UI
  renderDiagnosis();

  // Events
  const eventsEl = document.getElementById("events");
  eventsEl.innerHTML = (state.events || []).map(e => `<div class="event">${esc(e)}</div>`).join("");

  // Subtitle
  if(state.diagnosis_submitted){
    document.getElementById("subtitle").innerHTML = state.correct
      ? '<span style="color:#2a2">✓ Correct Diagnosis!</span>'
      : '<span style="color:#c44">✗ Wrong Diagnosis</span>';
  } else {
    document.getElementById("subtitle").textContent = `Find the faulty gate: ${state.n_inputs} inputs, ${state.n_gates} gates`;
  }
}
function renderDiagnosis(){
  const diagEl = document.getElementById("selectedDiagnosis");
  const targetEl = document.getElementById("diagTarget");
  const choicesEl = document.getElementById("faultChoices");
  if(selectedGate && !state.diagnosis_submitted){
    diagEl.classList.add("on");
    const label = state.nodes.find(n => n.id === selectedGate)?.label || `?${selectedGate}`;
    targetEl.textContent = `Fault at ${label}:`;
    choicesEl.innerHTML = `
      <button onclick="submitDiagnosis(${selectedGate},'stuck_at_0')">Stuck-at-0</button>
      <button onclick="submitDiagnosis(${selectedGate},'stuck_at_1')">Stuck-at-1</button>`;
  } else {
    diagEl.classList.remove("on");
    targetEl.textContent = "";
    choicesEl.innerHTML = "";
  }
}
async function onGateClick(nodeId){
  if(state.diagnosis_submitted) return;
  // Probe the gate (costs 1 probe if still available)
  if(state.input_vector === null){
    // Auto-set all-0 input if none set
    state = await api("api/set-input", {inputs: new Array(state.n_inputs).fill(false)});
    render();
  }
  if(state.probes_used < state.max_probes){
    state = await api("api/probe", {node_id: nodeId});
  }
  // Select this gate for diagnosis
  selectedGate = nodeId;
  render();
}
function toggleInput(idx){
  if(state.diagnosis_submitted) return;
  if(!state.input_vector) state.input_vector = new Array(state.n_inputs).fill(false);
  state.input_vector[idx] = !state.input_vector[idx];
  render();
}
function setAllInputs(val){
  state.input_vector = new Array(state.n_inputs).fill(val);
}
document.getElementById("setInputBtn").onclick = async () => {
  const bits = state.input_vector || new Array(state.n_inputs).fill(false);
  state = await api("api/set-input", {inputs: bits});
  selectedGate = null;
  render();
};
document.getElementById("submitBtn").onclick = () => {}; // submit via fault choices
document.getElementById("cancelDiag").onclick = () => {selectedGate = null; render();};
document.getElementById("newBtn").onclick = async () => {
  state = await api("api/new");
  selectedGate = null;
  render();
};
async function submitDiagnosis(nodeId, faultType){
  state = await api("api/submit", {node_id: nodeId, fault_type: faultType});
  selectedGate = null;
  render();
}
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
    args = parser.parse_args()

    server = DiagnosisServer((args.host, args.port), DiagnosisHandler, seed=args.seed)
    url = f"http://{args.host}:{args.port}"
    print(f"Logic Diagnosis web UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()