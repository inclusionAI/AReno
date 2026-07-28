"""Cartoon web UI server for the Battleship agentic example.

Run from the repository root, pointing at an OpenAI-compatible endpoint
(typically an `areno serve` instance) so the on-screen agent fires real shots:

    areno serve --model-path /path/to/model --port 8000 --world-size 1
    python examples/agentic/battleship/web_ui.py \
        --base-url http://127.0.0.1:8000/v1 --api-key token

Without --base-url the agent still plays in --agent-mode heuristic (hunt/target)
or you can play solo by clicking cells.

Run from a checkout to generate a seeded fleet:
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
import game  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768
MAX_EVENTS = 30

# Reuse the same tool contract the training agent (run_agent.py) exposes so
# behavior is identical between the UI and an actual agentic rollout.
FIRE_TOOL = {
    "type": "function",
    "function": {
        "name": "fire",
        "description": "Fire a shot at a coordinate on the Battleship board.",
        "parameters": {
            "type": "object",
            "properties": {
                "coordinate": {
                    "type": "string",
                    "description": "Coordinate such as 'A1', 'B7', 'H8'. Letter row A-H, number column 1-8.",
                    "pattern": "^[A-H][1-8]$",
                },
            },
            "required": ["coordinate"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are playing Battleship. Use the fire(coordinate) tool to sink all ships.\n\n"
    "Rules:\n"
    "- The grid is 8x8, columns numbered 1-8, rows labeled A-H.\n"
    "- Coordinates are letter+number, e.g. A1 (top-left), H8 (bottom-right).\n"
    "- The fire tool returns: miss (no ship), hit (ship not yet sunk), sunk (ship destroyed).\n"
    "- Do not fire at the same coordinate twice. Do not fire outside A1-H8.\n"
    "- Win by sinking all ships in as few shots as possible.\n"
    "- The board shows your hits as X, misses as o, and unknown cells as '.'."
)


class BattleshipServer(ThreadingHTTPServer):
    """Small stateful HTTP server for one local Battleship game."""

    def __init__(self, server_address, request_handler, *, seed: int, args):
        super().__init__(server_address, request_handler)
        self.args = args
        self.rng = random.Random(seed)
        self.seed = seed
        self.state: game.GameState | None = None
        self.agent_mode = args.agent_mode
        self.events: list[str] = []
        self.openai_client = None
        self._reset(seed=seed)


    def _reset(self, *, seed: int) -> None:
        """Start a fresh seeded fleet."""
        self.seed = seed
        self.rng = random.Random(seed)
        record = game.place_fleet(seed)
        self.state = game.init_state(record)
        self.state.seed = seed
        self.events = [
            f"New game. Fleet placed from seed {seed}. Ships remaining: {len(game.SHIPS)}.",
            f"Agent mode: {self.agent_mode}. Click a cell to fire, or use an agent action.",
        ]


class BattleshipHandler(BaseHTTPRequestHandler):
    server: BattleshipServer

    def do_GET(self) -> None:
        route = _route_path(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            self._send_json(_payload(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route_path(self.path)
        if route == "new":
            body = self._read_json()
            seed = body.get("seed") if isinstance(body, dict) else None
            agent_mode = body.get("agent_mode") if isinstance(body, dict) else None
            self.server.agent_mode = agent_mode or self.server.agent_mode
            self.server._reset(seed=_coerce_seed(seed, fallback=self.server.seed))
            self._send_json(_payload(self.server))
        elif route == "fire":
            body = self._read_json()
            coord = body.get("coordinate") if isinstance(body, dict) else None
            self._send_json(_fire(self.server, coord, source="Human"))
        elif route == "agent":
            self._send_json(_agent_move(self.server))
        elif route == "autoplay":
            body = self._read_json()
            steps = body.get("steps") if isinstance(body, dict) else None
            self._send_json(_autoplay(self.server, _coerce_steps(steps)))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("battleship-web: " + fmt % args + "\n")

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


def _route_path(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    if path.endswith("/api/state"):
        return "state"
    if path.endswith("/api/new"):
        return "new"
    if path.endswith("/api/fire"):
        return "fire"
    if path.endswith("/api/agent"):
        return "agent"
    if path.endswith("/api/autoplay"):
        return "autoplay"
    if "/api/" in path:
        return "missing"
    if path == "/" or not path.rsplit("/", 1)[-1].count("."):
        return "index"
    return "missing"


def _coerce_seed(value: Any, *, fallback: int) -> int:
    try:
        if value is None:
            return random.Random().randint(1, 1_000_000)
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_steps(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = game.MAX_TURNS
    return max(0, min(n, game.MAX_TURNS))


def _record_event(server: BattleshipServer, message: str) -> None:
    server.events.insert(0, message)
    del server.events[MAX_EVENTS:]


def _fire(server: BattleshipServer, coord: Any, *, source: str) -> dict[str, Any]:
    state = server.state
    if state is None:
        return _payload(server)
    if game.is_terminal(state):
        _record_event(server, "Game over. Start a new game.")
        return _payload(server)
    if coord in (None, ""):
        _record_event(server, f"{source} sent no coordinate.")
        return _payload(server)
    result = game.fire(state, coord)
    label = str(coord).upper()
    if result["status"] == "invalid":
        _record_event(server, f"{source} fired {label}: rejected ({result.get('reason', 'invalid')}).")
    elif result["status"] == "miss":
        _record_event(server, f"{source} fired {label}: miss. Shots used {result['shots_used']}/{game.MAX_TURNS}.")
    elif result["status"] == "hit":
        _record_event(server, f"{source} fired {label}: hit! Ships remaining {result['remaining']}.")
    elif result["status"] == "sunk":
        _record_event(server, f"{source} fired {label}: sunk a ship (length {result.get('ship_sunk')})! Remaining {result['remaining']}.")
    if game.is_win(state):
        _record_event(server, f"Victory! All ships sunk in {state.shots_used} shots.")
    elif game.is_terminal(state):
        _record_event(server, f"Turn cap reached. Ships remaining: {sum(1 for s in state.ships if not s.is_sunk)}.")
    return _payload(server)


def _agent_move(server: BattleshipServer) -> dict[str, Any]:
    state = server.state
    if state is None or game.is_terminal(state):
        return _payload(server)
    try:
        coord = _agent_shot(server)
    except Exception as exc:  # noqa: BLE001
        _record_event(server, f"Agent failed: {exc}")
        return _payload(server)
    return _fire(server, coord, source=f"Agent({server.agent_mode})")


def _autoplay(server: BattleshipServer, steps: int) -> dict[str, Any]:
    """Run up to `steps` agent shots, stopping at a terminal state."""
    state = server.state
    if state is None:
        return _payload(server)
    fired = 0
    while fired < steps and not game.is_terminal(state):
        try:
            coord = _agent_shot(server)
        except Exception as exc:  # noqa: BLE001
            _record_event(server, f"Agent failed during autoplay: {exc}")
            break
        _fire(server, coord, source=f"Agent({server.agent_mode})")
        fired += 1
    return _payload(server)


def _agent_shot(server: BattleshipServer) -> str:
    if server.agent_mode == "llm":
        return _llm_shot(server)
    return _heuristic_shot(server)


def _llm_shot(server: BattleshipServer) -> str:
    if not server.args.base_url:
        raise ValueError("LLM agent mode requires --base-url")
    if server.openai_client is None:
        server.openai_client = _make_openai_client(server.args)
    state = server.state
    response = server.openai_client.chat.completions.create(
        model=server.args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _turn_prompt(state)},
        ],
        tools=[FIRE_TOOL],
        tool_choice={"type": "function", "function": {"name": "fire"}},
    )
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    tool_calls = choices[0].get("message", {}).get("tool_calls", []) if choices else []
    for call in tool_calls:
        if call.get("function", {}).get("name") != "fire":
            continue
        args = call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        coord = args.get("coordinate")
        if coord and game.parse_coordinate(str(coord)) is not None:
            return str(coord).upper()
    raise ValueError("LLM did not return a valid fire tool call")


def _heuristic_shot(server: BattleshipServer) -> str:
    """Hunt/target heuristic: once a hit lands, fire neighbors of known hits."""
    state = server.state
    legal = game.legal_shots(state)
    if not legal:
        raise ValueError("no legal shots remain")
    # Target phase: collect cells adjacent to unresolved hits (hits not part of a sunk ship).
    hit_cells = set()
    sunk_cells = set()
    for ship in state.ships:
        for cell in ship.cells:
            if cell in ship.hits:
                (sunk_cells if ship.is_sunk else hit_cells).add(cell)
    open_hits = hit_cells - sunk_cells
    candidates = [n for hit in open_hits for n in _neighbors(hit) if n in legal]
    if candidates:
        return game.format_coordinate(*server.rng.choice(candidates))
    # Hunt phase: prefer a checkerboard spread so every ship cell is reachable.
    spread = [cell for cell in legal if (cell[0] + cell[1]) % 2 == 0]
    pool = spread or list(legal)
    return game.format_coordinate(*server.rng.choice(pool))


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]


def _turn_prompt(state: game.GameState) -> str:
    legal = sorted(game.format_coordinate(r, c) for (r, c) in game.legal_shots(state))
    return (
        "You are playing Battleship. Sink all hidden ships on the 8x8 grid.\n\n"
        "Legend: X = your hit, o = your miss, . = unknown.\n"
        "Call the fire tool with a coordinate like 'C5'. Do not repeat a cell.\n\n"
        f"Board:\n{game.board_text(state)}\n\n"
        f"Legal shots remaining: {len(legal)}\n\nYour shot:"
    )


def _payload(server: BattleshipServer) -> dict[str, Any]:
    state = server.state
    hits: list[list[int]] = [[0] * game.GRID for _ in range(game.GRID)]
    misses: list[list[int]] = [[0] * game.GRID for _ in range(game.GRID)]
    if state is not None:
        hit_set = set()
        for ship in state.ships:
            for cell in ship.cells:
                if cell in ship.hits:
                    hit_set.add(cell)
        for (r, c) in state.shots_history:
            if (r, c) in hit_set:
                hits[r][c] = 1
            else:
                misses[r][c] = 1
    cells = []
    for r in range(game.GRID):
        row = []
        for c in range(game.GRID):
            row.append("hit" if hits[r][c] else "miss" if misses[r][c] else "unknown")
        cells.append(row)
    return {
        "grid_size": game.GRID,
        "max_turns": game.MAX_TURNS,
        "total_ship_cells": game.TOTAL_SHIP_CELLS,
        "cells": cells,
        "shots_used": state.shots_used if state else 0,
        "hits": sum(len(s.hits) for s in state.ships) if state else 0,
        "sunk_ships": sum(1 for s in state.ships if s.is_sunk) if state else 0,
        "ships_total": len(game.SHIPS),
        "seed": server.seed,
        "agent_mode": server.agent_mode,
        "win": game.is_win(state) if state else False,
        "terminal": game.is_terminal(state) if state else False,
        "events": server.events[:MAX_EVENTS],
    }


def _make_openai_client(args):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("LLM mode requires `openai`. Install it with `pip install openai`.") from exc
    return OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Battleship</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#24313a}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#0b1f3a,#13325f 55%,#1f6f8b);display:grid;place-items:center;padding:18px}
.app{width:min(1020px,96vw);display:grid;grid-template-columns:minmax(360px,520px) 1fr;gap:22px;align-items:start}
.panel{background:#f4f7fb;border:4px solid #0b1f3a;border-radius:24px;box-shadow:8px 8px 0 rgba(0,0,0,.35);padding:18px}
h1{font-size:34px;line-height:1;margin:0 0 6px;color:#13325f;text-shadow:2px 2px 0 #7fb3d5}
.subtitle{font-weight:900;color:#3a4a5d;margin-bottom:14px}
.board{background:#0b1f3a;border:7px solid #0b1f3a;border-radius:18px;padding:10px;display:grid;grid-template-columns:auto repeat(8,1fr);grid-template-rows:auto repeat(8,1fr);gap:6px;box-shadow:inset 0 7px 0 rgba(255,255,255,.06),inset 0 -10px 0 rgba(0,0,0,.3)}
.lbl{display:grid;place-items:center;font-weight:1000;color:#9fc3dd;font-size:14px}
.cell{aspect-ratio:1;background:#2a5b8c;border:3px solid #0b1f3a;border-radius:10px;display:grid;place-items:center;position:relative;cursor:pointer;transition:.12s transform}
.cell:hover:not(.disabled){transform:translateY(-2px)}
.cell.disabled{cursor:default}
.cell.unknown::after{content:"";width:10px;height:10px;border-radius:50%;background:rgba(255,255,255,.18)}
.cell.miss{background:#5d7b91}
.cell.miss::after{content:"o";color:#dde7f0;font-weight:1000;font-size:20px}
.cell.hit{background:#e05d2f;animation:pop .3s ease}
.cell.hit::after{content:"X";color:#fff;font-weight:1000;font-size:22px}
.cell.recent{outline:3px solid #ffd166}
@keyframes pop{0%{transform:scale(.7)}70%{transform:scale(1.1)}100%{transform:scale(1)}}
.stats,.actions,.mode-control,.seed-control{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.pill{background:#fff;border:3px solid #0b1f3a;border-radius:999px;padding:8px 12px;font-weight:1000;font-size:14px}
button{border:3px solid #0b1f3a;border-radius:14px;background:#ffd166;box-shadow:4px 4px 0 #0b1f3a;color:#0b1f3a;font-weight:1000;padding:10px 14px;cursor:pointer}
button:hover:not(:disabled){transform:translateY(-1px)}
button:disabled{filter:grayscale(.7);opacity:.55;cursor:not-allowed}
.choice.active{background:#9be564}
.label{font-weight:1000;color:#3a4a5d;align-self:center}
.thinking{display:none;margin:10px 0;padding:10px 12px;border:3px solid #0b1f3a;border-radius:14px;background:#dceefb;font-weight:900}
.thinking.on{display:block}
.dots::after{content:"";animation:dots 1s steps(4,end) infinite}
@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}100%{content:""}}
.rules{font-weight:800;line-height:1.5}
.rules li{margin:7px 0}
.events{display:grid;gap:7px;margin-top:14px;max-height:300px;overflow:auto}
.event{background:#fff;border:3px solid #0b1f3a;border-radius:12px;padding:8px 10px;font-weight:800;font-size:13px}
input[type=number]{width:90px;border:3px solid #0b1f3a;border-radius:12px;padding:8px;font-weight:900}
@media(max-width:820px){.app{grid-template-columns:1fr}h1{font-size:28px}}
</style>
</head>
<body>
<main class="app">
  <section class="panel">
    <h1>Battleship</h1>
    <div class="subtitle">Sink the hidden fleet with an agentic LLM.</div>
    <div id="board" class="board" aria-label="Battleship board"></div>
    <div id="thinking" class="thinking">Agent is thinking<span class="dots"></span></div>
    <div class="stats">
      <div class="pill" id="turnPill"></div>
      <div class="pill" id="hitsPill"></div>
      <div class="pill" id="modePill"></div>
    </div>
    <div class="mode-control" aria-label="Agent mode selector">
      <span class="label">Agent</span>
      <button class="choice active" id="llmMode">LLM</button>
      <button class="choice" id="heurMode">Heuristic</button>
    </div>
    <div class="actions">
      <button id="agentBtn">Agent Fires Once</button>
      <button id="autoBtn">Auto-play</button>
      <button id="newBtn">New Game</button>
    </div>
    <div class="seed-control">
      <span class="label">Seed</span>
      <input id="seedInput" type="number" value="2026">
    </div>
  </section>
  <aside class="panel">
    <h1 style="font-size:26px;color:#1f6f8b">How to play</h1>
    <ul class="rules">
      <li>8x8 grid, fleet of [4,3,2,2] = 11 hidden cells. Max 64 shots.</li>
      <li><b>Human:</b> click any unknown cell to fire. X = hit, o = miss.</li>
      <li><b>Agent (LLM):</b> start an OpenAI-compatible server (e.g. <code>areno serve</code>) and pass <code>--base-url</code>. Then "Agent Fires Once" or "Auto-play".</li>
      <li><b>Agent (Heuristic):</b> no server needed; uses a hunt/target strategy.</li>
      <li>Change the seed and click New Game to replay a fixed fleet.</li>
    </ul>
    <h1 style="font-size:20px;color:#1f6f8b;margin-bottom:8px">Event log</h1>
    <div id="events" class="events"></div>
  </aside>
</main>
<script>
const api = (path) => new URL(path, window.location.href).toString();
let state = null, recent = null, agentBusy = false, autoPlay = false, agentMode = "llm", lastShots = 0;
async function request(path, body){
  const opts = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {};
  const res = await fetch(api(path), opts);
  state = await res.json();
  if (state.shots_used !== lastShots + 1 && state.shots_used !== lastShots) {
    // batch play; mark the most recent by nothing special
  }
  lastShots = state.shots_used;
  render();
}
function render(){
  const board = document.getElementById("board");
  board.innerHTML = "";
  const N = state.grid_size;
  board.appendChild(div("lbl", ""));
  for (let c = 1; c <= N; c++) board.appendChild(div("lbl", String(c)));
  const cols = "ABCDEFGH".slice(0, N);
  for (let r = 0; r < N; r++){
    board.appendChild(div("lbl", cols[r]));
    for (let c = 0; c < N; c++){
      const kind = state.cells[r][c];
      const coord = cols[r] + (c + 1);
      const d = document.createElement("button");
      d.className = `cell ${kind}`;
      d.dataset.coord = coord;
      d.disabled = agentBusy || state.terminal || kind !== "unknown";
      d.onclick = () => request("api/fire", {coordinate: coord});
      board.appendChild(d);
    }
  }
  agentMode = state.agent_mode || "llm";
  document.getElementById("llmMode").classList.toggle("active", agentMode === "llm");
  document.getElementById("heurMode").classList.toggle("active", agentMode === "heuristic");
  const status = state.terminal ? (state.win ? "Victory!" : "Out of turns") : `${state.shots_used}/${state.max_turns} shots`;
  document.getElementById("turnPill").textContent = status;
  document.getElementById("hitsPill").textContent = `Hits ${state.hits}/${state.total_ship_cells} · Sunk ${state.sunk_ships}/${state.ships_total}`;
  document.getElementById("modePill").textContent = `Agent: ${agentMode}`;
  document.getElementById("thinking").classList.toggle("on", agentBusy && agentMode === "llm");
  document.getElementById("agentBtn").disabled = agentBusy || state.terminal;
  document.getElementById("autoBtn").disabled = agentBusy || state.terminal;
  document.getElementById("autoBtn").textContent = autoPlay ? "Stop auto-play" : "Auto-play";
  document.getElementById("seedInput").value = state.seed;
  document.getElementById("events").innerHTML = state.events.map(e => `<div class="event">${escapeHtml(e)}</div>`).join("");
}
function div(cls, text){ const d = document.createElement("div"); d.className = cls; d.textContent = text; return d; }
function escapeHtml(text){return text.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));}
async function agentOnce(){
  if (!state || state.terminal || agentBusy) return;
  agentBusy = true; render();
  try { await request("api/agent", {}); } finally { agentBusy = false; render(); if (autoPlay && state && !state.terminal) setTimeout(agentOnce, 250); }
}
async function autoplayToggle(){
  if (autoPlay){ autoPlay = false; render(); return; }
  autoPlay = true; render();
  if (!agentBusy) agentOnce();
}
function setMode(m){
  agentMode = m;
  request("api/new", {seed: state.seed, agent_mode: m});
}
document.getElementById("llmMode").onclick = () => setMode("llm");
document.getElementById("heurMode").onclick = () => setMode("heuristic");
document.getElementById("agentBtn").onclick = agentOnce;
document.getElementById("autoBtn").onclick = autoplayToggle;
document.getElementById("newBtn").onclick = () => { autoPlay = false; request("api/new", {seed: document.getElementById("seedInput").value, agent_mode: agentMode}); };
request("api/state");
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Battleship cartoon web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--agent-mode", choices=("llm", "heuristic"), default="heuristic")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for LLM agent mode (e.g. areno serve).")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    server = BattleshipServer((args.host, args.port), BattleshipHandler, seed=args.seed, args=args)
    url = f"http://{args.host}:{args.port}"
    print(f"Battleship web UI running at {url}")
    print(f"Agent mode: {args.agent_mode}" + (f" | base_url={args.base_url}" if args.base_url else " | no --base-url (use heuristic or click to play)"))
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()