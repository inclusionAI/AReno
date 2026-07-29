"""Cartoon web UI server for the 2048 agentic example.

Run from the repository root:

    python examples/agentic/game2048/web_ui.py
    python examples/agentic/game2048/web_ui.py --base-url http://127.0.0.1:8000/v1 --api-key token
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

TILE_COLORS = {
    0: "#cdc1b4",
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}
TILE_TEXT_COLOR = {0: "#cdc1b4", 2: "#776e65", 4: "#776e65"}


class Game2048Server(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler, *, seed: int | None = None, args):
        super().__init__(server_address, request_handler)
        self.seed = seed if seed is not None else random.randint(0, 999999)
        self.rng = random.Random(self.seed)
        self.board = game.new_board(self.seed)
        self.score = 0
        self.moves = 0
        self.valid_moves = 0
        self.invalid_moves = 0
        self.agent_mode = args.agent_mode
        self.args = args
        self.openai_client = None
        self.events = [f"New game (seed={self.seed}). Use arrow keys or let the agent play."]


class Game2048Handler(BaseHTTPRequestHandler):
    server: Game2048Server

    def do_GET(self) -> None:
        route = _route_path(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            self._send_json(_payload(self.server))
        elif route == "agent":
            self._send_json(_agent_move(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route_path(self.path)
        if route == "new":
            body = self._read_json()
            agent_mode = body.get("agent_mode") if isinstance(body, dict) else None
            self._send_json(_reset(self.server, agent_mode=agent_mode))
        elif route == "move":
            body = self._read_json()
            direction = body.get("direction") if isinstance(body, dict) else None
            self._send_json(_move(self.server, direction))
        elif route == "agent":
            self._send_json(_agent_move(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("game2048-web: " + fmt % args + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _route_path(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    if path.endswith("/api/state"):
        return "state"
    if path.endswith("/api/new"):
        return "new"
    if path.endswith("/api/move"):
        return "move"
    if path.endswith("/api/agent"):
        return "agent"
    if "/api/" in path:
        return "missing"
    if path == "/" or not path.rsplit("/", 1)[-1].count("."):
        return "index"
    return "missing"


def _reset(server: Game2048Server, *, agent_mode: Any = None) -> dict[str, Any]:
    if agent_mode in {"llm", "random", "greedy"}:
        server.agent_mode = str(agent_mode)
    server.seed = random.randint(0, 999999)
    server.rng = random.Random(server.seed)
    server.board = game.new_board(server.seed)
    server.score = 0
    server.moves = 0
    server.valid_moves = 0
    server.invalid_moves = 0
    server.events = [f"New game (seed={server.seed}). Agent mode: {_agent_name(server)}."]
    return _payload(server)


def _move(server: Game2048Server, direction: Any) -> dict[str, Any]:
    if game.is_terminal(server.board):
        server.events.insert(0, "Game over. Start a new game.")
        return _payload(server)
    direction = str(direction).upper() if direction else ""
    if direction not in game.DIRECTIONS:
        server.events.insert(0, f"Invalid direction: {direction}")
        server.events = server.events[:10]
        return _payload(server)

    new_board, score, valid, terminal = game.move(server.board, direction, server.rng)
    server.moves += 1
    if valid:
        server.board = new_board
        server.score += score
        server.valid_moves += 1
        mt = game.max_tile(server.board)
        if score > 0:
            server.events.insert(0, f"{direction}: +{score} pts, max={mt}")
        else:
            server.events.insert(0, f"{direction}: moved, max={mt}")
    else:
        server.invalid_moves += 1
        server.events.insert(0, f"{direction}: no change (invalid)")

    if terminal:
        server.events.insert(0, f"Game over! Final score: {server.score}, max tile: {game.max_tile(server.board)}")
    server.events = server.events[:10]
    return _payload(server)


def _agent_move(server: Game2048Server) -> dict[str, Any]:
    if game.is_terminal(server.board):
        return _payload(server)
    try:
        direction = _agent_direction(server)
    except Exception as exc:
        server.events.insert(0, f"{_agent_name(server)} failed: {exc}")
        server.events = server.events[:10]
        return _payload(server)
    return _move(server, direction)


def _agent_direction(server: Game2048Server) -> str:
    if server.agent_mode == "random":
        return game.random_action(server.board, server.rng)
    if server.agent_mode == "greedy":
        return _greedy_direction(server.board, server.rng)
    return _llm_direction(server)


def _greedy_direction(board: game.Board, rng: random.Random) -> str:
    """Pick the direction that yields the highest merge score."""
    best_dir = None
    best_score = -1
    for direction in game.DIRECTIONS:
        _, score = game._slide(board, direction)
        if score > best_score or (score == best_score and best_dir is None):
            best_score = score
            best_dir = direction
    if best_score <= 0:
        return game.random_action(board, rng)
    return best_dir


def _llm_direction(server: Game2048Server) -> str:
    if not server.args.base_url:
        raise ValueError("LLM mode requires --base-url")
    if server.openai_client is None:
        server.openai_client = _make_openai_client(server.args)
    response = server.openai_client.chat.completions.create(
        model=server.args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": game.format_prompt(server.board)},
        ],
        tools=[game.MOVE_TOOL],
        tool_choice={"type": "function", "function": {"name": "move"}},
    )
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    tool_calls = choices[0].get("message", {}).get("tool_calls", []) if choices else []
    for call in tool_calls:
        if call.get("function", {}).get("name") != "move":
            continue
        args = call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        direction = str(args.get("direction", "")).upper()
        if direction in game.DIRECTIONS:
            return direction
    raise ValueError("response did not contain a valid move tool call")


SYSTEM_PROMPT = (
    "You are a strategic 2048 player. Call the move tool with a direction "
    "(UP, DOWN, LEFT, RIGHT) to swipe the board. Merge identical tiles to grow "
    "the largest tile possible. Avoid moves that do not change the board."
)


def _agent_name(server: Game2048Server) -> str:
    return {"llm": "LLM", "random": "Random", "greedy": "Greedy"}.get(server.agent_mode, server.agent_mode)


def _payload(server: Game2048Server) -> dict[str, Any]:
    return {
        "board": server.board,
        "score": server.score,
        "moves": server.moves,
        "valid_moves": server.valid_moves,
        "invalid_moves": server.invalid_moves,
        "max_tile": game.max_tile(server.board),
        "terminal": game.is_terminal(server.board),
        "legal_directions": game.legal_directions(server.board),
        "agent_mode": server.agent_mode,
        "agent_name": _agent_name(server),
        "seed": server.seed,
        "events": server.events,
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
<title>2048</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#776e65;background:#faf8ef}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#faf8ef,#f6f0e0 50%,#eee4da);display:grid;place-items:center}
.app{width:min(560px,94vw);display:flex;flex-direction:column;gap:16px;align-items:center}
.panel{background:#bbada0;border-radius:16px;padding:20px;box-shadow:0 4px 20px rgba(0,0,0,.15)}
h1{font-size:48px;line-height:1;margin:0;color:#776e65;font-weight:800}
.subtitle{font-weight:700;color:#9c8a7a;margin-bottom:12px}
.header{display:flex;justify-content:space-between;align-items:flex-end;width:100%;gap:12px}
.scores{display:flex;gap:8px}
.score-box{background:#bbada0;border-radius:10px;padding:8px 16px;text-align:center;min-width:80px}
.score-box .label{font-size:11px;text-transform:uppercase;color:#eee4da;font-weight:700;letter-spacing:1px}
.score-box .value{font-size:24px;color:#fff;font-weight:800}
.board{background:#bbada0;border-radius:12px;padding:12px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px;box-shadow:inset 0 2px 8px rgba(0,0,0,.15)}
.tile{aspect-ratio:1;border-radius:8px;display:grid;place-items:center;font-weight:800;transition:.12s;position:relative}
.tile.new{animation:pop .2s ease}@keyframes pop{0%{transform:scale(.7)}60%{transform:scale(1.08)}100%{transform:scale(1)}}
.tile .num{font-size:clamp(20px,5vw,36px)}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;justify-content:center}
.pill{background:#fff;border:2px solid #bbada0;border-radius:999px;padding:6px 14px;font-weight:700;color:#776e65;font-size:14px}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{border:2px solid #bbada0;border-radius:10px;background:#8f7a66;color:#fff;font-weight:700;padding:10px 16px;cursor:pointer;font-size:14px;transition:.12s}
button:hover{transform:translateY(-1px);background:#9f8b77}button:disabled{opacity:.5;cursor:not-allowed}
.mode-btn.active{background:#f59563;border-color:#f67c5f}
.mode-control{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:10px}
.label{font-weight:700;color:#9c8a7a;font-size:13px;align-self:center;margin-right:4px}
.thinking{display:none;margin:10px 0;padding:8px 14px;border:2px solid #bbada0;border-radius:10px;background:#eee4da;font-weight:700;text-align:center}
.thinking.on{display:block}
.dots::after{content:"";animation:dots 1s steps(4,end) infinite}@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}100%{content:""}}
.events{width:100%;display:grid;gap:6px;margin-top:14px;max-height:200px;overflow-y:auto}
.event{background:#eee4da;border-radius:8px;padding:8px 12px;font-weight:600;color:#776e65;font-size:13px}
.game-over{position:fixed;inset:0;background:rgba(0,0,0,.5);display:none;place-items:center;z-index:10}
.game-over.on{display:grid}
.game-over .box{background:#faf8ef;border-radius:16px;padding:32px 48px;text-align:center;box-shadow:0 8px 40px rgba(0,0,0,.3)}
.game-over .box h2{font-size:32px;margin:0 0 8px;color:#776e65}
.game-over .box p{font-size:16px;color:#9c8a7a;margin:0 0 20px}
@media(max-width:480px){h1{font-size:36px}.tile .num{font-size:18px}}
</style>
</head>
<body>
<main class="app">
  <div class="header">
    <div>
      <h1>2048</h1>
      <div class="subtitle">Play with arrow keys or let the agent play.</div>
    </div>
    <div class="scores">
      <div class="score-box"><div class="label">Score</div><div class="value" id="score">0</div></div>
      <div class="score-box"><div class="label">Max</div><div class="value" id="maxTile">0</div></div>
    </div>
  </div>
  <div class="panel">
    <div id="board" class="board" aria-label="2048 board"></div>
  </div>
  <div id="thinking" class="thinking">LLM is thinking<span class="dots"></span></div>
  <div class="stats">
    <div class="pill" id="moves">0 moves</div>
    <div class="pill" id="valid">0 valid</div>
    <div class="pill" id="invalid">0 invalid</div>
    <div class="pill" id="seed">seed: 0</div>
  </div>
  <div class="mode-control">
    <span class="label">Agent:</span>
    <button class="mode-btn active" id="llmMode">LLM</button>
    <button class="mode-btn" id="randomMode">Random</button>
    <button class="mode-btn" id="greedyMode">Greedy</button>
  </div>
  <div class="actions">
    <button id="agentMove">Agent Move</button>
    <button id="autoPlay">Auto Play</button>
    <button id="new">New Game</button>
  </div>
  <div id="events" class="events"></div>
</main>
<div id="gameOver" class="game-over">
  <div class="box">
    <h2>Game Over</h2>
    <p id="gameOverText"></p>
    <button onclick="document.getElementById('gameOver').classList.remove('on');request('api/new',{});">New Game</button>
  </div>
</div>
<script>
const api = (path) => new URL(path, window.location.href).toString();
let state = null, agentBusy = false, autoPlay = false, lastBoard = "";
const DIR_KEYS = {ArrowUp:"UP",ArrowDown:"DOWN",ArrowLeft:"LEFT",ArrowRight:"RIGHT"};
const TILE_COLORS = {0:"#cdc1b4",2:"#eee4da",4:"#ede0c8",8:"#f2b179",16:"#f59563",32:"#f67c5f",64:"#f65e3b",128:"#edcf72",256:"#edcc61",512:"#edc850",1024:"#edc53f",2048:"#edc22e"};
const TEXT_COLORS = {0:"#cdc1b4",2:"#776e65",4:"#776e65"};

async function request(path, body){
  const opts = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {};
  const res = await fetch(api(path), opts);
  state = await res.json();
  render();
  if(autoPlay && !state.terminal && !agentBusy) doAgentMove();
  if(state.terminal && autoPlay){ autoPlay = false; document.getElementById("autoPlay").textContent="Auto Play"; }
}
function render(){
  const board = document.getElementById("board");
  const key = JSON.stringify(state.board);
  board.innerHTML = "";
  state.board.flat().forEach((val, idx) => {
    const div = document.createElement("div");
    div.className = "tile" + (key !== lastBoard ? " new" : "");
    div.style.background = TILE_COLORS[val] || "#3c3a32";
    div.style.color = TEXT_COLORS[val] || "#fff";
    if(val > 0) div.innerHTML = `<span class="num">${val}</span>`;
    board.appendChild(div);
  });
  lastBoard = key;
  document.getElementById("score").textContent = state.score;
  document.getElementById("maxTile").textContent = state.max_tile;
  document.getElementById("moves").textContent = state.moves + " moves";
  document.getElementById("valid").textContent = state.valid_moves + " valid";
  document.getElementById("invalid").textContent = state.invalid_moves + " invalid";
  document.getElementById("seed").textContent = "seed: " + state.seed;
  document.getElementById("thinking").classList.toggle("on", agentBusy && state.agent_mode === "llm");
  const mode = state.agent_mode || "llm";
  document.getElementById("llmMode").classList.toggle("active", mode === "llm");
  document.getElementById("randomMode").classList.toggle("active", mode === "random");
  document.getElementById("greedyMode").classList.toggle("active", mode === "greedy");
  document.getElementById("agentMove").disabled = agentBusy || state.terminal;
  document.getElementById("autoPlay").disabled = state.terminal;
  document.getElementById("events").innerHTML = state.events.map(e => `<div class="event">${escapeHtml(e)}</div>`).join("");
  if(state.terminal){
    document.getElementById("gameOverText").textContent = `Score: ${state.score}, Max tile: ${state.max_tile}, Moves: ${state.moves}`;
    document.getElementById("gameOver").classList.add("on");
  }
}
async function doAgentMove(){
  if(!state || state.terminal || agentBusy) return;
  agentBusy = true;
  render();
  try { await request("api/agent", {}); }
  finally { agentBusy = false; render(); }
}
function escapeHtml(text){return text.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));}
function setMode(m){
  const body = {agent_mode: m};
  fetch(api("api/new"), {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json()).then(s=>{state=s;render();});
}
document.getElementById("agentMove").onclick = doAgentMove;
document.getElementById("autoPlay").onclick = () => {
  autoPlay = !autoPlay;
  document.getElementById("autoPlay").textContent = autoPlay ? "Stop" : "Auto Play";
  if(autoPlay && !state.terminal) doAgentMove();
};
document.getElementById("new").onclick = () => request("api/new", {});
document.getElementById("llmMode").onclick = () => setMode("llm");
document.getElementById("randomMode").onclick = () => setMode("random");
document.getElementById("greedyMode").onclick = () => setMode("greedy");
window.addEventListener("keydown", (e) => {
  const dir = DIR_KEYS[e.key];
  if(dir && state && !state.terminal && !agentBusy && !autoPlay) request("api/move", {direction: dir});
});
request("api/state");
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2048 cartoon web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent-mode", choices=("llm", "random", "greedy"), default="random")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for LLM mode.")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    server = Game2048Server((args.host, args.port), Game2048Handler, seed=args.seed, args=args)
    url = f"http://{args.host}:{args.port}"
    print(f"2048 web UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()