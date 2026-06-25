"""Cartoon web UI server for the Tic-Tac-Toe agentic example.

Run from the repository root:

    python examples/agentic/tictactoe/web_ui.py --base-url http://127.0.0.1:8000/v1 --api-key token
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
DEFAULT_PORT = 8767

CHOOSE_SQUARE_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_square",
        "description": "Choose the next Tic-Tac-Toe square.",
        "parameters": {
            "type": "object",
            "properties": {
                "square": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "The 1-indexed square to place your mark in.",
                }
            },
            "required": ["square"],
            "additionalProperties": False,
        },
    },
}


class TicTacToeServer(ThreadingHTTPServer):
    """Small stateful HTTP server for one local Tic-Tac-Toe game."""

    def __init__(self, server_address, request_handler, *, seed: int | None = None, args):
        super().__init__(server_address, request_handler)
        self.rng = random.Random(seed)
        self.board = _empty_board()
        self.turn = "X"
        self.agent_first = True
        self.agent_player = "X"
        self.agent_mode = args.agent_mode
        self.human_player = "O"
        self.events = [f"New game. {_agent_name(self)} controls X and moves first."]
        self.args = args
        self.openai_client = None


class TicTacToeHandler(BaseHTTPRequestHandler):
    server: TicTacToeServer

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
            agent_first = body.get("agent_first") if isinstance(body, dict) else None
            agent_mode = body.get("agent_mode") if isinstance(body, dict) else None
            _reset(self.server, agent_first=agent_first, agent_mode=agent_mode)
            self._send_json(_payload(self.server))
        elif route == "move":
            body = self._read_json()
            square = body.get("square") if isinstance(body, dict) else None
            self._send_json(_move(self.server, square, self.server.human_player))
        elif route == "agent":
            self._send_json(_agent_move(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("tictactoe-web: " + fmt % args + "\n")

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
    if path.endswith("/api/move"):
        return "move"
    if path.endswith("/api/agent"):
        return "agent"
    if "/api/" in path:
        return "missing"
    if path == "/" or not path.rsplit("/", 1)[-1].count("."):
        return "index"
    return "missing"


def _empty_board() -> game.Board:
    return [[game.EMPTY for _ in range(3)] for _ in range(3)]


def _reset(server: TicTacToeServer, *, agent_first: Any = None, agent_mode: Any = None) -> None:
    if agent_first is not None:
        server.agent_first = bool(agent_first)
    if agent_mode in {"llm", "best"}:
        server.agent_mode = str(agent_mode)
    server.agent_player = "X" if server.agent_first else "O"
    server.human_player = "O" if server.agent_first else "X"
    server.board = _empty_board()
    server.turn = "X"
    agent_name = _agent_name(server)
    first = agent_name if server.agent_first else "Player"
    server.events = [
        f"Started a fresh Tic-Tac-Toe board. {first} moves first.",
        f"You control {server.human_player}; {agent_name} controls {server.agent_player}.",
    ]


def _move(server: TicTacToeServer, square: Any, player: str) -> dict[str, Any]:
    if game.is_terminal(server.board):
        server.events.insert(0, "Game is over. Start a new board.")
        return _payload(server)
    if server.turn != player:
        server.events.insert(0, f"It is {server.turn}'s turn.")
        return _payload(server)
    try:
        square = int(square)
        server.board = game.apply_move(server.board, square, player)
    except (TypeError, ValueError):
        server.events.insert(0, f"Illegal square: {square}")
        server.events = server.events[:8]
        return _payload(server)
    won = game.winner(server.board)
    if won:
        server.events.insert(0, f"{won} made three in a row and won!")
    elif not game.legal_moves(server.board):
        server.events.insert(0, "Board filled. Draw.")
    else:
        server.turn = "O" if player == "X" else "X"
        server.events.insert(0, f"{player} placed on square {square}.")
    server.events = server.events[:8]
    return _payload(server)


def _agent_move(server: TicTacToeServer) -> dict[str, Any]:
    if server.turn != server.agent_player or game.is_terminal(server.board):
        return _payload(server)
    try:
        square = _agent_square(server)
    except Exception as exc:  # noqa: BLE001
        server.events.insert(0, f"{_agent_name(server)} failed: {exc}")
        server.events = server.events[:8]
        return _payload(server)
    return _move(server, square, server.agent_player)


def _agent_square(server: TicTacToeServer) -> int:
    if server.agent_mode == "best":
        moves = _best_moves_for_player(server.board, server.agent_player)
        if not moves:
            raise ValueError("no legal best move available")
        return server.rng.choice(moves)
    return _llm_square(server)


def _llm_square(server: TicTacToeServer) -> int:
    if not server.args.base_url:
        raise ValueError("LLM mode requires --base-url")
    if server.openai_client is None:
        server.openai_client = _make_openai_client(server.args)
    response = server.openai_client.chat.completions.create(
        model=server.args.model,
        messages=[
            {"role": "system", "content": _system_prompt(server.agent_player)},
            {"role": "user", "content": _turn_prompt(server.board, server.agent_player)},
        ],
        tools=[CHOOSE_SQUARE_TOOL],
        tool_choice={"type": "function", "function": {"name": "choose_square"}},
    )
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    tool_calls = choices[0].get("message", {}).get("tool_calls", []) if choices else []
    for call in tool_calls:
        if call.get("function", {}).get("name") != "choose_square":
            continue
        args = call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return int(args["square"])
    raise ValueError("response did not contain choose_square tool call")


def _best_moves_for_player(board: game.Board, player: str) -> list[int]:
    if player == "X":
        return game.best_moves(board)
    swapped = [["X" if cell == "O" else "O" if cell == "X" else cell for cell in row] for row in board]
    return game.best_moves(swapped)


def _agent_name(server: TicTacToeServer) -> str:
    return "Best Move" if server.agent_mode == "best" else "LLM"


def _system_prompt(player: str) -> str:
    opponent = "O" if player == "X" else "X"
    return (
        f"You are a careful Tic-Tac-Toe player. You play {player}. "
        "Choose exactly one legal square by calling the choose_square tool. "
        "Try to win immediately, block immediate opponent wins, and prefer the center and corners. "
        f"If {opponent} can win next turn, block {opponent}."
    )


def _turn_prompt(board: game.Board, player: str) -> str:
    legal = game.legal_moves(board)
    opponent = "O" if player == "X" else "X"
    return (
        f"You are playing Tic-Tac-Toe as {player}.\n\n"
        "Rules:\n"
        "- The board is 3 rows by 3 columns.\n"
        "- Empty squares are numbered 1 through 9.\n"
        f"- {player} is your mark and {opponent} is the opponent.\n"
        "- Choose exactly one legal square by calling the choose_square tool.\n"
        "- You win by placing three marks in a row, column, or diagonal.\n"
        "- You cannot choose an occupied square.\n\n"
        f"Board:\n{game.board_to_text(board)}\n\n"
        f"Legal squares: {legal}\n\nMove:"
    )


def _payload(server: TicTacToeServer) -> dict[str, Any]:
    return {
        "board": server.board,
        "turn": server.turn,
        "human_player": server.human_player,
        "agent_player": server.agent_player,
        "agent_first": server.agent_first,
        "agent_mode": server.agent_mode,
        "agent_name": _agent_name(server),
        "winner": game.winner(server.board),
        "terminal": game.is_terminal(server.board),
        "legal_moves": game.legal_moves(server.board),
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
<title>Tic-Tac-Toe</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#24313a;background:#ffe7b7}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#ffe8b6,#f6b36c 58%,#8ed8c2);display:grid;place-items:center}
.app{width:min(930px,94vw);display:grid;grid-template-columns:minmax(310px,430px) 1fr;gap:22px;align-items:start}
.panel{background:#fffaf0;border:4px solid #27313a;border-radius:24px;box-shadow:8px 8px 0 #27313a;padding:18px}
h1{font-size:38px;line-height:1;margin:0 0 8px;color:#e05d2f;text-shadow:2px 2px 0 #ffd977}
.subtitle{font-weight:900;color:#5d3d1d;margin-bottom:14px}
.board{background:#5ab7a7;border:7px solid #27313a;border-radius:24px;padding:12px;display:grid;grid-template-columns:repeat(3,1fr);gap:10px;box-shadow:inset 0 7px 0 rgba(255,255,255,.35),inset 0 -10px 0 rgba(31,58,65,.25),8px 10px 0 rgba(39,49,58,.28)}
.cell{aspect-ratio:1;background:#fff7df;border:4px solid #27313a;border-radius:18px;display:grid;place-items:center;position:relative;cursor:pointer;box-shadow:inset 0 -7px 0 rgba(39,49,58,.12);transition:.15s transform}
.cell:hover{transform:translateY(-2px)}.cell.disabled{cursor:not-allowed;filter:saturate(.8)}
.cell .mark{font-size:clamp(54px,12vw,96px);line-height:.8;font-weight:1000}
.cell.x .mark{color:#252525;text-shadow:3px 3px 0 #9ddfd3}.cell.o .mark{color:#fff;text-shadow:0 0 0 #fff,3px 3px 0 #f08a4b,-2px -2px 0 #27313a,2px -2px 0 #27313a,-2px 2px 0 #27313a,2px 2px 0 #27313a}
.cell.empty::after{content:attr(data-square);font-size:22px;font-weight:1000;color:rgba(39,49,58,.32)}
.cell.drop{animation:pop .28s ease}@keyframes pop{0%{transform:scale(.72) rotate(-8deg)}70%{transform:scale(1.08) rotate(3deg)}100%{transform:scale(1) rotate(0)}}
.stats,.actions,.first-control,.mode-control{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}.pill{background:#fff;border:3px solid #27313a;border-radius:999px;padding:8px 12px;font-weight:1000}
button{border:3px solid #27313a;border-radius:16px;background:#ffd166;box-shadow:4px 4px 0 #27313a;color:#27313a;font-weight:1000;padding:12px 14px;cursor:pointer}
button:hover{transform:translateY(-1px)}button:disabled{filter:grayscale(.75);opacity:.55;cursor:not-allowed}.choice.active{background:#9be564}.label{font-weight:1000;color:#5d3d1d;align-self:center}
.thinking{display:none;margin:10px 0;padding:10px 12px;border:3px solid #27313a;border-radius:16px;background:#dff6ff;font-weight:1000}.thinking.on{display:block}
.dots::after{content:"";animation:dots 1s steps(4,end) infinite}@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}100%{content:""}}
.rules{font-weight:800;line-height:1.45}.rules li{margin:7px 0}.events{display:grid;gap:8px;margin-top:14px}.event{background:#fff;border:3px solid #27313a;border-radius:14px;padding:10px;font-weight:800}
@media(max-width:760px){.app{grid-template-columns:1fr}h1{font-size:32px}}
</style>
</head>
<body>
<main class="app">
  <section class="panel">
    <h1>Tic-Tac-Toe</h1>
    <div class="subtitle">Classic three-in-a-row with an LLM opponent.</div>
    <div id="board" class="board" aria-label="Tic-Tac-Toe board"></div>
    <div id="thinking" class="thinking">LLM is thinking<span class="dots"></span></div>
    <div class="stats">
      <div class="pill" id="turn"></div>
      <div class="pill" id="mode"></div>
    </div>
    <div class="first-control" aria-label="First move selector">
      <span class="label">First move</span>
      <button class="choice active" id="agentFirst">Agent</button>
      <button class="choice" id="playerFirst">Player</button>
    </div>
    <div class="mode-control" aria-label="Agent mode selector">
      <span class="label">Agent mode</span>
      <button class="choice active" id="llmMode">LLM</button>
      <button class="choice" id="bestMode">Best Move</button>
    </div>
    <div class="actions">
      <button id="agent">Retry LLM Move</button>
      <button id="new">New Game</button>
    </div>
  </section>
  <aside class="panel">
    <h1 style="font-size:26px;color:#14866f">Rules</h1>
    <ul class="rules">
      <li id="playerRule">You are O. X is the agent.</li>
      <li>Click a square or press keys 1-9 on your turn.</li>
      <li>First player to make three in a row, column, or diagonal wins.</li>
      <li>LLM mode calls choose_square; Best Move mode uses minimax.</li>
    </ul>
    <div id="events" class="events"></div>
  </aside>
</main>
<script>
const api = (path) => new URL(path, window.location.href).toString();
let state = null, lastBoard = "", agentBusy = false, agentFirst = true, agentMode = "llm";
async function request(path, body){
  const opts = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {};
  const res = await fetch(api(path), opts);
  state = await res.json();
  render();
  maybeAgentMove();
}
function render(){
  const board = document.getElementById("board");
  const key = JSON.stringify(state.board);
  board.innerHTML = "";
  state.board.flat().forEach((cell, idx) => {
    const square = idx + 1;
    const div = document.createElement("button");
    div.className = `cell ${cell === "." ? "empty" : cell.toLowerCase()} ${key !== lastBoard ? "drop" : ""}`;
    div.dataset.square = square;
    div.disabled = agentBusy || state.turn !== state.human_player || state.terminal || !state.legal_moves.includes(square);
    div.onclick = () => request("api/move", {square});
    div.innerHTML = cell === "." ? "" : `<span class="mark">${cell}</span>`;
    board.appendChild(div);
  });
  lastBoard = key;
  agentFirst = Boolean(state.agent_first);
  agentMode = state.agent_mode || "llm";
  document.getElementById("agentFirst").classList.toggle("active", agentFirst);
  document.getElementById("playerFirst").classList.toggle("active", !agentFirst);
  document.getElementById("llmMode").classList.toggle("active", agentMode === "llm");
  document.getElementById("bestMode").classList.toggle("active", agentMode === "best");
  document.getElementById("turn").textContent = agentBusy ? `${state.agent_player} thinking` : (state.terminal ? (state.winner ? `${state.winner} wins` : "Draw") : `${state.turn}'s turn`);
  document.getElementById("mode").textContent = `You ${state.human_player} · ${state.agent_name} ${state.agent_player}`;
  document.getElementById("playerRule").textContent = `You are ${state.human_player}. ${state.agent_name} controls ${state.agent_player}.`;
  document.getElementById("agent").disabled = agentBusy || state.turn !== state.agent_player || state.terminal;
  document.getElementById("thinking").innerHTML = `${state.agent_player} is thinking<span class="dots"></span>`;
  document.getElementById("thinking").classList.toggle("on", agentBusy && agentMode === "llm");
  document.getElementById("events").innerHTML = state.events.map(e => `<div class="event">${escapeHtml(e)}</div>`).join("");
}
async function maybeAgentMove(){
  if(!state || state.turn !== state.agent_player || state.terminal || agentBusy) return;
  agentBusy = true;
  render();
  document.getElementById("agent").textContent = agentMode === "best" ? "Best Move..." : `${state.agent_player} is thinking...`;
  try {
    await request("api/agent", {});
  } finally {
    agentBusy = false;
    document.getElementById("agent").textContent = "Retry LLM Move";
    render();
  }
}
function escapeHtml(text){return text.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));}
function setFirst(nextAgentFirst){
  agentFirst = nextAgentFirst;
  document.getElementById("agentFirst").classList.toggle("active", agentFirst);
  document.getElementById("playerFirst").classList.toggle("active", !agentFirst);
}
function setMode(nextAgentMode){
  agentMode = nextAgentMode;
  document.getElementById("llmMode").classList.toggle("active", agentMode === "llm");
  document.getElementById("bestMode").classList.toggle("active", agentMode === "best");
}
document.getElementById("agentFirst").onclick = () => setFirst(true);
document.getElementById("playerFirst").onclick = () => setFirst(false);
document.getElementById("llmMode").onclick = () => setMode("llm");
document.getElementById("bestMode").onclick = () => setMode("best");
document.getElementById("new").onclick = () => request("api/new", {agent_first: agentFirst, agent_mode: agentMode});
document.getElementById("agent").onclick = () => request("api/agent", {});
window.addEventListener("keydown", (e) => {
  const square = Number(e.key);
  if(square >= 1 && square <= 9 && state && state.turn === state.human_player && !state.terminal && !agentBusy) request("api/move", {square});
});
request("api/state");
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tic-Tac-Toe cartoon web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent-mode", choices=("llm", "best"), default="llm")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for LLM mode.")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    server = TicTacToeServer((args.host, args.port), TicTacToeHandler, seed=args.seed, args=args)
    url = f"http://{args.host}:{args.port}"
    print(f"Tic-Tac-Toe web UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
