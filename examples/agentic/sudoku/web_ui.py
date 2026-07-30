"""Web UI server for the Sudoku agentic example.

Run from the repository root:

    python examples/agentic/sudoku/web_ui.py
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8769


class SudokuWebServer(ThreadingHTTPServer):
    """Small stateful HTTP server for one local Sudoku game."""

    def __init__(self, server_address, handler_cls, args: argparse.Namespace) -> None:
        super().__init__(server_address, handler_cls)
        self.args = args
        self.episode = _new_episode(args)
        self.events: list[str] = []
        self.llm_client = None
        if args.agent:
            self.llm_client = _make_openai_client(args)
        self.llm_messages: list[dict[str, Any]] = []


class SudokuHandler(BaseHTTPRequestHandler):
    server: SudokuWebServer

    def do_GET(self) -> None:
        path = _route_path(self.path)
        if path == "/" or path == "/index.html":
            self._send_html(INDEX_HTML)
        elif path == "/api/state":
            self._send_json(_payload(self.server))
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = _route_path(self.path)
        body = self._read_json()
        if path == "/api/action":
            self._send_json(_apply_action(self.server, body))
        elif path == "/api/agent":
            self._send_json(_run_agent_turn(self.server))
        elif path == "/api/new":
            self.server.episode = _new_episode(self.server.args)
            self.server.events = ["New puzzle generated."]
            self.server.llm_messages = []
            self._send_json(_payload(self.server))
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("sudoku-web: " + fmt % args + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        data = self.rfile.read(length).decode("utf-8")
        return json.loads(data)

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
    return urlparse(raw_path).path


def _new_episode(args: argparse.Namespace) -> game.SudokuEpisode:
    puzzle = game.generate_puzzle(args.difficulty, seed=args.seed)
    return game.SudokuEpisode(puzzle["puzzle"], max_actions=args.max_actions)


def _apply_action(server: SudokuWebServer, body: dict) -> dict:
    name = body.get("name", "")
    args = body.get("arguments", {})
    ep = server.episode
    if name == "inspect_candidates":
        result = game.inspect_candidates(ep.board, int(args.get("row", -1)), int(args.get("col", -1)))
    elif name == "place_digit":
        result = ep.place(int(args.get("row", -1)), int(args.get("col", -1)), int(args.get("digit", 0)))
    elif name == "undo":
        result = ep.undo()
    else:
        result = {"valid": False, "error": f"unknown tool: {name}"}
    server.events.insert(0, f"tool: {name}({args}) -> valid={result.get('valid')}")
    payload = _payload(server)
    payload["result"] = result
    return payload


def _run_agent_turn(server: SudokuWebServer) -> dict:
    if server.llm_client is None:
        return _payload(server)
    ep = server.episode
    if ep.is_done():
        return _payload(server)
    if not server.llm_messages:
        record = {"puzzle": ep.original, "difficulty": server.args.difficulty, "max_actions": ep.max_actions}
        server.llm_messages = [
            {"role": "system", "content": game.SYSTEM_PROMPT},
            {"role": "user", "content": game.make_prompt(record)},
        ]
    turn_prompt = {"role": "user", "content": f"Action {ep.actions_taken + 1}/{ep.max_actions}: call one tool now."}
    messages = [*server.llm_messages, turn_prompt]
    response = server.llm_client.chat.completions.create(
        model=server.args.model,
        messages=messages,
        tools=game.TOOLS,
        stream=False,
    )
    message = response.choices[0].message
    calls = list(message.tool_calls or [])
    if not calls:
        server.events.insert(0, "agent: no tool call returned")
        return _payload(server)
    call = calls[0]
    name = call.function.name
    raw_args = call.function.arguments or "{}"
    try:
        args_dict = json.loads(raw_args)
    except json.JSONDecodeError:
        args_dict = {}
    if name == "inspect_candidates":
        result = game.inspect_candidates(ep.board, int(args_dict.get("row", -1)), int(args_dict.get("col", -1)))
    elif name == "place_digit":
        result = ep.place(int(args_dict.get("row", -1)), int(args_dict.get("col", -1)), int(args_dict.get("digit", 0)))
    elif name == "undo":
        result = ep.undo()
    else:
        result = {"valid": False, "error": f"unknown tool: {name}"}
    server.llm_messages.extend([
        turn_prompt,
        {"role": "assistant", "content": message.content, "tool_calls": [
            {"id": call.id, "type": call.type, "function": {"name": name, "arguments": raw_args}}
        ]},
        {"role": "tool", "tool_call_id": call.id, "name": name, "content": json.dumps(result)},
    ])
    server.events.insert(0, f"agent: {name}({args_dict}) -> valid={result.get('valid')}")
    payload = _payload(server)
    payload["result"] = result
    return payload


def _make_openai_client(args: argparse.Namespace):
    from openai import OpenAI

    return OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)


def _payload(server: SudokuWebServer) -> dict[str, Any]:
    ep = server.episode
    return {
        "board": ep.board,
        "original": ep.original,
        "actions_taken": ep.actions_taken,
        "max_actions": ep.max_actions,
        "solved": ep.is_solved(),
        "done": ep.is_done(),
        "events": server.events[:20],
        "agent_enabled": server.llm_client is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--difficulty", choices=list(game.DIFFICULTY_EMPTY), default=game.DEFAULT_DIFFICULTY)
    parser.add_argument("--max-actions", type=int, default=game.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--agent", action="store_true", help="Enable LLM agent mode")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    server = SudokuWebServer((args.host, args.port), SudokuHandler, args)
    print(f"Sudoku web UI on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sudoku</title>
  <style>
    :root {
      --bg: #f0f4f8;
      --panel: #ffffff;
      --ink: #1a2a3a;
      --muted: #6b7c8c;
      --line: #c3d4e0;
      --blue: #2fa8e7;
      --green: #45bd68;
      --red: #ef6969;
      --gold: #f3b832;
      --cell-bg: #fafdff;
      --cell-given: #e8edf2;
      --cell-user: #e3f2fd;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #d6eaf8 0, var(--bg) 40%, #e8f5e9 100%);
      color: var(--ink);
      font: 15px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(960px, calc(100vw - 20px));
      margin: 0 auto;
      padding: 10px 0 18px;
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 300px;
      gap: 12px;
    }
    header {
      grid-column: 1 / -1;
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 8px 2px 6px;
    }
    h1 { margin: 0; font-size: 26px; }
    .subtitle { margin: 4px 0 0; color: var(--muted); }
    button {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff, #e3f2fd);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
      min-height: 36px;
    }
    button:hover:not(:disabled) { border-color: var(--blue); background: linear-gradient(180deg, #f0f7ff, #bbdefb); }
    button:disabled { opacity: .46; cursor: not-allowed; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,.06);
    }
    .board-wrap { margin: 0 auto; padding: 8px; }
    .board {
      display: grid;
      grid-template-columns: repeat(9, 1fr);
      gap: 0;
      width: 100%;
      max-width: 450px;
      aspect-ratio: 1;
      margin: 0 auto;
      border: 2px solid var(--ink);
    }
    .cell {
      display: grid;
      place-items: center;
      aspect-ratio: 1;
      border: 1px solid var(--line);
      background: var(--cell-bg);
      font-weight: 700;
      font-size: clamp(14px, 3vw, 24px);
      cursor: pointer;
      position: relative;
    }
    .cell.given { background: var(--cell-given); color: var(--ink); }
    .cell.user { background: var(--cell-user); color: var(--blue); }
    .cell.empty { color: var(--muted); }
    .cell.selected { box-shadow: inset 0 0 0 3px var(--gold); }
    .cell.box-right { border-right: 2px solid var(--ink); }
    .cell.box-bottom { border-bottom: 2px solid var(--ink); }
    .hud {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 13px;
      background: #f8fbfd;
    }
    .side { display: grid; gap: 10px; align-content: start; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
    .log {
      display: grid;
      gap: 6px;
      max-height: 220px;
      overflow: auto;
      padding-right: 4px;
    }
    .log-entry {
      padding: 6px 8px;
      background: #f8fbfd;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: #415056;
      font-size: 12px;
      white-space: pre-wrap;
    }
    .input-row { display: grid; gap: 6px; }
    .input-row label { font-size: 13px; color: var(--muted); }
    .input-row select, .input-row input {
      padding: 6px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
    }
    .digit-pad { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; }
    .digit-pad button { padding: 4px; min-height: 32px; font-weight: 700; }
    .top-actions { display: flex; flex-wrap: wrap; gap: 8px; }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Sudoku</h1>
        <p class="subtitle">Fill the grid so each row, column, and 3x3 box has 1-9.</p>
      </div>
      <div class="top-actions">
        <span class="pill" id="status-pill">Playing</span>
        <button id="new-game">New Puzzle</button>
        <button id="agent-turn" style="display:none">Agent Step</button>
      </div>
    </header>

    <section class="panel">
      <div class="board-wrap">
        <div id="board" class="board"></div>
      </div>
      <div class="hud">
        <span class="pill" id="actions-pill">Actions: 0/120</span>
        <span class="pill" id="solved-pill" style="display:none">Solved!</span>
      </div>
    </section>

    <aside class="side">
      <section class="panel">
        <strong>Tools</strong>
        <div class="input-row">
          <label>Row (0-8)</label>
          <input type="number" id="tool-row" min="0" max="8" value="0" />
        </div>
        <div class="input-row">
          <label>Col (0-8)</label>
          <input type="number" id="tool-col" min="0" max="8" value="0" />
        </div>
        <div class="input-row">
          <label>Digit (1-9)</label>
          <input type="number" id="tool-digit" min="1" max="9" value="1" />
        </div>
        <div class="actions" style="margin-top:8px">
          <button id="btn-inspect">Inspect</button>
          <button id="btn-place">Place</button>
          <button id="btn-undo" style="grid-column:1/-1">Undo</button>
        </div>
      </section>
      <section class="panel">
        <strong>History</strong>
        <div class="log" id="log"></div>
      </section>
    </aside>
  </main>

  <script>
    let state = null;
    let busy = false;

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function render(data) {
      state = data;
      renderBoard(data);
      document.getElementById('actions-pill').textContent = `Actions: ${data.actions_taken}/${data.max_actions}`;
      document.getElementById('status-pill').textContent = data.solved ? 'Solved!' : (data.done ? 'Game Over' : 'Playing');
      document.getElementById('solved-pill').style.display = data.solved ? 'inline-flex' : 'none';
      const agentBtn = document.getElementById('agent-turn');
      agentBtn.style.display = data.agent_enabled ? '' : 'none';
      agentBtn.disabled = data.done;
      renderLog(data.events || []);
      document.querySelectorAll('button').forEach(b => b.disabled = data.done && b.id !== 'new-game');
    }

    function renderBoard(data) {
      const board = document.getElementById('board');
      board.innerHTML = '';
      for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
          const val = data.board[r][c];
          const given = data.original[r][c] !== 0;
          const cell = document.createElement('div');
          cell.className = 'cell ' + (val === 0 ? 'empty' : (given ? 'given' : 'user'));
          if (c === 2 || c === 5) cell.classList.add('box-right');
          if (r === 2 || r === 5) cell.classList.add('box-bottom');
          cell.textContent = val === 0 ? '' : val;
          cell.dataset.row = r;
          cell.dataset.col = c;
          cell.onclick = () => {
            document.getElementById('tool-row').value = r;
            document.getElementById('tool-col').value = c;
          };
          board.appendChild(cell);
        }
      }
    }

    function renderLog(events) {
      const log = document.getElementById('log');
      log.innerHTML = '';
      for (const event of events) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.textContent = event;
        log.appendChild(entry);
      }
    }

    async function submitAction(name, args) {
      if (busy || !state || state.done) return;
      busy = true;
      try {
        const data = await api('api/action', { method: 'POST', body: JSON.stringify({ name, arguments: args }) });
        render(data);
      } catch (error) {
        console.error(error);
      } finally {
        busy = false;
      }
    }

    async function agentStep() {
      if (busy || !state || state.done) return;
      busy = true;
      try {
        const data = await api('api/agent', { method: 'POST' });
        render(data);
      } catch (error) {
        console.error(error);
      } finally {
        busy = false;
      }
    }

    document.getElementById('btn-inspect').onclick = () => {
      submitAction('inspect_candidates', {
        row: parseInt(document.getElementById('tool-row').value),
        col: parseInt(document.getElementById('tool-col').value)
      });
    };

    document.getElementById('btn-place').onclick = () => {
      submitAction('place_digit', {
        row: parseInt(document.getElementById('tool-row').value),
        col: parseInt(document.getElementById('tool-col').value),
        digit: parseInt(document.getElementById('tool-digit').value)
      });
    };

    document.getElementById('btn-undo').onclick = () => submitAction('undo', {});

    document.getElementById('agent-turn').onclick = agentStep;

    document.getElementById('new-game').onclick = async () => {
      if (busy) return;
      render(await api('api/new', { method: 'POST' }));
    };

    api('api/state').then(render).catch(console.error);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()