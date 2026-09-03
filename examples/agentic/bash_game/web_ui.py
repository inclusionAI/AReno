"""Cartoon web UI server for the Bash game (巴什博弈 / 取石子游戏) example.

Run from the repository root (after starting ``areno serve`` on :8000):

    python examples/agentic/bash_game/web_ui.py \
      --base-url http://127.0.0.1:8000/v1 --api-key token --port 8001

The human plays against either the served LLM policy or the perfect oracle
(``--agent-mode best``), taking 1..m stones per turn. The player who removes
the last stone wins.
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
DEFAULT_PORT = 8001

SUBMIT_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_move",
        "description": "Take k stones (winning) or resign (losing position).",
        "parameters": {
            "type": "object",
            "properties": {
                "take": {"type": "integer", "minimum": 1, "description": "Stones to take this turn."},
                "resign": {"type": "boolean", "description": "Set true exactly when there is no winning move."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

INDEX_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>取石子游戏 · Bash Game</title>
<style>  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: radial-gradient(1200px 700px at 20% -10%, #1b2a4a 0%, #0d1220 60%);
    color: #e8ecf7; padding: 24px;
  }
  .card {
    width: min(720px, 100%); background: #12192c; border: 1px solid #26334f;
    border-radius: 20px; box-shadow: 0 24px 80px rgba(0,0,0,.55); padding: 28px;
  }
  h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: .3px; }
  .sub { color: #8d9ac0; font-size: 13.5px; margin-bottom: 20px; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; align-items: center; }
  .btn {
    background: #232f4d; color: #eaf0ff; border: 1px solid #33436b; border-radius: 10px;
    padding: 9px 14px; font-size: 14px; cursor: pointer; transition: .15s;
  }
  .btn:hover { background: #2c3b63; transform: translateY(-1px); }
  .btn.primary { background: #3f6df6; border-color: #5b82ff; }
  .btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }
  select { background: #1a2338; color: #eaf0ff; border: 1px solid #33436b; border-radius: 8px; padding: 8px; font-size: 13px; }
  .pile { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 20px; min-height: 40px; }
  .stone {
    width: 32px; height: 32px; border-radius: 50%;
    background: radial-gradient(circle at 32% 28%, #f4f0e0, #8a8570 70%);
    box-shadow: inset 0 -3px 5px rgba(0,0,0,.4), 0 2px 3px rgba(0,0,0,.4);
    transition: transform .15s, opacity .15s;
  }
  .status { font-size: 15px; margin-bottom: 14px; min-height: 22px; }
  .status.win { color: #7ee787; font-weight: 600; }
  .status.lose { color: #ff7b72; font-weight: 600; }
  .take-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .take-row .btn { min-width: 46px; text-align: center; }
  .log { background: #0c1220; border: 1px solid #1d2840; border-radius: 12px; padding: 12px 14px;
         height: 150px; overflow-y: auto; font-size: 13px; color: #aab6d8; line-height: 1.6; }
  .log div { border-bottom: 1px dashed #1b2540; padding: 2px 0; }
  .foot { margin-top: 14px; font-size: 12px; color: #6c6c7aa0; }
</style>
</head>
<body>
<div class="card">
  <h1>取石子游戏 · Bash Game</h1>
  <div class="sub">单堆取石子 — 每次拿 1~m 颗，拿到最后一颗者胜。你 vs 智能体。</div>
  <div class="controls">
    <label>初始石子 n <select id="nSel"></select></label>
    <label>每次最多拿 m
      <select id="mSel">
        <option value="2">2</option><option value="3" selected>3</option>
        <option value="4">4</option><option value="5">5</option><option value="6">6</option>
      </select>
    </label>
    <label>智能体
      <select id="agentMode">
        <option value="llm" selected>LLM (served policy)</option>
        <option value="best">Perfect oracle</option>
      </select>
    </label>
    <label>先手
      <select id="firstSel">
        <option value="human" selected>我先</option>
        <option value="agent">智能体先</option>
      </select>
    </label>
    <button class="btn primary" id="newBtn">新游戏</button>
  </div>
  <div class="status" id="status">点击「新游戏」开始。</div>
  <div class="pile" id="pile"></div>
  <div class="take-row" id="takeRow"></div>
  <div class="log" id="log"></div>
  <div class="foot">必胜策略：拿 n mod (m+1) 颗（余 0 则为必败局面）。智能体应学会这一点。</div>
</div>
<script>
function $(id){ return document.getElementById(id); }
let state = null;
async function api(path, body) {
  const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
  return r.json();
}
function render() {
  if (!state) return;
  const pile = $('pile');
  pile.innerHTML = '';
  const remaining = state.n;
  for (let i = 0; i < remaining; i++) {
    const s = document.createElement('div');
    s.className = 'stone';
    pile.appendChild(s);
  }
  const takeRow = $('takeRow');
  takeRow.innerHTML = '';
  if (state.over) {
    $('status').className = 'status ' + (state.human_won ? 'win' : 'lose');
    $('status').textContent = state.over_message;
    return;
  }
  $('status').className = 'status';
  $('status').textContent = state.turn === 'human'
    ? `轮到你了：拿 1~${state.m} 颗石子。剩余 ${state.n} 颗。`
    : '智能体思考中…';
  const max = Math.min(state.m, state.n);
  for (let k = 1; k <= max; k++) {
    const b = document.createElement('button');
    b.className = 'btn';
    b.textContent = '拿 ' + k;
    b.disabled = state.turn !== 'human';
    b.onclick = () => move(k);
    takeRow.appendChild(b);
  }
}
function log(msg) {
  const el = $('log');
  const d = document.createElement('div');
  d.textContent = msg;
  el.prepend(d);
  while (el.children.length > 30) el.removeChild(el.lastChild);
}
async function move(k) {
  state = await api('/api/move', {take: k});
  if (state.events) state.events.slice().reverse().forEach(e => log(e));
  render();
  if (!state.over && state.turn === 'agent') await agent();
}
async function agent() {
  state = await api('/api/agent');
  if (state.events) state.events.slice().reverse().forEach(e => log(e));
  render();
}
async function newGame() {
  const body = {
    n: parseInt($('nSel').value),
    m: parseInt($('mSel').value),
    agent_mode: $('agentMode').value,
    agent_first: $('firstSel').value === 'agent',
  };
  state = await api('/api/new', body);
  $('log').innerHTML = '';
  log('新游戏：n=' + state.n + '，m=' + state.m + '，先手=' + (state.agent_first ? '智能体' : '你'));
  render();
  if (state.turn === 'agent') await agent();
}
function populateN() {
  const sel = $('nSel');
  for (let n = 1; n <= 40; n++) {
    const o = document.createElement('option');
    o.value = n; o.textContent = n;
    if (n === 12) o.selected = true;
    sel.appendChild(o);
  }
}
populateN();
$('newBtn').onclick = newGame;
</script>
</body>
</html>
"""


class BashServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler, *, seed=None, args):
        super().__init__(server_address, request_handler)
        self.rng = random.Random(seed)
        self.n = 12
        self.m = 3
        self.agent_mode = args.agent_mode
        self.agent_first = args.agent_first
        self.turn = "agent" if args.agent_first else "human"
        self.over = False
        self.human_won = False
        self.events: list[str] = []
        self.args = args
        self.openai_client = None


class BashHandler(BaseHTTPRequestHandler):
    server: BashServer

    def do_GET(self) -> None:
        route = _route_path(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            self._send_json(_payload(self.server))
        elif route == "agent":
            self._send_json(_agent_turn(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route_path(self.path)
        if route == "new":
            body = self._read_json()
            _reset(self.server, body)
            self._send_json(_payload(self.server))
        elif route == "move":
            body = self._read_json()
            take = body.get("take") if isinstance(body, dict) else None
            self._send_json(_human_turn(self.server, take))
        elif route == "agent":
            self._send_json(_agent_turn(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt, *args):
        sys.stderr.write("bash-web: " + fmt % args + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


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
    if path == "/" or "." not in path.rsplit("/", 1)[-1]:
        return "index"
    return "missing"


def _reset(server: BashServer, body: dict | None) -> None:
    body = body or {}
    n = int(body.get("n", 12))
    m = int(body.get("m", 3))
    if n < 1:
        n = 1
    if m < 1 or m not in game.VALID_MAX_TAKES:
        m = 3
    server.n = n
    server.m = m
    server.agent_mode = body.get("agent_mode", server.agent_mode)
    server.agent_first = bool(body.get("agent_first", False))
    server.turn = "agent" if server.agent_first else "human"
    server.over = False
    server.human_won = False
    server.events = []


def _apply_take(server: BashServer, take: int, player: str) -> None:
    take = int(take)
    server.n -= take
    server.events.append(f"{'你' if player == 'human' else '智能体'}拿了 {take} 颗石子，剩余 {server.n} 颗。")
    if server.n == 0:
        server.over = True
        server.human_won = (player == "human")
        who = "你" if server.human_won else "智能体"
        server.events.append(f"{who}拿到最后一颗，获胜！")
    else:
        server.turn = "agent" if player == "human" else "human"


def _human_turn(server: BashServer, take: Any) -> dict:
    if server.over:
        server.events.append("游戏已结束，请开始新游戏。")
        return _payload(server)
    if server.turn != "human":
        server.events.append("还没轮到你。")
        return _payload(server)
    try:
        take = int(take)
        if take < 1 or take > server.m or take > server.n:
            raise ValueError
    except (TypeError, ValueError):
        server.events.append(f"非法拿取：{take}（必须 1~{min(server.m, server.n)}）。")
        return _payload(server)
    _apply_take(server, take, "human")
    return _payload(server)


def _agent_turn(server: BashServer) -> dict:
    if server.over or server.turn != "agent":
        return _payload(server)
    try:
        move = _agent_move(server)
    except Exception as exc:  # noqa: BLE001
        server.events.append(f"智能体失败：{exc}")
        return _payload(server)
    _apply_take(server, move, "agent")
    return _payload(server)


def _agent_move(server: BashServer) -> int:
    if server.agent_mode == "best":
        g = game.BashGame(server.n, server.m)
        oracle = g.optimal_move()
        if oracle.get("resign"):
            return 1  # losing position: in UI we still must move, take the minimum
        return int(oracle["take"])
    return _llm_move(server)


def _llm_move(server: BashServer) -> int:
    if not server.args.base_url:
        raise ValueError("LLM mode requires --base-url")
    if server.openai_client is None:
        server.openai_client = _make_openai_client(server.args)
    response = server.openai_client.chat.completions.create(
        model=server.args.model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": game.format_prompt({"n": server.n, "m": server.m})},
        ],
        tools=[SUBMIT_MOVE_TOOL],
        tool_choice="required",
    )
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    tool_calls = choices[0].get("message", {}).get("tool_calls", []) if choices else []
    for call in tool_calls:
        fn = call.get("function", {})
        if fn.get("name") != "submit_move":
            continue
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if args.get("resign") is True:
            return 1  # losing position: take the minimum legal stone
        take = args.get("take")
        if isinstance(take, (int, float)):
            take = int(take)
            if 1 <= take <= server.m and take <= server.n:
                return take
    raise ValueError("response did not contain a valid submit_move take")


def _make_openai_client(args):
    from openai import OpenAI

    return OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=3)


def _system_prompt() -> str:
    return (
        "You are a perfect Bash-game (取石子游戏) strategist. Output exactly one "
        "submit_move tool call: take k stones for a winning move, or resign in a "
        "losing position. Do not narrate; only call the tool once."
    )


def _payload(server: BashServer) -> dict[str, Any]:
    return {
        "n": server.n,
        "m": server.m,
        "turn": server.turn,
        "over": server.over,
        "human_won": server.human_won,
        "agent_mode": server.agent_mode,
        "agent_first": server.agent_first,
        "over_message": ("你赢了！🎉" if server.human_won else "智能体赢了。") if server.over else None,
        "events": list(reversed(server.events))[-8:],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Bash game web UI.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--base-url", default=None, help="e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="token")
    p.add_argument("--model", default="policy")
    p.add_argument("--agent-mode", default="llm", choices=("llm", "best"))
    p.add_argument("--agent-first", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.agent_mode == "llm" and not args.base_url:
        p.error("--agent-mode llm requires --base-url")

    server = BashServer((args.host, args.port), BashHandler, seed=args.seed, args=args)
    print(f"Bash game web UI on http://{args.host}:{args.port}  (agent_mode={args.agent_mode})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()