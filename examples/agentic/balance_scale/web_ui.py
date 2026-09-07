"""Interactive web UI for the balance-scale agentic example.

Run a policy server first, then start this UI:

    areno serve --model-path /path/to/model --tp-size 1 --world-size 1 --port 8000
    python examples/agentic/balance_scale/web_ui.py \\
        --base-url http://127.0.0.1:8000/v1 --api-key token --model policy

Open http://127.0.0.1:8768 in a browser.  Set the number of balls, the odd-ball
index, and its direction (heavier/lighter).  Click "Run" to watch the model
solve the puzzle step by step.

The UI uses the XML no-tool variant: the model outputs <weigh> and <answer>
tags, and the UI parses them, executes the weighing locally, and feeds the
result back to the model.
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
import game as game_module  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768
MAX_TURNS = 20
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are solving a balance-scale odd-ball puzzle. "
    "You have a set of visually identical balls, one of which is heavier or lighter. "
    "Output a weigh tag to compare two equal-size disjoint groups of balls. "
    "Output an answer tag when you have identified the odd ball. "
    "Choose your weighings carefully — you have a limited budget."
)


class BalanceServer(ThreadingHTTPServer):
    """Stateful HTTP server for one balance-scale puzzle session."""

    def __init__(self, server_address, request_handler, *, args):
        super().__init__(server_address, request_handler)
        self.args = args
        self.reset(num_balls=9, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=3)

    def reset(self, *, num_balls: int, odd_ball_index: int, odd_ball_direction: str, max_weighings: int) -> None:
        self.puzzle = game_module.BalanceGame(
            num_balls=num_balls,
            odd_ball_index=odd_ball_index,
            odd_ball_direction=odd_ball_direction,
            max_weighings=max_weighings,
        )
        self.turns: list[dict[str, Any]] = []
        self.finished = False
        self.model_answer: tuple[int, str] | None = None
        self.error: str | None = None


class BalanceHandler(BaseHTTPRequestHandler):
    server: BalanceServer

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
        if route == "setup":
            body = self._read_json()
            self._handle_setup(body)
        elif route == "run":
            self._send_json(_handle_run(self.server))
        elif route == "random":
            body = self._read_json()
            self._handle_random(body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("balance-web: " + fmt % args + "\n")

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

    def _handle_setup(self, body: Any) -> None:
        if not isinstance(body, dict):
            self._send_json({"error": "invalid body"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            num_balls = int(body.get("num_balls", 9))
            odd_ball_index = int(body.get("odd_ball_index", 0))
            odd_ball_direction = body.get("odd_ball_direction", "heavier")
            max_weighings = int(body.get("max_weighings", 3))
            self.server.reset(
                num_balls=num_balls,
                odd_ball_index=odd_ball_index,
                odd_ball_direction=odd_ball_direction,
                max_weighings=max_weighings,
            )
            self._send_json(_payload(self.server))
        except (ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_random(self, body: Any) -> None:
        if not isinstance(body, dict):
            body = {}
        num_balls = int(body.get("num_balls", 9))
        max_weighings = int(body.get("max_weighings", 3))
        import random
        rng = random.Random()
        odd_index = rng.randint(0, num_balls - 1)
        direction = rng.choice(["heavier", "lighter"])
        self.server.reset(
            num_balls=num_balls,
            odd_ball_index=odd_index,
            odd_ball_direction=direction,
            max_weighings=max_weighings,
        )
        self._send_json(_payload(self.server))


def _route_path(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    if path.endswith("/api/state"):
        return "state"
    if path.endswith("/api/setup"):
        return "setup"
    if path.endswith("/api/run"):
        return "run"
    if path.endswith("/api/random"):
        return "random"
    if "/api/" in path:
        return "missing"
    if path == "/" or not path.rsplit("/", 1)[-1].count("."):
        return "index"
    return "missing"


def _payload(server: BalanceServer) -> dict[str, Any]:
    puzzle = server.puzzle
    result: dict[str, Any] = {
        "num_balls": puzzle.num_balls,
        "odd_ball_index": puzzle.odd_ball_index,
        "odd_ball_direction": puzzle.odd_ball_direction,
        "max_weighings": puzzle.max_weighings,
        "weighings_used": puzzle.weighings_used,
        "weighings_remaining": puzzle.weighings_remaining,
        "turns": server.turns,
        "finished": server.finished,
        "error": server.error,
    }
    if server.model_answer is not None:
        ball, direction = server.model_answer
        identity_correct, direction_correct = puzzle.check_answer(ball, direction)
        result["model_answer"] = {"ball_index": ball, "direction": direction}
        result["identity_correct"] = identity_correct
        result["direction_correct"] = direction_correct
        if identity_correct and direction_correct:
            result["verdict"] = "Fully correct! Ball index and direction both match."
        elif identity_correct:
            result["verdict"] = "Identity only: correct ball, wrong direction."
        else:
            result["verdict"] = "Wrong answer."
    return result


def _handle_run(server: BalanceServer) -> dict[str, Any]:
    """Run the full multi-turn agent loop synchronously and return the final state."""

    if server.finished:
        return _payload(server)

    args = server.args
    try:
        from openai import OpenAI
    except ImportError:
        server.error = "openai package not installed. Run: pip install openai"
        return _payload(server)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    puzzle = server.puzzle

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": game_module.format_xml_prompt(puzzle.num_balls, puzzle.max_weighings)},
    ]

    for turn_idx in range(MAX_TURNS):
        if puzzle.weighings_remaining <= 1 and puzzle.weighings_remaining > 0:
            messages.append({
                "role": "user",
                "content": "You have 1 weighing remaining. Use it wisely, then answer.",
            })
        if puzzle.weighings_remaining <= 0:
            messages.append({
                "role": "user",
                "content": "Your weighing budget is exhausted. You must output an answer tag now.",
            })

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=False,
                max_tokens=MAX_NEW_TOKENS,
            )
        except Exception as exc:
            server.error = f"Model request failed: {exc}"
            server.finished = True
            return _payload(server)

        text = ""
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            pass

        # Check for answer tag.
        answer = game_module.parse_xml_answer(text)
        if answer is not None:
            ball, direction = answer
            server.model_answer = (ball, direction)
            server.turns.append({"turn": turn_idx + 1, "action": "answer", "text": text, "ball": ball, "direction": direction})
            server.finished = True
            return _payload(server)

        # Check for weigh tag.
        weigh = game_module.parse_xml_weigh(text)
        if weigh is None:
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    'Your response did not contain a <weigh> or <answer> tag. '
                    'Please output one now. '
                    'To weigh: <weigh left="0,1" right="2,3"/>. '
                    'To answer: <answer ball="3" direction="heavier"/>.'
                ),
            })
            server.turns.append({"turn": turn_idx + 1, "action": "nudge", "text": text})
            continue

        left_group, right_group = weigh
        try:
            result = puzzle.weigh(left_group, right_group)
        except ValueError as exc:
            result = f"error: {exc}"

        server.turns.append({
            "turn": turn_idx + 1,
            "action": "weigh",
            "left": left_group,
            "right": right_group,
            "result": result,
            "text": text,
        })
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Result: {result}"})

    server.finished = True
    server.error = "Max turns reached without an answer."
    return _payload(server)


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Balance Scale Agent</title>
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; max-width: 800px; margin: 20px auto; padding: 16px; background: #f5f5f5; }
h1 { color: #333; }
.card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
label { display: inline-block; width: 140px; font-weight: 600; margin-bottom: 8px; }
input, select { padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
button { padding: 8px 20px; border: none; border-radius: 4px; background: #4a90d9; color: white; font-size: 14px; cursor: pointer; margin-right: 8px; }
button:hover { background: #357abd; }
button:disabled { background: #ccc; cursor: not-allowed; }
button.secondary { background: #6c757d; }
button.secondary:hover { background: #5a6268; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
th { background: #f0f0f0; font-weight: 600; }
.turn-entry { padding: 8px 0; border-bottom: 1px solid #eee; }
.turn-entry .turn-num { font-weight: 700; color: #4a90d9; }
.turn-entry .turn-action { font-weight: 600; color: #333; }
.turn-entry .turn-result { color: #e74c3c; font-weight: 600; }
.verdict-correct { color: #27ae60; font-weight: 700; }
.verdict-wrong { color: #e74c3c; font-weight: 700; }
.verdict-partial { color: #f39c12; font-weight: 700; }
.error { color: #e74c3c; font-weight: 600; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { width: auto; }
#run-btn { background: #27ae60; }
#run-btn:hover { background: #229954; }
</style>
</head>
<body>
<h1>Balance Scale Agent</h1>
<p>Set puzzle parameters, run the model, and watch it solve the odd-ball puzzle step by step.</p>

<div class="card">
  <h3>Puzzle Setup</h3>
  <div class="controls">
    <div class="field">
      <label>Number of balls</label>
      <input type="number" id="num-balls" value="9" min="3" max="20">
    </div>
    <div class="field">
      <label>Odd ball index (0-based)</label>
      <input type="number" id="odd-index" value="0" min="0" max="19">
    </div>
    <div class="field">
      <label>Odd ball direction</label>
      <select id="odd-direction">
        <option value="heavier">heavier</option>
        <option value="lighter">lighter</option>
      </select>
    </div>
    <div class="field">
      <label>Max weighings</label>
      <input type="number" id="max-weighings" value="3" min="1" max="10">
    </div>
    <button id="setup-btn">Apply Setup</button>
    <button id="random-btn" class="secondary">Random</button>
    <button id="run-btn">Run Model</button>
  </div>
</div>

<div class="card" id="result-card" style="display:none">
  <h3>Result</h3>
  <table>
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Number of balls</td><td id="r-num-balls"></td></tr>
    <tr><td>Odd ball</td><td id="r-odd-ball"></td></tr>
    <tr><td>True direction</td><td id="r-true-dir"></td></tr>
    <tr><td>Model answer</td><td id="r-model-answer"></td></tr>
    <tr><td>Verdict</td><td id="r-verdict"></td></tr>
    <tr><td>Weighings used</td><td id="r-weighings"></td></tr>
  </table>
</div>

<div class="card" id="error-card" style="display:none">
  <p class="error" id="error-text"></p>
</div>

<div class="card">
  <h3>Weighing Process</h3>
  <div id="turns"></div>
</div>

<script>
const $ = (id) => document.getElementById(id);

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {}),
  });
  return resp.json();
}

async function getState() {
  const resp = await fetch('/api/state');
  return resp.json();
}

function applySetup() {
  const body = {
    num_balls: parseInt($('num-balls').value),
    odd_ball_index: parseInt($('odd-index').value),
    odd_ball_direction: $('odd-direction').value,
    max_weighings: parseInt($('max-weighings').value),
  };
  postJSON('/api/setup', body).then(render);
}

function randomPuzzle() {
  const body = {
    num_balls: parseInt($('num-balls').value),
    max_weighings: parseInt($('max-weighings').value),
  };
  postJSON('/api/random', body).then(render);
}

function runModel() {
  $('run-btn').disabled = true;
  $('run-btn').textContent = 'Running...';
  postJSON('/api/run', {}).then((data) => {
    render(data);
    $('run-btn').disabled = false;
    $('run-btn').textContent = 'Run Model';
  }).catch(() => {
    $('run-btn').disabled = false;
    $('run-btn').textContent = 'Run Model';
  });
}

function render(data) {
  if (data.error) {
    $('error-card').style.display = '';
    $('error-text').textContent = data.error;
  } else {
    $('error-card').style.display = 'none';
  }

  if (data.model_answer) {
    $('result-card').style.display = '';
    $('r-num-balls').textContent = data.num_balls;
    $('r-odd-ball').textContent = '#' + data.odd_ball_index;
    $('r-true-dir').textContent = data.odd_ball_direction;
    $('r-model-answer').textContent = 'Ball ' + data.model_answer.ball_index + ' (' + data.model_answer.direction + ')';
    $('r-weighings').textContent = data.weighings_used + ' / ' + data.max_weighings;
    const v = $('r-verdict');
    v.textContent = data.verdict || '';
    v.className = '';
    if (data.identity_correct && data.direction_correct) v.className = 'verdict-correct';
    else if (data.identity_correct) v.className = 'verdict-partial';
    else v.className = 'verdict-wrong';
  } else {
    $('result-card').style.display = 'none';
  }

  const turnsDiv = $('turns');
  turnsDiv.innerHTML = '';
  if (!data.turns || data.turns.length === 0) {
    turnsDiv.innerHTML = '<p style="color:#999">No turns yet. Click "Run Model" to start.</p>';
    return;
  }
  for (const t of data.turns) {
    const div = document.createElement('div');
    div.className = 'turn-entry';
    let html = '<span class="turn-num">--- Turn ' + t.turn + ' ---</span><br>';
    if (t.action === 'weigh') {
      html += '<span class="turn-action">weigh(left=[' + t.left + '], right=[' + t.right + '])</span><br>';
      html += '<span class="turn-result">&rarr; ' + t.result + '</span>';
    } else if (t.action === 'answer') {
      html += '<span class="turn-action">answer(ball=' + t.ball + ', direction=' + t.direction + ')</span>';
    } else {
      html += '<span class="turn-action">(no valid tag, retrying)</span><br><span style="color:#999;font-size:12px">' + (t.text || '').substring(0, 100) + '...</span>';
    }
    div.innerHTML = html;
    turnsDiv.appendChild(div);
  }
}

$('setup-btn').addEventListener('click', applySetup);
$('random-btn').addEventListener('click', randomPuzzle);
$('run-btn').addEventListener('click', runModel);

getState().then(render);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI for the balance-scale agentic example.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="token", help="API key for the model server")
    parser.add_argument("--model", default="policy", help="Model name for chat completions")
    args = parser.parse_args()

    server = BalanceServer((args.host, args.port), BalanceHandler, args=args)
    print(f"Balance Scale Web UI: http://{args.host}:{args.port}")
    print(f"Model endpoint: {args.base_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()