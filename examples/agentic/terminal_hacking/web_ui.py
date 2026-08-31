"""Interactive CRT WebUI for the terminal-hacking agentic example."""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from contextlib import nullcontext
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
from game import (  # noqa: E402
    TerminalHackingSession,
    candidate_controller_action,
    candidate_filter_request_messages,
    candidate_tool,
    expert_decision,
    extra_body_for_base_url,
    make_filter_prompt,
    probe_is_preferred,
    tool_choice_for_base_url,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8771
MODEL_RETRY_INTERVAL_SECONDS = 10.0
MODEL_WAIT_MAX_SECONDS = 300.0


class TerminalServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler, *, args: argparse.Namespace) -> None:
        super().__init__(server_address, handler)
        self.args = args
        self.rng = random.Random(args.seed)
        self.clients: dict[tuple[int, int], Any] = {}
        self.agent_busy = False
        self.agent_job_id = 0
        self.state_lock = threading.RLock()
        self.error: str | None = None
        self.events: list[dict[str, str]] = []
        self.reset()

    def reset(self) -> None:
        self.agent_job_id += 1
        self.agent_busy = False
        clients = list(self.clients.values())
        self.clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        self.record = dataset_generator.generate_records(1, seed=self.rng.randrange(2**31), workers=1)[0]
        self.session = TerminalHackingSession(self.record)
        self.error = None
        self.events = [_event("NODE READY // SELECT AN ENTRY", "节点就绪 // 请选择条目")]

    def start_agent_step(self, mode: str, model_index: int = 0) -> bool:
        """Start one agent step without holding the initiating HTTP request open."""

        with self.state_lock:
            if self.agent_busy or self.session.done:
                return False
            self.agent_busy = True
            self.agent_job_id += 1
            job_id = self.agent_job_id

        def run() -> None:
            try:
                _agent_step(self, mode=mode, model_index=model_index, job_id=job_id)
            finally:
                with self.state_lock:
                    if self.agent_job_id == job_id:
                        self.agent_busy = False
                    client = self.clients.pop((job_id, model_index), None)
                if client is not None:
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass

        threading.Thread(target=run, name="terminal-hacking-agent-step", daemon=True).start()
        return True


class TerminalHandler(BaseHTTPRequestHandler):
    server: TerminalServer

    def do_GET(self) -> None:
        route = _route(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            with self.server.state_lock:
                self._send_json(_payload(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route(self.path)
        body = self._read_json()
        if route == "new":
            with self.server.state_lock:
                self.server.reset()
        elif route == "action":
            with self.server.state_lock:
                if not self.server.agent_busy:
                    _apply(self.server, body.get("action") if isinstance(body, dict) else None, actor="OPERATOR")
        elif route == "agent":
            mode = body.get("mode", "algo") if isinstance(body, dict) else "algo"
            try:
                model_index = int(body.get("model_index", 0)) if isinstance(body, dict) else 0
            except (TypeError, ValueError):
                model_index = -1
            self.server.start_agent_step(str(mode), model_index=model_index)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        with self.server.state_lock:
            self._send_json(_payload(self.server))

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("terminal-hacking-web: " + fmt % args + "\n")

    def _read_json(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, json.JSONDecodeError):
            return {}

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


def _route(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    for route in ("state", "new", "action", "agent"):
        if path.endswith(f"/api/{route}"):
            return route
    if "/api/" in path:
        return "missing"
    return "index" if path == "/" or "." not in path.rsplit("/", 1)[-1] else "missing"


def _event(en: str, zh: str) -> dict[str, str]:
    return {"en": en, "zh": zh}


def _apply(server: TerminalServer, action: object, *, actor: str) -> None:
    server.error = None
    result = server.session.execute(action)
    if not result["valid"]:
        server.error = str(result["error"])
        server.events.insert(0, _event(f"INPUT REJECTED // {result['error']}", f"输入被拒绝 // {result['error']}"))
        return
    event = result["event"]
    if event["kind"] == "guess":
        if event["solved"]:
            message = _event(f"{actor}: {event['word']} // ACCESS GRANTED", f"{actor}：{event['word']} // 访问授权")
        else:
            message = _event(
                f"{actor}: {event['word']} // POSITIONAL MATCH {event['likeness']}/{event['out_of']}",
                f"{actor}：{event['word']} // 同位匹配 {event['likeness']}/{event['out_of']}",
            )
    elif event["effect"] == "replenish":
        message = _event(f"{actor}: ATTEMPT BUFFER RESTORED", f"{actor}：尝试次数已恢复")
    else:
        removed = event["removed"] or "NONE"
        message = _event(f"{actor}: DECOY PURGED // {removed}", f"{actor}：干扰项已清除 // {removed}")
    server.events.insert(0, message)
    if server.session.locked:
        server.events.insert(0, _event("NODE LOCKED // SESSION TERMINATED", "节点锁定 // 会话终止"))
    server.events = server.events[:24]


class _AgentStepCancelled(Exception):
    """Internal signal used when New Node detaches an in-flight agent job."""


def _job_is_current(server: TerminalServer, job_id: int | None) -> bool:
    return job_id is None or server.agent_job_id == job_id


def _server_lock(server: TerminalServer):
    return getattr(server, "state_lock", nullcontext())


def _agent_step(
    server: TerminalServer,
    *,
    mode: str = "algo",
    model_index: int = 0,
    job_id: int | None = None,
) -> None:
    if server.session.done:
        return
    if mode == "algo":
        try:
            _target, action = expert_decision(server.session, server.rng)
            with _server_lock(server):
                if _job_is_current(server, job_id):
                    _apply(server, action, actor="ALGO")
        except (TypeError, ValueError) as exc:
            server.error = str(exc)
            server.events.insert(0, _event(f"ALGO ERROR // {exc}", f"算法错误 // {exc}"))
        return
    if mode != "model":
        server.error = f"Unknown agent mode: {mode}"
        return
    targets = server.args.inference_targets
    if not targets:
        server.error = "Model mode requires --base-url"
        return
    if not 0 <= model_index < len(targets):
        server.error = f"Invalid inference model index: {model_index}"
        return
    target = targets[model_index]
    try:
        state = server.session.public_state()
        if probe_is_preferred(server.session):
            probe = server.rng.choice(state["available_probes"])
            with _server_lock(server):
                if _job_is_current(server, job_id):
                    _apply(server, f"probe:{probe['id']}", actor="AGENT")
            return
        active = state["active_candidates"]
        if not any(event.get("kind") == "guess" for event in state["history"]):
            with _server_lock(server):
                if _job_is_current(server, job_id):
                    action = candidate_controller_action(server.session, active, server.rng)
                    _apply(server, action, actor="AGENT")
            return
        tool = candidate_tool(active)
        request_kwargs = {
            "model": target["model"],
            "messages": candidate_filter_request_messages(
                make_filter_prompt(server.record),
                state,
            ),
            "tools": [tool],
            "tool_choice": tool_choice_for_base_url(target["base_url"]),
            "temperature": server.args.temperature,
        }
        extra_body = extra_body_for_base_url(target["base_url"])
        if extra_body is None:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        request_kwargs["extra_body"] = extra_body
        response = _model_completion_with_retry(server, model_index, target, request_kwargs, job_id=job_id)
        if not _job_is_current(server, job_id):
            return
        calls = response.choices[0].message.tool_calls or []
        if len(calls) != 1 or calls[0].function.name != "submit_candidates":
            raise ValueError(f"expected one submit_candidates call, received {len(calls)}")
        arguments = json.loads(calls[0].function.arguments)
        candidates = arguments.get("candidates") if set(arguments) == {"candidates"} else None
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(not isinstance(candidate, str) for candidate in candidates)
            or len(candidates) != len(set(candidates))
            or any(candidate not in active for candidate in candidates)
        ):
            raise ValueError(f"invalid agent candidates: {arguments!r}")
        with _server_lock(server):
            if _job_is_current(server, job_id):
                _apply(server, candidate_controller_action(server.session, candidates, server.rng), actor="AGENT")
    except _AgentStepCancelled:
        return
    except Exception as exc:  # noqa: BLE001
        with _server_lock(server):
            if _job_is_current(server, job_id):
                server.error = str(exc)
                server.events.insert(0, _event(f"AGENT LINK ERROR // {exc}", f"Agent 链路错误 // {exc}"))


def _model_completion_with_retry(
    server: TerminalServer,
    model_index: int,
    target: dict[str, str],
    request_kwargs: dict[str, Any],
    *,
    job_id: int | None = None,
):
    """Retry transient upstream failures every ten seconds for at most five minutes."""

    from openai import OpenAI

    deadline = time.monotonic() + MODEL_WAIT_MAX_SECONDS
    retry_count = 0
    client_key = (job_id if job_id is not None else -1, model_index)
    while True:
        if not _job_is_current(server, job_id):
            raise _AgentStepCancelled
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("model connection did not recover within 5 minutes")
        try:
            if client_key not in server.clients:
                server.clients[client_key] = OpenAI(
                    base_url=target["base_url"],
                    api_key=target["api_key"],
                    timeout=None,
                    max_retries=0,
                )
            client = server.clients[client_key]
            return client.chat.completions.create(
                **request_kwargs,
                timeout=None,
            )
        except Exception as exc:  # noqa: BLE001
            if not _job_is_current(server, job_id):
                raise _AgentStepCancelled from exc
            if not _is_transient_model_error(exc):
                raise
            retry_count += 1
            with _server_lock(server):
                if not _job_is_current(server, job_id):
                    raise _AgentStepCancelled from exc
                server.error = f"MODEL LINK RETRY {retry_count} // {exc}"
            try:
                server.clients[client_key].close()
            except Exception:  # noqa: BLE001
                pass
            server.clients.pop(client_key, None)
            sleep_for = min(MODEL_RETRY_INTERVAL_SECONDS, deadline - time.monotonic())
            if sleep_for <= 0:
                raise TimeoutError("model connection did not recover within 5 minutes") from exc
            time.sleep(sleep_for)


def _is_transient_model_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
    }:
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429} or isinstance(status_code, int) and status_code >= 500


def _payload(server: TerminalServer) -> dict[str, Any]:
    return {
        **server.session.public_state(),
        "allowed_actions": server.session.allowed_actions(),
        "agent_busy": server.agent_busy,
        "inference_models": [
            {"index": index, "label": target["label"]} for index, target in enumerate(server.args.inference_targets)
        ],
        "events": server.events,
        "error": server.error,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en" data-theme="phosphor">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Civil Defense Data Terminal</title>
<style>
:root{--void:#030805;--glass:#07110b;--phosphor:#72ff8f;--bright:#c0ffca;--dim:#2e7f42;--ghost:rgba(114,255,143,.12);--shell:#171a16;--edge:#41453c;--warning:#ffdc73;--unit:4px}
html[data-theme="amber"]{--void:#0a0702;--glass:#151006;--phosphor:#ffba52;--bright:#ffe0a0;--dim:#9d6322;--ghost:rgba(255,186,82,.13);--warning:#fff0b2}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:#0b0c0a;color:var(--phosphor);font-family:"SFMono-Regular","Cascadia Mono","Liberation Mono",monospace;-webkit-font-smoothing:none}
button,select{font:inherit}.room{min-height:100vh;display:grid;place-items:center;padding:32px;background:radial-gradient(circle at 50% 20%,#303229 0,#161813 36%,#090a08 75%)}
.console{width:min(1180px,100%);padding:20px;border:1px solid #4b4f45;border-radius:26px;background:linear-gradient(145deg,#23261f,#11130f);box-shadow:0 28px 80px #000,0 0 0 5px #090a08, inset 0 1px #555b50}
.hardware{height:28px;display:flex;align-items:flex-start;justify-content:space-between;color:#8f9484;font-size:10px;letter-spacing:.18em;padding:0 8px}.brand{font-weight:700}.serial{opacity:.6}
.screen{position:relative;overflow:hidden;min-height:690px;border-radius:42px/30px;background:var(--glass);box-shadow:inset 0 0 0 2px #050805,inset 0 0 38px #000,inset 0 0 110px rgba(0,0,0,.72),0 0 0 1px #454a40;padding:34px 38px;text-shadow:0 0 7px currentColor}
.screen:before{content:"";position:absolute;inset:0;pointer-events:none;background:repeating-linear-gradient(to bottom,transparent 0,transparent 2px,rgba(0,0,0,.28) 3px,rgba(0,0,0,.28) 4px);opacity:.7;z-index:5}
.screen:after{content:"";position:absolute;inset:-20%;pointer-events:none;background:linear-gradient(100deg,transparent 42%,rgba(255,255,255,.035) 49%,transparent 56%);animation:sweep 8s linear infinite;z-index:6}@keyframes sweep{to{transform:translateX(45%)}}
.topline{position:relative;z-index:2;display:flex;justify-content:space-between;gap:24px;padding-bottom:20px;border-bottom:1px dashed var(--dim)}
.eyebrow{font-size:11px;letter-spacing:.22em;color:var(--dim)}h1{font-size:18px;line-height:1.25;margin:5px 0 0;letter-spacing:.08em;font-weight:600}.status{text-align:right;font-size:12px;line-height:1.55}.attempts{color:var(--bright);font-size:16px;letter-spacing:.14em}
.grid{position:relative;z-index:2;display:grid;grid-template-columns:minmax(520px,1.45fr) minmax(260px,.7fr);gap:32px;padding-top:24px}.dump{font-size:14px;line-height:1.65;white-space:nowrap}.dump-row{display:grid;grid-template-columns:68px 1fr 68px 1fr;gap:8px}.address{color:var(--dim)}.bytes{letter-spacing:.03em}.entry{display:inline;padding:1px 0;border:0;color:var(--bright);background:transparent;text-shadow:inherit;cursor:pointer}.entry:hover,.entry:focus-visible{outline:0;color:var(--void);background:var(--bright);text-shadow:none}.entry:active{transform:translateY(1px)}.entry[disabled]{cursor:not-allowed;color:var(--dim);text-decoration:line-through;background:transparent}
.side{border-left:1px dashed var(--dim);padding-left:24px;min-width:0}.side-title{font-size:11px;letter-spacing:.2em;color:var(--dim);margin-bottom:12px}.log{display:flex;flex-direction:column;gap:10px;min-height:360px;font-size:12px;line-height:1.5}.log-line{padding-left:12px;position:relative;color:var(--phosphor);overflow-wrap:anywhere}.log-line:before{content:">";position:absolute;left:0;color:var(--bright)}.log-line:first-child{color:var(--bright)}
.controls{position:relative;z-index:2;display:flex;align-items:center;justify-content:space-between;gap:12px;border-top:1px dashed var(--dim);padding-top:18px;margin-top:22px}.control-group{display:flex;gap:8px;flex-wrap:wrap}.control{min-height:40px;padding:0 14px;border:1px solid var(--dim);background:var(--ghost);color:var(--phosphor);cursor:pointer;letter-spacing:.08em}.control:hover,.control:focus-visible{border-color:var(--bright);color:var(--bright);outline:none}.control:active{transform:scale(.97)}.control.primary{background:var(--phosphor);color:var(--void);text-shadow:none}.control[disabled]{opacity:.35;cursor:not-allowed}select.control option{background:var(--glass);color:var(--phosphor)}
.help{font-size:11px;color:var(--dim);max-width:520px;line-height:1.55;text-align:right}.cursor{display:inline-block;width:8px;height:15px;background:var(--bright);vertical-align:-2px;animation:blink 1s steps(1) infinite}@keyframes blink{50%{opacity:0}}
@media(max-width:850px){.room{padding:0}.console{border-radius:0;padding:0;width:100%;min-height:100vh}.hardware{display:none}.screen{border-radius:0;min-height:100vh;padding:24px 18px}.grid{grid-template-columns:1fr}.dump{font-size:11px;overflow:auto}.dump-row{grid-template-columns:56px 1fr 56px 1fr;gap:5px}.side{border-left:0;border-top:1px dashed var(--dim);padding:18px 0 0}.log{min-height:160px}.controls{align-items:flex-start;flex-direction:column}.help{text-align:left}}
@media(prefers-reduced-motion:reduce){.screen:after,.cursor{animation:none}}
</style>
</head>
<body><main class="room"><section class="console"><header class="hardware"><span class="brand">CIVIL DEFENSE DATA SYSTEMS</span><span class="serial">MODEL VT-77 // NODE 04</span></header><div class="screen">
<header class="topline"><div><div class="eyebrow" data-i18n="eyebrow">REMOTE MEMORY ACCESS</div><h1 data-i18n="title">SECURE ARCHIVE // AUTHENTICATION</h1></div><div class="status"><div data-i18n="allowance">ATTEMPT BUFFER</div><div class="attempts" id="attempts"></div><div id="stateLabel"></div></div></header>
<div class="grid"><div class="dump" id="dump"></div><aside class="side"><div class="side-title" data-i18n="trace">ACCESS TRACE</div><div class="log" id="log"></div></aside></div>
<footer class="controls"><div class="control-group"><select class="control" id="agentMode" aria-label="Agent mode"><option value="algo">ALGO</option><option value="model">MODEL</option></select><select class="control" id="modelTarget" aria-label="Inference model"></select><button class="control primary" id="agent" data-i18n="agent">AGENT STEP</button><button class="control" id="reset" data-i18n="reset">NEW NODE</button><button class="control" id="theme">GREEN / AMBER</button><button class="control" id="lang">中文</button></div><div class="help" data-i18n="help">Select a word. POSITIONAL MATCH counts letters in the correct slots. Bracket sequences can purge a decoy or restore attempts.</div></footer>
</div></section></main>
<script>
const copy={en:{eyebrow:'REMOTE MEMORY ACCESS',title:'SECURE ARCHIVE // AUTHENTICATION',allowance:'ATTEMPT BUFFER',trace:'ACCESS TRACE',agent:'AGENT STEP',loading:'AGENT PROCESSING...',reset:'NEW NODE',help:'Select a word. POSITIONAL MATCH counts letters in the correct slots. Bracket sequences can purge a decoy or restore attempts.',ready:'AWAITING INPUT',granted:'ACCESS GRANTED',locked:'NODE LOCKED'},zh:{eyebrow:'远程内存访问',title:'安全档案 // 身份验证',allowance:'尝试缓冲区',trace:'访问记录',agent:'AGENT 单步',loading:'AGENT 推理中...',reset:'新建节点',help:'选择一个单词。“同位匹配”表示字母与密码位置同时正确；括号序列可清除干扰项或恢复尝试次数。',ready:'等待输入',granted:'访问授权',locked:'节点锁定'}};
let lang='en',state=null,busy=false,clientError='';
const AGENT_POLL_MS=1000,AGENT_WAIT_MAX_MS=300000;
const proxyPrefix=location.pathname==='/'?'':location.pathname.replace(/\/$/,'');
function apiPath(name){return `${proxyPrefix}/api/${name}`}
async function api(path,body){const options=body===undefined?{}:{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)};const response=await fetch(path,options);if(!response.ok)throw new Error(await response.text());return response.json()}
function t(key){return copy[lang][key]}
function syncModelOptions(){const select=document.getElementById('modelTarget');const signature=JSON.stringify(state.inference_models||[]);if(select.dataset.signature!==signature){const previous=select.value;select.innerHTML=(state.inference_models||[]).map(item=>`<option value="${item.index}">${escapeHtml(item.label)}</option>`).join('')||'<option value="0">NO MODEL CONFIGURED</option>';select.dataset.signature=signature;if([...select.options].some(option=>option.value===previous))select.value=previous}}
function renderSegments(segments){return segments.map(seg=>{if(!seg.action)return escapeHtml(seg.text);const enabled=state.allowed_actions.includes(seg.action);return `<button class="entry" data-action="${escapeHtml(seg.action)}" ${enabled?'':'disabled'}>${escapeHtml(seg.text)}</button>`}).join('')}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
function render(){document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=t(el.dataset.i18n));const waiting=busy||state.agent_busy;syncModelOptions();document.getElementById('lang').textContent=lang==='en'?'中文':'EN';document.getElementById('agent').textContent=waiting?t('loading'):t('agent');document.getElementById('attempts').textContent='■ '.repeat(state.attempts_remaining)+'□ '.repeat(state.attempts_max-state.attempts_remaining);document.getElementById('stateLabel').innerHTML=(waiting?t('loading'):state.solved?t('granted'):state.locked?t('locked'):t('ready'))+' <span class="cursor"></span>';document.getElementById('dump').innerHTML=state.dump_rows.map(row=>`<div class="dump-row"><span class="address">${row.left_address}</span><span class="bytes">${renderSegments(row.left_segments)}</span><span class="address">${row.right_address}</span><span class="bytes">${renderSegments(row.right_segments)}</span></div>`).join('');const loading=waiting?`<div class="log-line">${escapeHtml(t('loading'))}</div>`:'';const errorText=clientError||state.error||'';const errors=errorText?`<div class="log-line">${escapeHtml(errorText)}</div>`:'';document.getElementById('log').innerHTML=loading+errors+state.events.map(event=>`<div class="log-line">${escapeHtml(event[lang])}</div>`).join('');document.getElementById('agent').disabled=waiting||state.done;document.getElementById('agentMode').disabled=waiting;document.getElementById('modelTarget').disabled=waiting||document.getElementById('agentMode').value!=='model';document.querySelectorAll('.entry[data-action]').forEach(button=>button.addEventListener('click',()=>act(button.dataset.action)))}
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function waitForAgent(){const deadline=Date.now()+AGENT_WAIT_MAX_MS;while(state.agent_busy&&Date.now()<deadline){try{state=await api(apiPath('state'));clientError='';render()}catch(error){clientError=`STATE POLL RETRY // ${error.message}`;render()}if(state.agent_busy&&Date.now()<deadline)await delay(AGENT_POLL_MS)}if(state.agent_busy)throw new Error('AGENT WAIT EXCEEDED 5 MINUTES // SERVER STEP MAY STILL BE RUNNING')}
async function load(){state=await api(apiPath('state'));render();if(state.agent_busy){try{await waitForAgent()}catch(error){clientError=error.message;render()}}}
async function act(action){if(busy||state.agent_busy)return;busy=true;render();try{state=await api(apiPath('action'),{action})}finally{busy=false;render()}}
document.getElementById('agent').onclick=async()=>{if(busy||state.agent_busy)return;busy=true;clientError='';render();try{state=await api(apiPath('agent'),{mode:document.getElementById('agentMode').value,model_index:Number(document.getElementById('modelTarget').value)});busy=false;render();await waitForAgent()}catch(error){clientError=error.message}finally{busy=false;render()}};
document.getElementById('agentMode').onchange=()=>render();
document.getElementById('reset').onclick=async()=>{state=await api(apiPath('new'),{});render()};
document.getElementById('theme').onclick=()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='amber'?'phosphor':'amber'};
document.getElementById('lang').onclick=()=>{lang=lang==='en'?'zh':'en';render()};
load().catch(error=>{document.getElementById('log').textContent=error.message});
</script></body></html>"""


def _build_inference_targets(
    parser: argparse.ArgumentParser,
    base_urls: list[str],
    api_keys: list[str],
    models: list[str],
) -> list[dict[str, str]]:
    """Zip parallel inference options, broadcasting one key or model when requested."""

    if not base_urls:
        return []
    count = len(base_urls)
    if len(api_keys) == 1:
        api_keys = api_keys * count
    if len(models) == 1:
        models = models * count
    if len(api_keys) != count or len(models) != count:
        parser.error("--base-url, --api-key, and --model must have equal lengths (single key/model may broadcast)")
    targets = []
    for base_url, api_key, model in zip(base_urls, api_keys, models, strict=True):
        hostname = urlparse(base_url).hostname or base_url
        targets.append(
            {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "label": f"{model} @ {hostname}",
            }
        )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--base-url", nargs="+", help="One or more inference API base URLs")
    parser.add_argument("--api-key", nargs="+", default=["EMPTY"], help="Parallel API keys; one value broadcasts")
    parser.add_argument("--model", nargs="+", default=["policy"], help="Parallel model names; one value broadcasts")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    args.inference_targets = _build_inference_targets(parser, args.base_url or [], args.api_key, args.model)
    server = TerminalServer((args.host, args.port), TerminalHandler, args=args)
    print(f"Terminal hacking WebUI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
