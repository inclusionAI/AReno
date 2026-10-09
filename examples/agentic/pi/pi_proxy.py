"""Sample-local secondary proxy: pi -> this proxy -> AReno rollout proxy."""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


class PiProxy:
    """Capture exact upstream metadata while presenting pi with a chat API.

    SSE is buffered: AReno completes a turn before we emit its chunks. Each
    instance belongs to exactly one task/sample and uses a private bearer key.
    """

    def __init__(
        self,
        ctx,
        *,
        max_turns: int = 32,
        timeout: float = 300,
        bind_host: str = "127.0.0.1",
        connect_host: str = "127.0.0.1",
    ):
        if max_turns < 1 or timeout <= 0:
            raise ValueError("max_turns and timeout must be positive")
        self.ctx = ctx
        self.bind_host = bind_host
        self.connect_host = connect_host
        self.max_turns = max_turns
        self.timeout = timeout
        self.api_key = secrets.token_hex(24)
        self.trace: list[tuple[dict, dict]] = []
        self.errors: list[str] = []
        self.limit_reached = False
        self._lock = threading.Lock()
        self._active = False
        self._requests = 0
        self._server = None
        self._thread = None

    async def __aenter__(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                proxy._handle(self)

            def log_message(self, *args):
                pass

        # Non-daemon request threads are drained before the training phase.
        self._server = ThreadingHTTPServer((self.bind_host, 0), Handler)
        self._server.daemon_threads = False
        self._active = True
        port = self._server.server_address[1]
        self.base_url = f"http://{self.connect_host}:{port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    async def __aexit__(self, *args):
        self._active = False
        # The upstream rollout proxy needs the event loop to remain available.
        await asyncio.to_thread(self._server.shutdown)
        await asyncio.to_thread(self._server.server_close)
        await asyncio.to_thread(self._thread.join)

    def _handle(self, handler):
        handler.connection.settimeout(self.timeout)
        if handler.path != "/v1/chat/completions":
            self._json(handler, 404, {"error": {"message": "unsupported route"}})
            return
        if handler.headers.get("Authorization") != f"Bearer {self.api_key}":
            self._json(handler, 401, {"error": {"message": "invalid rollout key"}})
            return
        try:
            length = int(handler.headers.get("Content-Length", "0"))
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("request body must be between 1 byte and 16 MiB")
            body = json.loads(handler.rfile.read(length))
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                raise ValueError("request must contain a messages list")
            if body.get("n", 1) != 1:
                raise ValueError("pi rollout requires n=1")
            stream = bool(body.get("stream"))
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            # One sample's model calls stay ordered; different proxies run concurrently.
            with self._lock:
                if not self._active:
                    raise ValueError("pi rollout closed")
                if self._requests >= self.max_turns:
                    self.limit_reached = True
                    self._json(handler, 429, {"error": {"message": "pi turn limit reached"}})
                    return
                self._requests += 1
                upstream = dict(body, stream=False)
                # Generation length is owned by areno train, not pi's model catalog.
                for key in ("stream_options", "max_tokens", "max_completion_tokens"):
                    upstream.pop(key, None)
                request = Request(
                    f"{self.ctx.base_url}/chat/completions",
                    data=json.dumps(upstream).encode(),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.ctx.api_key}"},
                )
                # Loopback requests must not use an ambient HTTP(S)_PROXY.
                with build_opener(ProxyHandler({})).open(request, timeout=self.timeout) as result:
                    response = json.load(result)
                metadata = response.get("areno", {})
                tokens = metadata.get("response_tokens")
                logprobs = metadata.get("response_logprobs")
                if not isinstance(tokens, list) or not isinstance(logprobs, list) or len(tokens) != len(logprobs):
                    raise ValueError("upstream did not return aligned AReno tokens/logprobs")
                if tokens and not metadata.get("input_tokens"):
                    raise ValueError("upstream did not return exact input tokens")
                self.trace.append((upstream, response))
            if stream:
                self._stream(handler, response, include_usage)
            else:
                self._json(handler, 200, {k: v for k, v in response.items() if k != "areno"})
        except Exception as exc:
            status = 400 if isinstance(exc, (ValueError, UnicodeDecodeError)) else 502
            if isinstance(exc, HTTPError):
                detail = exc.read().decode(errors="replace")[:2000]
                message = f"upstream HTTP {exc.code}: {detail}"
            else:
                message = str(exc)
            self.errors.append(message)
            try:
                self._json(handler, status, {"error": {"message": message}})
            except (OSError, TimeoutError):
                pass

    @staticmethod
    def _json(handler, status, body):
        data = json.dumps(body).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def _stream(handler, response, include_usage):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        for chunk in stream_chunks(response, include_usage=include_usage):
            handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            handler.wfile.flush()
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()

    def turns(self, item):
        """Rebuild trainable turns from server-side metadata, never rendered text."""
        from areno.api.agentic import AgentTrajectoryTurn

        return [
            AgentTrajectoryTurn(
                item=item,
                messages=request["messages"],
                response=response,
                tools=request.get("tools") or [],
                tool_choice=request.get("tool_choice"),
                model=request.get("model") or "policy",
            )
            for request, response in self.trace
        ]


def stream_chunks(response, *, include_usage=False):
    base = {key: response[key] for key in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    for choice in response["choices"]:
        delta = dict(choice["message"])
        if delta.get("tool_calls"):
            delta["tool_calls"] = [dict(call, index=i) for i, call in enumerate(delta["tool_calls"])]
        yield {**base, "choices": [{"index": choice["index"], "delta": delta, "finish_reason": None}]}
        yield {**base, "choices": [{"index": choice["index"], "delta": {}, "finish_reason": choice["finish_reason"]}]}
    if include_usage:
        yield {**base, "choices": [], "usage": response["usage"]}
