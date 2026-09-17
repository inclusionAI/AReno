"""Dashboard-owned Modal application and a headless HTTP adapter for tests."""

from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlsplit

from areno.dashboard.flow.assets import save_upload
from areno.dashboard.flow.billing import fetch_billing
from areno.dashboard.flow.catalog import catalog
from areno.dashboard.flow.controller import Controller
from areno.dashboard.flow.credentials import Credentials
from areno.dashboard.flow.datasets import find, resolve_request, save_dataset, save_function, save_script_batch
from areno.dashboard.flow.gpu_estimate import recommend
from areno.dashboard.flow.inference import test_deployment
from areno.dashboard.flow.llm import ScriptGenerator
from areno.dashboard.flow.pricing import Pricing
from areno.dashboard.flow.provider import latest_image
from areno.dashboard.flow.sample_cache import SampleCache
from areno.dashboard.flow.store import Store
from areno.dashboard.flow.workflows import GPU_TYPES, plan


class Application:
    def __init__(self, directory, controller=None):
        self.catalog = catalog()
        self.controller = controller or Controller(Store(directory), self.catalog)
        self.csrf = secrets.token_urlsafe(32)
        self.billing_cache = None
        self.billing_time = 0
        self.billing_lock = threading.Lock()
        self.image_cache = None
        self.image_lock = threading.Lock()
        self.pricing = Pricing(directory)
        self.llm = ScriptGenerator()
        self.samples = SampleCache(self.controller.store)
        self.controller.cost_estimator = self.pricing.quote
        self.credentials = Credentials(directory)
        self.connection_lock = threading.Lock()
        self.connection_error = None
        self.reconnecting = False
        self.reconnect_stop = threading.Event()
        self.reconnect_thread = None
        if self.credentials.saved or all(self.controller.env_credentials):
            self.reconnecting = True
            self.reconnect_thread = threading.Thread(target=self._auto_reconnect, daemon=True)
            self.reconnect_thread.start()

    def _stored_credentials(self):
        return self.credentials.load() or self.controller.env_credentials

    def _auto_reconnect(self):
        delay = 5
        while not self.reconnect_stop.is_set():
            try:
                with self.connection_lock:
                    if self.controller.provider is not None:
                        self.reconnecting = False
                        return
                    token_id, token_secret = self._stored_credentials()
                    self.controller.connect(token_id, token_secret)
                    self.connection_error = None
                    self.reconnecting = False
                    return
            except Exception as exc:
                if self.controller.provider is not None:
                    self.reconnecting = False
                    return
                self.connection_error = self.controller.redact(str(exc))
            if self.reconnect_stop.wait(delay):
                break
            delay = min(delay * 2, 60)
        self.reconnecting = False

    def connect(self, body):
        with self.connection_lock:
            token_id, token_secret = body.get("token_id", ""), body.get("token_secret", "")
            if not isinstance(token_id, str) or not isinstance(token_secret, str):
                raise ValueError("Modal credentials must be text")
            token_id, token_secret = token_id.strip(), token_secret.strip()
            if not token_id and not token_secret:
                token_id, token_secret = self._stored_credentials()
            result = self.controller.connect(token_id, token_secret)
            if body.get("remember", True):
                self.credentials.save(token_id, token_secret)
            else:
                self.credentials.forget()
            self.connection_error = None
            self.reconnecting = False
            return {**result, "credentials_saved": self.credentials.saved}

    def get(self, path, query):
        if path == "/api/bootstrap":
            return dict(
                catalog=self.catalog,
                csrf=self.csrf,
                connected=self.controller.provider is not None,
                environment_credentials=all(self.controller.env_credentials),
                credentials_saved=self.credentials.saved,
                reconnecting=self.reconnecting,
                connection_error=self.connection_error,
                gpu_types=GPU_TYPES,
            )
        if path == "/api/llm":
            return self.llm.settings()
        if path == "/api/jobs":
            return self.controller.store.jobs()
        if path == "/api/datasets":
            return [{**d, "sample_status": self.samples.status(d)["status"]} for d in self.controller.store.datasets()]
        if path == "/api/functions":
            return self.controller.store.functions()
        if path == "/api/estimates":
            return self.pricing.all_runs(self.controller.store.jobs())
        if path == "/api/images/latest":
            with self.image_lock:
                if not self.image_cache or time.time() - self.image_cache["checked_at"] >= 60:
                    self.image_cache = latest_image(self.catalog["image"])
                return self.image_cache
        if path.startswith("/api/jobs/"):
            parts = path.split("/")
            job = self.controller.store.get(parts[3])
            if len(parts) == 5 and parts[4] == "events":
                return self.controller.store.events(job["id"], int(query.get("after", ["0"])[0]))
            if len(parts) == 4:
                return job
        if path == "/api/billing":
            if not self.controller.provider:
                raise ValueError("Connect Modal to load workspace billing")
            with self.billing_lock:
                if not self.billing_cache or time.time() - self.billing_time > 15:
                    result = fetch_billing(self.controller.provider)
                    self.billing_cache = json.loads(self.controller.redact(json.dumps(result)))
                    self.billing_time = time.time()
                return self.billing_cache
        raise KeyError("Route not found")

    def post(self, path, body):
        if path == "/api/llm":
            return self.llm.configure(body)
        if path == "/api/scripts/sample":
            dataset = find(self.controller.store.datasets(), body.get("dataset_id"), "Dataset")
            return self.samples.enqueue(dataset, retry=bool(body.get("retry")))
        if path == "/api/scripts/batch":
            return save_script_batch(self.controller.store, body)
        if path == "/api/scripts/generate":
            return self.llm.generate(body, self.controller.store, self.catalog)
        if path == "/api/recommend-gpu":
            return recommend(body, self.pricing.cache or self.pricing.rates())
        if path == "/api/estimate":
            return self.pricing.quote(body.get("resources", {}), body.get("hours", 1), body.get("run_count", 1))
        if path == "/api/functions":
            return save_function(self.controller.store, body)
        if path.startswith("/api/functions/") and path.endswith("/delete"):
            return self.controller.store.delete_function(path.split("/")[3])
        if path == "/api/datasets":
            dataset = save_dataset(self.controller.store, body)
            state = self.samples.enqueue(dataset)
            return {**dataset, "sample_status": state["status"]}
        if path.startswith("/api/datasets/") and path.endswith("/delete"):
            return self.controller.store.delete_dataset(path.split("/")[3])
        if path == "/api/connect":
            result = self.connect(body)
            self.billing_cache = None
            return result
        if path == "/api/forget-credentials":
            with self.connection_lock:
                self.reconnect_stop.set()
                self.credentials.forget()
            return {"credentials_saved": False, "connected": self.controller.provider is not None}
        if path == "/api/uploads":
            return save_upload(self.controller.store.directory, body.get("name", ""), body.get("content", ""))
        if path == "/api/preview":
            return plan(resolve_request(body, self.controller.store), self.catalog)
        if path == "/api/jobs":
            return self.controller.submit(body)
        if path.startswith("/api/jobs/") and path.endswith("/inference"):
            return test_deployment(self.controller.store.get(path.split("/")[3]), body)
        if path.startswith("/api/jobs/") and path.endswith("/stop"):
            return self.controller.stop(path.split("/")[3])
        raise KeyError("Route not found")


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def origin_allowed(self):
            host = self.headers.get("Host", "")
            if urlsplit("//" + host).hostname not in ("localhost", "127.0.0.1", "[::1]", "::1"):
                return False
            origin = self.headers.get("Origin")
            return not origin or origin == "http://" + host

        def reply(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if not self.origin_allowed():
                self.reply(403, {"error": "Local same-origin access only"})
                return
            url = urlsplit(self.path)
            if url.path.startswith("/api/"):
                self.dispatch(lambda: app.get(url.path, parse_qs(url.query)))
                return
            self.reply(404, {"error": "Use the AReno dashboard UI"})

        def do_POST(self):
            if not self.origin_allowed() or not secrets.compare_digest(
                self.headers.get("X-Arenoflow-CSRF", ""), app.csrf
            ):
                self.reply(403, {"error": "Invalid origin or session; reload ARenoflow"})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply(415, {"error": "Use application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 24 * 1024 * 1024:
                    raise ValueError("Request must be between 1 byte and 24 MiB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
            except (ValueError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            self.dispatch(lambda: app.post(urlsplit(self.path).path, body))

        def dispatch(self, operation):
            try:
                self.reply(200, operation())
            except KeyError as exc:
                self.reply(404, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self.reply(400, {"error": app.controller.redact(str(exc))})
            except Exception as exc:
                self.reply(502, {"error": app.controller.redact(str(exc))})

        def log_message(self, *_args):
            # Request URLs/bodies can contain user configuration; no access log by default.
            pass

    return Handler
