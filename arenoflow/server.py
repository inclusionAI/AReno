"""Same-origin local API and production React static server."""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from arenoflow.assets import save_upload
from arenoflow.billing import fetch_billing
from arenoflow.catalog import ROOT, catalog
from arenoflow.controller import Controller
from arenoflow.datasets import resolve_request, save_dataset, save_function
from arenoflow.pricing import Pricing
from arenoflow.provider import latest_image
from arenoflow.store import Store
from arenoflow.workflows import GPU_TYPES, plan

STATIC = Path(__file__).with_name("static")


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
        self.pricing = Pricing()
        self.controller.cost_estimator = self.pricing.quote

    def get(self, path, query):
        if path == "/api/bootstrap":
            return dict(
                catalog=self.catalog,
                csrf=self.csrf,
                connected=self.controller.provider is not None,
                environment_credentials=all(self.controller.env_credentials),
                gpu_types=GPU_TYPES,
            )
        if path == "/api/jobs":
            return self.controller.store.jobs()
        if path == "/api/datasets":
            return self.controller.store.datasets()
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
        if path == "/api/estimate":
            return self.pricing.quote(body.get("resources", {}), body.get("hours", 1), body.get("run_count", 1))
        if path == "/api/functions":
            return save_function(self.controller.store, body)
        if path.startswith("/api/functions/") and path.endswith("/delete"):
            return self.controller.store.delete_function(path.split("/")[3])
        if path == "/api/datasets":
            return save_dataset(self.controller.store, body)
        if path.startswith("/api/datasets/") and path.endswith("/delete"):
            return self.controller.store.delete_dataset(path.split("/")[3])
        if path == "/api/connect":
            result = self.controller.connect(body.get("token_id", ""), body.get("token_secret", ""))
            self.billing_cache = None
            return result
        if path == "/api/uploads":
            return save_upload(self.controller.store.directory, body.get("name", ""), body.get("content", ""))
        if path == "/api/preview":
            return plan(resolve_request(body, self.controller.store), self.catalog)
        if path == "/api/jobs":
            return self.controller.submit(body)
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
            if url.path == "/brand/logo.svg":
                file = ROOT / "docs/_static/asystem_areno_logo.svg"
            else:
                file = (STATIC / url.path.lstrip("/")).resolve()
                if STATIC.resolve() not in file.parents and file != STATIC.resolve():
                    self.reply(404, {"error": "Not found"})
                    return
                if not file.is_file():
                    file = STATIC / "index.html"
            if not file.is_file():
                self.reply(503, {"error": "Build the React app: cd arenoflow/web && npm ci && npm run build"})
                return
            data = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'",
            )
            self.end_headers()
            self.wfile.write(data)

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


def main():
    parser = argparse.ArgumentParser(description="ARenoflow local control plane")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "0.0.0.0"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name(".data"))
    options = parser.parse_args()
    application = Application(options.data_dir)
    server = ThreadingHTTPServer((options.host, options.port), handler_for(application))
    print(f"ARenoflow → http://127.0.0.1:{options.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Local server stopped. Remote jobs continue until stopped or their Modal timeout expires.")


if __name__ == "__main__":
    main()
