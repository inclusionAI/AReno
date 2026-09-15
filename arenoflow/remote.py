"""Sandbox entrypoint. Only this file is added to the published AReno image."""

from __future__ import annotations

import base64
import hmac
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PREFIX = "ARENOFLOW_EVENT "


def emit(kind, **data):
    print(PREFIX + json.dumps(dict(type=kind, time=time.time(), **data), allow_nan=False), flush=True)


def cache_model_refs(args):
    """Resolve original model snapshots in the mounted Volume before starting AReno."""
    result = list(args)
    hub = result[result.index("--model-hub") + 1] if "--model-hub" in result else "hf"
    if hub != "hf":
        raise ValueError("Only Hugging Face is supported")
    if "--model-hub" not in result:
        result.extend(["--model-hub", "hf"])
    resolved = {}
    for index, option in enumerate(result[:-1]):
        if option not in ("--ckpt", "--model-path", "--ref-ckpt", "--reward-ckpt", "--critic-ckpt"):
            continue
        reference = result[index + 1]
        if Path(reference).exists():
            continue
        if reference.startswith("/"):
            raise FileNotFoundError("The selected checkpoint is not available on the mounted Volume")
        if reference not in resolved:
            emit("phase", phase="caching_model")
            emit("model_cache", model=reference, hub=hub, status="checking")
            from huggingface_hub import snapshot_download

            directory = os.environ.get("HF_HUB_CACHE", "/artifacts/cache/hf/hub")
            resolved[reference] = snapshot_download(reference, cache_dir=directory)
            emit("model_cache", model=reference, hub=hub, status="ready")
        result[index + 1] = resolved[reference]
    return result


def stage_main(stage):
    args = cache_model_refs(stage["args"])
    emit("phase", phase="preparing_data")
    import areno.api
    import areno.api.metrics as metrics
    from areno import Trainer
    from areno.cli.train import train_command

    class FlowTrainer(Trainer):
        """Save the final successful state before the CLI releases its backend."""

        def init(self):
            emit("phase", phase="loading_model")
            super().init()
            emit("phase", phase="training")
            emit("stage", index=stage["index"], status="running", algo=stage["algo"])
            self.flow_initialized = True

        def close(self):
            initialized = getattr(self, "flow_initialized", False)
            self.flow_initialized = False
            try:
                if initialized and sys.exc_info()[0] is None:
                    emit("phase", phase="saving_checkpoint")
                    self.save_checkpoint(str(Path(stage["save_path"]) / "final"))
            finally:
                super().close()

    areno.api.Trainer = FlowTrainer

    original = metrics.create_tensorboard_writer

    class Writer:
        def __init__(self, writer):
            self.writer = writer

        def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
            import math

            value = float(scalar_value)
            if math.isfinite(value):
                emit("metric", tag=tag, value=value, step=global_step or 0, index=stage.get("index", 0))
            return self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.writer, name)

    metrics.create_tensorboard_writer = lambda directory: Writer(original(directory))
    train_command.main(args=args, prog_name="areno train")


def latest_checkpoint(directory):
    root = Path(directory)
    candidates = sorted(root.glob("step_*"))
    if (root / "final").is_dir():
        candidates.append(root / "final")
    for path in reversed(candidates):
        if (path / "adapter_config.json").is_file() and list(path.glob("*.safetensors")):
            return str(path)
        if (path / "config.json").is_file() and (list(path.glob("*.safetensors")) or list(path.glob("*.bin"))):
            return str(path)
    raise RuntimeError(
        "No full-model checkpoint was saved. Lower save_interval; adapter-only chaining requires a base model."
    )


def gateway():
    """Expose only an authenticated proxy; AReno itself binds to loopback."""

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            expected = "Bearer " + os.environ["ARENOFLOW_ENDPOINT_KEY"]
            if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                self.send_error(401, "Bearer token required")
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > 8 * 1024 * 1024:
                self.send_error(413)
                return
            connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=300)
            try:
                connection.request(
                    self.command,
                    self.path,
                    body=self.rfile.read(length) if length else None,
                    headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                )
                response = connection.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                self.close_connection = True
            except (OSError, http.client.HTTPException):
                self.close_connection = True
            finally:
                connection.close()

        def log_message(self, *_args):
            pass

    ThreadingHTTPServer(("0.0.0.0", 8080), Proxy).serve_forever()


def main(manifest):
    emit("runtime", image=manifest["image"], source_revision=manifest["revision"])
    if manifest["kind"] == "model_download":
        cache_model_refs(["--model-path", manifest["model"]["checkpoint"], "--model-hub", manifest["model_hub"]])
        emit("complete", model=manifest["model"]["checkpoint"])
        return
    if manifest["kind"] == "deployment":
        emit("phase", phase="loading_model")
        args = cache_model_refs(manifest["serve_args"])
        emit("phase", phase="loading_model")
        proc = subprocess.Popen(["areno", "serve", *args])
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError(f"Serving process exited with code {proc.returncode}")
            conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=2)
            try:
                conn.request("GET", "/openapi.json")
                if conn.getresponse().status == 200:
                    break
            except OSError:
                pass
            finally:
                conn.close()
            time.sleep(2)
        else:
            proc.terminate()
            raise RuntimeError("Endpoint did not become healthy within 30 minutes")
        threading.Thread(target=gateway, daemon=True).start()
        emit("ready")
        sys.exit(proc.wait())
    previous = None
    for index, stage in enumerate(manifest["stages"]):
        args = [previous if arg == "__previous__" else arg for arg in stage["args"]]
        if None in args:
            raise RuntimeError("A previous checkpoint is required")
        emit("stage", index=index, status="starting", algo=stage["algo"])
        emit("phase", phase="preparing_data")
        proc = subprocess.Popen(
            [sys.executable, "-u", __file__, "stage", json.dumps({**stage, "args": args, "index": index})],
            stderr=subprocess.STDOUT,
        )
        code = proc.wait()
        if code:
            emit("stage", index=index, status="failed", exit_code=code)
            sys.exit(code)
        previous = latest_checkpoint(stage["save_path"])
        adapter_only = (Path(previous) / "adapter_config.json").is_file()
        if adapter_only and index < len(manifest["stages"]) - 1:
            raise RuntimeError("Adapter-only artifacts cannot be chained as full model checkpoints")
        emit("checkpoint", index=index, path=previous, adapter_only=adapter_only)
        emit("stage", index=index, status="succeeded", algo=stage["algo"])
    emit("complete", checkpoint=previous)


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        stage_main(json.loads(sys.argv[2]))
    else:
        try:
            main(json.loads(base64.urlsafe_b64decode(sys.argv[1])))
        except Exception as exc:
            emit("error", message=str(exc))
            raise
