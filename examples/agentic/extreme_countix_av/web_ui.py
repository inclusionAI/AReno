"""Web workbench for inspecting and evaluating Extreme Countix-AV samples."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import tempfile
import wave
from array import array
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import count_similarity, discover_samples, prompt_text  # noqa: E402
from run_agent import REPORT_TOOL, SYSTEM_PROMPT  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770


class CountixServer(ThreadingHTTPServer):
    def __init__(self, address, handler, *, args: argparse.Namespace):
        super().__init__(address, handler)
        self.args = args
        self.samples = [] if args.live_only else discover_samples(args.dataset_root)
        if args.browser_video_root:
            self.samples = [sample for sample in self.samples if _browser_video_path(args, sample).is_file()]
        self.by_id = {sample_id(sample): sample for sample in self.samples}


class Handler(BaseHTTPRequestHandler):
    server: CountixServer

    def do_GET(self) -> None:
        route = _route(self.path)
        if route == "index":
            page = INDEX_HTML.replace("__LIVE_ONLY__", json.dumps(self.server.args.live_only)).replace(
                "__SEGMENT_SECONDS__", json.dumps(self.server.args.segment_seconds)
            )
            self._send(page.encode(), "text/html; charset=utf-8")
        elif route == "samples":
            self._json({"samples": [_public_sample(sample) for sample in self.server.samples]})
        elif route == "media":
            self._media()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route(self.path)
        if route == "live_predict":
            self._live_predict()
            return
        if route != "predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        body = self._read_json()
        sample = self.server.by_id.get(str(body.get("id", "")))
        if sample is None:
            self._json({"error": "unknown sample id"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            prediction = _predict(self.server.args, sample, request_origin=_origin(self))
            count_score = count_similarity(prediction.get("repetition_count"), sample.repetition_count)
            prediction["reward"] = round(count_score, 6)
            prediction["count_score"] = round(count_score, 6)
            self._json(prediction)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)

    def _live_predict(self) -> None:
        try:
            body = self._read_json()
            with tempfile.TemporaryDirectory(prefix="countix-live-") as directory:
                root = Path(directory)
                audio_path = _write_data_url(body.get("audio"), root / "capture.wav", "audio/")
                audio_rms = _wav_rms(audio_path)
                streaming = bool(body.get("streaming"))
                audio_only = bool(body.get("audio_only"))
                try:
                    if audio_only:
                        prediction = _predict_audio(
                            self.server.args,
                            audio_path,
                            request_origin=_origin(self),
                            live_segment=streaming,
                        )
                    else:
                        video_path = _write_video_data_url(body.get("video"), root)
                        prediction = _predict_media(
                            self.server.args,
                            video_path,
                            audio_path,
                            request_origin=_origin(self),
                            live_segment=streaming,
                        )
                except Exception as exc:
                    if streaming:
                        prediction = {"skipped": True, "error": f"skipped failed live segment: {exc}"}
                    else:
                        raise
                prediction["audio_rms"] = round(audio_rms, 6)
                prediction["audio_active"] = audio_rms >= 0.001
            self._json(prediction)
        except (TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)

    def _media(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        sample = self.server.by_id.get((query.get("id") or [""])[0])
        kind = (query.get("kind") or [""])[0]
        if sample is None or kind not in {"video", "audio"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown media")
            return
        path = sample.video_path if kind == "video" else sample.audio_path
        if kind == "video" and self.server.args.browser_video_root:
            path = _browser_video_path(self.server.args, sample)
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Media file missing")
            return
        file_size = path.stat().st_size
        try:
            byte_range = _parse_byte_range(self.headers.get("Range"), file_size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = byte_range if byte_range is not None else (0, file_size - 1)
        content_length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range is not None else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        if byte_range is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining and (chunk := handle.read(min(1024 * 1024, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 140 * 1024 * 1024:
            raise ValueError("request body exceeds 140 MiB")
        return json.loads(self.rfile.read(length) or b"{}")

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def _send(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("countix-web: " + fmt % args + "\n")


def sample_id(sample) -> str:
    return f"{sample.youtube_id}:{sample.condition}"


def _public_sample(sample) -> dict[str, Any]:
    return {
        "id": sample_id(sample),
        "youtube_id": sample.youtube_id,
        "condition": sample.condition,
        "action_class": sample.action_class,
        "repetition_count": sample.repetition_count,
        "repetition_start_frame": sample.repetition_start_frame,
        "repetition_end_frame": sample.repetition_end_frame,
        "start_crop_frame": sample.start_crop_frame,
        "end_crop_frame": sample.end_crop_frame,
    }


def _predict(args: argparse.Namespace, sample, *, request_origin: str) -> dict[str, Any]:
    return _predict_media(args, sample.video_path, sample.audio_path, request_origin=request_origin)


def _predict_media(
    args: argparse.Namespace,
    video_path: Path,
    audio_path: Path,
    *,
    request_origin: str,
    live_segment: bool = False,
) -> dict[str, Any]:
    prompt = (
        "Analyze only this non-overlapping live segment. Identify the main repeated physical activity and count only "
        "cycles completed inside this segment; do not estimate cycles outside it. Call report_repetitions exactly once."
        if live_segment
        else prompt_text()
    )
    return _predict_content(
        args,
        [
            {"type": "video_url", "video_url": {"url": str(video_path)}},
            {"type": "audio_url", "audio_url": {"url": str(audio_path)}},
            {"type": "text", "text": prompt},
        ],
        request_origin=request_origin,
    )


def _predict_audio(
    args: argparse.Namespace,
    audio_path: Path,
    *,
    request_origin: str,
    live_segment: bool = False,
) -> dict[str, Any]:
    prompt = (
        "Listen only to this non-overlapping live audio segment. Identify the main repeated audible activity and "
        "count only cycles completed inside this segment; do not estimate cycles outside it. Call "
        "report_repetitions exactly once."
        if live_segment
        else "Identify the repeated audible activity and count its completed repetitions. Call report_repetitions once."
    )
    return _predict_content(
        args,
        [{"type": "audio_url", "audio_url": {"url": str(audio_path)}}, {"type": "text", "text": prompt}],
        request_origin=request_origin,
    )


def _predict_content(args: argparse.Namespace, content: list[dict[str, Any]], *, request_origin: str) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("WebUI inference requires the openai package included with AReno") from exc
    base_url = args.base_url
    if base_url.startswith("/"):
        base_url = request_origin.rstrip("/") + base_url
    client = OpenAI(base_url=base_url, api_key=args.api_key, max_retries=0, timeout=1800)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=[REPORT_TOOL],
                tool_choice={"type": "function", "function": {"name": "report_repetitions"}},
            )
            message = response.choices[0].message
            for call in message.tool_calls or []:
                if call.function.name == "report_repetitions":
                    parsed = json.loads(call.function.arguments)
                    return {
                        "action_class": str(parsed["action_class"]),
                        "repetition_count": int(parsed["repetition_count"]),
                        "raw": call.function.arguments,
                    }
            last_error = ValueError("model returned no report_repetitions tool call")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"inference failed after 3 attempts: {last_error}")


def _route(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    if path.endswith("/api/samples"):
        return "samples"
    if path.endswith("/api/media"):
        return "media"
    if path.endswith("/api/predict"):
        return "predict"
    if path.endswith("/api/live-predict"):
        return "live_predict"
    if "/api/" in path:
        return "missing"
    if path == "/" or not path.rsplit("/", 1)[-1].count("."):
        return "index"
    return "missing"


def _origin(handler: Handler) -> str:
    forwarded_proto = handler.headers.get("X-Forwarded-Proto", "http")
    forwarded_host = handler.headers.get("X-Forwarded-Host") or handler.headers.get("Host")
    return f"{forwarded_proto}://{forwarded_host}"


def _browser_video_path(args: argparse.Namespace, sample) -> Path:
    relative_path = sample.video_path.relative_to(Path(args.dataset_root).expanduser().resolve() / "Videos")
    return Path(args.browser_video_root).expanduser().resolve() / relative_path


def _write_data_url(value: Any, path: Path, expected_prefix: str) -> Path:
    if not isinstance(value, str) or not value.startswith("data:") or ";base64," not in value:
        raise ValueError(f"expected a base64 {expected_prefix} data URL")
    metadata, encoded = value.split(",", 1)
    media_type = metadata.removeprefix("data:").split(";", 1)[0]
    if not media_type.startswith(expected_prefix):
        raise ValueError(f"expected {expected_prefix} media, got {media_type or 'unknown'}")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("invalid base64 capture") from exc
    if not payload or len(payload) > 100 * 1024 * 1024:
        raise ValueError("capture must be between 1 byte and 100 MiB")
    path.write_bytes(payload)
    return path


def _write_video_data_url(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value.startswith("data:video/"):
        raise ValueError("expected a base64 video/ data URL")
    media_type = value.split(";", 1)[0].removeprefix("data:").lower()
    suffix = ".mp4" if media_type in {"video/mp4", "video/x-m4v"} else ".webm"
    return _write_data_url(value, root / f"capture{suffix}", "video/")


def _wav_rms(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            return 0.0
        samples = array("h", wav.readframes(wav.getnframes()))
    if not samples:
        return 0.0
    return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5 / 32768.0


def _parse_byte_range(value: str | None, file_size: int) -> tuple[int, int] | None:
    """Parse one HTTP byte range, rejecting malformed or unsatisfiable ranges."""

    if value is None:
        return None
    if file_size <= 0 or not value.startswith("bytes=") or "," in value:
        raise ValueError("unsupported byte range")
    spec = value.removeprefix("bytes=").strip()
    if "-" not in spec:
        raise ValueError("malformed byte range")
    start_text, end_text = spec.split("-", 1)
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError("invalid suffix length")
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
    except ValueError as exc:
        raise ValueError("malformed byte range") from exc
    if start < 0 or start >= file_size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, min(end, file_size - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root")
    parser.add_argument("--live-only", action="store_true", help="Show only continuous camera and microphone counting")
    parser.add_argument("--segment-seconds", type=float, default=4.0, help="Live inference segment duration")
    parser.add_argument("--base-url", default="http://127.0.0.1:8014/v1", help="Absolute or relative OpenAI API base")
    parser.add_argument(
        "--browser-video-root",
        help="Optional H.264 mirror of dataset-root/Videos used only for browser playback",
    )
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if not args.live_only and not args.dataset_root:
        parser.error("--dataset-root is required unless --live-only is set")
    if args.segment_seconds < 2:
        parser.error("--segment-seconds must be at least 2")
    server = CountixServer((args.host, args.port), Handler, args=args)
    print(f"Extreme Countix-AV WebUI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Extreme Countix-AV Workbench</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#69737d;--line:#d9dee3;--soft:#f4f6f7;--panel:#fff;--accent:#176b58;--accent2:#c35a2c;--ok:#237a4b;--bad:#a83832}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#eef1f2;font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif;letter-spacing:0}.top{height:58px;padding:0 20px;display:flex;align-items:center;justify-content:space-between;background:#fff;border-bottom:1px solid var(--line)}h1{margin:0;font-size:18px}.status{color:var(--muted);font-size:12px}.layout{display:grid;grid-template-columns:280px minmax(440px,1fr) 330px;height:calc(100vh - 58px)}aside,main{min-width:0;min-height:0}.browser{border-right:1px solid var(--line);background:#fff;display:grid;grid-template-rows:auto 1fr}.filters{padding:12px;border-bottom:1px solid var(--line);display:grid;gap:8px}.filters input,.filters select{width:100%;height:34px;padding:0 9px;border:1px solid #cbd2d8;border-radius:4px;background:#fff}.list{overflow:auto}.sample{width:100%;padding:11px 12px;text-align:left;border:0;border-bottom:1px solid #e7eaed;background:#fff;cursor:pointer}.sample:hover{background:#f7f9f9}.sample.active{background:#e9f3f0;border-left:3px solid var(--accent);padding-left:9px}.sample strong,.sample span{display:block}.sample span{margin-top:3px;color:var(--muted);font-size:12px}.stage{padding:18px;overflow:auto}.media{background:#111;aspect-ratio:16/9;display:grid;place-items:center;overflow:hidden;border-radius:6px}.media video{width:100%;height:100%;object-fit:contain}.transport{display:flex;gap:8px;align-items:center;padding:10px 0;flex-wrap:wrap}.transport button,.run{height:36px;padding:0 13px;border:1px solid #bdc6cc;border-radius:4px;background:#fff;font-weight:650;cursor:pointer}.transport button:hover{background:var(--soft)}.transport button:disabled{opacity:.5;cursor:not-allowed}.transport .recording{color:#fff;background:var(--bad);border-color:var(--bad)}audio{width:100%;height:38px}.timeline{margin-top:14px;padding:12px 0;border-top:1px solid var(--line)}.timeline h2,.inspector h2{margin:0 0 9px;font-size:13px;text-transform:uppercase;color:#4d5963}.rail{border-left:1px solid var(--line);background:#fff;overflow:auto;padding:16px}.meta{display:grid;grid-template-columns:1fr auto;gap:8px 12px;padding-bottom:16px;border-bottom:1px solid var(--line)}.meta span{color:var(--muted)}.meta strong{text-align:right}.run{width:100%;margin:16px 0;color:#fff;background:var(--accent);border-color:var(--accent)}.run:disabled{opacity:.55;cursor:wait}.result{display:grid;gap:10px}.metric{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #e8ebed}.action-list{max-height:240px;overflow-y:auto;overscroll-behavior:contain;border-top:1px solid var(--line)}.action-list .metric:last-child{border-bottom:0}.score{font:700 22px ui-monospace,monospace;color:var(--ok)}.raw{max-height:180px;overflow:auto;margin:0;padding:10px;border-radius:4px;color:#dce6e3;background:#17211f;font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.empty{color:var(--muted)}body.live-only .layout{grid-template-columns:minmax(480px,1fr) 360px;max-width:1320px;margin:auto}body.live-only .browser{display:none}body.live-only .stage{padding:24px}body.live-only .rail{border-left:1px solid var(--line)}@media(max-width:980px){.layout{grid-template-columns:220px 1fr}.rail{grid-column:1/-1;border-left:0;border-top:1px solid var(--line)}body{overflow:auto}.layout{height:auto}.browser,.stage{height:620px}body.live-only .layout{display:flex;flex-direction:column}body.live-only .stage{height:auto;order:2}body.live-only .rail{order:1;max-height:42vh;border-top:0;border-bottom:1px solid var(--line)}}@media(max-width:680px){.layout{display:block}.browser{height:360px}.stage{height:auto}.top{padding:0 12px}.status{display:none}body.live-only .stage{padding:12px}body.live-only .rail{padding:12px}.action-list{max-height:42vh}}
@media(max-width:680px){body.live-only .layout{display:grid;grid-template-columns:minmax(120px,42%) minmax(0,58%);align-items:start;min-height:calc(100vh - 58px)}body.live-only .stage{position:sticky;z-index:20;top:0;width:auto;padding:8px;background:#fff;border-right:1px solid var(--line);overflow:visible}body.live-only .stage .media{width:100%;border-radius:5px}body.live-only .stage .transport{display:grid;grid-template-columns:1fr;padding:8px 0 0;gap:6px}body.live-only .stage .transport button{width:100%;height:34px;padding:0 6px}body.live-only .stage .transport .status,body.live-only .stage audio,body.live-only .stage .timeline{display:none}body.live-only .rail{order:2;max-height:calc(100vh - 58px);min-height:calc(100vh - 58px);padding:10px;border:0;overflow:auto}body.live-only #reference-title,body.live-only .meta{display:none}body.live-only #prediction-title{margin-bottom:6px}body.live-only .result{gap:4px}body.live-only .metric{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;padding:7px 0}body.live-only .metric span{overflow-wrap:anywhere}body.live-only .score{font-size:20px}}
</style>
<style>
:root{
  --canvas:#f2f4f3;--paper:#ffffff;--paper-muted:#f8faf9;--graphite:#17201d;
  --graphite-2:#4d5954;--graphite-3:#7b8581;--rule:rgba(23,32,29,.12);
  --rule-strong:rgba(23,32,29,.2);--signal:#087f5b;--signal-dark:#066a4c;
  --signal-wash:#e8f5f0;--record:#c2410c;--record-wash:#fff1e8;
  --focus:rgba(8,127,91,.24);--shadow:0 0 0 1px rgba(23,32,29,.04),0 2px 8px rgba(23,32,29,.06);
  --radius-sm:4px;--radius-md:8px;
}
html{-webkit-font-smoothing:antialiased;background:var(--canvas)}
body{color:var(--graphite);background:var(--canvas);font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select{font:inherit}
button{transition:background-color 140ms cubic-bezier(.23,1,.32,1),border-color 140ms cubic-bezier(.23,1,.32,1),color 140ms cubic-bezier(.23,1,.32,1),transform 100ms cubic-bezier(.23,1,.32,1)}
button:active:not(:disabled){transform:scale(.97)}
button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid var(--focus);outline-offset:1px}
.top{position:relative;z-index:40;height:56px;padding:0 20px;background:rgba(255,255,255,.96);border-color:var(--rule)}
.brand{display:flex;align-items:center;gap:10px;min-width:0}
.brand-mark{position:relative;width:24px;height:24px;flex:0 0 auto;border:1px solid var(--rule-strong);border-radius:6px;background:var(--paper-muted)}
.brand-mark::before,.brand-mark::after{content:"";position:absolute;left:5px;right:5px;height:2px;border-radius:1px;background:var(--signal)}
.brand-mark::before{top:7px;box-shadow:0 4px 0 rgba(8,127,91,.58),0 8px 0 rgba(8,127,91,.28)}
.brand-mark::after{top:5px;left:11px;right:auto;width:2px;height:14px;background:var(--record)}
h1{font-size:16px;font-weight:680;letter-spacing:0;text-wrap:balance}
.system-status{display:flex;align-items:center;gap:7px;color:var(--graphite-2);font-size:12px;white-space:nowrap}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--signal);box-shadow:0 0 0 3px var(--signal-wash)}
.layout{grid-template-columns:264px minmax(480px,1fr) 344px;height:calc(100dvh - 56px)}
.browser{border-color:var(--rule);background:var(--canvas)}
.browser-heading{display:flex;align-items:center;justify-content:space-between;padding:14px 12px 8px;background:var(--paper)}
.browser-heading strong{font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.browser-heading span{color:var(--graphite-3);font-size:11px;font-variant-numeric:tabular-nums}
.filters{padding:12px;border-color:var(--rule);background:var(--paper);gap:8px}
.filters input,.filters select{height:36px;padding:0 10px;color:var(--graphite);border-color:var(--rule);border-radius:var(--radius-sm);background:#f5f7f6}
.sample{padding:12px;border-color:rgba(23,32,29,.08);background:transparent}
.sample:hover{background:rgba(255,255,255,.72)}
.sample.active{color:var(--graphite);background:var(--paper);border-left-color:var(--signal);box-shadow:inset 0 1px rgba(23,32,29,.03),inset 0 -1px rgba(23,32,29,.03)}
.sample strong{font-weight:650}.sample span{color:var(--graphite-3)}
.stage{padding:20px 24px;background:var(--canvas)}
.media{position:relative;border-radius:var(--radius-md);background:#101513;box-shadow:var(--shadow)}
.media::after{content:"";position:absolute;inset:0;pointer-events:none;border-radius:inherit;box-shadow:inset 0 0 0 1px rgba(255,255,255,.1)}
.media video{position:relative;z-index:1}
#audio-mode{position:relative;z-index:1;display:grid;place-items:center;width:100%;height:100%;color:#d9e5e0;font-size:13px;font-weight:650;letter-spacing:.04em}
#audio-mode[hidden]{display:none}
#audio-mode::before{content:"";display:block;width:52px;height:52px;margin:0 auto 12px;border:1px solid rgba(255,255,255,.24);border-radius:50%;box-shadow:inset 0 0 0 9px rgba(255,255,255,.06),0 0 0 1px rgba(0,0,0,.3)}
.transport{gap:8px;padding:12px 0}
.transport button,.run{min-height:40px;padding:0 14px;border-color:var(--rule-strong);border-radius:var(--radius-sm);color:var(--graphite);background:var(--paper);font-size:13px;font-weight:650}
.transport button:hover,.run:hover{background:var(--paper-muted);border-color:rgba(23,32,29,.3)}
.transport button:disabled,.run:disabled{color:var(--graphite-3);background:#eef1ef;opacity:1}
#live,.run{color:#fff;background:var(--signal);border-color:var(--signal)}
#live:hover,.run:hover{color:#fff;background:var(--signal-dark);border-color:var(--signal-dark)}
#stop{color:var(--record);border-color:rgba(194,65,12,.28)}
#stop.recording{color:#fff;background:var(--record);border-color:var(--record)}
#clock{margin-left:auto;padding:0 2px;font-variant-numeric:tabular-nums}
.timeline{margin-top:8px;padding-top:16px;border-color:var(--rule)}
.timeline h2,.inspector h2{margin-bottom:8px;color:var(--graphite-3);font-size:11px;font-weight:700;letter-spacing:.08em}
.rail{padding:20px;background:var(--paper);border-color:var(--rule)}
.meta{gap:8px 16px;padding-bottom:16px;border-color:var(--rule);font-size:13px}
.meta span{color:var(--graphite-3)}.meta strong{font-weight:650;font-variant-numeric:tabular-nums}
.reference-primary,.prediction-primary{grid-column:1/-1;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 12px;align-items:end;padding:2px 0 12px;margin-bottom:4px;border-bottom:1px solid var(--rule)}
.reference-primary span,.prediction-primary span{grid-column:1/-1;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.reference-primary strong,.prediction-primary strong{font-size:22px;line-height:1.15;text-align:left;text-wrap:balance}
.reference-primary em,.prediction-primary em{color:var(--signal);font-size:28px;line-height:1;font-style:normal;font-weight:720;font-variant-numeric:tabular-nums}
.run{margin:16px 0}
.result{gap:0;min-height:0}
.metric{min-height:40px;padding:9px 0;border-color:rgba(23,32,29,.08);align-items:center}
.metric span{color:var(--graphite-2)}.metric strong{font-weight:680;font-variant-numeric:tabular-nums}
.session-strip{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px 16px;align-items:end;padding:14px 0 16px;border-bottom:1px solid var(--rule)}
.session-strip span{color:var(--graphite-3);font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.session-strip strong{grid-column:1;font-size:40px;line-height:1;font-weight:720;font-variant-numeric:tabular-nums}
.session-state{grid-column:2;grid-row:1/3;align-self:center;padding:5px 8px;border:1px solid var(--rule);border-radius:var(--radius-sm);color:var(--graphite-2);background:var(--paper-muted);font-size:11px;font-style:normal;font-weight:700}
.session-state.live{color:var(--record);border-color:rgba(194,65,12,.2);background:var(--record-wash)}
.action-list{max-height:320px;border-color:var(--rule);scrollbar-width:thin;scrollbar-color:rgba(23,32,29,.24) transparent}
.action-list .metric strong{min-width:30px;text-align:right;font-size:18px}
.score{color:var(--signal);font-family:ui-monospace,"SFMono-Regular",Consolas,monospace;font-variant-numeric:tabular-nums}
.raw{border-radius:var(--radius-sm);background:#17201d}
.empty{color:var(--graphite-3);text-wrap:pretty}
body:not(.live-only) .browser{grid-template-rows:auto auto 1fr}
body:not(.live-only) .stage{display:flex;min-height:0;flex-direction:column}
body:not(.live-only) .media{flex:0 1 auto;max-height:calc(100dvh - 250px)}
body:not(.live-only) .rail{display:flex;min-height:0;flex-direction:column;overflow:auto}
body:not(.live-only) #result>.empty{padding:16px 0;border-top:1px solid var(--rule)}
body:not(.live-only) .prediction-primary{margin-bottom:8px;padding-top:4px}
body:not(.live-only) .raw{margin-top:12px}
body.live-only .layout{grid-template-columns:minmax(520px,1fr) 360px;max-width:1280px}
body.live-only .stage{padding:24px 28px}
body.live-only .rail{display:flex;min-height:0;flex-direction:column;padding:24px}
body.live-only .result{display:flex;min-height:0;flex:1;flex-direction:column}
body.live-only .action-list{min-height:0;max-height:none;flex:1}
@media(max-width:980px) and (min-width:681px){
  body.live-only .layout{display:grid;grid-template-columns:minmax(440px,1fr) 320px;height:calc(100dvh - 56px)}
  body.live-only .stage{height:auto;order:initial;padding:20px}
  body.live-only .rail{order:initial;max-height:none;border-top:0;border-bottom:0;border-left:1px solid var(--rule)}
}
@media(max-width:680px){
  .top{position:sticky;top:0;height:48px;padding:0 12px}
  h1{font-size:14px}.brand{gap:8px}.brand-mark{width:22px;height:22px}.system-status{display:none}
  body.live-only{height:100dvh;overflow:hidden}
  body.live-only .layout{display:block;height:calc(100dvh - 48px);min-height:0}
  body.live-only .stage{position:static;width:0;height:0;padding:0;border:0;overflow:visible}
  body.live-only .stage .media{position:fixed;z-index:30;top:60px;right:12px;width:132px;aspect-ratio:3/4;border-radius:8px;box-shadow:0 0 0 1px rgba(255,255,255,.16),0 8px 24px rgba(23,32,29,.24)}
  body.live-only .stage .media video{object-fit:cover}
  body.live-only .stage .transport{position:fixed;z-index:40;left:8px;right:8px;bottom:max(8px,env(safe-area-inset-bottom));display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;padding:8px;border:1px solid var(--rule);border-radius:8px;background:rgba(255,255,255,.96);box-shadow:0 8px 28px rgba(23,32,29,.18);backdrop-filter:blur(14px)}
  body.live-only .stage .transport button{width:100%;min-height:38px;height:38px;padding:0 6px;font-size:11px}
  body.live-only .stage .transport #live,body.live-only .stage .transport #stop{grid-column:span 1}
  body.live-only .stage .transport #stop:last-of-type{grid-column:span 2}
  body.live-only .stage .transport .status,body.live-only .stage audio,body.live-only .stage .timeline{display:none}
  body.live-only .rail{height:100%;min-height:0;max-height:none;padding:12px 12px calc(132px + env(safe-area-inset-bottom));border:0;overflow:hidden;background:var(--paper)}
  body.live-only #reference-title,body.live-only .meta{display:none}
  body.live-only #prediction-title{margin-bottom:4px}
  body.live-only .session-strip{min-height:180px;margin-right:148px;padding:10px 0 12px;gap:6px 8px;align-content:start}
  body.live-only .session-strip strong{font-size:44px}
  body.live-only .session-state{padding:4px 6px}
  body.live-only .metric{min-height:36px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;padding:7px 0}
  body.live-only .metric span{overflow-wrap:anywhere}
  body.live-only .action-list{min-height:80px;max-height:none;flex:1;padding-bottom:12px;scroll-padding-bottom:12px}
  body.live-only .score{font-size:18px}
  body:not(.live-only){overflow:auto}
  body:not(.live-only) .layout{display:flex;height:auto;min-height:calc(100dvh - 48px);flex-direction:column}
  body:not(.live-only) .browser{height:auto;max-height:300px;border-right:0;border-bottom:1px solid var(--rule)}
  body:not(.live-only) .browser-heading{padding:12px 12px 6px}
  body:not(.live-only) .filters{grid-template-columns:minmax(0,1fr) minmax(112px,.55fr);padding:8px 12px}
  body:not(.live-only) .list{display:flex;gap:8px;padding:8px 12px 12px;overflow-x:auto;scroll-snap-type:x proximity}
  body:not(.live-only) .sample{min-width:184px;max-width:220px;padding:10px;border:1px solid var(--rule);border-radius:var(--radius-sm);scroll-snap-align:start}
  body:not(.live-only) .sample.active{padding-left:10px;border-color:rgba(8,127,91,.35);box-shadow:inset 3px 0 var(--signal)}
  body:not(.live-only) .stage{height:auto;padding:12px}
  body:not(.live-only) .media{max-height:none}
  body:not(.live-only) .transport{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
  body:not(.live-only) .transport button{min-width:0;padding:0 8px}
  body:not(.live-only) #clock{grid-column:1/-1;margin:0;text-align:right}
  body:not(.live-only) .timeline{margin-top:4px;padding-top:12px}
  body:not(.live-only) .rail{padding:16px 12px;border-top:1px solid var(--rule);border-left:0;overflow:visible}
  body:not(.live-only) .run{position:sticky;bottom:8px;z-index:5;box-shadow:0 2px 8px rgba(23,32,29,.14)}
}
@media(max-width:390px){
  body.live-only .stage .media{top:56px;right:8px;width:116px}
  body.live-only .rail{padding:10px 10px calc(132px + env(safe-area-inset-bottom))}
  body.live-only .session-strip{min-height:158px;margin-right:126px}
  body.live-only .session-strip strong{font-size:38px}
  body.live-only .stage .transport button{font-size:11px}
}
@media(prefers-reduced-motion:reduce){button{transition:none}}
</style></head><body>
<header class="top"><div class="brand"><span class="brand-mark" aria-hidden="true"></span><h1 id="brand-title">Countix Evaluator</h1></div><div class="system-status"><span class="status-dot" aria-hidden="true"></span><span id="status">Loading</span></div></header>
<div class="layout"><aside class="browser"><div class="browser-heading"><strong>Samples</strong><span id="sample-count">0</span></div><div class="filters"><input id="search" placeholder="Filter action or video ID"><select id="condition"><option value="">All conditions</option></select></div><div class="list" id="list"></div></aside>
<main class="stage"><div class="media"><video id="video" controls preload="metadata"></video><div id="audio-mode" class="empty" hidden>Audio only</div></div><div class="transport"><button id="play">Play synced</button><button id="pause">Pause</button><button id="mode">Audio only</button><button id="camera">Use camera</button><button id="switch-camera" disabled>Use back camera</button><button id="microphone" disabled>Enable microphone</button><button id="record" disabled>Record</button><button id="live">Start live</button><button id="stop" disabled>Stop</button><span class="status" id="clock">0.00 s</span></div><audio id="audio" preload="metadata" aria-hidden="true"></audio><section class="timeline"><h2 id="interval-title">Activity interval</h2><div id="interval" class="empty">Select a sample</div></section></main>
<aside class="rail inspector"><h2 id="reference-title">Reference</h2><div class="meta" id="meta"></div><button class="run" id="run">Run Gemma4</button><h2 id="prediction-title">Prediction</h2><div class="result" id="result" aria-live="polite"><p class="empty">No prediction yet.</p></div></aside></div>
<script>
const LIVE_ONLY=__LIVE_ONLY__,SEGMENT_SECONDS=__SEGMENT_SECONDS__;
const state={samples:[],filtered:[],current:null};
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[<>"'&]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const video=$('video'),audio=$('audio');
let syncFrame=0,liveStream=null,recorder=null,videoChunks=[],audioChunks=[];
let audioContext=null,audioSource=null,audioNode=null,audioSink=null,liveVideo=null,liveAudio=null;
let liveRunning=false,liveTimer=0,liveSegment=null,livePending=0,liveIndex=0,liveSession=0;
let cameraFacing='user',microphoneEnabled=true,audioOnly=false;
const liveCounts=new Map();

function api(path){return new URL(path,location.href).toString()}
function apply(){
  const q=$('search').value.toLowerCase(),c=$('condition').value;
  state.filtered=state.samples.filter(x=>(!c||x.condition===c)&&(!q||`${x.action_class} ${x.youtube_id}`.toLowerCase().includes(q)));
  $('sample-count').textContent=`${state.filtered.length} / ${state.samples.length}`;
  $('list').innerHTML=state.filtered.map(x=>`<button class="sample ${state.current&&x.id===state.current.id?'active':''}" data-id="${esc(x.id)}"><strong>${esc(x.action_class)}</strong><span>${esc(x.condition)} · ${esc(x.youtube_id)}</span></button>`).join('');
  document.querySelectorAll('.sample').forEach(b=>b.onclick=()=>select(state.samples.find(x=>x.id===b.dataset.id)));
}
async function load(){
  if(LIVE_ONLY){
    document.body.classList.add('live-only');
    $('brand-title').textContent='Countix Monitor';
    $('interval-title').textContent='Live activity';
    $('interval').textContent=`Press Start live to analyze consecutive ${SEGMENT_SECONDS}-second camera and microphone segments.`;
    $('status').textContent='Camera + microphone ready to start';
    $('reference-title').textContent='Session';
    $('prediction-title').textContent='Live count';
    $('meta').innerHTML=`<span>Segment length</span><strong>${SEGMENT_SECONDS} s</strong><span>Inference</span><strong>Parallel</strong>`;
    $('play').hidden=true;$('pause').hidden=true;$('camera').hidden=true;$('record').hidden=true;$('run').hidden=true;
    $('microphone').disabled=false;updateMicrophoneButton();updateModeButton();
    return;
  }
  $('live').hidden=true;
  const data=await fetch(api('api/samples')).then(r=>r.json());
  state.samples=data.samples;
  const conditions=[...new Set(data.samples.map(x=>x.condition))].sort();
  $('condition').innerHTML+=conditions.map(x=>`<option>${esc(x)}</option>`).join('');
  apply();select(state.filtered[0]);$('status').textContent=`${data.samples.length} labelled audiovisual samples`;
}
function select(x){
  if(!x)return;
  video.pause();audio.pause();closeCapture();liveVideo=null;liveAudio=null;video.srcObject=null;video.controls=true;video.muted=false;
  $('record').disabled=true;$('stop').disabled=true;$('camera').disabled=false;$('switch-camera').disabled=true;$('microphone').disabled=true;$('run').textContent='Run Gemma4';state.current=x;apply();
  const id=encodeURIComponent(x.id);video.src=api(`api/media?kind=video&id=${id}`);audio.src=api(`api/media?kind=audio&id=${id}`);video.load();audio.load();
  $('meta').innerHTML=`<div class="reference-primary"><span>Ground truth</span><strong>${esc(x.action_class)}</strong><em>${x.repetition_count}</em></div><span>Condition</span><strong>${esc(x.condition)}</strong><span>Video ID</span><strong>${esc(x.youtube_id)}</strong>`;
  $('interval').textContent=`Repetition frames ${x.repetition_start_frame}-${x.repetition_end_frame}; source crop ${x.start_crop_frame}-${x.end_crop_frame}.`;
  $('result').innerHTML='<p class="empty">No prediction yet.</p>';
}
function alignAudio(force=false){if(Number.isFinite(video.currentTime)&&(force||Math.abs(audio.currentTime-video.currentTime)>.08))audio.currentTime=video.currentTime;audio.playbackRate=video.playbackRate;$('clock').textContent=`${video.currentTime.toFixed(2)} s`}
function stopSync(){cancelAnimationFrame(syncFrame);syncFrame=0}
function syncLoop(){alignAudio();if(!video.paused&&!video.ended)syncFrame=requestAnimationFrame(syncLoop)}
async function startAudio(){if(liveStream)return;alignAudio(true);try{await audio.play()}catch(e){console.warn('Synchronized audio could not start',e)}stopSync();syncLoop()}
async function syncedPlay(){alignAudio(true);await video.play()}
$('play').onclick=syncedPlay;$('pause').onclick=()=>video.pause();
video.addEventListener('play',startAudio);video.addEventListener('playing',startAudio);video.addEventListener('seeking',()=>alignAudio(true));video.addEventListener('seeked',()=>alignAudio(true));video.addEventListener('ratechange',()=>alignAudio(true));video.addEventListener('timeupdate',()=>{if(!liveStream)alignAudio()});
video.addEventListener('pause',()=>{if(liveStream)return;audio.pause();stopSync();alignAudio(true)});video.addEventListener('ended',()=>{audio.pause();stopSync();alignAudio(true)});

function blobUrl(blob){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(blob)})}
function encodeWav(chunks,rate){
  const n=chunks.reduce((s,x)=>s+x.length,0),b=new ArrayBuffer(44+n*2),v=new DataView(b);
  function textAt(o,s){for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))}
  textAt(0,'RIFF');v.setUint32(4,36+n*2,true);textAt(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);textAt(36,'data');v.setUint32(40,n*2,true);
  let o=44;for(const x of chunks)for(const f of x){v.setInt16(o,Math.max(-1,Math.min(1,f))*(f<0?32768:32767),true);o+=2}
  return new Blob([b],{type:'audio/wav'});
}
function encodeSegmentAudio(chunks,rate,seconds){const target=Math.max(1,Math.round(rate*seconds)),actual=chunks.reduce((sum,chunk)=>sum+chunk.length,0),result=[...chunks];if(actual<target)result.push(new Float32Array(target-actual));return encodeWav(result,rate)}
async function useCamera(){
  if(!navigator.mediaDevices?.getUserMedia)throw Error('Camera and microphone require HTTPS or localhost');
  if(!liveStream)liveStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:cameraFacing}},audio:microphoneEnabled});
  video.pause();audio.pause();video.removeAttribute('src');video.srcObject=liveStream;video.muted=true;video.controls=false;await video.play();
  if(microphoneEnabled)await setupAudioCapture();
  $('record').disabled=false;$('camera').disabled=true;$('switch-camera').disabled=false;$('microphone').disabled=false;$('interval').textContent=`Camera ready; microphone ${microphoneEnabled?'enabled':'disabled'}.`;
  if(!LIVE_ONLY)$('result').innerHTML='<p class="empty">Record an activity, then run Gemma4.</p>';
}
function updateCameraButton(){$('switch-camera').textContent=cameraFacing==='user'?'Use back camera':'Use front camera'}
function updateMicrophoneButton(){$('microphone').textContent=microphoneEnabled?'Disable microphone':'Enable microphone'}
function updateModeButton(){$('mode').textContent=audioOnly?'Camera + audio':'Audio only';$('audio-mode').hidden=!audioOnly;video.hidden=audioOnly;$('switch-camera').hidden=audioOnly;$('microphone').hidden=audioOnly;document.body.classList.toggle('audio-only',audioOnly)}
async function toggleMode(){
  if(liveRunning)return;
  if(liveStream)await closeCapture();audioOnly=!audioOnly;microphoneEnabled=audioOnly||microphoneEnabled;updateModeButton();updateMicrophoneButton();$('status').textContent=audioOnly?'Audio-only mode':'Camera mode';
}
async function switchCamera(){
  if(!liveStream)return;
  const wasRunning=liveRunning,nextFacing=cameraFacing==='user'?'environment':'user';
  $('switch-camera').disabled=true;
  if(wasRunning){clearTimeout(liveTimer);liveTimer=0;liveRunning=false;if(liveSegment)await rotateLiveSegment()}
  try{
    const replacement=await navigator.mediaDevices.getUserMedia({video:{facingMode:{exact:nextFacing}},audio:false});
    const oldVideo=liveStream.getVideoTracks(),audioTracks=liveStream.getAudioTracks();
    liveStream=new MediaStream([...replacement.getVideoTracks(),...audioTracks]);video.srcObject=liveStream;await video.play();
    oldVideo.forEach(track=>track.stop());cameraFacing=nextFacing;updateCameraButton();
    $('status').textContent=cameraFacing==='environment'?'Back camera active':'Front camera active';
  }finally{
    if(wasRunning){liveRunning=true;beginLiveSegment();liveResult()}
    $('switch-camera').disabled=!liveStream;
  }
}
async function setupAudioCapture(){
  if(audioContext||!liveStream.getAudioTracks().length)return;
  audioContext=new AudioContext();await audioContext.resume();
  audioSource=audioContext.createMediaStreamSource(new MediaStream(liveStream.getAudioTracks()));
  audioNode=audioContext.createScriptProcessor(4096,1,1);
  audioNode.onaudioprocess=e=>audioChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  audioSink=audioContext.createGain();audioSink.gain.value=0;audioSource.connect(audioNode);audioNode.connect(audioSink);audioSink.connect(audioContext.destination);
}
async function teardownAudioCapture(){
  if(audioSource)audioSource.disconnect();if(audioNode)audioNode.disconnect();if(audioSink)audioSink.disconnect();
  if(audioContext){await audioContext.close();audioContext=null}audioSource=null;audioNode=null;audioSink=null;
}
async function toggleMicrophone(){
  $('microphone').disabled=true;
  try{
    if(!liveStream){microphoneEnabled=!microphoneEnabled;updateMicrophoneButton();$('status').textContent=`Microphone will start ${microphoneEnabled?'enabled':'disabled'}`;return}
    if(microphoneEnabled){
      await teardownAudioCapture();liveStream.getAudioTracks().forEach(track=>{liveStream.removeTrack(track);track.stop()});microphoneEnabled=false;
    }else{
      const stream=await navigator.mediaDevices.getUserMedia({video:false,audio:true});stream.getAudioTracks().forEach(track=>liveStream.addTrack(track));microphoneEnabled=true;await setupAudioCapture();
    }
    updateMicrophoneButton();$('status').textContent=`Microphone ${microphoneEnabled?'enabled':'disabled'}`;
  }finally{$('microphone').disabled=false}
}
function makeRecorder(){
  const mime=['video/mp4;codecs=avc1.42E01E','video/mp4','video/webm;codecs=vp8','video/webm'].find(x=>MediaRecorder.isTypeSupported(x))||'';
  const chunks=[],item=new MediaRecorder(new MediaStream(liveStream.getVideoTracks()),mime?{mimeType:mime}:undefined);
  item.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};item.start(250);
  return {recorder:item,chunks,mime:item.mimeType||mime||'video/webm',started:performance.now()};
}
function finishRecorder(item){return new Promise((resolve,reject)=>{item.recorder.onerror=e=>reject(e.error||Error('Recording failed'));item.recorder.onstop=()=>resolve(new Blob(item.chunks,{type:item.mime}));item.recorder.stop()})}
async function closeCapture(){
  clearTimeout(liveTimer);liveTimer=0;
  await teardownAudioCapture();
  if(liveStream)liveStream.getTracks().forEach(t=>t.stop());liveStream=null;
  $('switch-camera').disabled=true;$('microphone').disabled=!LIVE_ONLY;
}
async function startRecording(){
  await setupAudioCapture();videoChunks=[];audioChunks=[];const item=makeRecorder();recorder=item;
  $('record').disabled=true;$('stop').disabled=false;$('stop').classList.add('recording');$('interval').textContent='Recording camera and microphone...';
}
async function stopRecording(){
  liveVideo=await finishRecorder(recorder);liveAudio=encodeSegmentAudio(audioChunks,audioContext?.sampleRate||16000,(performance.now()-recorder.started)/1000);await closeCapture();
  video.srcObject=null;video.src=URL.createObjectURL(liveVideo);audio.src=URL.createObjectURL(liveAudio);video.controls=true;video.muted=false;video.load();audio.load();
  $('stop').disabled=true;$('stop').classList.remove('recording');$('camera').disabled=false;$('run').textContent='Analyze capture';$('interval').textContent='Capture ready; video progress controls the preview.';
}
function liveResult(latest=null,error=''){
  const sessionTotal=[...liveCounts.values()].reduce((sum,x)=>sum+x.count,0);
  const rows=[...liveCounts.values()].reverse().map(x=>`<div class="metric"><span>${esc(x.label)}</span><strong>${x.count}</strong></div>`).join('');
  const stateLabel=liveRunning?'Capturing':livePending?'Finishing':'Stopped';
  let top=`<div class="session-strip"><span>Session total</span><strong>${sessionTotal}</strong><em class="session-state ${liveRunning?'live':''}">${stateLabel}</em></div><div class="metric"><span>Completed segments</span><strong>${liveIndex}</strong></div><div class="metric"><span>In flight</span><strong>${livePending}</strong></div>`;
  if(error)top+=`<p class="empty">${esc(error)}</p>`;
  $('result').innerHTML=top+(rows?`<h2>By action</h2><div class="action-list" id="action-list">${rows}</div>`:'<p class="empty">Waiting for the first completed segment.</p>');
  const list=$('action-list');if(list)list.scrollTop=0;
}
async function analyzeLiveSegment(videoBlob,audioBlob,index,segmentAudioOnly){
  const started=performance.now();
  const payload={audio:await blobUrl(audioBlob),audio_only:segmentAudioOnly,streaming:true};if(videoBlob)payload.video=await blobUrl(videoBlob);
  const response=await fetch(api('api/live-predict'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw Error(data.error||'Inference failed');
  if(data.skipped)return {skipped:true,error:data.error||'Undecodable video segment',segment:index,latency_seconds:(performance.now()-started)/1000};
  data.action_class=String(data.action_class||'unknown').trim()||'unknown';data.repetition_count=Math.max(0,Math.round(Number(data.repetition_count)||0));data.latency_seconds=(performance.now()-started)/1000;data.segment=index;return data;
}
function enqueueLive(videoBlob,audioBlob,index,segmentAudioOnly){
  const session=liveSession;livePending++;liveResult();
  analyzeLiveSegment(videoBlob,audioBlob,index,segmentAudioOnly).then(data=>{
    if(session!==liveSession)return;
    livePending--;if(data.skipped){$('interval').textContent=`Segment ${data.segment} skipped; capture continues.`;liveResult(null,data.error);return}const key=data.action_class.toLowerCase(),entry=liveCounts.get(key)||{label:data.action_class,count:0};entry.count+=data.repetition_count;liveCounts.delete(key);liveCounts.set(key,entry);$('interval').textContent=`Segment ${data.segment}: ${data.action_class}, +${data.repetition_count}; inference ${data.latency_seconds.toFixed(1)} s.`;liveResult(data);
  }).catch(e=>{if(session!==liveSession)return;livePending--;liveResult(null,e.message)});
}
function beginLiveSegment(){audioChunks=[];liveSegment=audioOnly?{started:performance.now()}:makeRecorder();liveTimer=setTimeout(rotateLiveSegment,SEGMENT_SECONDS*1000)}
async function rotateLiveSegment(){
  if(!liveSegment)return;
  const old=liveSegment,oldAudio=audioChunks,sampleRate=audioContext?.sampleRate||16000,index=++liveIndex,segmentAudioOnly=audioOnly;
  liveSegment=null;if(liveRunning)beginLiveSegment();
  try{const videoBlob=segmentAudioOnly?null:await finishRecorder(old),audioBlob=encodeSegmentAudio(oldAudio,sampleRate,(performance.now()-old.started)/1000);enqueueLive(videoBlob,audioBlob,index,segmentAudioOnly)}catch(e){liveResult(null,e.message)}
}
async function startLive(){
  if(audioOnly){liveStream=await navigator.mediaDevices.getUserMedia({video:false,audio:true});microphoneEnabled=true;await setupAudioCapture()}else await useCamera();liveSession++;liveCounts.clear();liveIndex=0;livePending=0;liveRunning=true;
  $('live').disabled=true;$('mode').disabled=true;$('stop').disabled=false;$('stop').classList.add('recording');$('status').textContent=audioOnly?'Audio counting':'Live counting';beginLiveSegment();liveResult();
}
async function stopLive(){
  liveRunning=false;clearTimeout(liveTimer);liveTimer=0;$('stop').disabled=true;$('stop').classList.remove('recording');
  if(liveSegment){const elapsed=performance.now()-liveSegment.started;if(elapsed>=500)await rotateLiveSegment();else{if(liveSegment.recorder)liveSegment.recorder.stop();liveSegment=null}}
  await closeCapture();video.srcObject=null;$('live').disabled=false;$('mode').disabled=false;$('status').textContent='Live session stopped';liveResult();
}
$('mode').onclick=()=>toggleMode().catch(e=>liveResult(null,e.message));
$('camera').onclick=()=>useCamera().catch(e=>$('result').innerHTML=`<p class="empty">${esc(e.message)}</p>`);
$('switch-camera').onclick=()=>switchCamera().catch(e=>{liveRunning=Boolean(liveStream&&liveSegment);liveResult(null,e.message)});
$('microphone').onclick=()=>toggleMicrophone().catch(e=>liveResult(null,e.message));
$('record').onclick=()=>startRecording().catch(e=>$('result').innerHTML=`<p class="empty">${esc(e.message)}</p>`);
$('live').onclick=()=>startLive().catch(e=>{liveRunning=false;liveResult(null,e.message)});
$('stop').onclick=()=>{const action=LIVE_ONLY?stopLive():stopRecording();action.catch(e=>$('result').innerHTML=`<p class="empty">${esc(e.message)}</p>`)};
$('run').onclick=async()=>{
  if(!state.current&&!liveVideo)return;const b=$('run');b.disabled=true;b.textContent='Running';$('result').innerHTML='<p class="empty">Processing video and audio...</p>';
  try{let endpoint='api/predict',payload={id:state.current.id};if(liveVideo){endpoint='api/live-predict';payload={video:await blobUrl(liveVideo),audio:await blobUrl(liveAudio)}}const r=await fetch(api(endpoint),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.error||'Inference failed');let details=`<div class="prediction-primary"><span>Model prediction</span><strong>${esc(d.action_class)}</strong><em>${d.repetition_count}</em></div>`;if(d.reward!==undefined)details+=`<div class="metric"><span>Count reward</span><strong class="score">${d.reward.toFixed(3)}</strong></div>`;$('result').innerHTML=details+`<pre class="raw">${esc(d.raw)}</pre>`}catch(e){$('result').innerHTML=`<p class="empty">${esc(e.message)}</p>`}finally{b.disabled=false;b.textContent=liveVideo?'Analyze capture':'Run Gemma4'}
};
$('search').oninput=apply;$('condition').onchange=apply;load().catch(e=>$('status').textContent=e.message);
</script></body></html>"""


if __name__ == "__main__":
    main()
