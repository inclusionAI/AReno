"""One-shot inference requests to an existing deployment; no prompts or keys are stored."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from areno.dashboard.flow.llm import NoRedirect
from areno.dashboard.flow.workflows import bounded


def test_deployment(job, body):
    if job["kind"] != "deployment" or job["status"] != "ready" or not job.get("endpoint"):
        raise ValueError("Wait for the deployment to be ready before testing inference")
    base = job["endpoint"].rstrip("/")
    url = urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("The deployment endpoint is invalid")
    key = body.get("api_key", "")
    if not isinstance(key, str) or not key or len(key) > 4096 or "\n" in key or "\r" in key:
        raise ValueError("Enter the endpoint API key")
    prompt, system = body.get("prompt", ""), body.get("system", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
        raise ValueError("Enter a prompt of 1–32000 characters")
    if not isinstance(system, str) or len(system) > 16000:
        raise ValueError("System prompt must be at most 16000 characters")
    messages = ([{"role": "system", "content": system}] if system.strip() else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = dict(
        messages=messages,
        max_tokens=bounded(body.get("max_tokens", 256), "Maximum output tokens", 1, 32768, True),
        temperature=bounded(body.get("temperature", 0.7), "Temperature", 0, 2),
        stream=False,
    )
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    started = time.monotonic()
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=120) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Inference response is too large")
        result = json.loads(raw)
        if not isinstance(result, dict) or not result.get("choices"):
            raise ValueError("The endpoint returned no completion")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError("Endpoint authentication failed; check the endpoint API key") from None
        raise ValueError(f"Inference request failed (HTTP {exc.code}); check deployment logs") from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("Inference request failed or timed out; check deployment status") from None
    except json.JSONDecodeError:
        raise ValueError("The endpoint returned an invalid JSON response") from None
    return {"response": result, "elapsed_seconds": round(time.monotonic() - started, 3)}
