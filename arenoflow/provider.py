"""Modal adapter; no credentials are written to disk or embedded in manifests."""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def registry_headers(repo):
    query = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{repo}:pull"})
    with urllib.request.urlopen(f"https://ghcr.io/token?{query}", timeout=20) as response:
        token = json.load(response)["token"]
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json",
    }


def published_tag(repo, headers):
    """Prefer the highest stable semantic version, with latest as an alias fallback."""
    tags = []
    base = f"https://ghcr.io/v2/{repo}/tags/list"
    url = base + "?n=100"
    while url:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            tags.extend(json.load(response).get("tags") or [])
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = urllib.parse.urljoin(url, match.group(1)) if match else None
        if url and urllib.parse.urlsplit(url)._replace(query="").geturl() != base:
            raise ValueError("Registry returned an unexpected pagination URL")
    releases = [tag for tag in tags if re.fullmatch(r"v?\d+\.\d+\.\d+", tag)]
    if releases:
        return max(releases, key=lambda t: tuple(int(part) for part in t.lstrip("v").split(".")))
    if "latest" in tags:
        return "latest"
    raise ValueError("No stable release or latest tag is published; specify a tag manually")


def latest_image(reference):
    match = re.fullmatch(r"ghcr\.io/([a-z0-9_./-]+)(?::[\w.-]+)?", reference)
    if not match:
        raise ValueError("Automatic image discovery requires a public GHCR repository")
    repo = match.group(1)
    tag = published_tag(repo, registry_headers(repo))
    return {"reference": f"ghcr.io/{repo}:{tag}", "tag": tag, "checked_at": time.time()}


def resolve_image(reference: str) -> str:
    """Resolve the current GHCR tag on each launch, avoiding Modal tag cache staleness."""
    match = re.fullmatch(r"ghcr\.io/([a-z0-9_./-]+)(?::([\w.-]+)|@(sha256:[0-9a-f]{64}))", reference)
    if not match:
        raise ValueError("Use a public ghcr.io image with an explicit tag or sha256 digest")
    repo, tag, digest = match.groups()
    if digest:
        return reference
    headers = registry_headers(repo)

    def manifest_digest(selected):
        request = urllib.request.Request(f"https://ghcr.io/v2/{repo}/manifests/{selected}", headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.headers.get("Docker-Content-Digest", "")

    try:
        digest = manifest_digest(tag)
    except urllib.error.HTTPError as exc:
        if tag != "latest" or exc.code != 404:
            raise
        selected = published_tag(repo, headers)
        digest = manifest_digest(selected)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError("Registry did not return an image digest")
    return f"ghcr.io/{repo}@{digest}"


class ModalProvider:
    def __init__(self, token_id, token_secret):
        import modal

        self.modal = modal
        self.client = modal.Client.from_credentials(token_id, token_secret)

    def check(self):
        self.client.hello()

    def upload(self, files):
        volume = self.modal.Volume.from_name("arenoflow-artifacts", create_if_missing=True, client=self.client)
        with volume.batch_upload(force=True) as batch:
            for local, remote in files:
                batch.put_file(local, remote)

    def start(self, manifest, resources, endpoint_key=None, on_phase=None):
        progress = on_phase or (lambda phase: None)
        modal = self.modal
        progress("preparing_resources")
        app = modal.App.lookup("arenoflow", create_if_missing=True, client=self.client)
        volume = modal.Volume.from_name("arenoflow-artifacts", create_if_missing=True, client=self.client)
        image = modal.Image.from_registry(manifest["image"]).add_local_file(
            Path(__file__).with_name("remote.py"), "/opt/arenoflow/remote.py"
        )
        progress("building_image")
        image.build(app)
        if manifest["kind"] == "image_build":
            return None
        progress("starting_sandbox")
        secrets = []
        if endpoint_key:
            secrets.append(modal.Secret.from_dict({"ARENOFLOW_ENDPOINT_KEY": endpoint_key}))
        encoded = base64.urlsafe_b64encode(json.dumps(manifest).encode()).decode()
        return modal.Sandbox.create(
            "python",
            "-u",
            "/opt/arenoflow/remote.py",
            encoded,
            app=app,
            image=image,
            gpu=f"{resources['gpu']}:{resources['count']}" if resources.get("gpu") else None,
            cpu=resources["cpu"],
            memory=resources["memory_gib"] * 1024,
            timeout=resources["timeout_seconds"],
            volumes={"/artifacts": volume},
            secrets=secrets,
            client=self.client,
            workdir="/workspace/areno",
            env={
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": "/artifacts/cache/hf",
                "HF_HUB_CACHE": "/artifacts/cache/hf/hub",
                "HF_DATASETS_CACHE": "/artifacts/cache/hf/datasets",
            },
            encrypted_ports=[8080] if manifest["kind"] == "deployment" else [],
        )

    def attach(self, sandbox_id):
        return self.modal.Sandbox.from_id(sandbox_id, client=self.client)
