"""Local staging for user datasets and task hooks uploaded to Modal on launch."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
DATA_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow"}
MEDIA_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
}
ALLOWED_SUFFIXES = DATA_SUFFIXES | MEDIA_SUFFIXES | {".py"}


def save_upload(directory: Path, name: str, encoded: str) -> dict:
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Supported uploads: datasets, images, audio, video and Python task hooks")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        raise ValueError("Upload is not valid base64") from None
    if not 0 < len(content) <= MAX_UPLOAD_BYTES:
        raise ValueError(
            "Upload a non-empty file up to 16 MiB; use a dataset repository or Modal Volume for larger data"
        )
    identifier = hashlib.sha256(content).hexdigest() + suffix
    uploads = directory / "uploads"
    uploads.mkdir(mode=0o700, exist_ok=True)
    file = uploads / identifier
    file.write_bytes(content)
    file.chmod(0o600)
    return {"name": Path(name).name, "path": "/artifacts/uploads/" + identifier, "bytes": len(content)}


def referenced_uploads(manifest: dict, directory: Path) -> list[tuple[Path, str]]:
    found = set()

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.startswith("/artifacts/datasets/"):
            key = value.removeprefix("/artifacts/datasets/")
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError("Invalid cached dataset reference")
            folder = directory / "dataset_cache" / key
            marker = folder.with_suffix(".json")
            if not marker.is_file():
                raise ValueError("Dataset cache is missing; download it again")
            names = json.loads(marker.read_text())["files"]
            for name in names:
                local = folder / name
                if local.parent != folder or not local.is_file():
                    raise ValueError("Dataset cache is incomplete; download it again")
                found.add((local, f"/datasets/{key}/{name}"))
        elif isinstance(value, str) and value.startswith("/artifacts/uploads/"):
            # Loader function syntax may include a :function suffix.
            filename = value.removeprefix("/artifacts/uploads/").split(":")[0]
            if Path(filename).name != filename:
                raise ValueError("Invalid uploaded asset reference")
            local = directory / "uploads" / filename
            if not local.is_file():
                raise ValueError("An uploaded file is missing locally; upload it again before launching")
            found.add((local, "/uploads/" + filename))

    visit(manifest)
    return sorted(found)
