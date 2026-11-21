"""Integrity-checked SFT snapshot publication and input fingerprints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from areno.api.multimodal import record_has_image


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("SFT training state requires JSON-serializable text dataset rows") from exc


def fingerprint_inputs(config: Any, dataset: Any, validation_dataset: Any, tokenizer: Any, processor: Any) -> str:
    """Fingerprint materialized text rows and immutable local model/tokenizer inputs."""

    if processor is not None and type(processor) is not type(tokenizer):
        raise ValueError("SFT training state currently supports text-only models without a processor")
    model_path = Path(config.ckpt)
    if not model_path.is_dir():
        raise ValueError("SFT training state requires a resolved local model directory")
    if config.save_path and Path(config.save_path).resolve().is_relative_to(model_path.resolve()):
        raise ValueError("SFT training state output must be outside the model directory")
    config_values = asdict(config)
    for key in ("save_path", "save_training_state", "resume_from", "metrics_log_dir", "export_gguf_path"):
        config_values.pop(key, None)
    digest = hashlib.sha256()

    def update(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    update(_canonical(config_values))
    for spec in (config.dataset_loader_fn, config.validation_dataset_loader_fn):
        if spec:
            loader_path = Path(spec.split(":", 1)[0])
            if not loader_path.is_file():
                raise ValueError("SFT training state requires a local dataset loader file")
            update(_hash_file(loader_path).encode())
    source_root = Path(__file__).resolve().parents[2]
    for path in sorted(
        p for p in source_root.rglob("*") if p.is_file() and p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".so"}
    ):
        update(str(path.relative_to(source_root)).encode())
        update(_hash_file(path).encode())
    versions = {"torch": str(torch.__version__), "numpy": np.__version__}
    for package in ("transformers", "tokenizers", "safetensors", "datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    update(_canonical(versions))
    for path in sorted(p for p in model_path.rglob("*") if p.is_file()):
        update(str(path.relative_to(model_path)).encode())
        update(_hash_file(path).encode())
    update(
        _canonical(
            {
                "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "vocab": tokenizer.get_vocab(),
                "special_tokens": tokenizer.special_tokens_map,
                "chat_template": tokenizer.chat_template,
            }
        )
    )
    for rows in (dataset, validation_dataset):
        if rows is None:
            update(b"none")
            continue
        if not isinstance(rows, (list, tuple)) and type(rows).__module__ != "datasets.arrow_dataset":
            raise ValueError("SFT training state requires a list or a Hugging Face Dataset")
        update(str(len(rows)).encode())
        for index in range(len(rows)):
            row = dict(rows[index])
            if record_has_image(row) or row.get("features") is not None:
                raise ValueError("SFT training state currently supports text-only dataset rows")
            update(_canonical(row))
    return digest.hexdigest()


def capture_driver_rng(rng: np.random.RandomState) -> dict[str, Any]:
    def numpy_state(state):
        return (state[0], state[1].tolist(), *state[2:])

    return {
        "python": random.getstate(),
        "numpy": numpy_state(np.random.get_state()),
        "trainer": numpy_state(rng.get_state()),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_driver_rng(state: dict[str, Any], rng: np.random.RandomState) -> None:
    def numpy_state(value):
        return (value[0], np.asarray(value[1], dtype=np.uint32), *value[2:])

    random.setstate(state["python"])
    np.random.set_state(numpy_state(state["numpy"]))
    rng.set_state(numpy_state(state["trainer"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_driver_rng(state: dict[str, Any]) -> None:
    """Check RNG payloads without changing the live process's random streams."""

    random.Random().setstate(state["python"])
    for name in ("numpy", "trainer"):
        value = state[name]
        np.random.RandomState().set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.Generator(device="cpu").set_state(state["torch"])
    cuda_rng = state["cuda"]
    if cuda_rng is not None and not all(isinstance(value, torch.Tensor) for value in cuda_rng):
        raise ValueError("invalid CUDA RNG state")


def save_snapshot(path: str, trainer: Any, driver_state: dict[str, Any], fingerprint: str) -> None:
    """Publish the manifest only after the state files are in their final directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"training snapshot already exists: {target}")
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        trainer.save_training_state(str(temp / "worker.pt"))
        torch.save(driver_state, temp / "driver.pt")
        for name in ("worker.pt", "driver.pt"):
            _sync_file(temp / name)
        files = {name: _hash_file(temp / name) for name in ("worker.pt", "driver.pt")}
        _sync_directory(temp)
        if target.exists():
            raise FileExistsError(f"training snapshot already exists: {target}")
        os.rename(temp, target)
        manifest = target / ".manifest.tmp"
        manifest.write_text(
            json.dumps({"version": 2, "fingerprint": fingerprint, "files": files}, sort_keys=True) + "\n"
        )
        _sync_file(manifest)
        os.replace(manifest, target / "manifest.json")
        _sync_directory(target)
        _sync_directory(target.parent)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_snapshot(
    path: str, trainer: Any, fingerprint: str, validate_driver: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    """Check a complete snapshot before applying the backend state."""

    root = Path(path)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["version"] not in (1, 2) or manifest["fingerprint"] != fingerprint:
            raise ValueError("training snapshot input or configuration mismatch")
        if manifest["version"] == 1 and root.name.startswith("."):
            raise ValueError("training snapshot is unpublished")
        if set(manifest["files"]) != {"driver.pt", "worker.pt"}:
            raise ValueError("training snapshot file list is invalid")
        for name, expected in manifest["files"].items():
            if _hash_file(root / name) != expected:
                raise ValueError(f"training snapshot file is corrupt: {name}")
        driver_state = torch.load(root / "driver.pt", map_location="cpu", weights_only=True)
        validate_driver(driver_state)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("training snapshot is incomplete or invalid") from exc
    trainer.load_training_state(str(root / "worker.pt"), completed_batches=driver_state["completed_batches"])
    return driver_state
