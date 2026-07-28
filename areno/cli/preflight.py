"""Preflight model-reference validation without loading weights.

Provides :func:`preflight_model_ref` to check that a local checkpoint directory
or remote hub reference has the artifacts AReno needs (``config.json``,
tokenizer files, safetensors shards) *before* expensive model or worker
initialisation.  The function never loads tensors, instantiates
``SafetensorsIndex``, or calls ``AutoTokenizer.from_pretrained``; it only
inspects file presence and readability.

Results are returned as :class:`PreflightResult` dataclasses that carry a
structured status, the exact missing artifacts, and an actionable next step.
CLI layers can render these as human-readable text or JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

ConfigT = TypeVar("ConfigT")

# Tokenizer file sets: at least one group must be fully present.
# Each tuple is a *conjunction* — all files in the group must exist.
# Across groups, only one group needs to match (disjunction).
_TOKENIZER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("tokenizer.json",),
    ("tokenizer_config.json", "tokenizer.json"),
    ("vocab.json", "merges.txt"),
    ("tokenizer.model",),
    ("spiece.model",),
    ("tokenizer_config.json",),
)

# Files that indicate a safetensors weight shard collection.
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_GLOB = "*.safetensors"


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of preflighting one model reference.

    Attributes
    ----------
    model_ref:
        The original reference string the user supplied.
    resolved_path:
        Local directory path when the ref resolved locally, ``None`` for
        remote refs that are not in the local cache.
    status:
        One of ``ok``, ``not_found``, ``permission``, ``network``, ``format``.
    stage:
        Where the check stopped: ``local``, ``remote``, ``config``,
        ``tokenizer``, or ``weights``.
    detail:
        Human-readable description of the outcome.
    missing_artifacts:
        Exact list of file names that are absent or unreadable.  Empty when
        status is ``ok``.
    next_step:
        Actionable suggestion to resolve a failure.  Empty when status is ``ok``.
    """

    model_ref: str
    resolved_path: str | None
    status: Literal["ok", "not_found", "permission", "network", "format"]
    stage: str
    detail: str
    missing_artifacts: list[str] = field(default_factory=list)
    next_step: str = ""


def preflight_model_ref(
    model_ref: str,
    *,
    model_hub: str = "modelscope",
) -> PreflightResult:
    """Validate *model_ref* without loading weights or tokenizers.

    For local paths, checks directory existence, read permission,
    ``config.json`` parseability, tokenizer file presence, and safetensors
    shard integrity (index references or direct file presence).

    For remote IDs, verifies the hub client is importable and reports
    cache status.  No network requests are made.
    """

    if not model_ref:
        return PreflightResult(
            model_ref=str(model_ref),
            resolved_path=None,
            status="format",
            stage="local",
            detail="empty model reference",
            next_step="Provide a non-empty --ckpt or --model-path value.",
        )

    path = Path(model_ref)
    if path.exists():
        return _preflight_local(path, model_ref=model_ref)

    return _preflight_remote(model_ref, model_hub=model_hub)


def preflight_model_refs_for_config(config: Any) -> list[PreflightResult]:
    """Preflight every model checkpoint reference in a trainer config.

    Returns one :class:`PreflightResult` per non-empty ``*_ckpt`` attribute.
    """

    model_hub = str(getattr(config, "model_hub", "modelscope"))
    refs: list[tuple[str, str]] = []
    ckpt = getattr(config, "ckpt", None)
    if ckpt:
        refs.append(("ckpt", ckpt))
    for attr in ("ref_ckpt", "reward_ckpt", "critic_ckpt"):
        value = getattr(config, attr, None)
        if value:
            refs.append((attr, value))

    results: list[PreflightResult] = []
    for label, ref in refs:
        result = preflight_model_ref(ref, model_hub=model_hub)
        results.append(result)
    return results


def format_preflight_text(result: PreflightResult) -> str:
    """Render one result as a single human-readable block."""

    status_label = result.status.upper() if result.status != "ok" else "OK"
    line = f"{status_label:<4} model preflight ({result.stage})  {result.model_ref}"
    if result.detail:
        line += f"\n     {result.detail}"
    if result.missing_artifacts:
        line += f"\n     missing: {', '.join(result.missing_artifacts)}"
    if result.next_step:
        line += f"\nNext:\n  {result.next_step}"
    return line


def preflight_results_to_json(results: list[PreflightResult]) -> str:
    """Serialise results to a JSON string for structured consumption."""

    return json.dumps([asdict(r) for r in results], indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preflight_local(path: Path, *, model_ref: str) -> PreflightResult:
    """Validate a local checkpoint directory."""

    if not path.is_dir():
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=str(path),
            status="format",
            stage="local",
            detail=f"{path} is not a directory",
            next_step="Provide a checkpoint directory, not a file.",
        )

    if not os.access(path, os.R_OK):
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=str(path),
            status="permission",
            stage="local",
            detail=f"no read permission for {path}",
            next_step=f"Check file permissions: chmod +r {path}",
        )

    # --- config.json ---
    config_path = path / "config.json"
    if not config_path.exists():
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=str(path),
            status="format",
            stage="config",
            detail="config.json not found",
            missing_artifacts=["config.json"],
            next_step=f"Download config.json into {path}",
        )
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=str(path),
            status="format",
            stage="config",
            detail=f"config.json is not valid JSON: {type(exc).__name__}: {exc}",
            missing_artifacts=["config.json"],
            next_step=f"Replace {config_path} with a valid HF config.json.",
        )
    if not isinstance(data, dict) or "model_type" not in data:
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=str(path),
            status="format",
            stage="config",
            detail="config.json missing required 'model_type' field",
            missing_artifacts=["config.json"],
            next_step=f"Ensure config.json contains a 'model_type' field.",
        )

    # --- tokenizer files ---
    missing_tokenizer = _check_tokenizer_files(path)
    if missing_tokenizer is not None:
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=str(path),
            status="format",
            stage="tokenizer",
            detail="no valid tokenizer file set found",
            missing_artifacts=missing_tokenizer,
            next_step=f"Download tokenizer files into {path}",
        )

    # --- weights (safetensors) ---
    missing_weights = _check_weight_files(path)
    if missing_weights is not None:
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=str(path),
            status="format",
            stage="weights",
            detail="safetensors shard files not found or incomplete",
            missing_artifacts=missing_weights,
            next_step=f"Download model weight files into {path}",
        )

    return PreflightResult(
        model_ref=model_ref,
        resolved_path=str(path),
        status="ok",
        stage="local",
        detail=f"config, tokenizer, and weights verified in {path}",
    )


def _check_tokenizer_files(path: Path) -> list[str] | None:
    """Return ``None`` if a valid tokenizer set exists, else the missing files."""

    for group in _TOKENIZER_GROUPS:
        if all((path / name).exists() for name in group):
            return None
    # Collect all unique expected names for the error report.
    all_names: list[str] = []
    for group in _TOKENIZER_GROUPS:
        for name in group:
            if name not in all_names:
                all_names.append(name)
    return all_names


def _check_weight_files(path: Path) -> list[str] | None:
    """Return ``None`` if safetensors shards are present, else missing files."""

    index_path = path / _SAFETENSORS_INDEX
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})
        except (json.JSONDecodeError, OSError):
            return [_SAFETENSORS_INDEX]
        if not isinstance(weight_map, dict) or not weight_map:
            return [_SAFETENSORS_INDEX]
        # Check that every referenced shard file exists.
        referenced_shards = set(weight_map.values())
        missing_shards = sorted(s for s in referenced_shards if not (path / s).exists())
        if missing_shards:
            return missing_shards
        return None

    # No index file — look for direct safetensors files.
    safetensors_files = sorted(path.glob(_SAFETENSORS_GLOB))
    if not safetensors_files:
        return [_SAFETENSORS_INDEX, "model.safetensors"]
    return None


def _preflight_remote(model_ref: str, *, model_hub: str) -> PreflightResult:
    """Validate a remote model reference without network access."""

    # Basic format check: a hub repo ID typically contains '/'.
    if "/" not in model_ref:
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=None,
            status="format",
            stage="remote",
            detail=f"'{model_ref}' does not look like a valid {model_hub} repo ID (missing '/')",
            next_step="Use a repo ID like 'Qwen/Qwen3-0.6B' or a local path.",
        )

    # Check hub client importability.
    if model_hub == "hf":
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return PreflightResult(
                model_ref=model_ref,
                resolved_path=None,
                status="network",
                stage="remote",
                detail="huggingface_hub is not installed",
                next_step="pip install huggingface_hub",
            )
    elif model_hub == "modelscope":
        try:
            import modelscope  # noqa: F401
        except ImportError:
            return PreflightResult(
                model_ref=model_ref,
                resolved_path=None,
                status="network",
                stage="remote",
                detail="modelscope is not installed",
                next_step="pip install modelscope",
            )
    else:
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=None,
            status="format",
            stage="remote",
            detail=f"unknown model_hub '{model_hub}'",
            next_step="Use --model-hub hf or --model-hub modelscope.",
        )

    # Check local cache (no network request).
    cache_path = _find_hub_cache(model_ref, model_hub=model_hub)
    if cache_path is not None and cache_path.exists():
        return PreflightResult(
            model_ref=model_ref,
            resolved_path=str(cache_path),
            status="ok",
            stage="remote",
            detail=f"found in local {model_hub} cache: {cache_path}",
        )

    return PreflightResult(
        model_ref=model_ref,
        resolved_path=None,
        status="ok",
        stage="remote",
        detail=f"hub client available; '{model_ref}' not in local cache (will download on use)",
        next_step=f"Run areno train or areno serve to download '{model_ref}' from {model_hub}.",
    )


def _find_hub_cache(model_ref: str, *, model_hub: str) -> Path | None:
    """Return the likely local cache path for *model_ref* if it exists."""

    if model_hub == "hf":
        cache_root = os.environ.get("HF_HUB_CACHE") or str(
            Path.home() / ".cache" / "huggingface" / "hub"
        )
        candidate = Path(cache_root) / f"models--{model_ref.replace('/', '--')}"
        if candidate.exists():
            return candidate
        return None

    if model_hub == "modelscope":
        cache_root = os.environ.get("MODELSCOPE_CACHE") or str(
            Path.home() / ".cache" / "modelscope" / "hub"
        )
        candidate = Path(cache_root) / model_ref.replace("/", "/")
        if candidate.exists():
            return candidate
        return None

    return None
