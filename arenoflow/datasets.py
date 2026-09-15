"""Named datasets and Python functions; filesystem details stay in the runtime layer."""

from __future__ import annotations

import ast
import base64
import copy
import csv
import inspect
import io
import json
import time
import uuid
from pathlib import Path

from arenoflow.assets import DATA_SUFFIXES, MEDIA_SUFFIXES, referenced_uploads, save_upload

FUNCTIONS = {
    "dataset_loader": ("load_training_dataset", 1),
    "reward": ("reward_fn", 1),
    "agentic": ("run_agent", 2),
}


def required_text(body, name, limit=4096):
    value = body.get(name, "")
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be non-empty text up to {limit} characters")
    return value.strip()


def find(records, identifier, label):
    result = next((r for r in records if r["id"] == identifier), None)
    if result is None:
        raise ValueError(f"{label} no longer exists; select it again")
    return result


def validate_source(source, kind):
    if kind not in FUNCTIONS:
        raise ValueError("Choose dataset_loader, reward or agentic")
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > 256 * 1024:
        raise ValueError("Enter Python code up to 256 KiB")
    name, count = FUNCTIONS[kind]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Python syntax error on line {exc.lineno}: {exc.msg}") from None
    node = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None
    )
    if node is None or (isinstance(node, ast.AsyncFunctionDef) and kind != "agentic"):
        raise ValueError(f"Define {'a synchronous ' if kind != 'agentic' else 'a '}function named {name}")
    # Bind the real hook call against an AST-derived signature, without importing code.
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults_start = len(positional) - len(args.defaults)
    parameters = [
        inspect.Parameter(
            a.arg,
            inspect.Parameter.POSITIONAL_ONLY if i < len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None if i >= defaults_start else inspect.Parameter.empty,
        )
        for i, a in enumerate(positional)
    ]
    if args.vararg:
        parameters.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    parameters.extend(
        inspect.Parameter(
            a.arg, inspect.Parameter.KEYWORD_ONLY, default=None if d is not None else inspect.Parameter.empty
        )
        for a, d in zip(args.kwonlyargs, args.kw_defaults)
    )
    if args.kwarg:
        parameters.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    kwargs = {"default_loader": None, "load_dataset": None, "load_from_disk": None} if kind == "dataset_loader" else {}
    try:
        inspect.Signature(parameters).bind(*([None] * count), **kwargs)
    except TypeError as exc:
        raise ValueError(f"{name} does not accept AReno's hook arguments: {exc}") from None


def function_record(store, body):
    identifier = body.get("id")
    previous = find(store.functions(), identifier, "Function") if identifier else None
    kind = body.get("kind")
    if previous and previous["kind"] != kind:
        raise ValueError("Create a new function to use another function type")
    name = required_text(body, "name", 120)
    source = body.get("source", "")
    validate_source(source, kind)
    record = dict(
        id=identifier or uuid.uuid4().hex[:16],
        name=name,
        kind=kind,
        source=source,
        dataset_id=body.get("dataset_id"),
        algorithm=body.get("algorithm"),
        created_at=previous["created_at"] if previous else time.time(),
        updated_at=time.time(),
    )
    return record


def save_function(store, body):
    return store.save_function(function_record(store, body))


def save_script_batch(store, body):
    scripts = body.get("scripts")
    if not isinstance(scripts, list) or not 1 <= len(scripts) <= 3:
        raise ValueError("Provide one to three scripts")
    if any(not isinstance(item, dict) or item.get("id") for item in scripts):
        raise ValueError("Batch save creates new scripts only")
    records = [function_record(store, item) for item in scripts]
    if len({r["kind"] for r in records}) != len(records):
        raise ValueError("Duplicate script types")
    return {"scripts": store.save_functions(records)}


def save_dataset(store, body):
    identifier = body.get("id")
    previous = find(store.datasets(), identifier, "Dataset") if identifier else None
    name = required_text(body, "name", 120)
    source = required_text(body, "source")
    source_type = body.get("source_type", "repository")
    if source_type == "upload":
        if not source.startswith("/artifacts/uploads/") or Path(source).suffix not in DATA_SUFFIXES:
            raise ValueError("Upload a dataset first")
        referenced_uploads({"source": source}, store.directory)
    elif source_type != "repository" or source.startswith(("/", ".", "~")):
        raise ValueError("Enter a dataset repository ID or upload a dataset")
    hub = body.get("model_hub", "hf")
    if hub not in ("hf", "modelscope"):
        raise ValueError("Select Hugging Face or ModelScope")
    loader_id = body.get("loader_id") or None
    if loader_id and find(store.functions(), loader_id, "Dataset loader")["kind"] != "dataset_loader":
        raise ValueError("Select a Dataset Loader function")
    modalities = body.get("modalities", ["text"])
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(m not in ("text", "image", "audio", "video") for m in modalities)
    ):
        raise ValueError("Select text, image, audio or video modalities")
    media = body.get("media", [])
    if not isinstance(media, list) or len(media) > 128:
        raise ValueError("Attach up to 128 media files; use a repository for larger collections")
    names = set()
    for item in media:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("path"), str):
            raise ValueError("Invalid media attachment")
        if item["name"] in names:
            raise ValueError("Each media attachment needs a unique filename")
        names.add(item["name"])
        if not item["path"].startswith("/artifacts/uploads/") or Path(item["path"]).suffix not in MEDIA_SUFFIXES:
            raise ValueError("Upload images, audio or video for media attachments")
    referenced_uploads({"media": media}, store.directory)
    if media and (source_type != "upload" or Path(source).suffix not in (".json", ".jsonl", ".csv", ".tsv")):
        raise ValueError(
            "Attach media to an uploaded JSON, JSONL, CSV or TSV manifest; use a dataset repository for packaged Parquet/Arrow media"
        )
    record = dict(
        id=identifier or uuid.uuid4().hex[:16],
        name=name,
        source=source,
        source_type=source_type,
        modalities=modalities,
        media=media,
        source_name=str(body.get("source_name", "Uploaded dataset"))[:255],
        model_hub=hub,
        loader_id=loader_id,
        created_at=previous["created_at"] if previous else time.time(),
        updated_at=time.time(),
    )
    return store.save_dataset(record)


def resolve_request(request, store):
    """Snapshot selected code into content-addressed assets for a reproducible run."""
    result = copy.deepcopy(request)
    functions = store.functions()

    def reference(identifier, kind):
        function = find(functions, identifier, "Function")
        if function["kind"] != kind:
            raise ValueError(f"Select a {kind} function")
        validate_source(function["source"], kind)
        asset = save_upload(store.directory, "hook.py", base64.b64encode(function["source"].encode()).decode())
        return asset["path"] + (":load_training_dataset" if kind == "dataset_loader" else "")

    for stage in result.get("stages", []):
        params = stage.setdefault("params", {})
        if stage.get("dataset_id"):
            dataset = find(store.datasets(), stage["dataset_id"], "Dataset")
            result.setdefault("input_assets", []).extend(item["path"] for item in dataset.get("media", []))
            params.update(
                dataset_path=materialize_dataset(dataset, store),
                dataset_loader_fn=reference(dataset["loader_id"], "dataset_loader") if dataset["loader_id"] else None,
            )
        if "dataset_loader_id" in stage:
            params["dataset_loader_fn"] = (
                reference(stage["dataset_loader_id"], "dataset_loader") if stage["dataset_loader_id"] else None
            )
        for field, kind, option in (
            ("reward_function_id", "reward", "reward_fn_path"),
            ("agentic_function_id", "agentic", "agent_fn"),
        ):
            if stage.get(field):
                if stage.get("algo") not in ("gspo", "grpo", "ppo"):
                    raise ValueError(f"{stage.get('algo', '').upper()} does not use {kind} functions")
                params[option] = reference(stage[field], kind)
    return result


def materialize_dataset(dataset, store):
    """Resolve media names recursively in raw dataset manifests, without decoding media."""
    if dataset.get("source_type", "repository") == "repository":
        from arenoflow.dataset_cache import cache_key, cached_files

        cached_files(dataset, store.directory)
        return "/artifacts/datasets/" + cache_key(dataset)
    if not dataset.get("media"):
        return dataset["source"]
    names = {item["name"]: item["path"] for item in dataset["media"]}

    def replace(value):
        if isinstance(value, str):
            return names.get(value, names.get(value.removeprefix("./"), value))
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    files = referenced_uploads({"source": dataset["source"]}, store.directory)
    source = files[0][0]
    try:
        content = source.read_text()
        if source.suffix == ".json":
            content = json.dumps(replace(json.loads(content)), ensure_ascii=False)
        elif source.suffix == ".jsonl":
            content = (
                "\n".join(
                    json.dumps(replace(json.loads(line)), ensure_ascii=False)
                    for line in content.splitlines()
                    if line.strip()
                )
                + "\n"
            )
        else:
            reader = csv.DictReader(io.StringIO(content), delimiter="\t" if source.suffix == ".tsv" else ",")
            output = io.StringIO()
            writer = csv.DictWriter(
                output, fieldnames=reader.fieldnames, delimiter="\t" if source.suffix == ".tsv" else ","
            )
            writer.writeheader()
            writer.writerows(replace(row) for row in reader)
            content = output.getvalue()
    except (ValueError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read dataset manifest: {exc}") from None
    return save_upload(store.directory, "dataset" + source.suffix, base64.b64encode(content.encode()).decode())["path"]
