"""Download complete raw dataset shards locally, independently of model weights."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit


def cache_key(dataset):
    identity = {key: dataset.get(key) for key in ("source", "source_type", "model_hub")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def cached_files(dataset, directory):
    folder = directory / "dataset_cache" / cache_key(dataset)
    marker = folder.with_suffix(".json")
    if not marker.is_file():
        raise ValueError("Wait for the dataset download to finish in Dataset Manager")
    metadata = json.loads(marker.read_text())
    files = [folder / name for name in metadata["files"]]
    if not files or any(file.parent != folder or not file.is_file() for file in files):
        raise ValueError("Dataset cache is incomplete; download the dataset again")
    return files, metadata


def select_config(files, config, split):
    choices = [item for item in files if item["split"] == split and (not config or item["config"] == config)]
    configs = sorted({item["config"] for item in choices})
    if not configs:
        raise ValueError("Dataset config or split is unavailable")
    selected = config or next((c for c in ("default", "main") if c in configs), None)
    if selected is None and len(configs) != 1:
        raise ValueError("Dataset has multiple configs; specify repository:config:split")
    selected = selected or configs[0]
    return [item for item in choices if item["config"] == selected], selected


def repository_files(dataset):
    from arenoflow.samples import request_json

    parts = dataset["source"].split(":")
    if len(parts) > 3 or not parts[0]:
        raise ValueError("Use repository:config:split")
    repo, config, split = parts[0], parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "train"
    split = split or "train"
    hub = dataset.get("model_hub", "hf")
    if hub == "hf":
        token = os.environ.get("HF_TOKEN")
        data = request_json("https://datasets-server.huggingface.co/parquet?" + urlencode({"dataset": repo}), token)
        if data.get("partial") or data.get("pending") or data.get("failed"):
            raise ValueError("A complete dataset export is unavailable; upload the original data files")
        files, config = select_config(data.get("parquet_files", []), config, split)
        for item in files:
            url = urlsplit(item["url"])
            if url.scheme != "https" or url.hostname != "huggingface.co":
                raise ValueError("Dataset provider returned an unexpected download URL")
        return [(item["url"], ".parquet", item.get("size")) for item in files], token, dict(config=config, split=split)
    if hub != "modelscope" or len(repo.split("/")) != 2:
        raise ValueError("Select Hugging Face or a ModelScope namespace/dataset")
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    base = "https://modelscope.cn/api/v1/datasets/" + quote(repo, safe="/")
    files = []
    for page in range(1, 1001):
        query = dict(Revision="master", Root="/", Recursive="True", PageNumber=page, PageSize=100)
        response = request_json(base + "/repo/tree?" + urlencode(query), token)
        if response.get("Code") != 200:
            raise ValueError("ModelScope dataset listing failed")
        rows = response.get("Data", {}).get("Files", [])
        files.extend(item for item in rows if item.get("Type") != "tree")
        if len(rows) < 100:
            break
    else:
        raise ValueError("Dataset listing is too large; specify a smaller repository")
    formats = (".parquet", ".jsonl", ".json", ".csv", ".tsv", ".arrow")
    files = [item for item in files if Path(item["Path"]).suffix in formats]
    if config:
        files = [item for item in files if config in Path(item["Path"]).parts[:-1] or Path(item["Path"]).stem == config]
    pattern = re.compile(r"(?:^|[/_.-])" + re.escape(split) + r"(?:$|[/_.-])")
    matching = [item for item in files if pattern.search(item["Path"])]
    if not matching and split == "train" and len(files) == 1:
        if not re.search(r"(?:^|[/_.-])(test|validation|eval)(?:$|[/_.-])", files[0]["Path"]):
            matching = files
    if not matching:
        raise ValueError("Dataset config or split has no supported raw data files")
    if not config:
        groups = {str(Path(item["Path"]).parent) for item in matching}
        group = next((g for g in ("main", "default", ".", "data") if g in groups), None)
        if group is None and len(groups) > 1:
            raise ValueError("Dataset has multiple configs; specify repository:config:split")
        group = group or next(iter(groups))
        matching = [item for item in matching if str(Path(item["Path"]).parent) == group]
        config = group
    # Prefer one representation when a repository exports the same split in several formats.
    suffix = next(ext for ext in formats if any(Path(item["Path"]).suffix == ext for item in matching))
    result = []
    for item in sorted(matching, key=lambda item: item["Path"]):
        if Path(item["Path"]).suffix == suffix:
            query = dict(Source="SDK", Revision="master", FilePath=item["Path"], View="False")
            result.append((base + "/repo?" + urlencode(query), suffix, item.get("Size")))
    return result, token, dict(config=config, split=split)


def download_repository(dataset, directory):
    root = directory / "dataset_cache"
    root.mkdir(exist_ok=True, mode=0o700)
    key = cache_key(dataset)
    if (root / (key + ".json")).exists():
        try:
            return cached_files(dataset, directory)
        except ValueError:
            (root / (key + ".json")).unlink()
    files, token, metadata = repository_files(dataset)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=root) as temporary:
        folder = Path(temporary)
        names = []
        headers = {"Authorization": "Bearer " + token} if token else {}
        for index, (url, suffix, expected) in enumerate(files):
            name = f"shard-{index:06d}{suffix}"
            target = folder / name
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                with target.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if expected is not None and target.stat().st_size != expected:
                raise ValueError("Dataset download was incomplete; retry from Dataset Manager")
            names.append(name)
        if not names:
            raise ValueError("Dataset has no downloadable data files")
        destination = root / key
        if destination.exists():
            shutil.rmtree(destination)
        folder.rename(destination)
        marker = root / (key + ".tmp")
        marker.write_text(json.dumps({**metadata, "files": names}))
        marker.replace(root / (key + ".json"))
    return cached_files(dataset, directory)
