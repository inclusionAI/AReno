"""Bounded dataset previews without executing dataset scripts or decoding media."""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlencode

from arenoflow.assets import referenced_uploads
from arenoflow.datasets import find

ROWS = 3


def request_json(url, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=25) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("Dataset preview response is too large")
        return json.loads(data)
    except urllib.error.HTTPError as exc:
        raise ValueError(
            f"Dataset sample request failed (HTTP {exc.code}); check repository access and dataset reference"
        ) from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("Dataset sample request failed or timed out; retry or paste a sample") from None


def remote_rows(dataset):
    parts = dataset["source"].split(":")
    if len(parts) > 3 or not parts[0]:
        raise ValueError("Use repository, repository:config, or repository:config:split")
    repo, config, split = parts[0], parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "train"
    split = split or "train"
    if dataset.get("model_hub", "hf") == "modelscope":
        return modelscope_rows(repo, config, split)
    token = os.environ.get("HF_TOKEN")
    base = "https://datasets-server.huggingface.co"
    available = request_json(base + "/splits?" + urlencode({"dataset": repo}), token).get("splits", [])
    choices = [item for item in available if item["split"] == split and (not config or item["config"] == config)]
    if not choices:
        raise ValueError("Dataset config or split is unavailable; check the dataset reference")
    choices.sort(key=lambda item: (item["config"] not in ("default", "main"), item["config"]))
    if not config and len(choices) > 1 and choices[0]["config"] not in ("default", "main"):
        raise ValueError("Dataset has multiple configs; specify repository:config:split")
    selected = choices[0]
    query = {"dataset": repo, "config": selected["config"], "split": split, "offset": 0, "length": ROWS}
    result = request_json(base + "/rows?" + urlencode(query), token)
    return [r["row"] for r in result.get("rows", [])[:ROWS]], {"config": selected["config"], "split": split}


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP reads for Parquet metadata and the first record batch."""

    def __init__(self, url, size, headers):
        self.url, self.size, self.headers = url, size, headers
        self.position = 0
        self.remaining = 32 * 1024 * 1024

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Invalid dataset seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(self.size - self.position, self.size if size < 0 else size)
        if size <= 0:
            return b""
        if size > self.remaining:
            raise ValueError("Dataset preview exceeds the 32 MiB range-read limit")
        headers = {**self.headers, "Range": f"bytes={self.position}-{self.position + size - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=25) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                f"bytes {self.position}-"
            ):
                raise ValueError("Dataset server does not support partial reads for this shard")
            content = response.read(size)
        self.position += len(content)
        self.remaining -= len(content)
        return content

    def readinto(self, buffer):
        content = self.read(len(buffer))
        buffer[: len(content)] = content
        return len(content)


def modelscope_rows(repo, config, split):
    if len(repo.split("/")) != 2:
        raise ValueError("ModelScope datasets require namespace/dataset")
    base = "https://modelscope.cn/api/v1/datasets/" + quote(repo, safe="/")
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    files = []
    for page in range(1, 21):
        query = {"Revision": "master", "Root": "/", "Recursive": "True", "PageNumber": page, "PageSize": 100}
        response = request_json(base + "/repo/tree?" + urlencode(query), token)
        if response.get("Code") != 200:
            raise ValueError("ModelScope dataset listing failed; check access and repository name")
        rows = response.get("Data", {}).get("Files", [])
        files.extend(f for f in rows if f.get("Type") != "tree")
        if len(rows) < 100:
            break
    else:
        raise ValueError("ModelScope dataset listing exceeds the preview limit")
    formats = (".parquet", ".jsonl", ".json", ".csv", ".tsv", ".arrow")
    candidates = [f for f in files if Path(f["Path"]).suffix in formats]
    if config:
        candidates = [f for f in candidates if config in Path(f["Path"]).parts[:-1] or Path(f["Path"]).stem == config]
    split_pattern = re.compile(r"(?:^|[/_.-])" + re.escape(split) + r"(?:$|[/_.-])")
    matching = [f for f in candidates if split_pattern.search(f["Path"])]
    if not matching and split == "train" and len(candidates) == 1:
        # A single unsplit file is used as training data by the native raw-file loader.
        item = candidates[0]
        if not re.search(r"(?:^|[/_.-])(test|validation|eval)(?:$|[/_.-])", item["Path"]):
            matching = candidates
    if not matching:
        raise ValueError("ModelScope config or split has no supported data file; check the dataset reference")
    if not config:
        groups = {str(Path(f["Path"]).parent) for f in matching}
        preferred = next((g for g in ("main", "default", ".", "data") if g in groups), None)
        if len(groups) > 1 and preferred is None:
            raise ValueError("ModelScope dataset has multiple configs; specify repository:config:split")
        if preferred is not None:
            matching = [f for f in matching if str(Path(f["Path"]).parent) == preferred]
        config = preferred if preferred not in (".", "data", None) else "default"
    selected = sorted(matching, key=lambda f: (formats.index(Path(f["Path"]).suffix), f["Path"]))[0]
    limit = 16 * 1024 * 1024
    query = {"Source": "SDK", "Revision": "master", "FilePath": selected["Path"], "View": "False"}
    headers = {"Authorization": "Bearer " + token} if token else {}
    url = base + "/repo?" + urlencode(query)
    if selected.get("Size", 0) > limit and Path(selected["Path"]).suffix == ".parquet":
        import pyarrow.parquet as pq

        with HTTPRangeReader(url, selected["Size"], headers) as stream, pq.ParquetFile(stream) as reader:
            batch = next(reader.iter_batches(batch_size=ROWS), None)
            result = batch.to_pylist() if batch is not None else []
        return result, {"config": config, "split": split}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(base + "/repo?" + urlencode(query), headers=headers), timeout=25
        ) as response:
            content = response.read(limit + 1)
        if len(content) > limit:
            raise ValueError("ModelScope preview shard exceeds 16 MiB")
    except urllib.error.HTTPError as exc:
        raise ValueError(f"ModelScope sample download failed (HTTP {exc.code})") from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("ModelScope sample download failed or timed out") from None
    with tempfile.TemporaryDirectory(prefix="arenoflow-preview-") as folder:
        file = Path(folder) / ("sample" + Path(selected["Path"]).suffix)
        file.write_bytes(content)
        result = local_rows(file)
    return result, {"config": config, "split": split}


def local_rows(file):
    if file.suffix in (".parquet", ".arrow"):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq

        if file.suffix == ".parquet":
            with pq.ParquetFile(file) as reader:
                return list(
                    itertools.chain.from_iterable(
                        batch.to_pylist() for batch in itertools.islice(reader.iter_batches(batch_size=ROWS), 1)
                    )
                )
        with pa.memory_map(str(file), "r") as stream:
            try:
                reader = ipc.open_file(stream)
                batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
            except pa.ArrowInvalid:
                stream.seek(0)
                batches = ipc.open_stream(stream)
            rows = []
            for batch in batches:
                rows.extend(batch.slice(0, ROWS - len(rows)).to_pylist())
                if len(rows) >= ROWS:
                    break
            return rows
    with file.open(encoding="utf-8-sig") as stream:
        if file.suffix == ".json":
            # Uploads are capped at 16 MiB; parse the array before taking records.
            value = json.load(stream)
            if isinstance(value, dict):
                for key in ("train", "data", "records", "rows"):
                    if isinstance(value.get(key), list):
                        value = value[key]
                        break
            return value[:ROWS] if isinstance(value, list) else [value]
        if file.suffix == ".jsonl":
            return list(itertools.islice((json.loads(line) for line in stream if line.strip()), ROWS))
        return list(itertools.islice(csv.DictReader(stream, delimiter="\t" if file.suffix == ".tsv" else ","), ROWS))


def preview_value(value, depth=0):
    if depth > 12:
        return "[nested content omitted]"
    if isinstance(value, bytes):
        return f"[binary media omitted: {len(value)} bytes]"
    if isinstance(value, str):
        return value if len(value) <= 1200 else value[:1200] + "… [truncated]"
    if isinstance(value, dict):
        return {str(k): preview_value(v, depth + 1) for k, v in itertools.islice(value.items(), 30)}
    if isinstance(value, (list, tuple)):
        return [preview_value(v, depth + 1) for v in value[:12]]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value if value is None or isinstance(value, (bool, int, float)) else str(value)


def dataset_sample(store, identifier):
    dataset = find(store.datasets(), identifier, "Dataset")
    metadata = {}
    if dataset.get("source_type") == "upload":
        file = referenced_uploads({"source": dataset["source"]}, store.directory)[0][0]
        rows = local_rows(file)
    else:
        rows, metadata = remote_rows(dataset)
    if not rows:
        raise ValueError("Dataset contains no sample records")
    rows = preview_value(rows)
    while len(json.dumps(rows, ensure_ascii=False, indent=2)) > 12000 and len(rows) > 1:
        rows.pop()
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if len(text) > 12000:
        raise ValueError("Sample record is too large; paste the relevant fields manually")
    return {"sample": text, "manual": False, "row_count": len(rows), **metadata}
