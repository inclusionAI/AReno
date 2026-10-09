"""Bounded dataset previews without executing dataset scripts or decoding media."""

from __future__ import annotations

import csv
import itertools
import json
import math
import urllib.error
import urllib.request

from areno.dashboard.flow.assets import referenced_uploads
from areno.dashboard.flow.datasets import find

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
        from areno.dashboard.flow.dataset_cache import download_repository

        files, metadata = download_repository(dataset, store.directory)
        metadata = {key: value for key, value in metadata.items() if key != "files"}
        rows = []
        for file in files:
            rows.extend(local_rows(file)[: ROWS - len(rows)])
            if len(rows) >= ROWS:
                break
    if not rows:
        raise ValueError("Dataset contains no sample records")
    rows = preview_value(rows)
    while len(json.dumps(rows, ensure_ascii=False, indent=2)) > 12000 and len(rows) > 1:
        rows.pop()
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if len(text) > 12000:
        raise ValueError("Sample record is too large; paste the relevant fields manually")
    return {"sample": text, "manual": False, "row_count": len(rows), **metadata}
