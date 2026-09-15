"""Dataset previews preserve configuration and never execute loader scripts."""

import base64
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from arenoflow.assets import save_upload
from arenoflow.datasets import save_dataset
from arenoflow.samples import dataset_sample, remote_rows
from arenoflow.store import Store


def upload_sample(tmp_path, name, data):
    store = Store(tmp_path)
    asset = save_upload(tmp_path, name, base64.b64encode(data).decode())
    dataset = save_dataset(store, {"name": "test", "source_type": "upload", "source": asset["path"]})
    return dataset_sample(store, dataset["id"])


@pytest.mark.parametrize("format", ["parquet", "arrow", "arrow-stream"])
def test_binary_table_previews_omit_media(tmp_path, format):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"question": ["one", "two", "three", "four"], "image": [b"binary"] * 4})
    sink = pa.BufferOutputStream()
    if format == "parquet":
        pq.write_table(table, sink)
    else:
        factory = pa.ipc.new_file if format == "arrow" else pa.ipc.new_stream
        with factory(sink, table.schema) as writer:
            writer.write_table(table)
    result = upload_sample(
        tmp_path, "sample." + ("parquet" if format == "parquet" else "arrow"), sink.getvalue().to_pybytes()
    )
    rows = json.loads(result["sample"])
    assert len(rows) == 3 and rows[0]["question"] == "one"
    assert rows[0]["image"] == "[binary media omitted: 6 bytes]"


def test_large_json_still_loads_first_records(tmp_path):
    content = json.dumps([{"question": "x" * 1000}] * 100).encode()
    result = upload_sample(tmp_path, "sample.json", content)
    assert result["row_count"] == 3


def test_remote_reference_preserves_config_and_split(monkeypatch):
    queries = []

    def fetch(url, token):
        query = parse_qs(urlsplit(url).query)
        queries.append(query)
        if "/splits?" in url:
            return {"splits": [{"config": "main", "split": "test"}]}
        return {"rows": [{"row": {"answer": "42"}}]}

    monkeypatch.setattr("arenoflow.samples.request_json", fetch)
    rows, info = remote_rows({"source": "openai/gsm8k:main:test", "model_hub": "hf"})
    assert info == {"config": "main", "split": "test"}
    assert queries[1]["length"] == ["3"]
    assert queries[1]["split"] == ["test"]
    assert rows == [{"answer": "42"}]


def test_missing_remote_split_does_not_silently_change_dataset(monkeypatch):
    monkeypatch.setattr(
        "arenoflow.samples.request_json", lambda *args: {"splits": [{"config": "main", "split": "test"}]}
    )
    with pytest.raises(ValueError, match="unavailable"):
        remote_rows({"source": "openai/gsm8k:main:train"})


def test_modelscope_subset_and_split_select_matching_file(monkeypatch):
    import io

    from arenoflow.samples import modelscope_rows

    monkeypatch.setattr(
        "arenoflow.samples.request_json",
        lambda *args: {
            "Code": 200,
            "Data": {
                "Files": [
                    {"Path": "main/train.jsonl", "Size": 40, "Type": "blob"},
                    {"Path": "main/test.jsonl", "Size": 40, "Type": "blob"},
                    {"Path": "other/train.jsonl", "Size": 40, "Type": "blob"},
                ]
            },
        },
    )

    def download(request, timeout):
        query = parse_qs(urlsplit(request.full_url).query)
        assert query["FilePath"] == ["main/test.jsonl"]
        return io.BytesIO(b'{"answer":"42"}\n')

    monkeypatch.setattr("arenoflow.samples.urllib.request.urlopen", download)
    rows, info = modelscope_rows("owner/data", "main", "test")
    assert rows == [{"answer": "42"}]
    assert info == {"config": "main", "split": "test"}


def test_range_reader_seeks_and_enforces_budget(monkeypatch):
    import io

    from arenoflow.samples import HTTPRangeReader

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 5-7/10"}

    def download(request, timeout):
        assert request.get_header("Range") == "bytes=5-7"
        return Response(b"567")

    monkeypatch.setattr("arenoflow.samples.urllib.request.urlopen", download)
    stream = HTTPRangeReader("https://example.com/data", 10, {})
    stream.seek(-5, 2)
    assert stream.read(3) == b"567" and stream.tell() == 8
    stream.remaining = 1
    with pytest.raises(ValueError, match="limit"):
        stream.read(2)
