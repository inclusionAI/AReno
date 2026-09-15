"""Dataset previews preserve configuration and never execute loader scripts."""

import base64
import json

import pytest

from arenoflow.assets import save_upload
from arenoflow.datasets import save_dataset
from arenoflow.samples import dataset_sample
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


def test_unsupported_dataset_hub_is_rejected_before_download(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("Unsupported dataset sources must not make network requests")

    monkeypatch.setattr("arenoflow.samples.request_json", unexpected)
    record = {"name": "Unsupported", "source": "owner/data", "model_hub": "unsupported"}
    with pytest.raises(ValueError, match="Select Hugging Face or ModelScope"):
        save_dataset(Store(tmp_path), record)
    from arenoflow.dataset_cache import repository_files

    with pytest.raises(ValueError, match="Select Hugging Face"):
        repository_files(record)
