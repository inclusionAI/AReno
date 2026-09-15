"""Complete local dataset downloads must be independent of model checkpoint hubs."""

import io
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from arenoflow.assets import referenced_uploads
from arenoflow.catalog import catalog
from arenoflow.dataset_cache import cache_key, download_repository, repository_files
from arenoflow.datasets import resolve_request, save_dataset
from arenoflow.samples import dataset_sample
from arenoflow.store import Store
from arenoflow.workflows import plan


def test_modelscope_complete_shards_are_uploaded_with_hf_model(tmp_path, monkeypatch):
    store = Store(tmp_path)
    dataset = save_dataset(store, {"name": "Math", "source": "owner/data:main:train", "model_hub": "modelscope"})
    files = ["main/train-0000.jsonl", "main/train-0001.jsonl", "main/test.jsonl", "other/train.jsonl"]
    content = b'{"answer":42}\n' * 4
    monkeypatch.setattr(
        "arenoflow.samples.request_json",
        lambda *args: {
            "Code": 200,
            "Data": {"Files": [{"Path": f, "Size": len(content), "Type": "blob"} for f in files]},
        },
    )
    downloads = []

    def download(request, timeout):
        downloads.append(parse_qs(urlsplit(request.full_url).query)["FilePath"][0])
        return io.BytesIO(content)

    monkeypatch.setattr("arenoflow.dataset_cache.urllib.request.urlopen", download)
    sample = dataset_sample(store, dataset["id"])
    assert sample["row_count"] == 3
    assert downloads == files[:2]
    request = {
        "model": {"adapter": "qwen3", "checkpoint": "Qwen/Qwen3-0.6B"},
        "stages": [{"algo": "sft", "dataset_id": dataset["id"], "params": {"model_hub": "hf"}}],
    }
    prepared = plan(resolve_request(request, store), catalog())
    stage = prepared["manifest"]["stages"][0]
    assert stage["params"]["model_hub"] == "hf"
    assert stage["params"]["dataset_path"].startswith("/artifacts/datasets/")
    staged = referenced_uploads(prepared["manifest"], tmp_path)
    assert len(staged) == 2
    assert all(path.read_bytes() == content for path, _ in staged)
    assert all(remote.startswith("/datasets/") for _, remote in staged)
    dataset_sample(store, dataset["id"])
    assert len(downloads) == 2


def test_training_requires_complete_cache(tmp_path):
    store = Store(tmp_path)
    dataset = save_dataset(store, {"name": "Not downloaded", "source": "owner/data", "model_hub": "modelscope"})
    with pytest.raises(ValueError, match="download to finish"):
        resolve_request({"stages": [{"dataset_id": dataset["id"]}]}, store)


def test_failed_shard_never_publishes_complete_cache(tmp_path, monkeypatch):
    dataset = {"source": "owner/data", "model_hub": "hf"}
    monkeypatch.setattr(
        "arenoflow.dataset_cache.repository_files",
        lambda _: ([("https://example.com/one", ".jsonl", 4), ("https://example.com/two", ".jsonl", 4)], None, {}),
    )
    monkeypatch.setattr("arenoflow.dataset_cache.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="incomplete"):
        download_repository(dataset, tmp_path)
    assert not (tmp_path / "dataset_cache" / (cache_key(dataset) + ".json")).exists()
    assert not list((tmp_path / "dataset_cache").iterdir())


def test_hf_selects_all_requested_split_shards_and_rejects_partial(monkeypatch):
    result = {
        "parquet_files": [
            {"config": "main", "split": split, "url": f"https://huggingface.co/{i}", "size": i}
            for i, split in enumerate(["train", "test", "train"])
        ],
        "partial": False,
    }
    monkeypatch.setattr("arenoflow.samples.request_json", lambda *args: result)
    files, _, metadata = repository_files({"source": "owner/data:main:train"})
    assert len(files) == 2 and metadata == {"config": "main", "split": "train"}
    result["partial"] = True
    with pytest.raises(ValueError, match="complete dataset export"):
        repository_files({"source": "owner/data"})


def test_cached_upload_path_traversal_rejected(tmp_path):
    key = "a" * 64
    folder = tmp_path / "dataset_cache" / key
    folder.mkdir(parents=True)
    folder.with_suffix(".json").write_text(json.dumps({"files": ["../secret"]}))
    with pytest.raises(ValueError, match="incomplete"):
        referenced_uploads({"data": "/artifacts/datasets/" + key}, tmp_path)
