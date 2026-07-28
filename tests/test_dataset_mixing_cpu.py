from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from areno.api.data import DATASET_MIX_METADATA_KEY, DatasetMixSource, WeightedMixedDataset
from areno.cli import train as train_cli


def _source(name: str, size: int, weight: float) -> DatasetMixSource:
    return DatasetMixSource(
        name=name,
        dataset=[{"prompt": f"{name}-{index}", "response": str(index)} for index in range(size)],
        weight=weight,
    )


def _identity(dataset: WeightedMixedDataset) -> list[tuple[str, int, int]]:
    return [
        (
            row[DATASET_MIX_METADATA_KEY]["source"],
            row[DATASET_MIX_METADATA_KEY]["source_index"],
            row[DATASET_MIX_METADATA_KEY]["cycle"],
        )
        for row in dataset
    ]


def test_weighted_mix_is_deterministic_for_seed_and_epoch():
    sources = [_source("math", 5, 0.7), _source("code", 4, 0.3)]
    first = WeightedMixedDataset(sources, seed=42, exhaustion="renormalize")
    second = WeightedMixedDataset(sources, seed=42, exhaustion="renormalize")

    assert _identity(first) == _identity(second)
    assert first.summary()["schedule_hash"] == second.summary()["schedule_hash"]

    first.set_epoch(1)
    assert _identity(first) != _identity(second)


def test_renormalize_emits_every_record_once():
    dataset = WeightedMixedDataset(
        [_source("small", 2, 0.8), _source("large", 7, 0.2)],
        seed=7,
        exhaustion="renormalize",
        shuffle_within_sources=False,
    )

    identities = _identity(dataset)
    assert len(identities) == 9
    assert len({(source, index) for source, index, _cycle in identities}) == 9
    assert all(cycle == 0 for _source_name, _index, cycle in identities)
    assert dataset.summary()["termination_reason"] == "all_sources_exhausted"
    assert all(source["duplicates"] == 0 for source in dataset.summary()["sources"])


def test_stop_never_repeats_records_and_may_leave_rows_unused():
    dataset = WeightedMixedDataset(
        [_source("small", 1, 0.9), _source("large", 8, 0.1)],
        seed=3,
        exhaustion="stop",
        shuffle_within_sources=False,
    )

    identities = _identity(dataset)
    assert len(identities) < 9
    assert len({(source, index) for source, index, _cycle in identities}) == len(identities)
    assert dataset.summary()["termination_reason"].startswith("source_exhausted:")


def test_cycle_visits_every_record_and_can_repeat_small_sources():
    dataset = WeightedMixedDataset(
        [_source("small", 1, 0.9), _source("large", 4, 0.1)],
        seed=11,
        exhaustion="cycle",
        shuffle_within_sources=False,
    )

    identities = _identity(dataset)
    unique = {(source, index) for source, index, _cycle in identities}
    assert unique == {("small", 0), ("large", 0), ("large", 1), ("large", 2), ("large", 3)}
    assert len(identities) > len(unique)
    assert dataset.summary()["termination_reason"] == "all_sources_exhausted_once"
    assert next(item for item in dataset.summary()["sources"] if item["name"] == "small")["duplicates"] > 0


def test_max_samples_per_epoch_bounds_cycle():
    dataset = WeightedMixedDataset(
        [_source("small", 1, 0.99), _source("rare", 2, 0.01)],
        seed=5,
        exhaustion="cycle",
        max_samples_per_epoch=3,
    )

    assert len(dataset) == 3
    assert dataset.summary()["termination_reason"] == "max_samples_per_epoch"


def test_max_samples_per_epoch_is_rejected_for_non_cycle_policies():
    with pytest.raises(ValueError, match="only supported with exhaustion='cycle'"):
        WeightedMixedDataset(
            [_source("first", 1, 1.0), _source("second", 1, 1.0)],
            seed=5,
            exhaustion="renormalize",
            max_samples_per_epoch=1,
        )


def test_observed_proportions_follow_weights_with_documented_tolerance():
    dataset = WeightedMixedDataset(
        [_source("major", 10_000, 0.7), _source("minor", 10_000, 0.3)],
        seed=42,
        exhaustion="cycle",
        shuffle_within_sources=False,
        max_samples_per_epoch=2_000,
    )

    observed = {source["name"]: source["observed_proportion"] for source in dataset.summary()["sources"]}
    assert observed["major"] == pytest.approx(0.7, abs=0.05)
    assert observed["minor"] == pytest.approx(0.3, abs=0.05)


@pytest.mark.parametrize("weight", [0, -1, math.nan, math.inf])
def test_weighted_mix_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="weight must be finite and positive"):
        WeightedMixedDataset(
            [_source("bad", 1, weight), _source("good", 1, 1.0)],
            seed=1,
            exhaustion="stop",
        )


def test_weighted_mix_rejects_reserved_metadata_field():
    with pytest.raises(ValueError, match="reserved field"):
        WeightedMixedDataset(
            [
                DatasetMixSource("bad", [{DATASET_MIX_METADATA_KEY: {}, "prompt": "x"}], 1.0),
                _source("good", 1, 1.0),
            ],
            seed=1,
            exhaustion="stop",
        )


def test_weighted_mix_rejects_bool_weight_and_non_bool_shuffle():
    with pytest.raises(ValueError, match="weight must be finite and positive"):
        WeightedMixedDataset(
            [_source("good", 1, 1.0), DatasetMixSource("bad", [{"prompt": "x"}], True)],
            seed=1,
            exhaustion="stop",
        )
    with pytest.raises(ValueError, match="shuffle_within_sources must be a boolean"):
        WeightedMixedDataset(
            [_source("first", 1, 1.0), _source("second", 1, 1.0)],
            seed=1,
            exhaustion="stop",
            shuffle_within_sources=1,
        )


def test_weighted_mix_normalizes_large_finite_weights_without_overflow():
    dataset = WeightedMixedDataset(
        [_source("first", 2, 1e308), _source("second", 2, 1e308)],
        seed=1,
        exhaustion="renormalize",
    )

    assert [source["weight_requested"] for source in dataset.summary()["sources"]] == [0.5, 0.5]


def test_weighted_mix_rejects_unrepresentable_weight_range():
    with pytest.raises(ValueError, match="unsupported numeric range"):
        WeightedMixedDataset(
            [_source("tiny", 1, 1e-300), _source("huge", 1, 1e300)],
            seed=1,
            exhaustion="renormalize",
        )


def test_mix_manifest_loads_two_local_sft_sources_through_shared_loader(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"instruction":"math","output":"1"}\n', encoding="utf-8")
    second.write_text('{"instruction":"code","output":"2"}\n', encoding="utf-8")
    loader = tmp_path / "loader.py"
    loader.write_text(
        "def load_training_dataset(dataset_path, *, default_loader, **kwargs):\n"
        "    return [\n"
        "        {'prompt': row['instruction'], 'response': row['output']}\n"
        "        for row in default_loader(dataset_path)\n"
        "    ]\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "mix.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 42,
                "exhaustion": "renormalize",
                "sources": [
                    {"name": "math", "path": first.name, "weight": 7},
                    {"name": "code", "path": second.name, "weight": 3},
                ],
            }
        ),
        encoding="utf-8",
    )

    def load_dataset(_builder, *, data_files, split):
        assert split == "train"
        return [json.loads(line) for line in Path(data_files).read_text(encoding="utf-8").splitlines()]

    dataset = train_cli._load_mixed_dataset_for_training(
        str(manifest),
        model_hub="hf",
        dataset_loader_fn=str(loader),
        load_dataset=load_dataset,
        load_from_disk=lambda _path: None,
    )

    assert len(dataset) == 2
    assert {row["prompt"] for row in dataset} == {"math", "code"}
    assert {row[DATASET_MIX_METADATA_KEY]["source"] for row in dataset} == {"math", "code"}


def test_dataset_source_shorthand_preserves_remote_ref_colons():
    manifest = train_cli._dataset_mix_manifest_from_sources(
        (
            "math=gsm8k:main:train:0.7",
            "code=org/code-dataset:default:train:0.3",
        )
    )

    assert manifest["seed"] == 42
    assert manifest["exhaustion"] == "renormalize"
    assert manifest["shuffle_within_sources"] is True
    assert manifest["sources"] == [
        {"name": "math", "path": "gsm8k:main:train", "weight": 0.7},
        {"name": "code", "path": "org/code-dataset:default:train", "weight": 0.3},
    ]


@pytest.mark.parametrize(
    ("source_specs", "message"),
    [
        (("one=dataset:1",), "at least twice"),
        (("broken", "two=dataset:1"), "NAME=PATH:WEIGHT"),
        (("same=one:1", "same=two:1"), "duplicate source name"),
        (("one=dataset:nan", "two=dataset:1"), "finite and positive"),
    ],
)
def test_dataset_source_shorthand_rejects_invalid_entries(source_specs, message):
    with pytest.raises(ValueError, match=message):
        train_cli._dataset_mix_manifest_from_sources(source_specs)


def test_dataset_source_shorthand_loads_remote_sources_through_shared_loader(tmp_path):
    loader = tmp_path / "loader.py"
    loader.write_text(
        "def load_training_dataset(dataset_path, *, default_loader, **kwargs):\n"
        "    return [\n"
        "        {'prompt': row['instruction'], 'response': row['output']}\n"
        "        for row in default_loader(dataset_path)\n"
        "    ]\n",
        encoding="utf-8",
    )
    loaded = []

    def load_dataset(name, **_kwargs):
        loaded.append(name)
        return [{"instruction": name, "output": "ok"}]

    dataset = train_cli._load_dataset_sources_for_training(
        ("first=org/first:0.7", "second=org/second:0.3"),
        model_hub="hf",
        dataset_loader_fn=str(loader),
        load_dataset=load_dataset,
        load_from_disk=lambda _path: None,
    )

    assert loaded == ["org/first", "org/second"]
    assert {row["prompt"] for row in dataset} == {"org/first", "org/second"}
    assert {row[DATASET_MIX_METADATA_KEY]["source"] for row in dataset} == {"first", "second"}


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_mix_manifest_requires_integer_version_one(tmp_path, version):
    manifest = tmp_path / "mix.json"
    manifest.write_text(json.dumps({"version": version}), encoding="utf-8")

    with pytest.raises(ValueError, match="version must be 1"):
        train_cli._read_dataset_mix_manifest(manifest)


def test_mix_loader_reports_incompatible_source_without_sample_contents(tmp_path):
    manifest = tmp_path / "mix.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 1,
                "exhaustion": "stop",
                "sources": [
                    {"name": "good", "path": "good", "weight": 1},
                    {"name": "bad", "path": "bad", "weight": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    def load_dataset(name, **_kwargs):
        if name == "good":
            return [{"prompt": "safe", "response": "ok"}]
        return [{"prompt": "do-not-log-this-secret"}]

    with pytest.raises(ValueError, match=r"stage=dataset_mix_validation source=bad.*missing required SFT field"):
        train_cli._load_mixed_dataset_for_training(
            str(manifest),
            model_hub="hf",
            dataset_loader_fn=None,
            load_dataset=load_dataset,
            load_from_disk=lambda _path: None,
        )


def test_mix_validation_checks_later_rows_before_backend_initialization(tmp_path):
    manifest = tmp_path / "mix.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 1,
                "exhaustion": "renormalize",
                "sources": [
                    {"name": "good", "path": "good", "weight": 1},
                    {"name": "bad", "path": "bad", "weight": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    def load_dataset(name, **_kwargs):
        if name == "good":
            return [{"prompt": "q", "response": "a"}]
        return [{"prompt": "q", "response": "a"}, {"prompt": "later-row-is-invalid"}]

    with pytest.raises(ValueError, match=r"source=bad.*row 1 missing required SFT field"):
        train_cli._load_mixed_dataset_for_training(
            str(manifest),
            model_hub="hf",
            dataset_loader_fn=None,
            load_dataset=load_dataset,
            load_from_disk=lambda _path: None,
        )


def test_mix_imports_shared_loader_only_once(tmp_path):
    marker = tmp_path / "imports.txt"
    loader = tmp_path / "loader.py"
    loader.write_text(
        f"from pathlib import Path\nwith Path({str(marker)!r}).open('a') as marker:\n"
        "    marker.write('imported\\n')\n"
        "def load_training_dataset(dataset_path, *, default_loader, **kwargs):\n"
        "    return default_loader(dataset_path)\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "mix.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 1,
                "exhaustion": "renormalize",
                "sources": [
                    {"name": "first", "path": "first", "weight": 1},
                    {"name": "second", "path": "second", "weight": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    train_cli._load_mixed_dataset_for_training(
        str(manifest),
        model_hub="hf",
        dataset_loader_fn=str(loader),
        load_dataset=lambda *_args, **_kwargs: [{"prompt": "q", "response": "a"}],
        load_from_disk=lambda _path: None,
    )

    assert marker.read_text(encoding="utf-8").splitlines() == ["imported"]


def test_mix_artifact_is_structured_and_sample_free(tmp_path):
    dataset = WeightedMixedDataset(
        [
            DatasetMixSource("first", [{"prompt": "secret-one", "response": "answer-one"}], 1.0),
            DatasetMixSource("second", [{"prompt": "secret-two", "response": "answer-two"}], 1.0),
        ],
        seed=1,
        exhaustion="renormalize",
    )

    train_cli._write_dataset_mix_artifact(dataset, str(tmp_path))

    artifact_path = next(tmp_path.glob("dataset_mix_plan.*.json"))
    artifact_text = artifact_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_text)
    assert artifact["policy"] == "renormalize"
    assert artifact["planned_rows"] == 2
    assert "schedule_hash" in artifact
    assert "secret-one" not in artifact_text
    assert "answer-two" not in artifact_text
