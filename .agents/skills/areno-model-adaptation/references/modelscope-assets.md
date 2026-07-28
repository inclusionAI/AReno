# ModelScope Assets

Use ModelScope for model and dataset downloads in development and validation.
Do not switch to Hugging Face automatically when a ModelScope reference fails.

## Models

Normal AReno commands resolve remote model references through ModelScope:

```bash
areno serve --model-path <organization/model> --model-hub modelscope
areno train --ckpt <organization/model> --model-hub modelscope ...
```

For checkpoint inspection, download only required metadata first:

```bash
python - <<'PY'
from modelscope import snapshot_download

path = snapshot_download(
    "<organization/model>",
    local_dir="/tmp/areno-model-metadata",
    allow_file_pattern=[
        "config.json",
        "generation_config.json",
        "tokenizer*",
        "special_tokens_map.json",
    ],
)
print(path)
PY
```

Download the complete checkpoint only after its config and architecture have
been accepted:

```bash
python - <<'PY'
from modelscope import snapshot_download

print(snapshot_download("<organization/model>", local_dir="<local-model-dir>"))
PY
```

## Datasets

Prefer an AReno dataset reference with `--model-hub modelscope`. Its accepted
form is `name[:subset][:split]` as defined by the current train CLI. For direct
inspection use the same API as the runtime:

```python
from modelscope.msdatasets import MsDataset

dataset = MsDataset.load(
    "<organization/dataset>",
    subset_name="<optional-subset>",
    split="train",
    trust_remote_code=True,
)
```

Record the ModelScope model or dataset ID, resolved local directory, revision
when pinned, and files inspected. If the asset is unavailable on ModelScope,
stop with a concrete error or request a local path from the user.
