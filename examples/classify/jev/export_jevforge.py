"""Convert an AReno classify checkpoint into JevForge's checkpoint layout.

Input:  `<ckpt>/` = HF backbone saved by AReno + `score_head.safetensors`.
Output: `<out>/best.safetensors` (`backbone.*` + `head.*`), `<out>/config.json`,
and `<out>/backbone/` (HF config + tokenizer, no weights), which is what
`jevforge.predict.Predictor`, `jevforge.evaluate`, and `jevforge.serve` load.

Backbone keys are produced by `transformers.AutoModel.from_pretrained`, so they
match `JevForgeModel.from_backbone_config` for plain and multimodal configs.

    python examples/classify/jev/export_jevforge.py \
        --ckpt runs/jev_qwen35_08b/step_000600 --out runs/jev_qwen35_08b/jevforge
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="AReno checkpoint directory (step_XXXXXX)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-model", default=None, help="recorded as base_model; defaults to --ckpt")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0, help="calibration temperature to record")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoModel

    ckpt = Path(args.ckpt).resolve(strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    head = load_file(str(ckpt / "score_head.safetensors"), device="cpu")
    backbone = AutoModel.from_pretrained(str(ckpt), dtype=torch.bfloat16, local_files_only=True)
    config = AutoConfig.from_pretrained(str(ckpt), local_files_only=True)
    hidden_size = getattr(config, "hidden_size", None) or config.text_config.hidden_size
    if tuple(head["0.weight"].shape) != (hidden_size, hidden_size):
        raise ValueError(f"score head width {tuple(head['0.weight'].shape)} does not match hidden_size={hidden_size}")

    state = {f"backbone.{name}": tensor.contiguous() for name, tensor in backbone.state_dict().items()}
    state.update({f"head.{name}": tensor.to(torch.bfloat16).contiguous() for name, tensor in head.items()})
    save_file(state, str(out / "best.safetensors"))

    backbone_dir = out / "backbone"
    backbone_dir.mkdir(exist_ok=True)
    for item in ckpt.iterdir():
        if item.is_file() and not item.name.endswith(WEIGHT_SUFFIXES) and not item.name.endswith(".index.json"):
            shutil.copy2(item, backbone_dir / item.name)

    (out / "config.json").write_text(
        json.dumps(
            {
                "schema": "jevforge-checkpoint-v1",
                "base_model": str(args.base_model or ckpt),
                "hidden_size": hidden_size,
                "max_length": args.max_length,
                "temperature": args.temperature,
                "exported_from": str(ckpt),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "tensors": len(state)}))


if __name__ == "__main__":
    main()
