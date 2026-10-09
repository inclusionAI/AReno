"""Exercise real worker rollout, mixed image/text batches, and slot reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from validate_vision import prepare_inputs

from areno.api.tokenizer import eos_token_ids
from areno.engine.api import ArenoEngine
from areno.engine.config import RuntimeConfig
from areno.engine.data import SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--eager-decode", action="store_true")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    names = ["single", "text", "multi", "portrait"]
    cases = prepare_inputs(processor, names)
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    runtime = RuntimeConfig(
        attn_backend="native", compile_model=False, eager_decode=args.eager_decode, decode_graph_buckets=[1, 2]
    )
    report = {"eager_decode": args.eager_decode, "rounds": []}
    with ArenoEngine.from_pretrained(
        str(args.model_path), role="rollout", devices=[args.device], runtime_config=runtime
    ) as engine:
        for order in (names, list(reversed(names))):
            output = engine.generate_rollout(
                [cases[name]["tokens"] for name in order],
                prompt_features=[cases[name]["features"] for name in order],
                max_new_tokens=16,
                max_running_prompts=2,
                eos_token_id=eos_token_ids(args.model_path, processor.tokenizer),
                sampling_params=SamplingParams(temperature=0.0, suppress_special_tokens=False),
            )
            rows = []
            for name, tokens in zip(order, output.response_ids, strict=True):
                expected = reference[name]["tokens"]
                rows.append(
                    {
                        "case": name,
                        "tokens": tokens,
                        "reference_tokens": expected,
                        "tokens_equal": tokens == expected,
                        "text": processor.tokenizer.decode(tokens),
                    }
                )
            report["rounds"].append({"rows": rows, "metrics": output.metrics})
    report["passed"] = all(row["tokens_equal"] for result in report["rounds"] for row in result["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("worker rollout differs from the official reference")


if __name__ == "__main__":
    main()
