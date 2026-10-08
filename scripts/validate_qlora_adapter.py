"""Reload an actual QLoRA adapter and measure rollout cache/graph allocation.

Run eager and graph modes in separate processes with the same checkpoint.
This probe does not train: optimizer moments have not been allocated, and
its reported allocation is not a training or model-loading peak.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from areno import Trainer
from areno.adapters import LoraConfig
from areno.api.config import CudaConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eager-decode", action="store_true")
    args = parser.parse_args()
    adapter = LoraConfig(adapter_path=args.adapter)
    if not adapter.qlora:
        raise ValueError("adapter must contain AReno NF4 quantization metadata")
    trainer = Trainer(
        1,
        args.model,
        custom_config=CudaConfig(
            devices=[0],
            lora=adapter,
            max_running_prompts=16,
            optimizer={"adam_4bit": True},
            runtime={"eager_decode": args.eager_decode},
        ),
    )
    trainer.init()
    exported = str(Path(args.output).with_suffix(".adapter"))
    try:
        tokens = trainer.get_tokenizer().encode("Choose one legal Tic-Tac-Toe move: 1 2 3 / 4 X 6 / O 8 9.")
        first = trainer.score_logprobs("actor", [tokens], microbatch_size=1)[0]
        second = trainer.score_logprobs("actor", [tokens], microbatch_size=1)[0]
        torch.testing.assert_close(torch.tensor(first), torch.tensor(second), atol=1e-5, rtol=0)
        assert torch.isfinite(torch.tensor(first)).all()
        trainer.export_adapter(exported)
        # Probe exactly the same cache capacity in both modes; this is the
        # engine's allocated-memory probe, not free machine DRAM or RSS.
        fraction = trainer.probe_rollout_cache(max_new_tokens=128, max_running_prompts=16, max_prompt_len=1024)
        allocated = fraction * torch.cuda.get_device_properties(0).total_memory
    finally:
        trainer.close()
    original = load_file(str(Path(args.adapter) / "adapter_model.safetensors"))
    restored = load_file(str(Path(exported) / "adapter_model.safetensors"))
    assert original.keys() == restored.keys()
    assert all(torch.equal(value, restored[key]) for key, value in original.items())
    result = {
        "config": vars(args),
        "adapter_roundtrip_exact": True,
        "logprobs_finite_and_repeatable": True,
        "logprobs": first,
        "rollout_probe_allocated_bytes": allocated,
        "optimizer_state_initialized": False,
        "qlora_restored_from_metadata": adapter.qlora,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
