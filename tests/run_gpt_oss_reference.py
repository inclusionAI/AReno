"""Generate bounded full-model HF references; validation only, never runtime."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fp32", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        attn_implementation="eager",
    )
    # The upstream eager training implementation avoids the all-expert inference expansion.
    if args.fp32:
        model.float()
        torch.backends.cuda.matmul.allow_tf32 = False
    model.train()
    prompts = ["The capital of France is", "2 + 2 ="]
    stages = {}

    def capture(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            stages[name] = value.detach().float().cpu().numpy()

        return hook

    handles = []
    for layer_id, layer in enumerate(model.model.layers):
        for suffix, module in (("attn", layer.self_attn), ("moe", layer.mlp), ("out", layer)):
            handles.append(module.register_forward_hook(capture(f"{layer_id}:{suffix}")))
    records = []
    for index, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        with torch.no_grad():
            logits = model(ids).logits
        np.savez(out / f"reference-{index}.npz", tokens=ids.cpu().numpy(), logits=logits.float().cpu().numpy())
        np.savez(out / f"layers-{index}.npz", **stages)
        records.append({"index": index, "prompt": prompt, "tokens": ids.tolist(), "argmax": logits.argmax(-1).tolist()})
    for handle in handles:
        handle.remove()
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).cuda()
    with torch.no_grad():
        answer = model.generate(chat, max_new_tokens=32, do_sample=False)
    records.append({"chat_response": tokenizer.decode(answer[0, chat.shape[-1] :])})
    (out / "reference.json").write_text(json.dumps(records, indent=2))
    print("HF_REFERENCE_OK", json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
