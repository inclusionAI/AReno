"""Compare a local official Phi4MM checkpoint with AReno's image/text path.

Run from the repository root; see README.md for the reference environment.
The reference imports checkpoint remote code. AReno uses its native adapter.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from areno.api.multimodal import encode_multimodal_prompt
from areno.api.tokenizer import eos_token_ids
from areno.engine.data.rollout_state import InferenceBatchState, payload_to_infer_meta
from areno.engine.modeling import skip_torch_init
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.runtime.metadata import InferMeta
from areno.models.phi4mm.checkpoint import load_phi4mm_weights
from areno.models.phi4mm.model import Phi4MMAdapter


def fixtures():
    return {
        "single": ([Image.new("RGB", (64, 64), "red")], "What color is this image? Answer briefly."),
        "portrait": ([Image.new("RGB", (80, 160), "blue")], "What color is this image? Answer briefly."),
        "multi": (
            [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (160, 80), "blue")],
            "Name the color of each image in order. Answer briefly.",
        ),
        "text": ([], "What is two plus two? Answer briefly."),
    }


def prepare_inputs(processor, names):
    cases = {}
    for name in names:
        images, prompt = fixtures()[name]
        content = "".join(f"<|image_{idx + 1}|>\n" for idx in range(len(images))) + prompt
        text = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        official = dict(processor(text=text, images=images or None, return_tensors="pt"))
        if images:
            encoded_images = []
            for image in images:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            tokens, features = encode_multimodal_prompt(
                processor.tokenizer, processor, {"prompt": prompt, "images_base64": encoded_images}
            )
            assert tokens == official["input_ids"][0].tolist(), f"{name}: processor token mismatch"
            for key in ("input_image_embeds", "image_sizes", "image_attention_mask"):
                assert torch.equal(features[key], official[key]), f"{name}: processor {key} mismatch"
            from areno.cli.serve import ChatMessage, _encode_messages_with_features

            message = ChatMessage(
                role="user",
                content=[
                    *[
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{value}"}}
                        for value in encoded_images
                    ],
                    {"type": "text", "text": prompt},
                ],
            )
            served_tokens, served_features = _encode_messages_with_features(processor.tokenizer, processor, [message])
            assert served_tokens == tokens, f"{name}: serving processor token mismatch"
            assert served_features["image_token_id"] == features["image_token_id"]
            for key in ("input_image_embeds", "image_sizes", "image_attention_mask"):
                assert torch.equal(served_features[key], features[key]), f"{name}: serving {key} mismatch"
        else:
            tokens, features = official["input_ids"][0].tolist(), None
        cases[name] = {"tokens": tokens, "features": features, "official": official}
    return cases


@contextmanager
def stream_weights(model, device, enabled):
    """Validation-only weight staging; execute unchanged modules on one GPU.

    This is not a serving/offload implementation or a performance benchmark.
    Staging whole decoder layers also preserves direct parameter accesses.
    """
    handles = []
    if enabled:
        modules = [model.model.embed_tokens, model.model.embed_tokens_extend.image_embed]
        modules.extend(model.model.layers)
        modules.extend([model.model.norm, model.lm_head])
        cpu_parameters = {
            module: [(parameter, parameter.data) for parameter in module.parameters()] for module in modules
        }

        def onload(module, inputs):
            module.to(device)

        def offload(module, inputs, output):
            # Inference never changes these weights. Retain the CPU storage
            # instead of copying every decoder weight back after every token.
            for parameter, storage in cpu_parameters[module]:
                parameter.data = storage
            for child in module.modules():
                for name, buffer in child.named_buffers(recurse=False):
                    setattr(child, name, buffer.cpu())

        for module in modules:
            handles.append(module.register_forward_pre_hook(onload))
            handles.append(module.register_forward_hook(offload))
    else:
        model.to(device)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_stages(model):
    stages = {}
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            stages.setdefault(name, []).append(tensor.detach().cpu())

        return hook

    def embedding(module, inputs):
        stages["embedding"] = [inputs[0].detach().cpu()]

    vision = model.model.embed_tokens_extend.image_embed
    handles.append(vision.img_processor.encoder.layers[-2].register_forward_hook(capture("vision_patch")))
    handles.append(vision.img_projection.register_forward_hook(capture("projector")))
    handles.append(model.model.layers[0].register_forward_pre_hook(embedding))
    for idx in (0, len(model.model.layers) // 2, len(model.model.layers) - 1):
        handles.append(model.model.layers[idx].register_forward_hook(capture(f"layer_{idx}")))
    handles.append(model.model.norm.register_forward_hook(capture("norm")))
    try:
        yield stages
    finally:
        for handle in handles:
            handle.remove()


def move_inputs(values, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in values.items()}


@torch.inference_mode()
def run_reference(model, case, device, steps, eos_ids):
    inputs = move_inputs(case["official"], device)
    with capture_stages(model) as stages:
        output = model(**inputs, use_cache=True, num_logits_to_keep=1)
    logits = [output.logits[0, -1].float().cpu()]
    tokens = [int(logits[-1].argmax())]
    for _ in range(1, steps):
        if tokens[-1] in eos_ids:
            break
        output = model(
            input_ids=torch.tensor([[tokens[-1]]], device=device),
            input_mode=inputs["input_mode"],
            past_key_values=output.past_key_values,
            use_cache=True,
            num_logits_to_keep=1,
        )
        logits.append(output.logits[0, -1].float().cpu())
        tokens.append(int(logits[-1].argmax()))
    return {"stages": stages, "logits": torch.stack(logits), "tokens": tokens}


def native_logits(model, tokens, positions=None, features=None, meta=None):
    hidden = model.model(tokens, position_ids=positions, features=features, infer_meta=meta)
    # The decoder is unchanged; project only the sampled positions to bound
    # the validation artifact and avoid an entire prompt x vocabulary tensor.
    if meta is not None and meta.sample_indices is not None:
        hidden = hidden.reshape(-1, hidden.shape[-1])[meta.sample_indices].unsqueeze(0)
    else:
        hidden = hidden[:, -1:]
    logits = model.lm_head(hidden).reshape(-1, model.lm_head.weight.shape[0])
    context = get_tp_context()
    if context.world_size > 1:
        shards = [torch.empty_like(logits) for _ in range(context.world_size)]
        dist.all_gather(shards, logits.contiguous(), group=context.group)
        logits = torch.cat(shards, dim=-1)
    return logits.float().cpu()


@torch.inference_mode()
def run_native(model, case, device, steps, chunk_size, eos_ids):
    tokens = torch.tensor([case["tokens"]], device=device)
    with capture_stages(model) as stages:
        full_logits = native_logits(model, tokens, features=case["features"])[0]
    block_size = 64
    cache_len = len(case["tokens"]) + steps
    blocks = math.ceil(cache_len / block_size)
    model.set_kv_caches(model.allocate_kv_caches(blocks, block_size, device), num_slots=1)
    model.model.vision_lora_slots = model.model.vision_lora_slots.to(device)
    state = InferenceBatchState(
        [case["tokens"]],
        steps,
        max_cache_len=cache_len,
        max_prefill_tokens=chunk_size,
        kv_block_size=block_size,
        num_cache_blocks=blocks,
        prompt_features=[case["features"]],
    )
    while state.has_pending_prompts:
        payload = state.build_prefill_payload()
        assert payload is not None
        meta = payload_to_infer_meta(payload, device)
        last = native_logits(
            model,
            payload["input_ids"].to(device).unsqueeze(0),
            payload["position_ids"].to(device).unsqueeze(0),
            payload.get("features"),
            meta,
        )
    logits = [last[0]]
    generated = [int(last[0].argmax())]
    for step in range(1, steps):
        if generated[-1] in eos_ids:
            break
        position = len(case["tokens"]) + step - 1
        state.ensure_decode_blocks([0], [position])
        # A single sequence owns contiguous pages in this bounded fixture.
        meta = InferMeta(
            mode="decode",
            cache_seqlens=torch.tensor([position], dtype=torch.int32, device=device),
            block_table=torch.arange(blocks, dtype=torch.int32, device=device).unsqueeze(0),
            recurrent_slots=torch.zeros(1, dtype=torch.long, device=device),
        )
        last = native_logits(
            model,
            torch.tensor([[generated[-1]]], device=device),
            torch.tensor([[position]], device=device),
            meta=meta,
        )
        logits.append(last[0])
        generated.append(int(last[0].argmax()))
    model.clear_kv_caches()
    return {"stages": stages, "logits": torch.stack(logits), "tokens": generated, "full_logits": full_logits}


def metrics(actual, expected):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    actual, expected = actual.double().flatten(), expected.double().flatten()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = (actual - expected).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "cosine": float(F.cosine_similarity(actual, expected, dim=0)),
    }


def compare(reference, native, tokenizer, min_cosine, min_logits_cosine):
    report = {}
    passed = True
    for name, expected in reference.items():
        actual = native[name]
        stages = {}
        assert expected["stages"].keys() == actual["stages"].keys()
        for key in expected["stages"]:
            dim = 1 if key == "projector" else 0
            stages[key] = metrics(torch.cat(actual["stages"][key], dim), torch.cat(expected["stages"][key], dim))
        step_metrics = [metrics(a, b) for a, b in zip(actual["logits"], expected["logits"])]
        token_equal = actual["tokens"] == expected["tokens"]
        cache_metrics = metrics(actual["logits"][0], actual["full_logits"])
        case_passed = token_equal and all(item["cosine"] >= min_logits_cosine for item in step_metrics)
        case_passed &= all(item["cosine"] >= min_cosine for item in stages.values())
        case_passed &= cache_metrics["cosine"] >= min_logits_cosine
        passed &= case_passed
        report[name] = {
            "passed": case_passed,
            "stages": stages,
            "decode_logits": step_metrics,
            "cache_vs_full_prefill": cache_metrics,
            "tokens_equal": token_equal,
            "official_tokens": expected["tokens"],
            "areno_tokens": actual["tokens"],
            "official_text": tokenizer.decode(expected["tokens"]),
            "areno_text": tokenizer.decode(actual["tokens"]),
        }
    return {"passed": passed, "min_cosine": min_cosine, "min_logits_cosine": min_logits_cosine, "cases": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stream-weights", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--cases", nargs="+", choices=tuple(fixtures()), default=list(fixtures()))
    parser.add_argument("--phase", choices=("reference", "areno", "compare", "all"), default="all")
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--min-logits-cosine", type=float, default=0.999)
    parser.add_argument("--tp-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument(
        "--save-dir", type=Path, help="Save the loaded native model, including original processor/audio assets"
    )
    args = parser.parse_args()
    if args.steps < 1 or args.chunk_size < 1:
        parser.error("steps and chunk-size must be positive")
    if args.save_dir is not None and args.save_dir.exists() and any(args.save_dir.iterdir()):
        parser.error("save-dir must be empty to avoid overwriting an existing checkpoint")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    rank = int(os.environ.get("RANK", "0"))
    if args.tp_size > 1 and args.phase != "compare":
        if args.phase != "areno" or int(os.environ.get("WORLD_SIZE", "1")) != args.tp_size:
            parser.error("TP validation requires torchrun and --phase areno; compare runs in a single process")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    cases = prepare_inputs(processor, args.cases)
    eos_ids = eos_token_ids(args.model_path, processor.tokenizer)
    print("Processor parity passed:", {name: len(case["tokens"]) for name, case in cases.items()}, flush=True)
    suffix = "" if args.tp_size == 1 else f"-tp{args.tp_size}"
    reference_path, native_path = args.output_dir / "reference.pt", args.output_dir / f"areno{suffix}.pt"
    if args.phase in ("reference", "all"):
        reference_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        # This snapshot stores the private field in config.json; passing only
        # attn_implementation to from_pretrained can leave FlashAttention active.
        reference_config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=reference_config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()
        print(
            "Official attention:",
            type(model.model.layers[0].self_attn).__name__,
            type(model.model.embed_tokens_extend.image_embed.img_processor.encoder.layers[0].self_attn).__name__,
            flush=True,
        )
        results = {}
        with stream_weights(model, device, args.stream_weights):
            for name, case in cases.items():
                print("Official:", name, flush=True)
                results[name] = run_reference(model, case, device, args.steps, eos_ids)
                torch.save(results, reference_path)
        del model, results
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.phase in ("areno", "all"):
        set_tp_context(
            TPContext(
                rank=rank,
                world_size=args.tp_size,
                device=device,
                group=dist.group.WORLD if args.tp_size > 1 else None,
                global_rank=rank,
                global_world_size=args.tp_size,
            )
        )
        adapter = Phi4MMAdapter()
        config = adapter.config_from_hf(json.loads((args.model_path / "config.json").read_text()))
        config.attn_backend = "native"
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(config.dtype)
            with skip_torch_init(enabled=True):
                model = adapter.build(config).eval()
        finally:
            torch.set_default_dtype(previous_dtype)
        audit = load_phi4mm_weights(model, args.model_path)
        if rank == 0:
            (args.output_dir / f"checkpoint_audit{suffix}.json").write_text(json.dumps(asdict(audit), indent=2) + "\n")
        if args.save_dir is not None:
            adapter.save_weights(model, args.save_dir, args.model_path)
        results = {}
        with stream_weights(model, device, args.stream_weights):
            for name, case in cases.items():
                print("AReno:", name, flush=True)
                results[name] = run_native(model, case, device, args.steps, args.chunk_size, eos_ids)
                if rank == 0:
                    torch.save(results, native_path)
    if args.phase in ("compare", "all"):
        report = compare(
            torch.load(reference_path, map_location="cpu", weights_only=True),
            torch.load(native_path, map_location="cpu", weights_only=True),
            processor.tokenizer,
            args.min_cosine,
            args.min_logits_cosine,
        )
        report["environment"] = {
            "torch": torch.__version__,
            "device": str(device),
            "stream_weights": args.stream_weights,
            "steps": args.steps,
            "chunk_size": args.chunk_size,
            "tp_size": args.tp_size,
            "reference_source_sha256": {
                name: hashlib.sha256((args.model_path / name).read_bytes()).hexdigest()
                for name in ("modeling_phi4mm.py", "processing_phi4mm.py", "vision_siglip_navit.py", "config.json")
            },
        }
        (args.output_dir / f"report{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not report["passed"]:
            raise SystemExit("Phi4MM parity gate failed; inspect report.json")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
