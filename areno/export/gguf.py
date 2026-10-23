"""Write live canonical model tensors directly to an Ollama GGUF package.

No converter subprocess or intermediate checkpoint is used. Architecture
handlers register their exact tensor contract before any output is written.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

_ARCHITECTURES: dict[str, type] = {}


def register_architecture(name: str):
    def register(cls):
        if name in _ARCHITECTURES:
            raise ValueError(f"duplicate GGUF architecture: {name}")
        _ARCHITECTURES[name] = cls
        return cls

    return register


@register_architecture("qwen3")
class Qwen3Export:
    """Dense, standard-RoPE Qwen3 text models in canonical weight layout."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        if config.get("rope_scaling") or config.get("use_sliding_window"):
            raise ValueError("GGUF Qwen3 export does not support scaled RoPE or sliding-window attention")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("GGUF Qwen3 export requires hidden_act='silu'")
        if not config.get("qk_layernorm", config.get("qk_norm", True)):
            raise ValueError("GGUF Qwen3 export requires Q/K RMSNorm")

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        c = self.config
        hidden, intermediate = int(c["hidden_size"]), int(c["intermediate_size"])
        head_dim = int(c.get("head_dim", hidden // int(c["num_attention_heads"])))
        q_size = head_dim * int(c["num_attention_heads"])
        kv_size = head_dim * int(c["num_key_value_heads"])
        shapes = {"model.embed_tokens.weight": (int(c["vocab_size"]), hidden), "model.norm.weight": (hidden,)}
        if not c.get("tie_word_embeddings", False):
            shapes["lm_head.weight"] = shapes["model.embed_tokens.weight"]
        for index in range(int(c["num_hidden_layers"])):
            prefix = f"model.layers.{index}"
            entries = {
                "input_layernorm.weight": (hidden,),
                "post_attention_layernorm.weight": (hidden,),
                "self_attn.q_proj.weight": (q_size, hidden),
                "self_attn.k_proj.weight": (kv_size, hidden),
                "self_attn.v_proj.weight": (kv_size, hidden),
                "self_attn.o_proj.weight": (hidden, q_size),
                "self_attn.q_norm.weight": (head_dim,),
                "self_attn.k_norm.weight": (head_dim,),
                "mlp.gate_proj.weight": (intermediate, hidden),
                "mlp.up_proj.weight": (intermediate, hidden),
                "mlp.down_proj.weight": (hidden, intermediate),
            }
            if c.get("attention_bias", c.get("qkv_bias", False)):
                entries.update(
                    {
                        f"self_attn.{name}_proj.bias": (size,)
                        for name, size in (("q", q_size), ("k", kv_size), ("v", kv_size))
                    }
                )
            shapes.update({f"{prefix}.{name}": shape for name, shape in entries.items()})
        return shapes

    def metadata(self, writer) -> None:
        c = self.config
        hidden, heads = int(c["hidden_size"]), int(c["num_attention_heads"])
        head_dim = int(c.get("head_dim", hidden // heads))
        writer.add_name("AReno trained Qwen3")
        writer.add_context_length(int(c["max_position_embeddings"]))
        writer.add_embedding_length(hidden)
        writer.add_block_count(int(c["num_hidden_layers"]))
        writer.add_feed_forward_length(int(c["intermediate_size"]))
        writer.add_head_count(heads)
        writer.add_head_count_kv(int(c["num_key_value_heads"]))
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)
        writer.add_rope_dimension_count(head_dim)
        writer.add_rope_freq_base(float(c.get("rope_theta", 1_000_000)))
        writer.add_layer_norm_rms_eps(float(c["rms_norm_eps"]))


def _write_tokenizer(writer, model_path: Path, config: dict[str, Any]) -> None:
    import gguf

    tokenizer = json.loads((model_path / "tokenizer.json").read_text())
    if tokenizer["model"]["type"] != "BPE":
        raise ValueError("Qwen3 GGUF export requires its standard BPE tokenizer")
    # The qwen2 pre-tokenizer is also used by standard Qwen3 text models.
    expected_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    pre = tokenizer.get("pre_tokenizer", {})
    parts = pre.get("pretokenizers", [])
    if not (
        pre.get("type") == "Sequence"
        and len(parts) == 2
        and parts[0].get("type") == "Split"
        and parts[0].get("pattern", {}).get("Regex") == expected_regex
        and parts[0].get("behavior") == "Isolated"
        and not parts[0].get("invert", False)
        and parts[1].get("type") == "ByteLevel"
        and not parts[1].get("add_prefix_space", False)
        and not parts[1].get("use_regex", True)
    ):
        raise ValueError("unrecognized Qwen3 tokenizer pre-tokenizer; refusing to change tokenization")
    count = int(config["vocab_size"])
    tokens = [f"[PAD{index}]" for index in range(count)]
    types = [int(gguf.TokenType.UNUSED)] * count
    for token, index in tokenizer["model"]["vocab"].items():
        if not 0 <= index < count:
            raise ValueError(f"token id {index} is outside model vocabulary")
        tokens[index], types[index] = token, int(gguf.TokenType.NORMAL)
    for token in tokenizer.get("added_tokens", []):
        index = int(token["id"])
        if not 0 <= index < count:
            raise ValueError(f"added token id {index} is outside model vocabulary")
        tokens[index] = token["content"]
        types[index] = int(gguf.TokenType.CONTROL if token.get("special") else gguf.TokenType.USER_DEFINED)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    special = gguf.SpecialVocab(model_path, load_merges=True, n_vocab=count)
    if not special.merges:
        raise ValueError("Qwen3 tokenizer has no BPE merges")
    # Recent Ollama versions prefer GGUF's Jinja template over Modelfile's Go
    # template. Both must describe the same text-only, thinking-off profile.
    special.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}"
    )
    special.add_special_token.update(bos=False, eos=False)
    special.add_to_gguf(writer)


def qwen3_modelfile(context_length: int) -> str:
    # This chat template implements ordinary text messages with thinking off,
    # matching apply_chat_template(..., enable_thinking=False). Tool calling
    # is deliberately not advertised by this export profile.
    return (
        "FROM ./model.gguf\n"
        'TEMPLATE """{{ range .Messages }}<|im_start|>{{ .Role }}\n'
        "{{ .Content }}<|im_end|>\n"
        '{{ end }}<|im_start|>assistant\n<think>\n\n</think>\n\n"""\n'
        f"PARAMETER num_ctx {min(context_length, 4096)}\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|endoftext|>"\n'
    )


class GGUFPackage:
    """Plan tensor headers first; then consume one canonical tensor at a time."""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        tensor_shapes: dict[str, tuple[int, ...]],
        *,
        file_type: str = "f16",
        lora_merged: bool = False,
    ):
        import gguf

        if file_type not in {"f16", "q8_0"}:
            raise ValueError("GGUF file_type must be 'f16' or 'q8_0'")
        self.source = Path(model_path)
        self.destination = Path(output_dir).absolute()
        if self.destination.exists():
            raise FileExistsError(f"GGUF output directory already exists: {self.destination}")
        self.config = json.loads((self.source / "config.json").read_text())
        model_type = self.config.get("model_type")
        if model_type not in _ARCHITECTURES:
            raise ValueError(
                f"unsupported GGUF model architecture: {model_type!r}; supported: {sorted(_ARCHITECTURES)}"
            )
        architecture = _ARCHITECTURES[model_type](self.config)
        expected = architecture.tensor_shapes()
        if tensor_shapes != expected:
            missing, unexpected = (
                sorted(expected.keys() - tensor_shapes.keys()),
                sorted(tensor_shapes.keys() - expected.keys()),
            )
            mismatched = [key for key in expected.keys() & tensor_shapes.keys() if expected[key] != tensor_shapes[key]]
            raise ValueError(
                f"GGUF tensor contract mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}, shapes={mismatched[:5]}"
            )
        self.file_type = file_type
        self.lora_merged = lora_merged
        self.shapes = tensor_shapes
        self.pending = list(tensor_shapes)
        self.names: dict[str, str] = {}
        self.types: dict[str, Any] = {}
        name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, int(self.config["num_hidden_layers"]))
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(prefix=".areno-gguf-", dir=self.destination.parent)
        try:
            self.stage = Path(self.temporary.name)
            self.writer = gguf.GGUFWriter(self.stage / "model.gguf", "qwen3")
            architecture.metadata(self.writer)
            _write_tokenizer(self.writer, self.source, self.config)
            ft = gguf.LlamaFileType.MOSTLY_F16 if file_type == "f16" else gguf.LlamaFileType.MOSTLY_Q8_0
            self.writer.add_file_type(int(ft))
            self.writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
            for name, shape in tensor_shapes.items():
                target = name_map.get_name(name, try_suffixes=(".weight", ".bias"))
                if target is None or target in self.names.values():
                    raise ValueError(f"missing or duplicate GGUF tensor mapping for {name}")
                self.names[name] = target
                qtype = gguf.GGMLQuantizationType.F32 if len(shape) == 1 else gguf.GGMLQuantizationType.F16
                if file_type == "q8_0" and len(shape) >= 2 and shape[-1] % 32 == 0:
                    qtype = gguf.GGMLQuantizationType.Q8_0
                self.types[name] = qtype
                if qtype == gguf.GGMLQuantizationType.Q8_0:
                    packed = (*shape[:-1], shape[-1] // 32 * 34)
                    self.writer.add_tensor_info(target, packed, np.dtype(np.uint8), math.prod(packed), raw_dtype=qtype)
                else:
                    dtype = np.dtype(np.float32 if qtype == gguf.GGMLQuantizationType.F32 else np.float16)
                    self.writer.add_tensor_info(target, shape, dtype, math.prod(shape) * dtype.itemsize)
            self.writer.write_header_to_file()
            self.writer.write_kv_data_to_file()
            self.writer.write_ti_data_to_file()
        except BaseException:
            self.abort()
            raise

    def write(self, name: str, tensor: np.ndarray) -> None:
        import gguf

        if not self.pending or name != self.pending[0]:
            raise ValueError(f"out-of-order GGUF tensor: {name}")
        if tuple(tensor.shape) != self.shapes[name] or not np.isfinite(tensor).all():
            raise ValueError(f"invalid shape or non-finite GGUF tensor: {name}")
        from gguf.quants import quantize

        encoded = np.ascontiguousarray(quantize(tensor, self.types[name]))
        if self.types[name] != gguf.GGMLQuantizationType.Q8_0 and not np.isfinite(encoded).all():
            raise ValueError(f"GGUF precision conversion overflow: {name}")
        self.writer.write_tensor_data(encoded)
        self.pending.pop(0)

    def finish(self) -> str:
        if self.pending:
            raise ValueError(f"GGUF tensors were not written: {self.pending[:5]}")
        self.writer.close()
        (self.stage / "Modelfile").write_text(qwen3_modelfile(int(self.config["max_position_embeddings"])))
        digest = hashlib.sha256()
        with (self.stage / "model.gguf").open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest = {
            "format": "gguf",
            "architecture": self.config["model_type"],
            "file_type": self.file_type,
            "source": str(self.source.resolve()),
            "lora_merged": self.lora_merged,
            "tensor_count": len(self.shapes),
            "sha256": digest.hexdigest(),
            "chat_profile": "text, thinking disabled",
            "intermediate_checkpoint": False,
        }
        (self.stage / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.stage.rename(self.destination)
        self.temporary.cleanup()
        return str(self.destination / "model.gguf")

    def abort(self) -> None:
        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
        temporary = getattr(self, "temporary", None)
        if temporary is not None:
            temporary.cleanup()
