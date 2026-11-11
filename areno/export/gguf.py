"""Write live model tensors directly to an atomic GGUF/Ollama package."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from areno.export.architectures import TensorRule, build_architecture, collection_blocks
from areno.export.tokenizer import modelfile, write_tokenizer


class _TensorFile:
    """GGUF headers plus bounded writes to reserved final tensor regions.

    The Python GGUF writer normally accepts a complete tensor. Reserving its
    documented, aligned payload layout lets us stream experts to their final
    positions without keeping all experts in memory or writing a checkpoint.
    """

    def __init__(self, path: Path, arch: str, specs: dict, metadata, file_type: str):
        import gguf

        self.specs = specs
        self.types = {}
        self.offsets = {}
        self.bytes_written = dict.fromkeys(specs, 0)
        writer = gguf.GGUFWriter(path, arch, endianess=gguf.GGUFEndian.LITTLE)
        try:
            metadata(writer)
            writer.add_file_type(
                int(gguf.LlamaFileType.MOSTLY_F16 if file_type == "f16" else gguf.LlamaFileType.MOSTLY_Q8_0)
            )
            writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
            offset = 0
            for name, spec in specs.items():
                shape = spec.shape
                qtype = (
                    gguf.GGMLQuantizationType.F32
                    if len(shape) <= 1 or spec.force_f32
                    else gguf.GGMLQuantizationType.F16
                )
                if file_type == "q8_0" and not spec.force_f32 and len(shape) in {2, 3} and shape[-1] % 32 == 0:
                    qtype = gguf.GGMLQuantizationType.Q8_0
                self.types[name] = qtype
                if qtype == gguf.GGMLQuantizationType.Q8_0:
                    encoded_shape = (*shape[:-1], shape[-1] // 32 * 34)
                    nbytes, dtype = math.prod(encoded_shape), np.dtype(np.uint8)
                else:
                    encoded_shape = shape
                    dtype = np.dtype(np.float32 if qtype == gguf.GGMLQuantizationType.F32 else np.float16)
                    nbytes = math.prod(shape) * dtype.itemsize
                writer.add_tensor_info(name, encoded_shape, dtype, nbytes, raw_dtype=qtype)
                self.offsets[name] = (offset, nbytes)
                offset += gguf.GGUFWriter.ggml_pad(nbytes, writer.data_alignment)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
        finally:
            writer.close()
        self.stream = path.open("r+b")
        try:
            self.data_start = gguf.GGUFWriter.ggml_pad(self.stream.seek(0, 2), writer.data_alignment)
            self.stream.truncate(self.data_start + offset)
        except BaseException:
            self.stream.close()
            raise

    def write(self, name: str, value: np.ndarray, expert: int | None = None) -> None:
        import gguf
        from gguf.quants import quantize

        spec = self.specs[name]
        expected = spec.shape if expert is None else spec.shape[1:]
        if tuple(value.shape) != expected or not np.isfinite(value).all():
            raise ValueError(f"invalid shape or non-finite transformed GGUF tensor: {name}")
        encoded = np.ascontiguousarray(quantize(value, self.types[name]))
        if self.types[name] != gguf.GGMLQuantizationType.Q8_0 and not np.isfinite(encoded).all():
            raise ValueError(f"GGUF precision conversion overflow: {name}")
        if encoded.dtype.itemsize > 1:
            encoded = encoded.astype(encoded.dtype.newbyteorder("<"), copy=False)
        offset, nbytes = self.offsets[name]
        relative = 0 if expert is None else expert * encoded.nbytes
        if relative + encoded.nbytes > nbytes:
            raise ValueError(f"GGUF tensor write exceeds reserved payload: {name}")
        self.stream.seek(self.data_start + offset + relative)
        encoded.tofile(self.stream)
        self.bytes_written[name] += encoded.nbytes

    def close(self) -> None:
        self.stream.close()


class GGUFPackage:
    """Validate a registered architecture, then consume live canonical tensors."""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        tensor_shapes: dict[str, tuple[int, ...]],
        *,
        file_type: str = "f16",
        lora_merged: bool = False,
        model_type: str | None = None,
    ):
        if file_type not in {"f16", "q8_0"}:
            raise ValueError("GGUF file_type must be 'f16' or 'q8_0'")
        self.source = Path(model_path)
        self.destination = Path(output_dir).absolute()
        if self.destination.exists():
            raise FileExistsError(f"GGUF output directory already exists: {self.destination}")
        self.config = json.loads((self.source / "config.json").read_text())
        self.architecture = build_architecture(self.config, tensor_shapes, model_type=model_type)
        architecture = self.architecture
        expected = architecture.shapes
        if tensor_shapes != expected:
            missing, unexpected = (
                sorted(expected.keys() - tensor_shapes.keys()),
                sorted(tensor_shapes.keys() - expected.keys()),
            )
            mismatched = [key for key in expected.keys() & tensor_shapes.keys() if expected[key] != tensor_shapes[key]]
            raise ValueError(
                f"GGUF tensor contract mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}, shapes={mismatched[:5]}"
            )
        self.file_type, self.lora_merged, self.shapes = file_type, lora_merged, tensor_shapes
        self.pending = [
            (name, index, block_shape)
            for name, shape in tensor_shapes.items()
            for index, _, block_shape in collection_blocks(name, shape)
        ]
        self.file: _TensorFile | None = None
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(prefix=".areno-gguf-", dir=self.destination.parent)
        self.stage = Path(self.temporary.name)
        try:
            specs = {}
            for rules in architecture.rules.values():
                for rule in rules:
                    if rule.name in specs and specs[rule.name].shape != rule.shape:
                        raise ValueError(f"inconsistent GGUF output tensor: {rule.name}")
                    specs[rule.name] = rule
            for name, value in architecture.extra.items():
                if name in specs:
                    raise ValueError(f"duplicate generated GGUF tensor: {name}")
                specs[name] = TensorRule(name, value.shape, force_f32=True)

            def metadata(writer):
                architecture.metadata(writer)
                self.chat_profile = write_tokenizer(writer, self.source, architecture)

            self.file = _TensorFile(
                self.stage / "model.gguf",
                architecture.arch,
                specs,
                metadata,
                file_type,
            )
            for name, value in architecture.extra.items():
                self.file.write(name, value)
        except BaseException:
            self.abort()
            raise

    def write(self, name: str, tensor: np.ndarray, *, expert: int | None = None) -> None:
        if not self.pending or (name, expert) != self.pending[0][:2]:
            raise ValueError(f"out-of-order GGUF tensor: {name}, expert={expert}")
        if tuple(tensor.shape) != self.pending[0][2] or not np.isfinite(tensor).all():
            raise ValueError(f"invalid shape or non-finite GGUF tensor: {name}")
        for rule in self.architecture.rules[name]:
            value = rule.transform(tensor) if rule.transform else tensor
            self.file.write(rule.name, value, expert if rule.packed_experts else rule.expert)
        self.pending.pop(0)

    def finish(self) -> str:
        if self.pending:
            raise ValueError(f"GGUF tensors were not written: {self.pending[:5]}")
        for name, (_, nbytes) in self.file.offsets.items():
            if self.file.bytes_written[name] != nbytes:
                raise ValueError(f"incomplete GGUF output tensor: {name}")
        self.file.close()
        (self.stage / "Modelfile").write_text(modelfile(self.architecture))
        digest = hashlib.sha256()
        with (self.stage / "model.gguf").open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest = {
            "format": "gguf",
            "architecture": self.architecture.arch,
            "file_type": self.file_type,
            "source": str(self.source.resolve()),
            "lora_merged": self.lora_merged,
            "tensor_count": len(self.file.specs),
            "sha256": digest.hexdigest(),
            "chat_profile": self.chat_profile,
            "modalities": ["text"],
            "intermediate_checkpoint": False,
        }
        (self.stage / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.stage.rename(self.destination)
        self.temporary.cleanup()
        return str(self.destination / "model.gguf")

    def abort(self) -> None:
        if self.file is not None:
            self.file.close()
        temporary = getattr(self, "temporary", None)
        if temporary is not None:
            temporary.cleanup()
