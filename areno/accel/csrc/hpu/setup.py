"""Explicit native HPU build; run from the repository root with build_ext --inplace.

The regular CUDA/MLX installation path does not import or execute this file.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup

SOURCE = Path(__file__).resolve().parent


def activation_specs():
    return [
        (name, int(kind), int(backward), int(gated))
        for name, kind, backward, gated in re.findall(
            r"ARENO_ACTIVATION\((\w+), (\d), (\d), (\d)\)", (SOURCE / "activation_ops.inc").read_text()
        )
    ]


def kernel_specs():
    for name, kind, backward, gated in activation_specs():
        yield name, "activation.c", 0, kind, backward, gated
    for name, kind, direction in re.findall(
        r"ARENO_NORMALIZATION\((\w+), (\d), (\d)\)", (SOURCE / "normalization_ops.inc").read_text()
    ):
        yield name, "normalization.c", 1, int(kind), int(direction), 0
    yield "areno_linear_bias", "linear.c", 2, 0, 0, 0
    yield "areno_linear_bias_grad", "linear.c", 2, 0, 1, 0
    for grad_type, suffix in enumerate(("f32", "bf16")):
        yield f"areno_adamw_state_g{suffix}", "optimizer.c", 3, 0, grad_type, 0
        yield f"areno_adamw_master_g{suffix}", "optimizer.c", 3, 1, grad_type, 0
        for bits in (4, 8):
            yield f"areno_adamw{bits}_g{suffix}", "quantized_optimizer.c", 4, bits, grad_type, 0
        yield f"areno_adamw4_factored_g{suffix}", "quantized_optimizer.c", 4, 3, grad_type, 0
    yield "areno_adamw4_stats", "factored_stats.c", 5, 0, 0, 0
    yield "areno_adamw4_invalid", "factored_stats.c", 5, 1, 0, 0
    yield "areno_embedding", "embedding.c", 6, 0, 0, 0
    yield "areno_embedding_grad", "embedding.c", 6, 0, 1, 0
    for kind, name in enumerate(("areno_attention", "areno_attention_packed")):
        yield name, "attention.c", 7, kind, 0, 0
        yield f"{name}_grad", "attention.c", 7, kind, 1, 0
    yield "areno_attention_paged", "attention.c", 7, 2, 0, 0
    yield "areno_cache_owners", "paged_cache.c", 8, 0, 0, 0
    yield "areno_cache_update", "paged_cache.c", 8, 1, 0, 0
    for kind, name in enumerate(("areno_conv", "areno_conv_packed", "areno_conv_decode")):
        yield name, "conv.c", 9, kind, 0, 0
        if kind != 2:
            yield f"{name}_grad", "conv.c", 9, kind, 1, 0
    yield "areno_topk", "topk.c", 10, 0, 0, 0
    yield "areno_topk_grad", "topk.c", 10, 0, 1, 0
    yield "areno_topk_grouped", "topk.c", 10, 1, 0, 0
    for kind, name in enumerate(
        (
            "areno_moe_dense_counts",
            "areno_moe_topk_counts",
            "areno_moe_dense_permute",
            "areno_moe_topk_permute",
            "areno_moe_weight_grad",
        )
    ):
        yield name, "moe.c", 11, kind, 0, 0
    yield "areno_moe_align", "moe_align.c", 12, 0, 0, 0
    yield "areno_moe_weighted", "fused_moe.c", 13, 0, 0, 0
    yield "areno_moe_reduce", "fused_moe.c", 13, 1, 0, 0
    yield "areno_kda_prepare", "kda_prepare.c", 14, 0, 0, 0
    yield "areno_kda_prepare_grad", "kda_prepare.c", 14, 0, 1, 0
    yield "areno_kda", "kda.c", 15, 0, 0, 0
    yield "areno_kda_grad", "kda.c", 15, 0, 1, 0
    yield "areno_state_update", "state_update.c", 16, 0, 0, 0
    for kind, name in enumerate(
        ("areno_seg_la_prefill", "areno_seg_la_decode", "areno_seg_la_mtp", "areno_seg_la_spec")
    ):
        yield name, "seg_la.c", 17, kind, 0, 0


def compile_kernels(build_dir: Path, compiler: str, arch: str) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    declarations = []
    entries = []
    for name, source, family, kind, direction, gated in kernel_specs():
        for dtype, suffix in enumerate(("f32", "bf16", "f16")):
            if family == 3 and (dtype == 2 or (kind == 1 and dtype != 1)):
                continue
            if family == 4 and dtype == 2:
                continue
            if family == 5 and (dtype == 2 or (kind == 1 and dtype != 0)):
                continue
            if family == 8 and kind == 0 and dtype != 0:
                continue
            if family == 11 and (kind < 2 or kind == 4) and dtype != 0:
                continue
            if family in (12, 14, 15, 17) and dtype != 0:
                continue
            guid = f"{name}_{suffix}"
            target = build_dir / f"{guid}.o"
            subprocess.run(
                [
                    compiler,
                    f"-march={arch}",
                    "-O2",
                    "-Wall",
                    "-Werror",
                    f"-I{SOURCE}",
                    f"-DARENO_KIND={kind}",
                    f"-DARENO_BACKWARD={direction}",
                    f"-DARENO_DIRECTION={direction}",
                    f"-DARENO_GRAD_DTYPE={direction}",
                    f"-DARENO_GATED={gated}",
                    f"-DARENO_DTYPE={dtype}",
                    "-c",
                    str(SOURCE / source),
                    "-o",
                    str(target),
                ],
                check=True,
            )
            data = target.read_bytes()
            if not data.startswith(b"\x7fELF"):
                raise RuntimeError(f"TPC compiler did not produce an ELF kernel: {target}")
            content = ",".join(f"0x{byte:02x}" for byte in data)
            declarations.append(f"static const unsigned char {guid}[] = {{{content}}};")
            entries.append(
                f'{{"{guid}", {guid}, sizeof({guid}), {dtype}, {int(direction == 1)}, {gated}, {family}, {kind}, {direction}}}'
            )
    header = "\n".join(declarations) + "\nstatic const KernelBinary binaries[] = {\n" + ",\n".join(entries) + "\n};\n"
    (build_dir / "kernel_binaries.h").write_text(header)


def build_extensions():
    if platform.system() != "Linux":
        raise RuntimeError("Native HPU kernels must be built on Linux with the Gaudi TPC SDK and PyTorch bridge")
    compiler = shutil.which(os.environ.get("TPC_COMPILER", "tpc-clang"))
    if compiler is None:
        raise RuntimeError("tpc-clang was not found; install the Gaudi TPC SDK or set TPC_COMPILER")
    sdk = Path(os.environ.get("TPC_INCLUDE_DIR", "/usr/lib/habanatools/include"))
    for header in ("gc_interface.h", "tpc_kernel_lib_interface.h"):
        if not (sdk / header).is_file():
            raise RuntimeError(f"Missing TPC SDK header {sdk / header}; set TPC_INCLUDE_DIR")
    arch = os.environ.get("ARENO_HPU_ARCH", "gaudi2")
    if arch not in {"gaudi2", "gaudi3"}:
        raise ValueError("ARENO_HPU_ARCH must be gaudi2 or gaudi3")
    lazy_mode = os.environ.get("PT_HPU_LAZY_MODE")
    if lazy_mode not in {"0", "1"}:
        raise ValueError("Set PT_HPU_LAZY_MODE=0 (eager) or 1 (lazy) for both build and execution")

    from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    build_dir = Path("build") / f"areno_hpu_{arch}_{lazy_mode}"
    plugin = "habana_pytorch_plugin" if lazy_mode == "1" else "habana_pytorch2_plugin"
    lib_dir = get_lib_dir()

    class BuildHpuExtension(BuildExtension):
        def build_extensions(self):
            compile_kernels(build_dir, compiler, arch)
            super().build_extensions()

    kernel_library = Extension(
        "areno.accel._areno_hpu_kernels",
        sources=[str(SOURCE / "kernel_library.cpp")],
        include_dirs=[str(sdk), str(build_dir.resolve())],
        define_macros=[("ARENO_TPC_DEVICE", f"tpc_lib_api::DEVICE_ID_{arch.upper()}")],
        language="c++",
        extra_compile_args=["-O2", "-std=c++17"],
    )
    binding = CppExtension(
        "areno.accel._areno_accel_hpu",
        sources=[
            str(SOURCE / name)
            for name in (
                "extension.cpp",
                "normalization.cpp",
                "linear.cpp",
                "optimizer.cpp",
                "quantized_optimizer.cpp",
                "factored_optimizer.cpp",
                "embedding.cpp",
                "attention.cpp",
                "paged_cache.cpp",
                "grouped_linear.cpp",
                "conv.cpp",
                "topk.cpp",
                "moe.cpp",
                "moe_align.cpp",
                "fused_moe.cpp",
                "kda.cpp",
                "seg_la.cpp",
            )
        ],
        include_dirs=[get_include_dir(), "/usr/include/habanalabs"],
        libraries=[plugin],
        library_dirs=[lib_dir],
        runtime_library_dirs=[lib_dir],
        define_macros=[("ARENO_HPU_LAZY_MODE", lazy_mode)],
        # BuildExtension selects the C++ standard required by the installed torch.
        extra_compile_args=["-O2"],
    )
    return [kernel_library, binding], {"build_ext": BuildHpuExtension}


if __name__ == "__main__":
    extensions, commands = build_extensions()
    setup(ext_modules=extensions, cmdclass=commands)
