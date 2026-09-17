"""Check CANN's generated-header boundary, without emulating device kernels."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "areno/accel/csrc/npu"
ENTRY = re.compile(
    r"(?P<template>template <[^\n]+>\s*)?__global__\s+__aicore__\s+void\s+"
    r"(?P<name>\w+)\((?P<args>[^)]*)\)\s*\{"
)


def test_kernel_entries_are_visible_to_generated_host_launchers():
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the generated-header contract")
    declarations = ["#include <cstdint>\nusing GM_ADDR = uint8_t*;"]
    calls = []
    for path in sorted(ROOT.glob("*_kernel.cpp")):
        source = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", path.read_text())
        entries = list(ENTRY.finditer(source))
        assert entries, f"no kernel entries found in {path.name}"
        for entry in entries:
            prefix = source[: entry.start()]
            assert prefix.count("{") == prefix.count("}"), (
                f"{path.name}: {entry['name']} must be global so CANN can call *_origin "
                "and host code can resolve the generated launch overload"
            )
            template = entry["template"] or ""
            # Compile the declarations in isolation, just as CANN includes its
            # combined header before the source's own headers and namespaces.
            signature = f"void {entry['name']}({entry['args']});"
            declarations.append(template + signature)
            declarations.append(template + f"void {entry['name']}(uint32_t, void*, void*, {entry['args']});")
            specialization = ""
            if template:
                parameters = template[template.index("<") + 1 : template.rindex(">")].split(",")
                specialization = (
                    "<"
                    + ", ".join(
                        "float" if parameter.strip().startswith("typename ") else "0" for parameter in parameters
                    )
                    + ">"
                )
            arguments = ["nullptr" if "GM_ADDR" in parameter else "0" for parameter in entry["args"].split(",")]
            calls.append(f"{entry['name']}{specialization}(1, nullptr, nullptr, {', '.join(arguments)});")
    # Host launch helpers live in areno_npu, while the injected overloads are
    # global. Verify that ordinary lookup finds the configuration-argument form.
    unit = "\n".join(declarations) + "\nnamespace areno_npu { void check() {\n" + "\n".join(calls) + "\n}}"
    result = subprocess.run(
        [compiler, "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
        input=unit,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
