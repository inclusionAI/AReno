"""Install the recipe into the invoking Python environment."""

import os
import platform
import subprocess
import sys
from pathlib import Path

MLX_LM_REVISION = "1840cbba9cb3f8125b8cf7e137606256a2f162d1"


def install() -> None:
    root = Path(__file__).resolve().parents[2]
    system = platform.system()
    if system == "Darwin" and platform.machine().lower() != "arm64":
        raise RuntimeError("This recipe requires Apple Silicon on macOS.")
    if system not in {"Darwin", "Linux"}:
        raise RuntimeError("Use Apple Silicon macOS or Linux with NVIDIA CUDA.")
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run(pip + ["setuptools>=69", "wheel", "psutil", "ninja", "packaging"], check=True)
    if system == "Linux":
        # Hosted runtimes already provide a CUDA-compatible torch installation.
        subprocess.run([sys.executable, "-c", (
            "import torch; from packaging.version import Version; "
            "assert Version(torch.__version__.split('+')[0]) >= Version('2.6'); "
            "assert torch.cuda.is_available(), 'Select an NVIDIA GPU runtime first'"
        )], check=True)
    env = dict(os.environ)
    env.setdefault("MAX_JOBS", "4")
    subprocess.run(pip + ["-e", str(root), "--no-build-isolation"], env=env, check=True)
    if system == "Darwin":
        # The released minimum in AReno does not yet contain the Ling adapter.
        subprocess.run(pip + [f"mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@{MLX_LM_REVISION}"], check=True)
        subprocess.run([sys.executable, "-c", "import mlx_lm.models.bailing_moe_v3"], check=True)
    subprocess.run(pip + ["nbclient>=0.10,<1", "nbformat>=5.10,<6", "ipykernel>=6,<8"], check=True)


if __name__ == "__main__":
    install()
