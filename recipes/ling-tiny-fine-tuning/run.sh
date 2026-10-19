#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This launcher requires an Apple Silicon Mac. For CUDA, use Ling-Tiny-SFT-Colab.ipynb." >&2
    exit 1
fi

recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
recipe_python="${PYTHON:-python3}"
recipe_venv="${ARENO_RECIPE_VENV:-$recipe_dir/.venv}"

if [[ ! -x "$recipe_venv/bin/python" ]]; then
    "$recipe_python" -m venv "$recipe_venv"
fi
"$recipe_venv/bin/python" "$recipe_dir/install.py"
"$recipe_venv/bin/python" -m ipykernel install --sys-prefix \
    --name areno-ling-tiny --display-name "AReno Ling Tiny"
export ARENO_RECIPE_READY=1
exec "$recipe_venv/bin/python" "$recipe_dir/execute.py" "$@"
