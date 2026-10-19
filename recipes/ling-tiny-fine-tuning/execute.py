"""Execute the Mac notebook without a browser; retain partial outputs."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="Existing unquantized Ling Tiny checkpoint")
    parser.add_argument("--data-hub", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--output-dir", type=Path, help="New directory for this run")
    args = parser.parse_args()
    if args.model_path:
        os.environ["LING_MODEL_PATH"] = str(args.model_path.expanduser().resolve())
    os.environ["LING_DATA_HUB"] = args.data_hub
    if args.output_dir:
        os.environ["LING_OUTPUT_DIR"] = str(args.output_dir.expanduser().resolve())
    recipe = Path(__file__).resolve().parent
    notebook = nbformat.read(recipe / "Ling-Tiny-SFT-Mac.ipynb", as_version=4)
    # The notebook chooses its run directory; this separate execution directory
    # also survives errors in the first installation/configuration cells.
    from datetime import datetime, timezone

    execution_dir = recipe.parents[1] / "runs" / ("notebook-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    execution_dir.mkdir(parents=True)
    output = execution_dir / "Ling-Tiny-SFT-Mac.executed.ipynb"
    client = NotebookClient(
        notebook, kernel_name="areno-ling-tiny", timeout=None,
        resources={"metadata": {"path": str(recipe.parents[1])}},
        allow_errors=False,
    )

    def checkpoint(**kwargs) -> None:
        nbformat.write(notebook, output)
        print(f"Notebook progress saved: {output}", flush=True)

    client.on_cell_executed = checkpoint
    print(f"Executing notebook; outputs: {output}", flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
        print(f"Notebook saved: {output}", flush=True)


if __name__ == "__main__":
    main()
