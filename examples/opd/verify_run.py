"""Verify a bounded OPD run's logged stages, token counts, and checkpoint files.

The verifier deliberately fails closed: a successful process exit without a
student checkpoint and OPD loss is *not* an end-to-end smoke-test success.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

_REQUIRED_STAGES = (
    "stage=rollout_start", "stage=logprob_score_end",
    "stage=train_end", "stage=save_checkpoint_end",
)


def read_step_stats(log_text: str) -> list[dict]:
    """Extract repr(dict) train summaries from the existing AReno INFO log."""
    stats: list[dict] = []
    for line in log_text.splitlines():
        if "train_stats=" not in line:
            continue
        try:
            raw = line.split("train_stats=", 1)[1]
            entry = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(entry, dict):
            stats.append(entry)
    return stats


def checkpoint_is_nonempty(path: Path) -> bool:
    """Require saved model metadata and nonempty safetensors weights."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    try:
        json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    weights = list(path.glob("*.safetensors"))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            used = set(index["weight_map"].values())
        except (KeyError, TypeError, ValueError, OSError):
            return False
        if not used or not all((path / name).is_file() and (path / name).stat().st_size > 0 for name in used):
            return False
    return True


def verify_run(output_dir: Path, *, steps: int = 2) -> dict:
    """Return a machine-readable success report or raise ValueError."""
    output_dir = Path(output_dir)
    if steps <= 0:
        raise ValueError("steps must be positive")
    logfile = output_dir / "run.log"
    if not logfile.is_file():
        raise ValueError(f"missing run.log in {output_dir}")
    log = logfile.read_text(encoding="utf-8", errors="replace")
    positions = [log.find(marker) for marker in _REQUIRED_STAGES]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("missing or out-of-order rollout / teacher / train / checkpoint log markers")
    if len(re.findall(r"stage=save_checkpoint_end\b", log)) < steps:
        raise ValueError("fewer checkpoint-save completions than requested steps")
    stats = read_step_stats(log)
    valid = []
    for entry in stats:
        if "opd_loss" not in entry:
            continue
        try:
            loss = float(entry["opd_loss"])
            tokens = float(entry["opd_response_token_count"])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(loss) and math.isfinite(tokens) and tokens > 0:
            valid.append({"opd_loss": loss, "response_tokens": int(tokens)})
    if len(valid) < steps:
        raise ValueError("missing finite OPD loss or nonzero response tokens in training logs")

    ckpts = []
    for i in range(1, steps + 1):
        directory = output_dir / "checkpoints" / f"step_{i:06d}"
        if not checkpoint_is_nonempty(directory):
            raise ValueError(f"missing or incomplete saved checkpoint: {directory}")
        ckpts.append(str(directory))
    return {"ok": True, "steps": steps, "checkpoint_paths": ckpts,
            "train_steps": valid[:steps]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        result = verify_run(args.output_dir, steps=args.steps)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"OPD run verification failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
