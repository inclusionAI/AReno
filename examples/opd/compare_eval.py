"""Compare held-out initial vs OPD-updated student predictions with paired CI.

Control is an optional separately evaluated UNCHANGED student checkpoint.
This is not a matched optimizer-zero training run, which AReno does not yet
support (CLI requires positive learning rate and OPD coefficient).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_core import aligned_rows, paired_bootstrap_deltas, read_prediction_file, summary_from_rows


def compare(before: dict, after: dict, control: dict | None = None, *, bootstrap_replicates: int = 2000) -> dict:
    b, a = aligned_rows(before, after)
    # Ensure generation settings and tokenizers are genuinely comparable.
    if before.get("tokenizer") != after.get("tokenizer"):
        raise ValueError("evaluation tokenizers differ; token-level NLL is not directly comparable")
    for field in ("max_new_tokens", "decoding", "disable_thinking"):
        if before.get(field) != after.get(field):
            raise ValueError(f"evaluation settings differ: {field}")
    s_b = summary_from_rows(b)
    s_a = summary_from_rows(a)
    effect = paired_bootstrap_deltas(b, a, replicates=bootstrap_replicates)
    sources = {"before": before.get("evidence_kind", "unverified"),
               "after": after.get("evidence_kind", "unverified")}
    result = {"schema_version": 1,
              "evidence_kind": "real_model_comparison" if all(v == "real_model_inference" for v in sources.values()) else "synthetic_or_unverified_comparison",
              "source_evidence_kinds": sources,
              "dataset_sha256": before["dataset_sha256"],
              "models": {"before": before.get("model"), "after": after.get("model")},
              "before": s_b, "after": s_a,
              "delta": {**effect,
                        "answer_nll_per_token": s_a["answer_nll_per_token"] - s_b["answer_nll_per_token"],
                        "answer_perplexity": s_a["answer_perplexity"] - s_b["answer_perplexity"]},
              "no_update_control": {"kind": "identity_reference", "note": "Unchanged pretrained student; not a matched no-update training run.", "summary": s_b}}
    if control is not None:
        c, _ = aligned_rows(control, before)
        result["source_evidence_kinds"]["control"] = control.get("evidence_kind", "unverified")
        if result["evidence_kind"] == "real_model_comparison" and control.get("evidence_kind") != "real_model_inference":
            result["evidence_kind"] = "synthetic_or_unverified_comparison"
        if control.get("tokenizer") != before.get("tokenizer"):
            raise ValueError("control tokenizer differs from baseline")
        for field in ("max_new_tokens", "decoding", "disable_thinking"):
            if control.get(field) != before.get(field):
                raise ValueError(f"control settings differ: {field}")
        summary_c = summary_from_rows(c)
        result["no_update_control"] = {
            "kind": "independent_unchanged_checkpoint_evaluation",
            "note": "Independent inference on the unchanged model; does not execute OPD optimizer with zero updates.",
            "model": control.get("model"), "summary": summary_c,
            "delta_exact_match_vs_before": summary_c["exact_match"] - s_b["exact_match"],
        }
    return result


def render_markdown(report: dict) -> str:
    b, a, delta = report["before"], report["after"], report["delta"]
    return ("# OPD Stage 7 — held-out model evaluation\n\n"
            f"Evidence: **{report['evidence_kind']}**\n\n"
            f"Dataset SHA256: `{report['dataset_sha256']}`\n\n"
            "| Metric | Before OPD | After OPD | Difference (after − before) |\n"
            "|---|---:|---:|---:|\n"
            f"| Exact match | {b['exact_match']:.4f} | {a['exact_match']:.4f} | {delta['delta_exact_match']:+.4f} |\n"
            f"| Gold answer NLL/token | {b['answer_nll_per_token']:.4f} | {a['answer_nll_per_token']:.4f} | {delta['answer_nll_per_token']:+.4f} |\n"
            f"| Gold answer perplexity | {b['answer_perplexity']:.4f} | {a['answer_perplexity']:.4f} | {delta['answer_perplexity']:+.4f} |\n\n"
            f"Paired bootstrap 95% CI for exact-match change: {delta['bootstrap_95pct_ci']}. "
            "With small evaluation samples, interpret this interval cautiously.\n\n"
            f"Control: {report['no_update_control']['kind']} — {report['no_update_control']['note']}\n\n"
            "This evaluation is separate from the OPD training objective and does not establish broad model quality improvements.\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", required=True, type=Path)
    p.add_argument("--after", required=True, type=Path)
    p.add_argument("--control", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(argv)
    before, after = read_prediction_file(args.before), read_prediction_file(args.after)
    control = read_prediction_file(args.control) if args.control else None
    report = compare(before, after, control, bootstrap_replicates=args.bootstrap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = args.output_dir / "comparison.json", args.output_dir / "comparison.md"
    if json_path.exists() or md_path.exists():
        raise SystemExit("Stage 7 comparison already exists; use a fresh --output-dir")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
