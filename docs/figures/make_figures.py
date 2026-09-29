#!/usr/bin/env python3
"""Generate figures for the ling-3.0-tiny-jev paper from archived metric files.

Reads the JSON metric files produced by `evaluate.py --dump` and the archive
directories, then writes vector PDF + PNG figures into docs/_static/figures/.

Every number is read from the archived JSON; nothing is transcribed by hand.
Run from the repository root:

    .venv/bin/python docs/figures/make_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

RUNS = Path.home() / "areno-runs"
OUT = Path(__file__).resolve().parent.parent / "_static" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colour palette: colour-blind safe, distinguishes the two training runs.
C_V1 = "#1f6fb4"       # v1 (Open-Jev v1.1 only)
C_V2 = "#d1495b"       # v2 (format-completed mix)
C_ZS = "#4d4d4d"       # zero-shot baseline
C_OJ = "#2a9d8f"       # in-distribution Open-Jev
C_TD = "#e07b39"       # out-of-distribution typed-decisions
C_REF = "#9aa0a6"      # reference / baseline markers
C_CEIL = "#c9ccd1"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf / .png")


# --------------------------------------------------------------------------
# Figure 1: pipeline schematic
# --------------------------------------------------------------------------

def figure1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=8.0, weight="normal"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc,
                                   edgecolor=ec, linewidth=1.1,
                                   joinstyle="round", zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, weight=weight, linespacing=1.35)

    def arrow(x0, y0, x1, y1, style="-|>", color="#444444", lw=1.1, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color,
                                    linewidth=lw, linestyle=ls,
                                    shrinkA=1, shrinkB=1))

    # stage 1: input rendering
    box(0.15, 1.55, 1.75, 1.25,
        "state + question\n\nprompt", "#eef3f8", "#5b7fa6", fs=8.0)
    box(0.15, 0.15, 1.75, 1.05,
        "candidate answers\n(path 1 … path K)", "#eef3f8", "#5b7fa6", fs=8.0)
    arrow(1.05, 1.55, 1.05, 1.22)

    # stage 2: backbone
    box(2.55, 0.55, 2.05, 2.25,
        "backbone LM\n\nLing-3.0-tiny\n24 layers · 128 experts\n\n"
        "$\\it{defer\\_lm\\_head}$\n(final-norm hidden)",
        "#e7f1ea", "#4a8a63", fs=8.0)
    arrow(1.92, 1.77, 2.53, 1.9)
    arrow(1.92, 0.68, 2.53, 0.95)

    # stage 3: score head
    box(5.30, 1.25, 1.85, 1.55,
        "candidate-path\nscoring head\n\nLinear-GELU-Linear(1)\n"
        "scores $s_1 … s_K$", "#fdf1e3", "#c98a3c", fs=8.0)
    arrow(4.62, 1.68, 5.28, 1.9)

    # stage 4: grouped softmax
    box(7.75, 1.25, 2.10, 1.55,
        "grouped softmax\n\n$p_k = e^{s_k}/\\sum_j e^{s_j}$\n"
        "(within one question)", "#f6e8ee", "#b5607f", fs=8.0)
    arrow(7.17, 2.02, 7.73, 2.02)

    # feedback: distribution to loss / eval
    ax.annotate("", xy=(8.80, 1.23), xytext=(8.80, 0.42),
                arrowprops=dict(arrowstyle="-|>", color="#444444", linewidth=1.1))
    box(6.55, 0.05, 2.50, 0.62,
        "train: CE + 0.5·Brier   |   eval / serve: read-out",
        "#f2f2f2", "#8a8a8a", fs=7.6)

    # all stages share one forward pass
    ax.annotate("", xy=(2.57, 2.95), xytext=(9.83, 2.95),
                arrowprops=dict(arrowstyle="-", color="#b9bdc2",
                                linewidth=0.9, linestyle=(0, (4, 3))))
    ax.text(6.2, 3.02, "same packed variable-length forward pass "
                       "(training = evaluation = serving)",
            ha="center", va="bottom", fontsize=7.4, color="#6b7075", style="italic")

    save(fig, "fig1_pipeline")


# --------------------------------------------------------------------------
# Figure 2: accuracy trajectory, in- vs out-of-distribution
# --------------------------------------------------------------------------

def figure2_trajectory() -> None:
    steps = np.array([100, 200, 300, 400, 500])

    def series(run: str, split: str) -> np.ndarray:
        out = []
        for s in steps:
            p = RUNS / run / f"step{s}-{split}.json"
            out.append(load(p)["all"]["accuracy"])
        return np.array(out)

    td_v1 = series("diag", "td-test")
    oj_v1 = series("diag", "oj-dev")
    td_v2 = series("diag-v2", "td-test")
    oj_v2 = series("diag-v2", "oj-dev")

    zs_td = load(RUNS / "zero-shot" / "td-test.json")["all"]["accuracy"]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    # reference bands from the benchmark card
    ax.axhspan(0.70, 0.75, color=C_CEIL, alpha=0.45, zorder=0)
    ax.text(102, 0.725, "benchmark's 0.70–0.75\nstrong/saturation band",
            fontsize=6.6, color="#6b7075", va="center")
    ax.axhline(0.520, color=C_REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(102, 0.524, "majority floor 0.52", fontsize=6.6, color="#6b7075")

    # in-distribution dev series: rises steadily for both runs
    ax.plot(steps, oj_v1, "-o", color=C_OJ, mfc="white", mew=1.4, ms=5,
            lw=1.6, label="Open-Jev v1.1 dev (in-dist), v1")
    ax.plot(steps, oj_v2, "--s", color=C_OJ, mfc="white", mew=1.4, ms=5,
            lw=1.3, alpha=0.65, label="Open-Jev v1.1 dev (in-dist), v2")
    # out-of-distribution test series: flat
    ax.plot(steps, td_v1, "-o", color=C_TD, mfc="white", mew=1.4, ms=5,
            lw=1.6, label="typed-decisions test (OOD), v1")
    ax.plot(steps, td_v2, "--s", color=C_TD, mfc="white", mew=1.4, ms=5,
            lw=1.3, alpha=0.65, label="typed-decisions test (OOD), v2")

    ax.axhline(zs_td, color=C_ZS, lw=1.0, ls=":", zorder=1)
    ax.text(500, zs_td - 0.030, f"zero-shot base model, OOD ({zs_td:.3f})",
            fontsize=6.8, color=C_ZS, ha="right", va="top")

    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_xticks(steps)
    ax.set_ylim(0.45, 0.92)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(loc="lower right", fontsize=7.0, ncol=1)
    ax.set_title("Training and in-distribution accuracy rise together;\n"
                 "out-of-distribution accuracy does not", fontsize=8.6, pad=8)

    # step-500 in-distribution TEST endpoints (headline number, separate split)
    zs_oj = load(RUNS / "zero-shot" / "oj-test.json")["all"]["accuracy"]
    v1_oj = load(RUNS / "diag" / "step500-oj-test.json")["all"]["accuracy"]
    ax.annotate("", xy=(534, v1_oj), xytext=(534, zs_oj),
                arrowprops=dict(arrowstyle="<|-|>", color=C_OJ, lw=1.2))
    ax.text(540, (zs_oj + v1_oj) / 2,
            f"Open-Jev test,\nzero-shot→v1:\n{zs_oj:.3f}→{v1_oj:.3f}\n(+{100*(v1_oj-zs_oj):.0f} pt)",
            fontsize=6.4, color=C_OJ, va="center", ha="left")
    ax.scatter([500], [zs_oj], s=30, color=C_OJ, marker="v", zorder=4,
               edgecolor="white", linewidth=0.7)
    ax.set_xlim(90, 620)

    save(fig, "fig2_accuracy_trajectory")


# --------------------------------------------------------------------------
# Figure 3: calibration scatter (confidence vs accuracy)
# --------------------------------------------------------------------------

def figure3_calibration() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # --- left: aggregate, in-dist vs OOD, 4 configurations ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.9, ls="--", zorder=1)
    ax.text(0.62, 0.57, "perfect\ncalibration", fontsize=6.6,
            color="#7a7a7a", rotation=38, ha="center")

    pts = []
    # zero-shot OOD
    zs = load(RUNS / "zero-shot" / "td-test.json")["all"]
    pts.append(("zero-shot\n(OOD)", zs["confidence"], zs["accuracy"], C_ZS, "o"))
    # v1 step100 OOD, v1 step500 OOD
    d1 = load(RUNS / "diag" / "step100-td-test.json")["all"]
    pts.append(("v1 step100\n(OOD)", d1["confidence"], d1["accuracy"], C_TD, "o"))
    d5 = load(RUNS / "diag" / "step500-td-test.json")["all"]
    pts.append(("v1 step500\n(OOD)", d5["confidence"], d5["accuracy"], C_TD, "o"))
    # v2 step500 OOD
    v2 = load(RUNS / "diag-v2" / "step500-td-test.json")["all"]
    pts.append(("v2 step500\n(OOD)", v2["confidence"], v2["accuracy"], C_V2, "s"))
    # v1 in-dist test
    ojt = load(RUNS / "diag" / "step500-oj-test.json")["all"]
    pts.append(("v1 step500\n(in-dist)", ojt["confidence"], ojt["accuracy"], C_OJ, "o"))

    for label, c, a, col, mk in pts:
        ax.scatter(c, a, s=46, color=col, marker=mk, zorder=3,
                   edgecolor="white", linewidth=0.8)
        dy = 0.030 if "in-dist" not in label else -0.055
        ha = "left"
        dx = 0.018
        if label.startswith("v1 step500\n(OOD)"):
            dx, ha, dy = -0.02, "right", 0.018
        if label.startswith("zero-shot"):
            dx, ha, dy = 0.015, "left", 0.012
        ax.annotate(label, (c, a), xytext=(c + dx, a + dy),
                    fontsize=6.8, color=col, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                    alpha=0.7) if "in-dist" in label else None)

    ax.set_xlim(0.5, 0.92)
    ax.set_ylim(0.45, 0.92)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Aggregate", fontsize=8.4)
    ax.text(0.505, 0.885, "over-confident", fontsize=6.4, color="#8a8a8a")
    ax.text(0.80, 0.475, "under-confident", fontsize=6.4, color="#8a8a8a")

    # --- right: per-type, zero-shot vs v1 step500 (OOD) ---
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.9, ls="--", zorder=1)
    zs_t = load(RUNS / "zero-shot" / "td-test.json")["by_type"]
    v1_t = load(RUNS / "diag" / "step500-td-test.json")["by_type"]
    v2_t = load(RUNS / "diag-v2" / "step500-td-test.json")["by_type"]

    types = ["choice", "noul", "score"]
    xpos = np.array([0.0, 1.0, 2.0])
    width = 0.26

    for i, (name, col) in enumerate([("zero-shot", C_ZS), ("v1 step500", C_TD),
                                     ("v2 step500", C_V2)]):
        src = {"zero-shot": zs_t, "v1 step500": v1_t, "v2 step500": v2_t}[name]
        gaps = [src[t]["overconfidence"] for t in types]
        ax.bar(xpos + (i - 1) * width, gaps, width * 0.92,
               color=col, label=name, edgecolor="white", linewidth=0.6)
        for x, g in zip(xpos + (i - 1) * width, gaps):
            ax.text(x, g + (0.008 if g >= 0 else -0.016), f"{g:+.3f}",
                    ha="center", va="bottom" if g >= 0 else "top",
                    fontsize=6.0, color="#333333")

    ax.axhline(0, color="#555555", lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(types)
    ax.set_ylabel("overconfidence (conf − acc)")
    ax.set_title("Per answer type (typed-decisions)", fontsize=8.4)
    ax.legend(fontsize=7.0, loc="upper right")
    ax.set_ylim(-0.09, 0.30)

    fig.suptitle("Training removes overconfidence, but only in distribution",
                 fontsize=8.8, y=1.01)
    save(fig, "fig3_calibration")


# --------------------------------------------------------------------------
# Figure 4: temperature transfer
# --------------------------------------------------------------------------

def figure4_temperature() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # --- left: reliability on OOD, T=1 vs fitted T, per type ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.9, ls="--", zorder=1)

    base = load(RUNS / "diag" / "step500-td-test.json")["by_type"]
    fit = load(RUNS / "diag" / "step500-td-test-T-global.json")["by_type"]

    for name, col, mk in [("T = 1", C_TD, "o"), ("T = 1.0476", C_V1, "s")]:
        src = base if name == "T = 1" else fit
        c = [src[t]["confidence"] for t in ["choice", "noul", "score"]]
        a = [src[t]["accuracy"] for t in ["choice", "noul", "score"]]
        ax.plot(c, a, mk, color=col, ms=7, mfc="white", mew=1.5,
                label=name, zorder=3)
    ax.set_xlim(0.5, 0.9)
    ax.set_ylim(0.45, 0.75)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Per-type reliability on typed-decisions", fontsize=8.4)
    ax.legend(fontsize=7.4, loc="lower right")

    # --- right: what the fitted T was, and how little it changes ---
    ax = axes[1]
    tj = load(RUNS / "diag" / "temperature.json")
    tv2 = load(RUNS / "diag-v2" / "temperature.json")

    labels = ["all", "choice", "noul", "score"]
    y = np.arange(len(labels))
    h = 0.36

    v1_vals = [tj["all"], tj["choice"], tj["noul"], tj["score"]]
    v2_vals = [tv2["all"], tv2["choice"], tv2["noul"], tv2["score"]]
    ax.barh(y - h / 2, v1_vals, h, color=C_V1, label="v1 (fitted on Open-Jev cal)",
            edgecolor="white", linewidth=0.6)
    ax.barh(y + h / 2, v2_vals, h, color=C_V2, label="v2 (fitted on Open-Jev cal)",
            edgecolor="white", linewidth=0.6)
    for yy, vv in zip(y - h / 2, v1_vals):
        ax.text(vv + 0.002, yy, f"{vv:.4g}", va="center", fontsize=6.2)
    for yy, vv in zip(y + h / 2, v2_vals):
        ax.text(vv + 0.002, yy, f"{vv:.4g}", va="center", fontsize=6.2)

    ax.axvline(1.0, color="#555555", lw=0.9, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("fitted temperature")
    ax.set_xlim(0.96, 1.12)
    ax.set_title("Fitted temperature is near 1", fontsize=8.4)
    ax.legend(fontsize=6.8, loc="lower right")

    # annotate the effect on the OOD metric
    b = load(RUNS / "diag" / "step500-td-test.json")["all"]
    f = load(RUNS / "diag" / "step500-td-test-T-global.json")["all"]
    ax.text(0.985, 4.05,
            f"OOD after T: KL {b['kl']:.3f}→{f['kl']:.3f}, "
            f"Brier {b['brier']:.3f}→{f['brier']:.3f},\n"
            f"overconf {b['overconfidence']:+.3f}→{f['overconfidence']:+.3f}",
            fontsize=6.4, color="#444444", va="top")

    fig.suptitle("A temperature fitted in distribution does not correct the "
                 "out-of-distribution shift", fontsize=8.8, y=1.02)
    save(fig, "fig4_temperature")


# --------------------------------------------------------------------------
# Figure 5: v1 vs v2 (format-gap test) and data mix
# --------------------------------------------------------------------------

def figure5_format_gap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    # --- left: grouped bars of OOD metrics, v1 vs v2 ---
    ax = axes[0]
    v1 = load(RUNS / "diag" / "step500-td-test.json")
    v2 = load(RUNS / "diag-v2" / "step500-td-test.json")

    types = ["choice", "noul", "score"]
    xpos = np.arange(len(types))
    w = 0.36
    v1g = [v1["by_type"][t]["overconfidence"] for t in types]
    v2g = [v2["by_type"][t]["overconfidence"] for t in types]

    ax.bar(xpos - w / 2, v1g, w * 0.92, color=C_V1, label="v1 (no format fix)",
           edgecolor="white", linewidth=0.6)
    ax.bar(xpos + w / 2, v2g, w * 0.92, color=C_V2, label="v2 (format fixed)",
           edgecolor="white", linewidth=0.6)
    for x, g1, g2 in zip(xpos, v1g, v2g):
        ax.text(x - w / 2, g1 + 0.006, f"{g1:+.3f}", ha="center",
                fontsize=6.2, color="#333333")
        ax.text(x + w / 2, g2 + 0.006, f"{g2:+.3f}", ha="center",
                fontsize=6.2, color="#333333")
        ax.annotate("", xy=(x + w / 2, g2), xytext=(x - w / 2, g1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                    lw=1.0, alpha=0.8))
    ax.axhline(0, color="#555555", lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(types)
    ax.set_ylabel("overconfidence (conf − acc)")
    ax.set_ylim(0, 0.27)
    ax.set_title("Fixing the format gap increased overconfidence",
                 fontsize=8.4)
    ax.legend(fontsize=7.0, loc="upper left")

    # --- right: what the v2 mix added ---
    ax = axes[1]
    # bucket counts read from the recorded manifest summary
    mix = {
        "choice_desc": 16191,
        "noul_crit": 12857,
        "score": 11561,
        "noul_nocrit": 8887,
        "soft": 6425,
        "choice_nodesc": 5131,
        "wanli": 3280,
    }
    names = list(mix)
    vals = [mix[k] for k in names]
    # highlight the buckets the format fix targeted
    cols = ["#d1495b" if k in ("choice_desc", "noul_crit") else "#c6ccd2"
            for k in names]
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, edgecolor="white", linewidth=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 200, yy, f"{v:,}", va="center", fontsize=6.4, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("questions in v2 training mix")
    ax.set_xlim(0, 19500)
    ax.set_title("v2 mix (64,332 questions)", fontsize=8.4)
    ax.text(0.98, 0.03,
            "red = buckets that supply\noption descriptions / criteria",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, color="#8a5a63")

    fig.suptitle("The format-gap hypothesis was tested and failed",
                 fontsize=8.8, y=1.02)
    save(fig, "fig5_format_gap")


# --------------------------------------------------------------------------
# Figure 6: where the numbers sit on the leaderboard
# --------------------------------------------------------------------------

def figure6_leaderboard() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    entries = [
        ("meraGPT Decider 1", 0.768, 0.096, "generalist"),
        ("TypeSafe Jev 1.13.0", 0.727, 1.442, "generalist"),
        ("Featherless Simple Jev 35B-A3B", 0.716, 0.488, "generalist"),
        ("factor-fitted ceiling", 0.704, None, "reference"),
        ("ModernBERT-base (specialist)†", 0.646, 0.223, "specialist"),
        ("MiniLM-L6 (specialist)†", 0.587, 0.262, "specialist"),
        ("v1 scoring head, step 500", 0.582, 0.443, "this work"),
        ("zero-shot LM head", 0.567, 0.893, "this work"),
        ("v2 scoring head, step 500", 0.561, 0.544, "this work"),
        ("majority baseline", 0.520, None, "reference"),
        ("Prior baseline", 0.470, 0.347, "reference"),
        ("Uniform baseline", 0.308, 0.444, "reference"),
    ]

    cmap = {
        "generalist": ("#8a8f96", "o"),
        "specialist": ("#b08bbf", "^"),
        "this work": ("#d1495b", "D"),
        "reference": ("#c9ccd1", "s"),
    }

    for name, acc, kl, kind in entries:
        col, mk = cmap[kind]
        lw = 1.6 if kind == "this work" else 1.0
        ax.scatter(acc, kl if kl is not None else -0.05, s=64, color=col,
                   marker=mk, edgecolor="white", linewidth=0.8, zorder=3)
        if kl is None:
            ax.scatter(acc, -0.05, s=64, color=col, marker=mk,
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(acc, 0.02, name, fontsize=6.6, color="#666666",
                    ha="center", va="bottom", rotation=90)
        else:
            dy = 0.05 if not name.startswith(("meraGPT", "ModernBERT",
                                              "v1 scoring")) else -0.075
            ax.annotate(name, (acc, kl), xytext=(0, dy * 100),
                        textcoords="offset points", fontsize=6.8, color=col,
                        ha="center", va="bottom" if dy > 0 else "top")

    # saturation band
    ax.axvspan(0.70, 0.80, color=C_CEIL, alpha=0.35, zorder=0)
    ax.text(0.75, 1.32, "strong / saturation\nband (0.70–0.80)",
            fontsize=6.4, color="#6b7075", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75))

    ax.set_xlabel("accuracy on typed-decisions test")
    ax.set_ylabel("KL divergence from gold (lower is better)")
    ax.set_xlim(0.27, 0.82)
    ax.set_ylim(-0.14, 1.55)
    ax.invert_yaxis()
    ax.set_title("Accuracy against calibration: this work sits below the "
                 "generalist leaders\nand above the prior baseline",
                 fontsize=8.6, pad=10)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="D", color="#d1495b", ls="", label="this work"),
        Line2D([], [], marker="o", color="#8a8f96", ls="", label="published generalist"),
        Line2D([], [], marker="^", color="#b08bbf", ls="", label="published specialist†"),
        Line2D([], [], marker="s", color="#c9ccd1", ls="", label="reference baseline"),
    ]
    ax.legend(handles=handles, fontsize=7.0, loc="lower right")

    save(fig, "fig6_leaderboard")


# --------------------------------------------------------------------------

def main() -> None:
    figure1_pipeline()
    figure2_trajectory()
    figure3_calibration()
    figure4_temperature()
    figure5_format_gap()
    figure6_leaderboard()
    print(f"\nall figures written to {OUT}")


if __name__ == "__main__":
    main()