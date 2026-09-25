"""Build the paper's vector figures from result files (no hand-copied numbers).

Inputs: results/e1_toy_bias.json, results/e2_summary.json,
results/e2_row_analysis.json. Outputs: docs/paper/figures/fig_*.pdf, sized for one
IEEE column (3.5 in) with >= 8 pt text.

    uv run python docs/paper/figures/make_figures.py
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RES = ROOT / "results"

# Validated categorical slots 1-3 (dataviz reference palette, all-pairs PASS);
# aqua is < 3:1 on white, so every series is also direct-labelled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W = 3.5

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def label_end(ax, x, y, text, color_ink=INK, dx=4, dy=0):
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=7.5,
        color=color_ink,
    )


def fig_e1():
    e1 = json.loads((RES / "e1_toy_bias.json").read_text())
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    series = [
        ("log score", [r for r in e1["baseline_replication"]], BLUE, "o", "-"),
        (
            "full reward, choice",
            [r for r in e1["full_reward"] if r["qtype"] == "choice"],
            ORANGE,
            "s",
            "--",
        ),
        (
            "full reward, score",
            [r for r in e1["full_reward"] if r["qtype"] == "score"],
            AQUA,
            "^",
            ":",
        ),
    ]
    for name, rows, c, m, ls in series:
        xs = [r["sigma"] for r in rows]
        ys = [r["sharpness_max_p"] for r in rows]
        ax.plot(xs, ys, color=c, marker=m, ls=ls, label=name)
    ax.axhline(0.7, color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(
        1.25, 0.705, r"target $\max_i t_i = 0.7$", fontsize=7, color=INK2, va="bottom"
    )
    ax.set_xlabel(r"noise scale $\sigma$")
    ax.set_ylabel(r"noise-free $\max_i\,\mathrm{softmax}(z^\star)_i$")
    ax.set_xticks([0, 0.5, 1, 2])
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "fig_e1_toy_bias.pdf")
    plt.close(fig)


def fig_ece_crossing():
    ra = json.loads((RES / "e2_row_analysis.json").read_text())["per_config"]
    sig = [0.5, 1.0, 2.0]
    keys = ["e2_rlce_sigma0p5_fixed", "e2_rlce_sigma1_fixed", "e2_rlce_sigma2_fixed"]
    soft = [ra[k]["soft_ece"] for k in keys]
    hard = [ra[k]["hard_ece"] for k in keys]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.plot(sig, soft, color=BLUE, marker="o", label="target-referenced ECE")
    ax.plot(sig, hard, color=ORANGE, marker="s", ls="--", label="hard-label ECE")
    ce = ra["e2_ce_only"]
    ax.axhline(ce["soft_ece"], color=BLUE, lw=0.8, ls=(0, (1, 2)))
    ax.axhline(ce["hard_ece"], color=ORANGE, lw=0.8, ls=(0, (1, 2)))
    label_end(ax, 2.0, soft[-1], "vs. soft target", dy=0)
    label_end(ax, 2.0, hard[-1], "vs. hard label", dy=0)
    ax.text(
        0.47, ce["soft_ece"] + 0.004, "CE-only", fontsize=7, color=INK2, va="bottom"
    )
    ax.text(
        0.47, ce["hard_ece"] + 0.004, "CE-only", fontsize=7, color=INK2, va="bottom"
    )
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(0, 0.16)
    ax.set_xticks(sig)
    ax.set_xlabel(r"fixed noise scale $\sigma$ (RL+CE)")
    ax.set_ylabel("raw ECE (15 bins)")
    fig.savefig(OUT / "fig_ece_crossing.pdf")
    plt.close(fig)


def _t_by_config():
    runs = json.loads((RES / "e2_summary.json").read_text())["runs"]
    g = defaultdict(lambda: defaultdict(list))
    for r in runs:
        name = r["run"]
        if name.endswith("_rep2"):
            continue
        cfg = name.rsplit("_seed", 1)[0]
        for k, v in r.items():
            if k.startswith("T[") and v is not None:
                g[cfg][k].append(v)
    return g


def fig_temperature():
    g = _t_by_config()
    sig = [0.5, 1.0, 2.0]
    keys = ["e2_rlce_sigma0p5_fixed", "e2_rlce_sigma1_fixed", "e2_rlce_sigma2_fixed"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for tk, name, c, m, ls in [
        ("T[noul/b0]", "noul", ORANGE, "s", "--"),
        ("T[score/b1]", "score", AQUA, "^", ":"),
        ("T[choice/b1]", "choice", BLUE, "o", "-"),
    ]:
        ys = [statistics.mean(g[k][tk]) for k in keys]
        ax.plot(sig, ys, color=c, marker=m, ls=ls)
        label_end(ax, 2.0, ys[-1], name)
        ce = statistics.mean(g["e2_ce_only"][tk])
        ax.plot([0.42], [ce], marker=m, color=c, mfc="white", ls="none")
    ax.text(0.42, 1.012, "CE-only", fontsize=6.5, color=INK2, ha="center", va="bottom")
    ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(0.3, 2.55)
    ax.set_xticks(sig)
    ax.set_xlabel(r"fixed noise scale $\sigma$ (RL+CE)")
    ax.set_ylabel(r"fitted temperature $T$")
    fig.savefig(OUT / "fig_temperature.pdf")
    plt.close(fig)


def fig_gradients():
    grad = json.loads((RES / "e2_summary.json").read_text())["grad"]
    fixed = defaultdict(list)
    for r in grad:
        if "_fixed_" in r["run"]:
            fixed[r["sigma"]].append(r)
    sig = sorted(fixed)
    ratio = [statistics.mean(x["ratio_mean"] for x in fixed[s]) for s in sig]
    cos = [statistics.mean(x["cos_mean"] for x in fixed[s]) for s in sig]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.7))
    a1.loglog(sig, ratio, color=BLUE, marker="o")
    k = statistics.mean(r * s for r, s in zip(ratio, sig))
    a1.loglog([0.45, 2.2], [k / 0.45, k / 2.2], color=INK2, lw=0.8, ls=(0, (2, 2)))
    a1.text(1.25, k / 1.25 * 1.25, r"$\propto 1/\sigma$", fontsize=7, color=INK2)
    from matplotlib.ticker import (
        FixedLocator,
        NullFormatter,
        NullLocator,
        ScalarFormatter,
    )

    for axis in (a1.xaxis, a1.yaxis):
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    a1.xaxis.set_major_locator(FixedLocator(sig))
    a1.set_xticklabels(["0.5", "1", "2"])
    a1.yaxis.set_major_locator(FixedLocator([2, 5, 10]))
    a1.yaxis.set_major_formatter(ScalarFormatter())
    a1.set_xlabel(r"$\sigma$")
    a1.set_ylabel(r"$\|\nabla_z L_{RL}\| / \|\nabla_z L_{CE}\|$")
    a1.set_title("(a) norm ratio", fontsize=8)
    a2.plot(sig, cos, color=BLUE, marker="o")
    a2.set_ylim(0, 1)
    a2.set_xticks(sig)
    a2.set_xlabel(r"$\sigma$")
    a2.set_ylabel(r"$\cos(\nabla_z L_{RL}, \nabla_z L_{CE})$")
    a2.set_title("(b) alignment", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_gradients.pdf")
    plt.close(fig)


def fig_reliability():
    per = json.loads((RES / "e2_row_analysis.json").read_text())["per_run"]
    runs = [
        ("e2_ce_only_seed42", "CE-only", BLUE, "o", "-"),
        ("e2_rlce_sigma2_fixed_seed42", r"RL+CE, $\sigma{=}2$", ORANGE, "s", "--"),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.85), sharey=True)
    for ax, ref, title in (
        (a1, "soft", r"(a) vs. soft target $t[\arg\max p]$"),
        (a2, "hard", "(b) vs. hard label"),
    ):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (2, 2)))
        for run, name, c, m, ls in runs:
            curve = [b for b in per[run][ref]["curve"] if b["count"] >= 20]
            ax.plot(
                [b["conf"] for b in curve],
                [b["ref"] for b in curve],
                color=c,
                marker=m,
                ls=ls,
                ms=3.5,
                label=name,
            )
        ax.set_xlim(0.2, 1)
        ax.set_ylim(0.2, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"model $\max_i p_i$")
        ax.set_title(title, fontsize=8)
    a1.set_ylabel("mean reference in bin")
    a1.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig_reliability.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for f in (
        fig_ece_crossing,
        fig_temperature,
        fig_gradients,
        fig_reliability,
        fig_e1,
    ):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
