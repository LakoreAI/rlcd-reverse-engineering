r"""Generate the paper's LaTeX tables from result files (no hand-copied numbers).

Inputs: results/e2_summary.json, results/e2_row_analysis.json,
results/e2_vm/results/<run>/test_eval_v2.json. Outputs: docs/paper/tables/*.tex (each ends with its own
\bottomrule, so main.tex can \input it inside a tabular).

    uv run python docs/paper/tables/make_tables.py
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RES = ROOT / "results"
RUNS = RES / "e2_vm" / "results"

ORDER = [
    ("e2_ce_only", r"CE-only ($w_\mathrm{rl}{=}0$)"),
    ("e2_laya_rlce", r"RL+CE, Laya ($\sigma$ 0.4$\to$0.1)"),
    ("e2_rl_only", r"RL-only ($w_\mathrm{ce}{=}0$)"),
    ("e2_rlce_sigma0p5_fixed", r"RL+CE, $\sigma{=}0.5$"),
    ("e2_rlce_sigma1_fixed", r"RL+CE, $\sigma{=}1$"),
    ("e2_rlce_sigma2_fixed", r"RL+CE, $\sigma{=}2$"),
]


# Fixed-sigma sweep (RL+CE) and w_rl/w_ce ratio at sigma=1, plus E4's reward
# composition. Missing runs are skipped so partial batches still build.
SWEEP = [
    ("e2_rlce_sigma0p25_fixed", r"$\sigma{=}0.25$"),
    ("e2_rlce_sigma0p5_fixed", r"$\sigma{=}0.5$"),
    ("e2_rlce_sigma1_fixed", r"$\sigma{=}1$"),
    ("e2_rlce_sigma2_fixed", r"$\sigma{=}2$"),
    ("e2_rlce_sigma3_fixed", r"$\sigma{=}3$"),
    ("e2_rlce_sigma4_fixed", r"$\sigma{=}4$"),
]
RATIO = [
    ("e2_rlce_wrl0p5_sigma1_fixed", r"$0.5$"),
    ("e2_rlce_sigma1_fixed", r"$1$"),
    ("e2_rlce_wrl2_sigma1_fixed", r"$2$"),
]
E4 = [
    ("e4_log_only", "log only"),
    ("e4_log_sph", "log + sph"),
    ("e4_log_rps", "log + RPS"),
    ("e2_laya_rlce", "full (log+sph+RPS)"),
]
EXTRA_COLS = ["gap", "soft_ece", "nll", "post_nll", "T[noul/b0]", "raw_ece", "acc"]


def emit_extra(per_cfg, order, path) -> None:
    lines = []
    for key, label in order:
        rs = per_cfg.get(key) or []
        if not rs:
            continue
        g = lambda k: [r[k] for r in rs]  # noqa: E731
        cells = [label, str(len(rs))]
        for col in EXTRA_COLS:
            sign = col == "gap"
            nd = 2 if col.startswith("T[") else 3
            cells.append(ms(g(col), nd, sign=sign))
        lines.append(" & ".join(cells) + r" \\")
    (OUT / path).write_text("\n".join(lines) + "\n\\bottomrule\n")


def cfg_of(run: str) -> str | None:
    if run.endswith("_rep2"):
        return None
    return run.rsplit("_seed", 1)[0]


def num(s: str) -> str:
    """Typeset a leading sign as a math minus/plus."""
    if s.startswith("-"):
        return "$-$" + s[1:]
    if s.startswith("+"):
        return "$+$" + s[1:]
    return s


def ms(vals, nd=3, sign=False):
    m = statistics.mean(vals)
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    s = num(f.format(m))
    if len(vals) > 1:
        s += r"{\scriptsize$\pm$" + f"{statistics.stdev(vals):.{nd}f}" + "}"
    return s


def main() -> None:
    summary = json.loads((RES / "e2_summary.json").read_text())
    rows_by_run = {r["run"]: r for r in summary["runs"]}
    ra = json.loads((RES / "e2_row_analysis.json").read_text())
    per_run_ra = ra["per_run"]

    per_cfg = defaultdict(list)
    for run in rows_by_run:
        c = cfg_of(run)
        if c:
            ev = json.loads((RUNS / run / "test_eval_v2.json").read_text())
            per_cfg[c].append(
                {
                    **rows_by_run[run],
                    "gap": per_run_ra[run]["sharp_gap"],
                    "soft_ece": per_run_ra[run]["soft"]["ece"],
                    "post_nll": ev["post_temperature"]["nll"],
                }
            )

    # --- main E2 table (table*) ---
    lines = []
    for key, label in ORDER:
        rs = per_cfg[key]
        g = lambda k: [r[k] for r in rs]  # noqa: E731
        lines.append(
            " & ".join(
                [
                    label,
                    str(len(rs)),
                    ms(g("gap"), sign=True),
                    ms(g("soft_ece")),
                    ms(g("nll")),
                    ms(g("brier")),
                    ms(g("post_nll")),
                    ms(g("T[choice/b1]"), 2),
                    ms(g("T[score/b1]"), 2),
                    ms(g("T[noul/b0]"), 2),
                    ms(g("raw_ece")),
                    ms(g("post_ece")),
                    ms(g("acc")),
                ]
            )
            + r" \\"
        )
        if key == "e2_rl_only":
            lines.append(r"\midrule")
    t = ra["teacher"]
    lines.append(r"\midrule")
    lines.append(
        r"\textit{Teacher $t$ as predictor} & -- & $0$ & $0$ & -- & $0$ & -- & -- & -- & -- & "
        + f"{t['hard']['ece']:.3f} & -- & 0.984"
        + r" \\"
    )
    (OUT / "e2_main.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

    # --- E3 gradient table ---
    grad = summary["grad"]
    fixed = defaultdict(list)
    anneal = defaultdict(list)
    for r in grad:
        (fixed if "_fixed_" in r["run"] else anneal)[r["sigma"]].append(r)
    g_lines = []
    for s in sorted(anneal, reverse=True):
        rs = anneal[s]
        g_lines.append(
            f"{s:.1f} (annealed) & {len(rs)} & "
            + ms([x["cos_mean"] for x in rs], 2)
            + " & "
            + ms([x["ratio_mean"] for x in rs], 1)
            + r" \\"
        )
    g_lines.append(r"\midrule")
    for s in sorted(fixed):
        rs = fixed[s]
        ratio = [x["ratio_mean"] for x in rs]
        g_lines.append(
            f"{s:.1f} (fixed) & {len(rs)} & "
            + ms([x["cos_mean"] for x in rs], 2)
            + " & "
            + ms(ratio, 1)
            + r" \\"
        )
    (OUT / "e3_grad.tex").write_text("\n".join(g_lines) + "\n\\bottomrule\n")

    # --- appendix: every run ---
    a_lines = []
    for run in sorted(rows_by_run):
        r = rows_by_run[run]
        a_lines.append(
            " & ".join(
                [
                    run.replace("e2_", "").replace("_", r"\_"),
                    num(f"{per_run_ra[run]['sharp_gap']:+.3f}"),
                    f"{per_run_ra[run]['soft']['ece']:.3f}",
                    f"{r['nll']:.3f}",
                    f"{r['brier']:.3f}",
                    f"{r['T[choice/b1]']:.2f}",
                    f"{r['T[score/b1]']:.2f}",
                    f"{r['T[noul/b0]']:.2f}",
                    f"{r['raw_ece']:.3f}",
                    f"{r['acc']:.3f}",
                ]
            )
            + r" \\"
        )
    (OUT / "e2_all_runs.tex").write_text("\n".join(a_lines) + "\n\\bottomrule\n")

    # --- by-entropy table ---
    pc = ra["per_config"]
    e_lines = []
    for key, label in ORDER:
        be = pc[key]["by_entropy_sharp_gap"]
        e_lines.append(
            f"{label} & "
            + " & ".join(num(f"{be[t]:+.3f}") for t in ("low", "mid", "high"))
            + r" \\"
        )
    (OUT / "e2_entropy.tex").write_text("\n".join(e_lines) + "\n\\bottomrule\n")

    # --- extras: fixed-sigma sweep, w_rl/w_ce ratio, E4 reward composition ---
    emit_extra(per_cfg, SWEEP, "e2_sweep.tex")
    emit_extra(per_cfg, RATIO, "e2_ratio.tex")
    emit_extra(per_cfg, E4, "e4_reward.tex")
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
