r"""Generate the paper's result tables from `results/e2_vm/results/*/test_eval.json`.

Self-contained: reads each run's `test_eval.json` (raw/post/fitted_temperature),
groups by configuration, and writes mean ± std LaTeX rows. Used for the
sigma-sweep, ratio, budget and E4 tables (the older make_tables.py needs
`logits.pt` for a target-referenced ECE that not every new run has).

    uv run python docs/paper/tables/make_run_tables.py
"""

import glob
import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "e2_vm" / "results"

# Config label -> the run-name prefix before `_seedN`.
ORDER = {
    "e2_ce_only": r"CE-only ($w_\mathrm{rl}{=}0$)",
    "e2_ce_only_1024": r"CE-only, 1024/256",
    "e2_ce_only_typed_init": r"CE-only, 1024/256, typed-init",
    "e2_laya_rlce": r"RL+CE, Laya",
    "e2_rl_only": r"RL-only ($w_\mathrm{ce}{=}0$)",
    "e2_rlce_sigma0p25_fixed": r"$\sigma{=}0.25$",
    "e2_rlce_sigma0p5_fixed": r"$\sigma{=}0.5$",
    "e2_rlce_sigma1_fixed": r"$\sigma{=}1$",
    "e2_rlce_sigma2_fixed": r"$\sigma{=}2$",
    "e2_rlce_sigma3_fixed": r"$\sigma{=}3$",
    "e2_rlce_sigma4_fixed": r"$\sigma{=}4$",
    "e2_rlce_wrl0p5_sigma1_fixed": r"$w_\mathrm{rl}{=}0.5$",
    "e2_rlce_wrl2_sigma1_fixed": r"$w_\mathrm{rl}{=}2$",
    "e4_log_only": r"E4 log only",
    "e4_log_sph": r"E4 log+sph",
    "e4_log_rps": r"E4 log+RPS",
}


def load() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    dirs = sorted(glob.glob(str(RUNS / "e2_*")) + glob.glob(str(RUNS / "e4_*")))
    for d in dirs:
        # The original 13 runs only have the richer metrics in test_eval_v2.json
        # (written by dump_logits.py); newer runs have them in test_eval.json.
        f = os.path.join(d, "test_eval_v2.json")
        if not os.path.exists(f):
            f = os.path.join(d, "test_eval.json")
        if not os.path.exists(f):
            continue
        name = os.path.basename(d)
        if "rep2" in name or "attempt" in name:  # repeat/excluded runs
            continue
        cfg = re.sub(r"_seed\d+.*$", "", name)
        if cfg not in ORDER:
            continue
        ev = json.load(open(f))
        raw, post = ev["raw"], ev["post_temperature"]
        T = ev.get("fitted_temperature", {})
        groups.setdefault(cfg, []).append(
            {
                "gap": raw["mean_confidence"] - raw["mean_target_max"],
                "nll": raw["nll"],
                "brier": raw["brier"],
                "post_nll": post["nll"],
                "raw_ece": raw["ece"],
                "post_ece": post["ece"],
                "acc": raw["accuracy"],
                "Tc": T.get("type0_bucket1"),
                "Ts": T.get("type1_bucket1"),
                "Tn": T.get("type2_bucket0"),
            }
        )
    return groups


def ms(values, nd=3, sign=False):
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    fmt = f"{{:{'+' if sign else ''}.{nd}f}}"
    out = fmt.format(statistics.mean(values))
    if out.startswith("-"):
        out = "$-$" + out[1:]
    elif out.startswith("+"):
        out = "$+$" + out[1:]
    if len(values) > 1:
        out += "{\\scriptsize$\\pm$" + f"{statistics.stdev(values):.{nd}f}" + "}"
    return out


def emit(order, cols, path):
    groups = load()
    lines = []
    for cfg in order:
        rows = groups.get(cfg)
        if not rows:
            continue
        label = ORDER[cfg]
        cells = [label, str(len(rows))]
        for col in cols:
            nd = 2 if col in ("Tc", "Ts", "Tn") else 3
            cells.append(ms([r[col] for r in rows], nd, sign=(col == "gap")))
        lines.append(" & ".join(cells) + r" \\")
    (OUT / path).write_text("\n".join(lines) + "\n\\bottomrule\n")
    print("wrote", path, "->", len(lines), "rows")


def emit_noise():
    rows = []
    for sigma in (1, 2):
        d = json.load(open(RUNS / f"noise_avg_sigma{sigma}.json"))
        rows.append(
            f"${sigma}$ & {d['sharp0']:.3f} & {d['sharpbar']:.3f} & "
            f"{d['sharpt']:.3f} & ${d['gap0']:+.3f}$ & ${d['gapbar']:+.3f}$ & "
            f"{d['kl0']:.3f} & {d['klbar']:.3f} & {d['moved']:.2f} \\\\"
        )
    (OUT / "noise_avg.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    print("wrote noise_avg.tex ->", len(rows), "rows")


if __name__ == "__main__":
    emit(
        [
            "e2_ce_only",
            "e2_ce_only_1024",
            "e2_ce_only_typed_init",
            "e2_laya_rlce",
            "e2_rl_only",
            "e2_rlce_sigma0p5_fixed",
            "e2_rlce_sigma1_fixed",
            "e2_rlce_sigma2_fixed",
        ],
        [
            "gap",
            "nll",
            "brier",
            "post_nll",
            "Tc",
            "Ts",
            "Tn",
            "raw_ece",
            "post_ece",
            "acc",
        ],
        "e2_main.tex",
    )
    emit(
        [
            "e2_rlce_sigma0p25_fixed",
            "e2_rlce_sigma0p5_fixed",
            "e2_rlce_sigma1_fixed",
            "e2_rlce_sigma2_fixed",
            "e2_rlce_sigma3_fixed",
            "e2_rlce_sigma4_fixed",
        ],
        ["gap", "brier", "nll", "Tn", "Tc", "Ts", "acc"],
        "e2_sweep.tex",
    )
    emit(
        [
            "e2_rlce_wrl0p5_sigma1_fixed",
            "e2_rlce_sigma1_fixed",
            "e2_rlce_wrl2_sigma1_fixed",
        ],
        ["gap", "brier", "nll", "Tn", "Tc", "Ts", "acc"],
        "e2_ratio.tex",
    )
    emit(
        ["e4_log_only", "e4_log_sph", "e4_log_rps", "e2_laya_rlce"],
        ["gap", "brier", "nll", "Tn", "Tc", "Ts", "acc"],
        "e4_reward.tex",
    )
    emit_noise()
