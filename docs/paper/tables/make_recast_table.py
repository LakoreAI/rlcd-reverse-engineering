r"""Generate the E6 cross-task (recast) table from `results/recast/*.json`.

Each row is one recast classification task at two option-token budgets
(`head_max_len` 256 vs 512). Scores come from `scripts/eval/recast_bench.py`.

    uv run python docs/paper/tables/make_recast_table.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "docs" / "paper" / "data" / "recast"

# (task file key, display label, K)
TASKS = [
    ("ag_news", "AG News", 4),
    ("emotion", "Emotion", 6),
    ("sst5", "SST-5", 5),
    ("banking77", "Banking77", 77),
]


def acc(model, task, budget):
    f = RUNS / f"{model}_{task}_h{budget}.json"
    d = json.loads(f.read_text())
    return f"{100 * d['accuracy']:.1f}"


def main() -> None:
    lines = []
    for task, label, k in TASKS:
        row = [
            label,
            str(k),
            acc("laya", task, 256),
            acc("laya", task, 512),
            acc("sev", task, 256),
            acc("sev", task, 512),
        ]
        lines.append(" & ".join(row) + r" \\")
    (OUT / "recast.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    print("wrote recast.tex ->", len(lines), "rows")


if __name__ == "__main__":
    main()
