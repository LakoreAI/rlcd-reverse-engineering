# Paper: *Reverse-Engineering RLCD*

IEEE conference-format report (`\documentclass[conference]{IEEEtran}`) for
this project. `main.pdf` is the built copy.

| file | what |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | references (IEEEtran style; web sources as `@misc` with access dates) |
| `figures/make_figures.py` → `figures/fig_*.pdf` | all figures, from `results/*.json` |
| `tables/make_tables.py` → `tables/*.tex` | all result tables, from `results/*.json` and `results/e2_vm/` |

No number in the paper is typed by hand: every figure and table
regenerates from the result files.

## Rebuild

```bash
# 1. result files (E1 is CPU-only; E2 inputs come from the rented-GPU runs,
#    see docs/reports/2026-09-25/e2_minimal_runs.md)
uv run python experiments/e1_toy_bias.py                    # results/e1_toy_bias.json
uv run python scripts/e2/summarize.py --runs_dir results/e2_vm/results --json results/e2_summary.json
uv run python scripts/e2/row_analysis.py                    # results/e2_row_analysis.json (needs checkpoints/e2/*/logits.pt)

# 2. figures and tables
uv run python docs/paper/figures/make_figures.py
uv run python docs/paper/tables/make_tables.py

# 3. PDF — either engine works
cd docs/paper && tectonic -X compile main.tex               # XeTeX, fetches packages on demand
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main   (arXiv's engine)
```

Under XeTeX the preamble loads TeX Gyre Termes (a Times clone). Under
pdfLaTeX, IEEEtran's own Times setup applies.

## Guardrails (docs/PLAN.md §7)

Jev is described only as "consistent with TypeSafe's public claims". Raw
ECE is always shown next to post-temperature ECE. There are no "beats Jev"
claims, Laya's README is credited, and the paper is framed as analysis.
