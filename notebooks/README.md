Notebooks for experiments and analysis.

Place exploratory work here. Keep anything that must be reproducible in
`src/` or `scripts/` and import it from the notebook, so a notebook never
becomes the only copy of a result.

## Pipeline notebooks

Run `uv sync` (and `uv run jupyter lab`) from the repo root, then open:

| Notebook | What it does |
|---|---|
| `01_smoke_test.ipynb` | Tiny-encoder one-epoch train + eval + inference; validates the whole pipeline quickly. |
| `02_train.ipynb` | Configurable real training with the E2 ablation knobs (`w_rl`, `w_ce`, `sigma_*`) and learning curves. |
| `03_evaluate_and_infer.ipynb` | Rebuild a checkpoint, fit temperature, report raw/post-T metrics, and score `choice`/`score`/`noul` questions. |
| `04_e1_toy_bias.ipynb` | Runs `experiments/e1_toy_bias.py` and plots the noise-smoothing bias vs. sigma. |

These import the implementation from `src/` (and `experiments/`) rather than
reimplementing it, so the notebooks stay thin and reproducible from the
command line.
