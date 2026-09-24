# Experiments

Runbook for gated experiments. Each experiment records: hypothesis, config,
command, result, and decision.

## Template

### <YYYY-MM-DD> — <short name>

- **Hypothesis:**
- **Config:** `configs/<file>.yaml`
- **Command:**

  ```bash
  uv run python scripts/training/train.py --config configs/<file>.yaml
  ```

- **Result:** _accuracy / macro-F1 / loss, plus where the log lives._
- **Decision:** _adopt / discard / follow up with ...
