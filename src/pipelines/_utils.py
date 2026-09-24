"""Presentation helpers for the training entrypoint.

Kept separate from `src.pipelines.train` so the training loop stays readable and
this introspection can be reused (a debug script) or tested on its own. Nothing
here affects training — it only prints.

Output is rendered with `rich` when it is installed (panels, tables, JSON
highlighting) and falls back to aligned plain text otherwise. `rich` is an
optional extra: `uv sync --extra rich`.
"""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import DecisionModelConfig
from src.data import TypedDecisionDataset
from src.pipelines.config import TrainingConfig

try:  # optional pretty-printing dependency
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_RICH = False

WIDTH = 72


def banner(title: str) -> None:
    print(f"\n{title}\n{'-' * WIDTH}")


def indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def qtype_counts(dataset: TypedDecisionDataset) -> Counter:
    """Question-type distribution — the typed-decision analogue of the
    inherited scaffold's per-class label counts."""
    return Counter(question["type"] for _state, question, _gold in dataset._rows)


def announce_training(
    *,
    model: torch.nn.Module,
    model_cfg: DecisionModelConfig,
    train_cfg: TrainingConfig,
    train_loader: DataLoader,
    train_dataset: TypedDecisionDataset,
    device: torch.device,
    run_name: str,
    ckpt_dir: Path,
    result_dir: Path,
) -> None:
    """Print configs, a data sample, and the model layout before training."""
    common = dict(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        train_dataset=train_dataset,
        device=device,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
    )
    if _HAS_RICH:
        _announce_rich(**common)
    else:
        _announce_plain(**common)


def _forward_check(model: torch.nn.Module, batch: dict, device) -> str:
    """A no-grad forward on a real batch; never raises."""
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(
                batch["ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
        model.train(was_training)
        preview = " ".join(
            f"{v:+.3f}" for v in logits[0, : min(6, logits.shape[1])].tolist()
        )
        return f"logits {tuple(logits.shape)}\nlogits[0][:6] = [{preview}]"
    except Exception as e:  # noqa: BLE001 - a shape peek must never block training
        return f"forward check skipped: {e}"


# --- plain-text fallback ------------------------------------------------------


def _announce_plain(
    *,
    model,
    model_cfg,
    train_cfg,
    train_loader,
    train_dataset,
    device,
    run_name,
    ckpt_dir,
    result_dir,
) -> None:
    print("\n" + "=" * WIDTH)
    print(f"RUN  {run_name}")
    print("=" * WIDTH)
    print(f"device:      {device}   seed: {train_cfg.seed}")
    print(f"checkpoints: {ckpt_dir}")
    print(f"results:     {result_dir}")

    banner("model config (DecisionModelConfig)")
    print(indent(json.dumps(asdict(model_cfg), indent=2, default=str)))
    banner("training config (TrainingConfig)")
    print(indent(json.dumps(asdict(train_cfg), indent=2, default=str)))

    counts = qtype_counts(train_dataset)
    banner(f"data: train sample  ({len(train_dataset)} question-rows)")
    print(
        "qtype counts: "
        + "  ".join(f"{name}:{n}" for name, n in sorted(counts.items()))
    )

    batch = next(iter(train_loader))
    print(
        f"batch: ids {tuple(batch['ids'].shape)}   "
        f"markers {tuple(batch['marker_pos'].shape)}   "
        f"target {tuple(batch['target'].shape)}"
    )

    banner(f"model: {type(model).__name__}")
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
    print(f"  {'TOTAL params':<34s} {total:>14,d}")

    banner("forward check (untrained)")
    print(_forward_check(model, batch, device))

    print("\n" + "=" * WIDTH + "\n")


# --- rich rendering -----------------------------------------------------------


def _announce_rich(
    *,
    model,
    model_cfg,
    train_cfg,
    train_loader,
    train_dataset,
    device,
    run_name,
    ckpt_dir,
    result_dir,
) -> None:
    console = Console(highlight=False)
    console.print()
    console.print(Rule(f"[bold]RUN  {run_name}[/bold]"))

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan", justify="right")
    info.add_column()
    info.add_row("device", str(device))
    info.add_row("seed", str(train_cfg.seed))
    info.add_row("checkpoints", str(ckpt_dir))
    info.add_row("results", str(result_dir))
    console.print(info)

    console.print(
        Panel(
            JSON(json.dumps(asdict(model_cfg), default=str)),
            title="model config (DecisionModelConfig)",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            JSON(json.dumps(asdict(train_cfg), default=str)),
            title="training config (TrainingConfig)",
            border_style="blue",
        )
    )

    counts = qtype_counts(train_dataset)
    batch = next(iter(train_loader))
    table = Table(
        title=f"data: train sample  ({len(train_dataset)} question-rows)",
        border_style="green",
        caption="  ".join(f"{name}:{n}" for name, n in sorted(counts.items())),
    )
    table.add_column("tensor")
    table.add_column("shape", justify="right")
    table.add_row("ids", str(tuple(batch["ids"].shape)))
    table.add_row("marker_pos", str(tuple(batch["marker_pos"].shape)))
    table.add_row("target", str(tuple(batch["target"].shape)))
    console.print(table)

    model_table = Table(title=f"model: {type(model).__name__}", border_style="magenta")
    model_table.add_column("component")
    model_table.add_column("params", justify="right")
    total = sum(p.numel() for p in model.parameters())
    model_table.add_row(
        "encoder", f"{sum(p.numel() for p in model.encoder.parameters()):,d}"
    )
    model_table.add_row("head", f"{sum(p.numel() for p in model.head.parameters()):,d}")
    model_table.add_row(
        "scorer", f"{sum(p.numel() for p in model.scorer.parameters()):,d}"
    )
    model_table.add_section()
    model_table.add_row("TOTAL", f"{total:,d}", style="bold")
    console.print(model_table)

    console.print(
        Panel(
            _forward_check(model, batch, device),
            title="forward check (untrained)",
            border_style="yellow",
        )
    )
    console.print()
