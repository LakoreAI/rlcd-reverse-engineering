"""MLP training entrypoint.

Feature classification with a plain cross-entropy loss. Validation is the
held-out split's accuracy / macro-F1 (optionally its loss). Controlled via a
YAML config (see configs/train.yaml) with individual CLI overrides. Callbacks
(LR schedule, early stopping, best checkpoint, W&B) are wired from that file.

Usage:
    uv run python -m src.pipelines.train --config configs/train.yaml
    uv run python -m src.pipelines.train --config configs/train.yaml --epochs 50 --lr 5e-4
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)
from src.callbacks.wandb_callback import WandbCallback
from src.config import MLPConfig
from src.data import dataset_from_npz, split_train_val
from src.modules.loss import ClassificationLoss
from src.modules.model import MLPClassifier
from src.pipelines._utils import announce_training
from src.pipelines.config import TrainingConfig, load_training_config
from src.pipelines.eval import eval_per_epoch, format_report
from src.utils.io_utils import load_env, save_checkpoint, save_json
from src.utils.model_utils import detect_device, get_run_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(data_root: str, name: Optional[str]) -> Optional[Path]:
    """`<data_root>/<name>` if it exists, else None (optional splits)."""
    if not name:
        return None
    path = Path(data_root) / name
    return path if path.exists() else None


def build_model(model_cfg: MLPConfig, device: torch.device) -> MLPClassifier:
    return MLPClassifier(model_cfg).to(device)


def build_callbacks(
    train_cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    run_name: str,
    wandb_config: dict,
):
    callbacks = []

    lr_cb = build_lr_scheduler(optimizer, train_cfg.lr_scheduler)
    if lr_cb is not None:
        callbacks.append(lr_cb)

    if train_cfg.early_stopping and train_cfg.early_stopping.get("enabled", True):
        es_kwargs = {
            k: v for k, v in train_cfg.early_stopping.items() if k != "enabled"
        }
        callbacks.append(EarlyStopping(**es_kwargs))

    if train_cfg.save_best:
        callbacks.append(
            BestCheckpoint(monitor=train_cfg.best_metric, mode=train_cfg.best_mode)
        )

    # after BestCheckpoint: relies on <ckpt_dir>/best.pt already existing
    if train_cfg.wandb and train_cfg.wandb.get("enabled", False):
        wb = train_cfg.wandb
        callbacks.append(
            WandbCallback(
                project_name=wb.get("project", "ai-research-template"),
                run_name=run_name,
                config=wandb_config,
                entity=wb.get("entity"),
                monitor=wb.get("monitor", train_cfg.best_metric),
                mode=wb.get("mode", train_cfg.best_mode),
                log_artifacts=wb.get("log_artifacts", True),
                group=wb.get("group"),
            )
        )

    return callbacks


def build_loaders(
    train_cfg: TrainingConfig,
    device: torch.device,
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dataset]:
    """Returns (train_loader, val_loader, test_loader, train_dataset).

    Preference order for validation: an explicit `val_file`; otherwise a
    deterministic split of the train set when `val_fraction > 0`.
    """
    train_path = _resolve(train_cfg.data_root, train_cfg.train_file)
    if train_path is None:
        raise FileNotFoundError(
            f"training split not found: {Path(train_cfg.data_root) / train_cfg.train_file}"
        )
    train_dataset = dataset_from_npz(train_path)

    val_path = _resolve(train_cfg.data_root, train_cfg.val_file)
    if val_path is not None:
        val_dataset: Optional[Dataset] = dataset_from_npz(val_path)
    else:
        train_dataset, val_dataset = split_train_val(
            train_dataset, train_cfg.val_fraction, train_cfg.seed
        )

    test_path = _resolve(train_cfg.data_root, train_cfg.test_file)
    test_dataset = dataset_from_npz(test_path) if test_path is not None else None

    pin_memory = train_cfg.pin_memory and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
        )
        if test_dataset is not None
        else None
    )
    return train_loader, val_loader, test_loader, train_dataset


def _optimizer_step(
    optimizer, scaler, callbacks, ctx, pending_losses, step, epoch, log_every
):
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    step += 1
    avg_loss = sum(pending_losses) / len(pending_losses)
    state = TrainerState(step=step, train_loss=avg_loss, epoch=epoch)
    for cb in callbacks:
        cb.on_step_end(ctx, state)
    if step % log_every == 0 or step == 1:
        print(f"  epoch {epoch:3d}  step {step:5d}  loss={avg_loss:.4f}")
    return step, avg_loss


def train_per_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    accum_steps,
    device,
    callbacks,
    ctx,
    step,
    epoch,
    log_every,
):
    model.train()
    epoch_losses, pending = [], []
    optimizer.zero_grad()

    for x, labels in loader:
        x = x.float().to(device)
        labels = labels.reshape(-1).long().to(device)

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits, _ = model(x)
            loss = criterion(logits, labels)

        scaler.scale(loss / accum_steps).backward()
        pending.append(loss.item())

        if len(pending) == accum_steps:
            step, avg_loss = _optimizer_step(
                optimizer, scaler, callbacks, ctx, pending, step, epoch, log_every
            )
            epoch_losses.append(avg_loss)
            pending = []

    if pending:
        step, avg_loss = _optimizer_step(
            optimizer, scaler, callbacks, ctx, pending, step, epoch, log_every
        )
        epoch_losses.append(avg_loss)

    return (
        sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan"),
        step,
    )


def train(train_cfg: TrainingConfig):
    load_env(REPO_ROOT / ".env")  # WANDB_API_KEY for callbacks
    device = detect_device()
    print(f"device: {device}")

    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
    elif device.type == "mps":
        torch.mps.manual_seed(train_cfg.seed)

    train_loader, val_loader, test_loader, train_dataset = build_loaders(
        train_cfg, device
    )
    num_classes = train_dataset.num_classes
    input_dim = train_dataset.input_dim
    print(f"classes: {num_classes}  input_dim: {input_dim}")

    model_cfg = MLPConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        **(train_cfg.arch or {}),
    )
    model = build_model(model_cfg, device)
    criterion = ClassificationLoss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu",
        enabled=(train_cfg.amp and device.type == "cuda"),
    )

    run_name = train_cfg.run_name or get_run_name(
        "mlp",
        Path(train_cfg.data_root).name,
        train_cfg.lr,
        train_cfg.batch_size,
    )
    ckpt_dir = Path(train_cfg.ckpt_dir) / run_name
    result_dir = Path(train_cfg.result_dir) / run_name

    callbacks = build_callbacks(
        train_cfg, optimizer, run_name, {**asdict(model_cfg), **asdict(train_cfg)}
    )
    early_stoppers = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]

    wandb_run_id = None
    for cb in callbacks:
        if isinstance(cb, WandbCallback) and cb.enabled and cb.run is not None:
            wandb_run_id = cb.run.id

    ctx = TrainContext(
        model=model,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        cfg_dict=asdict(model_cfg),
        wandb_run_id=wandb_run_id,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    if train_cfg.resume_from:
        raw = torch.load(train_cfg.resume_from, map_location=str(device))
        model.load_state_dict(raw["model"])
        if "optimizer" in raw:
            optimizer.load_state_dict(raw["optimizer"])
        step = int(raw.get("step", 0))
        start_epoch = int(raw.get("extra", {}).get("epoch", 0)) + 1
        print(f"resumed at step {step}, continuing from epoch {start_epoch}")
    else:
        step, start_epoch = 0, 1

    announce_training(
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

    history = []
    stopped_early = False
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        train_loss, step = train_per_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            train_cfg.accum_steps,
            device,
            callbacks,
            ctx,
            step,
            epoch,
            train_cfg.log_every,
        )
        record = {"epoch": epoch, "step": step, "loss": train_loss}
        print(f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={train_loss:.4f}")

        should_validate = epoch % train_cfg.eval_every == 0 or epoch == train_cfg.epochs
        if should_validate:
            val_result = (
                eval_per_epoch(
                    model,
                    val_loader,
                    device,
                    num_classes=num_classes,
                    criterion=criterion,
                )
                if val_loader is not None
                else None
            )
            # Report the test split when present, else fall back to the val
            # split so a run without a held-out test still has metrics.
            metric_result = (
                eval_per_epoch(
                    model,
                    test_loader,
                    device,
                    num_classes=num_classes,
                    criterion=criterion,
                )
                if test_loader is not None
                else val_result
            )
            extra = {}
            if metric_result is not None:
                extra = {
                    "accuracy": metric_result["accuracy"],
                    "f1": metric_result["f1"],
                    "metric_loss": metric_result["loss"],
                }
                record.update(extra)
                print(format_report(metric_result))
            state = TrainerState(
                step=step,
                train_loss=train_loss,
                epoch=epoch,
                val_loss=val_result["loss"] if val_result is not None else None,
                extra=extra,
            )
            record["val_loss"] = state.val_loss
            for cb in callbacks:
                cb.on_validation_end(ctx, state)
            if any(es.should_stop for es in early_stoppers):
                stopped_early = True

        history.append(record)

        if epoch % train_cfg.ckpt_every == 0 or epoch == train_cfg.epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                step,
                ckpt_path,
                extra={
                    "cfg": asdict(model_cfg),
                    "epoch": epoch,
                    "wandb_run_id": ctx.wandb_run_id,
                },
            )
            print(f"  saved checkpoint: {ckpt_path}")

        if stopped_early:
            break

    for cb in callbacks:
        cb.on_train_end(ctx)

    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")
    print(
        f"\n{'stopped early' if stopped_early else 'finished'} "
        f"at epoch {epoch} (step {step})"
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML, see configs/train.yaml"
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--ckpt_every", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--pin_memory", action="store_true", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train_cfg = load_training_config(args.config, **overrides)
    train(train_cfg)
