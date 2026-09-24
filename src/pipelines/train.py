"""Typed-decision (RLCD/Laya-style) training entrypoint.

Fine-tunes a pretrained bidirectional encoder plus a small transformer head
on `LocalLLaMA/typed-decisions` (or an equivalent dataset) with Laya's
combined RL+CE loss. Validation is the raw (pre-temperature) calibration
split's ECE/Brier/NLL/accuracy. Controlled via
a YAML config (see `configs/train.yaml`, `configs/rlcd_smoke.yaml`) with
individual CLI overrides. Callbacks (LR schedule, early stopping, best
checkpoint, W&B) are wired from that file and are unmodified from the
inherited scaffold — see `src/callbacks/`.

Usage:
    uv run python -m src.pipelines.train --config configs/train.yaml
    uv run python -m src.pipelines.train --config configs/rlcd_smoke.yaml --epochs 1
"""

import argparse
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)
from src.callbacks.wandb_callback import WandbCallback
from src.config import DecisionModelConfig
from src.data import TypedDecisionDataset, collate_fn, split_train_calib
from src.modules.loss import rlcd_loss
from src.modules.model import DecisionModel
from src.pipelines._utils import announce_training
from src.pipelines.config import TrainingConfig, load_training_config
from src.pipelines.eval import eval_per_epoch, evaluate, format_report
from src.utils.io_utils import load_env, save_checkpoint, save_json
from src.utils.model_utils import detect_device, get_run_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_model(model_cfg: DecisionModelConfig, device: torch.device) -> DecisionModel:
    return DecisionModel(model_cfg).to(device)


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
    model_cfg: DecisionModelConfig,
    tokenizer,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, DataLoader, TypedDecisionDataset]:
    """Returns (train_loader, calib_loader, test_loader, train_dataset).

    `calib_loader` is a case-level held-out slice of the train split (never
    the test split) used for raw-ECE validation during training and for
    post-hoc temperature fitting.
    """
    raw = load_dataset(train_cfg.dataset_name, train_cfg.dataset_config)
    train_hf, test_hf = raw["train"], raw["test"]
    if train_cfg.max_examples is not None:
        train_hf = train_hf.select(range(min(len(train_hf), train_cfg.max_examples)))
        test_hf = test_hf.select(range(min(len(test_hf), train_cfg.max_examples)))
    train_hf, calib_hf = split_train_calib(
        train_hf, train_cfg.calib_fraction, train_cfg.seed
    )

    train_dataset = TypedDecisionDataset(
        train_hf, tokenizer, model_cfg.max_len, model_cfg.head_max_len
    )
    calib_dataset = (
        TypedDecisionDataset(
            calib_hf, tokenizer, model_cfg.max_len, model_cfg.head_max_len
        )
        if calib_hf is not None
        else None
    )
    test_dataset = TypedDecisionDataset(
        test_hf, tokenizer, model_cfg.max_len, model_cfg.head_max_len
    )

    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    pin_memory = train_cfg.pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    calib_loader = (
        DataLoader(
            calib_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        if calib_dataset is not None and len(calib_dataset) > 0
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    return train_loader, calib_loader, test_loader, train_dataset


def _sigma_at_epoch(train_cfg: TrainingConfig, epoch: int) -> float:
    if not train_cfg.anneal_sigma or train_cfg.epochs <= 1:
        return train_cfg.sigma_start
    progress = (epoch - 1) / (train_cfg.epochs - 1)
    return train_cfg.sigma_start + progress * (
        train_cfg.sigma_end - train_cfg.sigma_start
    )


def _optimizer_step(optimizer, callbacks, ctx, pending_losses, step, epoch, log_every):
    optimizer.step()
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
    loader,
    optimizer,
    accum_steps,
    device,
    callbacks,
    ctx,
    step,
    epoch,
    log_every,
    sigma,
    train_cfg: TrainingConfig,
):
    model.train()
    epoch_losses, pending = [], []
    optimizer.zero_grad()

    for batch in loader:
        ids = batch["ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        marker_pos = batch["marker_pos"].to(device)
        marker_mask = batch["marker_mask"].to(device)
        target = batch["target"].to(device)
        qtype = batch["qtype"].to(device)

        logits = model(ids, attention_mask, marker_pos, marker_mask, qtype)
        loss, _parts = rlcd_loss(
            logits,
            target,
            qtype,
            marker_mask,
            sigma,
            num_noise_samples=train_cfg.num_noise_samples,
            w_ce=train_cfg.w_ce,
            w_rl=train_cfg.w_rl,
            w_sph=train_cfg.reward_w_spherical,
            w_rps=train_cfg.reward_w_rps,
        )

        (loss / accum_steps).backward()
        pending.append(loss.item())

        if len(pending) == accum_steps:
            step, avg_loss = _optimizer_step(
                optimizer, callbacks, ctx, pending, step, epoch, log_every
            )
            epoch_losses.append(avg_loss)
            pending = []

    if pending:
        step, avg_loss = _optimizer_step(
            optimizer, callbacks, ctx, pending, step, epoch, log_every
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

    model_cfg = DecisionModelConfig(**(train_cfg.arch or {}))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.encoder_name)

    train_loader, calib_loader, test_loader, train_dataset = build_loaders(
        train_cfg, model_cfg, tokenizer, device
    )
    print(f"train rows: {len(train_dataset)}")

    model = build_model(model_cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    run_name = train_cfg.run_name or get_run_name(
        "rlcd",
        train_cfg.dataset_name.split("/")[-1],
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
        sigma = _sigma_at_epoch(train_cfg, epoch)
        train_loss, step = train_per_epoch(
            model,
            train_loader,
            optimizer,
            train_cfg.accum_steps,
            device,
            callbacks,
            ctx,
            step,
            epoch,
            train_cfg.log_every,
            sigma,
            train_cfg,
        )
        record = {"epoch": epoch, "step": step, "loss": train_loss, "sigma": sigma}
        print(
            f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={train_loss:.4f}  sigma={sigma:.3f}"
        )

        should_validate = epoch % train_cfg.eval_every == 0 or epoch == train_cfg.epochs
        if should_validate:
            extra = eval_per_epoch(model, calib_loader, device)
            record.update(extra)
            if extra:
                print(
                    f"  calib raw: ECE={extra['raw_ece'] * 100:6.2f}  "
                    f"Brier={extra['raw_brier']:.4f}  acc={extra['raw_accuracy'] * 100:6.2f}"
                )
            state = TrainerState(
                step=step,
                train_loss=train_loss,
                epoch=epoch,
                val_loss=None,
                extra=extra,
            )
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

    final_result = None
    if calib_loader is not None:
        final_result = evaluate(
            model,
            calib_loader,
            test_loader,
            device,
            save_json_path=result_dir / "test_eval.json",
        )
        print("\nfinal test evaluation:")
        print(format_report(final_result))

    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")
    print(
        f"\n{'stopped early' if stopped_early else 'finished'} "
        f"at epoch {epoch} (step {step})"
    )
    return history, final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML, see configs/train.yaml"
    )
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--calib_fraction", type=float, default=None)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="cap train/test to this many cases",
    )
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
    parser.add_argument("--w_rl", type=float, default=None)
    parser.add_argument("--w_ce", type=float, default=None)
    parser.add_argument("--sigma_start", type=float, default=None)
    parser.add_argument("--sigma_end", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train_cfg = load_training_config(args.config, **overrides)
    train(train_cfg)
