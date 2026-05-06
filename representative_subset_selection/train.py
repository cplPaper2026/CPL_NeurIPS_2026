"""Training and evaluation orchestration.

Top-level entry point :func:`run` builds the loaders + model + comet
experiment, loops epochs (or short-circuits to eval-only), and persists
checkpoints + visualisations to ``cfg.run_dir``.
"""

from __future__ import annotations

import logging
import os
import random

import comet_utils
import numpy as np
import torch
import viz
from data import Loaders, build_loaders
from gt import batched_indices_to_mask, random_gt_permutations
from losses import ar_loss, bce_loss, cpl_loss, hungarian_loss
from metrics import MetricRow, evaluate, print_table, rows_to_dict
from models import build_model
from tqdm import tqdm

from config import Config, save_config_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch (CPU+CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Per-batch step helpers
# ---------------------------------------------------------------------------


def _step_cpl(model, optimizer, X, pad, gt_masks, gt_indices, mask, cfg: Config) -> tuple[float, dict[str, float]]:
    theta, W = model(X, pad)
    loss, stats = cpl_loss(
        theta,
        W,
        gt_masks,
        mask,
        cfg.selection.iterations,
        cfg.cpl.pre_eos_weight,
        cfg.cpl.post_eos_weight,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), stats


def _step_bce(model, optimizer, X, pad, gt_masks, gt_indices, mask, cfg: Config) -> tuple[float, dict[str, float]]:
    logits = model(X, pad)
    loss, stats = bce_loss(logits, gt_masks, mask, pos_weight=cfg.bce.pos_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), stats


def _step_hungarian(model, optimizer, X, pad, gt_masks, gt_indices, mask, cfg: Config) -> tuple[float, dict[str, float]]:
    logits, h = model(X, pad, return_h=True)
    loss, stats = hungarian_loss(logits, h, X, gt_indices, mask, cfg.hungarian)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), stats


def _step_ar(model, optimizer, X, pad, gt_masks, gt_indices, mask, cfg: Config) -> tuple[float, dict[str, float]]:
    n_it = max(1, int(cfg.selection.iterations))
    h = model.encode(X, pad)
    loss_sum = None
    stats_acc: dict[str, float] = {}
    for _ in range(n_it):
        perm, perm_valid = random_gt_permutations(gt_indices)
        logits = model(X, pad, perm, perm_valid, encoder_hidden=h)
        loss, stats = ar_loss(logits, gt_indices, perm, perm_valid, mask, cfg.ar)
        loss_sum = loss if loss_sum is None else loss_sum + loss
        for k, v in stats.items():
            stats_acc[k] = stats_acc.get(k, 0.0) + float(v)
    loss = loss_sum / float(n_it)
    stats = {k: v / float(n_it) for k, v in stats_acc.items()}
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), stats


_STEP_FNS = {
    "cpl": _step_cpl,
    "bce": _step_bce,
    "hungarian": _step_hungarian,
    "ar": _step_ar,
}


def train_one_epoch(model, loader, optimizer, cfg: Config, device: torch.device) -> tuple[float, dict[str, float]]:
    """One pass over the training loader; returns mean loss and mean per-batch stats."""
    model.train()
    total = 0.0
    stat_acc: dict[str, float] = {}
    n = 0
    step = _STEP_FNS[cfg.train.method]
    for X, cluster_labels, _k, _ds, gt_indices, mask in tqdm(loader, desc="  train", leave=False):
        X = X.to(device)
        mask = mask.to(device)
        gt_indices = gt_indices.to(device)
        pad = ~mask
        # Derived boolean mask for the cpl/bce losses; hungarian uses gt_indices directly.
        gt_masks = batched_indices_to_mask(gt_indices, n=X.shape[1])
        loss_val, batch_stats = step(model, optimizer, X, pad, gt_masks, gt_indices, mask, cfg)
        total += loss_val
        for k, v in batch_stats.items():
            stat_acc[k] = stat_acc.get(k, 0.0) + float(v)
        n += 1
    denom = max(n, 1)
    mean_stats = {k: stat_acc[k] / denom for k in sorted(stat_acc)}
    return total / denom, mean_stats


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _ckpt_path(run_dir: str, name: str) -> str:
    return os.path.join(run_dir, f"{name}.pth")


def save_checkpoint(
    run_dir: str,
    name: str,
    model: torch.nn.Module,
    cfg: Config,
    epoch: int,
    metrics: dict[str, float] | None = None,
) -> str:
    """Persist a checkpoint dict containing model + cfg snapshot + metrics."""
    os.makedirs(run_dir, exist_ok=True)
    path = _ckpt_path(run_dir, name)
    payload = {
        "model": model.state_dict(),
        "method": cfg.train.method,
        "dataset": cfg.data.dataset,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device) -> dict:
    """Load weights into ``model`` and return the full checkpoint dict."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt if isinstance(ckpt, dict) else {"model": ckpt}


# ---------------------------------------------------------------------------
# Best-row selection (used for "best.pth")
# ---------------------------------------------------------------------------


def _select_best_row(rows: list[MetricRow], cfg: Config) -> MetricRow | None:
    """Pick the row that summarises the run for best-checkpoint tracking.

    Uses ``clu_f1`` as the primary criterion. For ``cpl`` / ``ar`` we
    return the matching greedy-decode row; for ``bce`` / ``hungarian`` we
    return the best sweep row (ignoring the KMeans reference).
    """
    if not rows:
        return None
    if cfg.train.method == "cpl":
        for r in rows:
            if r.name == "CPL":
                return r
        return rows[0]
    if cfg.train.method == "ar":
        for r in rows:
            if r.name == "AR":
                return r
        return rows[0]
    prefix = "BCE-" if cfg.train.method == "bce" else "HUN-"
    candidates = [r for r in rows if r.name.startswith(prefix)]
    if not candidates:
        return rows[0]
    return max(candidates, key=lambda r: r.clu_f1)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _train_comet_metrics(cfg: Config, train_loss: float, train_stats: dict[str, float]) -> dict[str, float]:
    """Metrics logged each epoch: overall train loss plus per-component stats under train/<method>/."""
    out: dict[str, float] = {"train/loss": float(train_loss)}
    method = cfg.train.method
    for k, v in train_stats.items():
        out[f"train/{method}/{k}"] = float(v)
    return out


def _maybe_log_eval(
    experiment,
    epoch: int,
    train_metrics: dict[str, float],
    rows: list[MetricRow],
    timing_report: dict[str, float] | None = None,
) -> None:
    metrics = dict(train_metrics)
    metrics.update(rows_to_dict(rows, prefix="val"))
    if timing_report:
        for k, v in timing_report.items():
            metrics[f"val/timing/{k}"] = float(v)
    comet_utils.log_metrics(experiment, metrics, step=epoch)


def _save_pr_plot(rows: list[MetricRow], run_dir: str, epoch: int, experiment) -> None:
    plot_path = os.path.join(run_dir, f"pr_epoch_{epoch:04d}.png")
    viz.save_pr_scatter(rows, plot_path)
    comet_utils.log_image(experiment, plot_path, name="pr_curve", step=epoch)


def _evaluate_and_log(
    model,
    loaders: Loaders,
    cfg: Config,
    device: torch.device,
    out_dir: str,
    epoch: int,
    train_metrics: dict[str, float],
    experiment,
) -> tuple[list[MetricRow], MetricRow | None]:
    timing_report: dict[str, float] | None = {} if cfg.train.log_inference_timing else None
    rows = evaluate(
        model,
        loaders.val_loader,
        cfg,
        device,
        timing_report=timing_report,
    )
    print_table(rows)
    _maybe_log_eval(experiment, epoch, train_metrics, rows, timing_report=timing_report)
    _save_pr_plot(rows, out_dir, epoch, experiment)
    viz.render_validation_visualizations(
        model,
        loaders.val_sets,
        loaders.test_ds_raw,
        loaders.test_labels,
        cfg,
        device,
        out_dir=out_dir,
        epoch=epoch,
        experiment=experiment,
    )
    return rows, _select_best_row(rows, cfg)


def run(cfg: Config) -> list[MetricRow]:
    """Entry point: builds everything, trains/eval-only, returns final rows."""
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = cfg.run_dir
    os.makedirs(run_dir, exist_ok=True)
    save_config_json(os.path.join(run_dir, "config.json"), cfg)

    loaders = build_loaders(cfg, device)
    model = build_model(cfg, device)
    experiment = comet_utils.build_experiment(cfg)

    if cfg.train.load_model or cfg.train.eval_only:
        ckpt_path = cfg.train.checkpoint_path or _ckpt_path(run_dir, "best")
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"--load-model/--eval-only: checkpoint not found at {ckpt_path}")
        load_checkpoint(ckpt_path, model, device)
        print(f"Loaded weights from {ckpt_path}")

    if cfg.train.eval_only:
        rows, _ = _evaluate_and_log(
            model,
            loaders,
            cfg,
            device,
            run_dir,
            epoch=0,
            train_metrics=_train_comet_metrics(cfg, float("nan"), {}),
            experiment=experiment,
        )
        comet_utils.end(experiment)
        return rows

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    print(f"Training {cfg.train.method.upper()} on {cfg.data.dataset} for {cfg.train.epochs} epochs ...")

    best_clu_f1 = -1.0
    last_rows: list[MetricRow] = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_stats = train_one_epoch(model, loaders.train_loader, optimizer, cfg, device)
        train_metrics = _train_comet_metrics(cfg, train_loss, train_stats)
        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f}")

        do_eval = cfg.train.eval_every > 0 and (epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs)
        if not do_eval:
            comet_utils.log_metrics(experiment, train_metrics, step=epoch)
            continue

        last_rows, best_row = _evaluate_and_log(model, loaders, cfg, device, run_dir, epoch, train_metrics, experiment)
        save_checkpoint(
            run_dir,
            f"epoch_{epoch:04d}",
            model,
            cfg,
            epoch,
            metrics=rows_to_dict(last_rows, prefix="val"),
        )
        if best_row is not None and best_row.clu_f1 > best_clu_f1:
            best_clu_f1 = best_row.clu_f1
            ckpt_path = save_checkpoint(run_dir, "best", model, cfg, epoch, metrics=rows_to_dict(last_rows, prefix="val"))
            print(f"  new best CluF1={best_clu_f1:.4f} ({best_row.name}) -> {ckpt_path}")

    comet_utils.end(experiment)
    return last_rows


__all__ = ["load_checkpoint", "run", "save_checkpoint", "seed_everything", "train_one_epoch"]
