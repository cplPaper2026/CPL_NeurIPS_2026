"""Cluster-level evaluation metrics and inference-rule dispatch.

The :func:`evaluate` entry point runs a single pass over the validation
loader, caches per-batch model outputs, and reports one :class:`MetricRow`
per inference rule. Rows contain four scalars only:

* ``CluRec``  : fraction of GT clusters covered by at least one prediction.
* ``CluPrec`` : (# distinct cluster ids hit by preds) / (# preds).
* ``CluF1``   : harmonic mean of ``CluRec`` and ``CluPrec``.
* ``CardErr`` : mean ``|pred_count - gt_count|`` across the batch.

The exact ground-truth token index is *not* required for any of these
metrics; only ``cluster_labels`` are consumed. This matches the
random-instance task definition: any token in the right cluster counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from inference import (
    ar_greedy,
    bce_threshold,
    cpl_greedy,
    kmeans_oracle_k_pred_mask,
)
from runtime_profile import (
    CudaTimerStats,
    aggregate_inference_timing,
    attach_inference_hooks,
    cuda_timing_section,
    detach_forward_hooks,
    print_inference_timing_report,
    warmup_inference,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config

# ---------------------------------------------------------------------------
# Per-sample metric primitives
# ---------------------------------------------------------------------------


def compute_cluster_metrics(
    pred_mask: torch.Tensor,
    cluster_labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Return ``(clu_rec, clu_prec, clu_f1, card_err)`` averaged over the batch.

    * ``clu_rec``  : fraction of GT clusters with >=1 predicted token.
    * ``clu_prec`` : (# unique clusters hit by preds) / (# preds). When a
                     batch element has 0 predictions this counts as 0
                     (coverage-first convention).
    * ``clu_f1``   : harmonic mean of ``clu_rec`` and ``clu_prec`` computed
                     per-sample and then averaged (so it tracks the row-wise
                     quantities rather than the harmonic mean of the means).
    * ``card_err`` : mean ``|pred_count - gt_count|``.
    """
    B = pred_mask.shape[0]
    device = pred_mask.device

    rec_vals: list[float] = []
    prec_vals: list[float] = []
    f1_vals: list[float] = []
    card_errs: list[float] = []

    for b in range(B):
        vm = valid_mask[b]
        lab_full = cluster_labels[b]
        gt_ids = lab_full[vm].unique()
        gt_ids = gt_ids[gt_ids >= 0]
        n_clusters = int(gt_ids.numel())
        if n_clusters == 0:
            continue

        pred_idx = (pred_mask[b] & vm).nonzero(as_tuple=True)[0]
        n_pred = int(pred_idx.numel())
        card_errs.append(abs(float(n_pred - n_clusters)))

        if n_pred == 0:
            rec_vals.append(0.0)
            prec_vals.append(0.0)
            f1_vals.append(0.0)
            continue

        pred_labs = lab_full[pred_idx]
        hit = torch.isin(gt_ids, pred_labs)
        rec_b = float(hit.float().mean().item())
        uniq = int(pred_labs.unique().numel())
        prec_b = uniq / float(n_pred)
        f1_b = 2.0 * rec_b * prec_b / max(rec_b + prec_b, 1e-8) if (rec_b + prec_b) > 0 else 0.0
        rec_vals.append(rec_b)
        prec_vals.append(prec_b)
        f1_vals.append(f1_b)

    if not rec_vals:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(rec_vals)),
        float(np.mean(prec_vals)),
        float(np.mean(f1_vals)),
        float(np.mean(card_errs)),
    )


# ---------------------------------------------------------------------------
# MetricRow + aggregation
# ---------------------------------------------------------------------------


@dataclass
class MetricRow:
    """All scalar metrics produced for one named inference rule."""

    name: str
    clu_rec: float
    clu_prec: float
    clu_f1: float
    card_err: float

    def as_tuple(self) -> tuple:
        return (self.name, self.clu_rec, self.clu_prec, self.clu_f1, self.card_err)


def _row_for_pred(
    name: str,
    pred: torch.Tensor,
    cluster_labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> MetricRow:
    cr, cp, cf1, ce = compute_cluster_metrics(pred, cluster_labels, valid_mask)
    return MetricRow(name, cr, cp, cf1, ce)


def _aggregate(rows: list[MetricRow]) -> MetricRow:
    """Average a list of per-batch :class:`MetricRow` values."""
    if not rows:
        return MetricRow("AGG", 0.0, 0.0, 0.0, 0.0)
    n = float(len(rows))
    return MetricRow(
        rows[0].name,
        sum(r.clu_rec for r in rows) / n,
        sum(r.clu_prec for r in rows) / n,
        sum(r.clu_f1 for r in rows) / n,
        sum(r.card_err for r in rows) / n,
    )


# ---------------------------------------------------------------------------
# Snapshot collection (single forward pass)
# ---------------------------------------------------------------------------


@dataclass
class _Snap:
    """Per-batch cached forward outputs used to build every metric row.

    For CPL ``theta`` is ``(B, N+1)`` and ``W`` is ``(B, N+1, N+1)``.
    For BCE / Hungarian ``theta`` holds the per-token ``(B, N)`` logits
    and ``W`` is None. For AR we cache the final greedy-decode prediction
    mask in ``ar_pred`` (one autoregressive run per batch); ``theta`` is
    not used and is kept as a dummy zero tensor.
    """

    X: torch.Tensor
    mask: torch.Tensor
    cluster_labels: torch.Tensor
    theta: torch.Tensor  # (B, N+1) for CPL; (B, N) logits for BCE/Hungarian
    W: torch.Tensor | None  # (B, N+1, N+1) only for CPL
    ar_pred: torch.Tensor | None = None  # (B, N) bool for AR


@torch.no_grad()
def _collect_snapshots(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    stats: CudaTimerStats | None = None,
) -> tuple[list[_Snap], int, int]:
    """One pass over ``loader``: per batch run model, cache outputs.

    Returns ``(snaps, num_batches, num_examples)``.
    """
    model.eval()
    snaps: list[_Snap] = []
    num_examples = 0
    for X, cluster_labels, _k, _ds, _gt, mask in tqdm(loader, desc="  eval ", leave=False):
        X_dev = X.to(device)
        mask_dev = mask.to(device)
        pad = ~mask_dev
        B = int(X_dev.shape[0])
        num_examples += B
        if method == "cpl":
            with cuda_timing_section(stats, "forward_total_ms"):
                theta_full, W = model(X_dev, pad)
            snaps.append(
                _Snap(
                    X=X_dev.cpu(),
                    mask=mask_dev.cpu(),
                    cluster_labels=cluster_labels.cpu(),
                    theta=theta_full.cpu(),
                    W=W.cpu(),
                )
            )
        elif method == "ar":
            # AR is autoregressive at inference: we cannot replay it from a
            # cached tensor, so we run greedy decode once here and stash the
            # final per-token prediction mask. The ``theta`` slot stays a
            # zero placeholder to keep the dataclass shape consistent.
            ar_max_steps = int(getattr(model, "max_selection_steps", 20))
            with cuda_timing_section(stats, "ar_inference_total_ms"):
                pred = ar_greedy(
                    model,
                    X_dev,
                    mask_dev,
                    ar_max_steps,
                    timing_stats=stats,
                )
            B_ar, N, _ = X_dev.shape
            snaps.append(
                _Snap(
                    X=X_dev.cpu(),
                    mask=mask_dev.cpu(),
                    cluster_labels=cluster_labels.cpu(),
                    theta=torch.zeros(B_ar, N),
                    W=None,
                    ar_pred=pred.detach().cpu(),
                )
            )
        else:
            with cuda_timing_section(stats, "forward_total_ms"):
                logits = model(X_dev, pad)
            snaps.append(
                _Snap(
                    X=X_dev.cpu(),
                    mask=mask_dev.cpu(),
                    cluster_labels=cluster_labels.cpu(),
                    theta=logits.cpu(),
                    W=None,
                )
            )
    num_batches = len(snaps)
    return snaps, num_batches, num_examples


# ---------------------------------------------------------------------------
# Per-method evaluators
# ---------------------------------------------------------------------------


def _evaluate_cpl(
    snaps: list[_Snap],
    cfg: Config,
    device: torch.device,
    stats: CudaTimerStats | None = None,
) -> list[MetricRow]:
    """CPL: greedy decode + KMeans-oracleK reference."""
    rows_per_rule: dict[str, list[MetricRow]] = {"CPL": [], "KMeans-oracleK": []}
    km_rng = np.random.default_rng(cfg.train.seed)
    for s in snaps:
        theta_full = s.theta.to(device)
        W = s.W.to(device)  # type: ignore[union-attr]
        mask = s.mask.to(device)
        cl = s.cluster_labels.to(device)
        if stats is not None:
            with cuda_timing_section(stats, "cpl_greedy_ms"):
                pred = cpl_greedy(theta_full, W, cfg.cpl.max_selection_steps, mask)
        else:
            pred = cpl_greedy(theta_full, W, cfg.cpl.max_selection_steps, mask)
        rows_per_rule["CPL"].append(_row_for_pred("CPL", pred, cl, mask))
        km_pred = kmeans_oracle_k_pred_mask(s.X, s.mask, s.cluster_labels, km_rng).to(device)
        rows_per_rule["KMeans-oracleK"].append(_row_for_pred("KMeans-oracleK", km_pred, cl, mask))
    return [_aggregate(rs) for rs in rows_per_rule.values()]


def _build_percentile_thresholds(snaps: list[_Snap], percentiles: tuple[int, ...]) -> dict[int, float]:
    """Pool ``sigmoid(logit)`` over all valid tokens to compute percentiles."""
    parts: list[torch.Tensor] = []
    for s in snaps:
        scores = torch.sigmoid(s.theta)
        for b in range(s.mask.shape[0]):
            parts.append(scores[b][s.mask[b]])
    if not parts:
        return dict.fromkeys(percentiles, 0.0)
    pool = torch.cat(parts).numpy()
    return {p: float(np.percentile(pool, p)) for p in percentiles}


def _evaluate_sigmoid_sweep(
    snaps: list[_Snap],
    cfg: Config,
    device: torch.device,
    rule_prefix: str,
    thresholds: tuple[float, ...],
    percentiles: tuple[int, ...],
) -> list[MetricRow]:
    """Shared BCE / Hungarian eval: threshold + percentile sweep + KMeans."""
    out: list[MetricRow] = []
    km_rng = np.random.default_rng(cfg.train.seed)

    pct_map = _build_percentile_thresholds(snaps, percentiles)

    rules: list[tuple[str, callable]] = []
    for thr in thresholds:
        rules.append(
            (
                f"{rule_prefix}-t{thr:.2f}",
                lambda s, t=thr: bce_threshold(s.theta.to(device), t, s.mask.to(device)),
            )
        )
    for pct in percentiles:
        thr = pct_map[pct]
        rules.append(
            (
                f"{rule_prefix}-p{pct}",
                lambda s, t=thr: (torch.sigmoid(s.theta.to(device)) >= t) & s.mask.to(device),
            )
        )
    rules.append(
        (
            "KMeans-oracleK",
            lambda s, _rng=km_rng: kmeans_oracle_k_pred_mask(s.X, s.mask, s.cluster_labels, _rng).to(device),
        )
    )

    for name, rule in rules:
        rows: list[MetricRow] = []
        for s in snaps:
            pred = rule(s)
            mask = s.mask.to(device)
            cl = s.cluster_labels.to(device)
            rows.append(_row_for_pred(name, pred, cl, mask))
        out.append(_aggregate(rows))
    return out


def _evaluate_bce(snaps: list[_Snap], cfg: Config, device: torch.device) -> list[MetricRow]:
    return _evaluate_sigmoid_sweep(
        snaps,
        cfg,
        device,
        rule_prefix="BCE",
        thresholds=cfg.bce.thresholds,
        percentiles=cfg.bce.percentiles,
    )


def _evaluate_hungarian(snaps: list[_Snap], cfg: Config, device: torch.device) -> list[MetricRow]:
    return _evaluate_sigmoid_sweep(
        snaps,
        cfg,
        device,
        rule_prefix="HUN",
        thresholds=cfg.hungarian.thresholds,
        percentiles=cfg.hungarian.percentiles,
    )


def _evaluate_ar(snaps: list[_Snap], cfg: Config, device: torch.device) -> list[MetricRow]:
    """AR: cached greedy-decode predictions + KMeans-oracleK reference."""
    rows_per_rule: dict[str, list[MetricRow]] = {"AR": [], "KMeans-oracleK": []}
    km_rng = np.random.default_rng(cfg.train.seed)
    for s in snaps:
        if s.ar_pred is None:
            continue
        mask = s.mask.to(device)
        cl = s.cluster_labels.to(device)
        pred = s.ar_pred.to(device)
        rows_per_rule["AR"].append(_row_for_pred("AR", pred, cl, mask))
        km_pred = kmeans_oracle_k_pred_mask(s.X, s.mask, s.cluster_labels, km_rng).to(device)
        rows_per_rule["KMeans-oracleK"].append(_row_for_pred("KMeans-oracleK", km_pred, cl, mask))
    return [_aggregate(rs) for rs in rows_per_rule.values()]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    timing_report: dict[str, float] | None = None,
) -> list[MetricRow]:
    """Run the full evaluation pipeline and return one row per inference rule.

    When ``cfg.train.log_inference_timing`` is True (or ``timing_report`` is
    passed), runs ``cfg.train.inference_timing_warmup_batches`` batches
    without hooks (see :func:`runtime_profile.warmup_inference`), then attaches
    forward hooks for encoder/decoder timing and prints an aggregate breakdown.
    Optional ``timing_report`` is filled with the same floats for Comet / callers.
    """
    use_timing = cfg.train.log_inference_timing or timing_report is not None
    stats = CudaTimerStats() if use_timing else None
    hooks: list = []
    if stats is not None:
        n_warm = int(cfg.train.inference_timing_warmup_batches)
        if n_warm > 0:
            warmup_inference(
                model,
                loader,
                device,
                n_warm,
                cfg.train.method,
                cpl_max_selection_steps=int(cfg.cpl.max_selection_steps),
                ar_max_selection_steps=int(getattr(model, "max_selection_steps", cfg.ar.max_selection_steps)),
            )
        hooks = attach_inference_hooks(model, cfg.train.method, stats)

    try:
        snaps, num_batches, num_examples = _collect_snapshots(model, loader, device, cfg.train.method, stats=stats)
        if cfg.train.method == "cpl":
            rows = _evaluate_cpl(snaps, cfg, device, stats=stats)
        elif cfg.train.method == "bce":
            rows = _evaluate_bce(snaps, cfg, device)
        elif cfg.train.method == "hungarian":
            rows = _evaluate_hungarian(snaps, cfg, device)
        elif cfg.train.method == "ar":
            rows = _evaluate_ar(snaps, cfg, device)
        else:
            raise ValueError(f"unknown method {cfg.train.method!r}")

        if stats is not None:
            rep = aggregate_inference_timing(stats, cfg.train.method, num_batches, num_examples)
            print_inference_timing_report(rep)
            if timing_report is not None:
                timing_report.clear()
                timing_report.update(rep)
    finally:
        detach_forward_hooks(hooks)

    return rows


# ---------------------------------------------------------------------------
# Pretty-printing + logging
# ---------------------------------------------------------------------------


def rows_to_dict(rows: list[MetricRow], prefix: str = "val") -> dict[str, float]:
    """Flat ``{prefix}/{rule}/{metric} -> value`` dict for Comet logging."""
    out: dict[str, float] = {}
    for r in rows:
        for k, v in (
            ("clu_rec", r.clu_rec),
            ("clu_prec", r.clu_prec),
            ("clu_f1", r.clu_f1),
            ("card_err", r.card_err),
        ):
            out[f"{prefix}/{r.name}/{k}"] = float(v)
    return out


def print_table(rows: list[MetricRow]) -> None:
    """Compact aligned table of the four cluster-level metrics."""
    name_width = max(15, max((len(r.name) for r in rows), default=15))
    header = f"{'Method':<{name_width}} {'CluRec':>7} {'CluPrec':>8} {'CluF1':>7} {'CardErr':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.name:<{name_width}} {r.clu_rec:7.4f} {r.clu_prec:8.4f} {r.clu_f1:7.4f} {r.card_err:8.2f}")
    print()


__all__ = [
    "MetricRow",
    "compute_cluster_metrics",
    "evaluate",
    "print_table",
    "rows_to_dict",
]
