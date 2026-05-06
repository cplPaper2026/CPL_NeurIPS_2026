"""Validation-time visualisations for the random-GT cluster-selection task.

Each cluster contributes one randomly-sampled token as the ground truth
(see :mod:`gt`). The visualiser highlights, per validation sample:

* ``embed_resnet_tsne_*.png`` : t-SNE 2D scatter of raw ResNet features.
* ``embed_hidden_*.png``      : configurable 2D projection of transformer
  hidden states (t-SNE, UMAP, or PCA via ``viz.embedding``). Each cluster
  has its own ``tab20`` colour *and* its own marker shape; a green ring
  marks model-predicted tokens.
* ``hist_*.png``      : score distribution with vertical lines for each
  active threshold and a green tick per *GT-sample* score (the score
  the network assigns to the random GT token of each cluster).
* ``grid_*.png``      : compact image grid of the set; the inner spine
  is cluster-coloured (one ``tab20`` colour per cluster), the outer
  thick border encodes the GT/prediction status (blue = GT, green =
  prediction, purple = both).
* ``W_*.png``         : (CPL only) ``W`` heatmap restricted to the valid
  tokens of the sample, sorted so each cluster occupies a contiguous
  block, with the cluster's CIFAR class name as a tick label.
* ``cpl_*.png``       : (CPL only) same row-per-step layout as ``ar_*.png``:
  chosen tile + softmax over ``N+1`` slots per greedy step (cluster-grouped);
  ``EOS`` is labelled beside the last bar; the chosen slot has a black arrow.
* ``ar_*.png``        : (AR only) same layout: chosen tile + softmax over
  ``N+1`` pointer logits per step (masked like greedy ``argmax``).
* ``hungarian_*.png`` : (Hungarian only) three image columns per GT row:
  random-GT tile, Hungarian-matched prediction, and a **runner-up**
  token (cheapest alternative in the same cost-matrix column, excluding
  the winning row) for debugging. Edges from GT to both predictions are
  annotated with weighted total cost plus cls/dist components. A bottom
  strip shows ``unmatched predicted positives`` (tokens above the primary
  threshold that weren't part of the K Hungarian winners).

The aggregate :func:`save_pr_scatter` writes a single-panel
``CluPrec / CluRec`` scatter combining every inference rule (CPL,
BCE-t*/BCE-p*, HUN-t*/HUN-p*, KMeans-oracleK) on one set of axes.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import comet_utils
import matplotlib.pyplot as plt
from data import collate_sets
from gt import batched_indices_to_mask
from inference import ar_greedy, bce_threshold, cpl_greedy
from losses import _apply_hungarian_identity_column_mask
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import linear_sum_assignment

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color palette: one tab20 entry per cluster
# ---------------------------------------------------------------------------


GT_COLOR = "#1f6feb"  # blue
PRED_COLOR = "#27ae60"  # green
BOTH_COLOR = "#8e44ad"  # purple (GT and prediction collide)
RUNNER_COLOR = "#e67e22"  # runner-up pred border / dashed edge


def _cluster_color(cid: int) -> tuple[float, float, float, float]:
    """Stable ``tab20`` color per cluster id."""
    cmap = plt.colormaps.get_cmap("tab20")
    return cmap(int(cid) % 20)


_CLUSTER_MARKERS: tuple[str, ...] = (
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "<",
    ">",
    "p",
    "h",
    "8",
    "d",
    "*",
)


def _cluster_marker(cid: int) -> str:
    """Stable marker shape per cluster id (filled markers only)."""
    return _CLUSTER_MARKERS[int(cid) % len(_CLUSTER_MARKERS)]


# ---------------------------------------------------------------------------
# Dimensionality reduction (t-SNE / UMAP / PCA with PCA fallback)
# ---------------------------------------------------------------------------


def _project_2d(feats: np.ndarray, method: str) -> np.ndarray:
    """Reduce ``feats`` (N, D) to (N, 2).

    ``method`` can be ``pca``, ``umap``, or ``tsne``. Missing optional
    dependencies or numerical failures fall back to PCA (same as choosing
    ``pca`` explicitly).
    """
    n = feats.shape[0]
    if n < 3:
        out = np.zeros((n, 2), dtype=np.float32)
        if n >= 1:
            out[:, 0] = np.arange(n)
        return out

    m = (method or "pca").strip().lower()

    if m == "umap":
        try:
            import umap  # type: ignore[import-not-found]

            n_neighbors = max(2, min(15, n - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="euclidean",
                random_state=42,
            )
            return reducer.fit_transform(feats).astype(np.float32)
        except Exception as e:  # pragma: no cover - umap not always installed
            logger.warning("UMAP failed (%s); falling back to PCA", e)

    if m == "tsne":
        try:
            from sklearn.manifold import TSNE

            perplexity = float(min(30.0, max(1.0, (n - 1) / 3.0)))
            perplexity = min(perplexity, float(n - 1) - 1e-6)
            kwargs = dict(
                n_components=2,
                perplexity=perplexity,
                metric="euclidean",
                init="pca",
                random_state=42,
                max_iter=750,
            )
            try:
                reducer = TSNE(learning_rate="auto", **kwargs)
            except TypeError:
                reducer = TSNE(learning_rate=200, **kwargs)
            return reducer.fit_transform(feats).astype(np.float32)
        except Exception as e:  # pragma: no cover - sklearn optional in some envs
            logger.warning("t-SNE failed (%s); falling back to PCA", e)

    centered = feats - feats.mean(axis=0, keepdims=True)
    try:
        u, _s, _vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return centered[:, :2].astype(np.float32)
    return u[:, :2].astype(np.float32)


# ---------------------------------------------------------------------------
# Sample inference cache: collect everything once
# ---------------------------------------------------------------------------


@dataclass
class _HungarianMatch:
    """Per-sample Hungarian assignment recovered for the visualisation.

    All indices are token indices in ``[0, N)`` (i.e. valid-mask space).
    ``cls_costs`` and ``dist_costs`` contain the unweighted raw costs
    (multiply by ``cls_weight`` / ``dist_weight`` from
    :class:`HungarianConfig` to recover the assignment cost).

    ``second_pred_tokens[k]`` is ``-1`` when no column runner-up exists
    (e.g. only one valid token in the bag); otherwise it is the global
    token index that minimises the weighted cost in GT column ``k``
    among rows other than the Hungarian winner.
    """

    pred_tokens: list[int]
    gt_tokens: list[int]
    cls_costs: list[float]
    dist_costs: list[float]
    second_pred_tokens: list[int]
    second_cls_costs: list[float]
    second_dist_costs: list[float]


@dataclass
class _VizCache:
    """Per-batch cache of the data the visualiser needs.

    ``scores`` is the per-token score that's overlaid in the scatter and
    histogram (raw ``theta`` for CPL; ``sigmoid(logit)`` for BCE / Hungarian;
    for AR, the step-1 marginal ``softmax`` probability on each bag token,
    i.e. ``P(bag i)`` from the same masked ``N+1`` logits as greedy decode).
    ``cpl_step_probs`` are None for non-CPL methods. ``hungarian_matches`` is
    populated only when ``cfg.train.method == "hungarian"``. For CPL,
    ``cpl_orders[b]`` may end with the sentinel ``N`` (= EOS slot) and
    ``cpl_step_probs[b][t]`` is a length-``(N+1)`` softmax distribution.
    ``ar_orders`` and ``ar_step_probs`` are populated only when
    ``cfg.train.method == "ar"``; ``ar_orders[b]`` may end with the sentinel
    ``N`` (= EOS slot) and ``ar_step_probs[b][t]`` is a length-``(N+1)``
    softmax distribution (same masked logits as greedy ``argmax``).
    """

    X: torch.Tensor  # (B, N, D)  raw features
    cluster_labels: torch.Tensor  # (B, N)
    ds_idx: torch.Tensor  # (B, N)
    mask: torch.Tensor  # (B, N) bool
    h: torch.Tensor  # (B, N, D) transformer hidden states
    scores: torch.Tensor  # (B, N) per-token score
    gt_indices: torch.Tensor  # (B, K_max) random-GT indices, -1 = pad
    gt_mask: torch.Tensor  # (B, N) bool, derived from gt_indices
    pred_masks_primary: torch.Tensor  # (B, N) bool
    cpl_W: torch.Tensor | None  # (B, N+1, N+1) for CPL else None
    cpl_orders: list[list[int]] | None
    hungarian_matches: list[_HungarianMatch] | None
    cpl_step_probs: list[list[torch.Tensor]] | None = None
    ar_orders: list[list[int]] | None = None
    ar_step_probs: list[list[torch.Tensor]] | None = None


def _hungarian_match_per_sample(
    h: torch.Tensor,
    logits: torch.Tensor,
    gt_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: Config,
    X: torch.Tensor | None = None,
) -> list[_HungarianMatch]:
    """Replay the Hungarian assignment used by the loss for visualisation.

    The matcher is the same as :func:`losses.hungarian_loss` (same cost
    matrix, same scipy call), including
    ``cfg.hungarian.exclude_identity_match`` and
    ``cfg.hungarian.dist_feature_space``. All inputs are CPU tensors; the
    function is only called from inside the no-grad viz path so we
    don't need the sample-level numerical stability tricks.

    When ``dist_feature_space`` is ``resnet``, ``X`` must be supplied
    (bag features, same shape as ``h``).

    For each matched column ``c``, the runner-up is the row ``r2`` that
    minimises the weighted cost in that column among all rows except the
    Hungarian winner ``r`` (column-wise second best for debugging).
    """
    B, N = logits.shape
    out: list[_HungarianMatch] = []
    neg_logsig = F.logsigmoid(logits).neg().clamp_max(50.0)

    if cfg.hungarian.dist_feature_space == "resnet" and X is None:
        raise ValueError("_hungarian_match_per_sample: dist_feature_space='resnet' requires X")

    for b in range(B):
        vm = valid_mask[b]
        gt_row = gt_indices[b]
        gt_slots = gt_row[gt_row >= 0]
        K = int(gt_slots.numel())
        if K == 0:
            out.append(_HungarianMatch([], [], [], [], [], [], []))
            continue
        valid_idx = vm.nonzero(as_tuple=True)[0]
        N_valid = int(valid_idx.numel())
        if N_valid < K:
            out.append(_HungarianMatch([], [], [], [], [], [], []))
            continue

        if cfg.hungarian.dist_feature_space == "resnet":
            assert X is not None
            feats_valid = X[b, valid_idx]
            feats_gt = X[b, gt_slots]
        else:
            feats_valid = h[b, valid_idx]
            feats_gt = h[b, gt_slots]

        if cfg.hungarian.distance == "cosine":
            a_n = F.normalize(feats_valid, dim=-1)
            b_n = F.normalize(feats_gt, dim=-1)
            dist_cost = 1.0 - a_n @ b_n.T
        else:
            a2 = (feats_valid * feats_valid).sum(dim=-1, keepdim=True)
            b2 = (feats_gt * feats_gt).sum(dim=-1, keepdim=True).T
            ab = feats_valid @ feats_gt.T
            dist_cost = (a2 + b2 - 2.0 * ab).clamp_min(0.0)

        cls_cost = neg_logsig[b, valid_idx].unsqueeze(1).expand(-1, K)
        C = cfg.hungarian.cls_weight * cls_cost + cfg.hungarian.dist_weight * dist_cost
        _apply_hungarian_identity_column_mask(C, valid_idx, gt_slots, cfg.hungarian.exclude_identity_match)
        row_ind, col_ind = linear_sum_assignment(C.numpy())
        C_np = C.detach().numpy()

        pred_tokens: list[int] = []
        gt_tokens: list[int] = []
        cls_list: list[float] = []
        dist_list: list[float] = []
        second_pred: list[int] = []
        second_cls: list[float] = []
        second_dist: list[float] = []
        for r, c in zip(row_ind, col_ind):
            pred_tokens.append(int(valid_idx[r].item()))
            gt_tokens.append(int(gt_slots[c].item()))
            cls_list.append(float(cls_cost[r, c].item()))
            dist_list.append(float(dist_cost[r, c].item()))

            col = C_np[:, c].copy()
            col[r] = np.inf
            r2 = int(np.argmin(col))
            if N_valid <= 1 or not np.isfinite(col[r2]):
                second_pred.append(-1)
                second_cls.append(0.0)
                second_dist.append(0.0)
            else:
                second_pred.append(int(valid_idx[r2].item()))
                second_cls.append(float(cls_cost[r2, c].item()))
                second_dist.append(float(dist_cost[r2, c].item()))

        out.append(
            _HungarianMatch(
                pred_tokens,
                gt_tokens,
                cls_list,
                dist_list,
                second_pred,
                second_cls,
                second_dist,
            )
        )

    return out


@torch.no_grad()
def _build_cache(
    model: torch.nn.Module,
    val_sets,
    cfg: Config,
    device: torch.device,
    n_samples: int,
) -> _VizCache:
    """One forward pass over the first ``n_samples`` validation sets."""
    items = [val_sets[i] for i in range(min(n_samples, len(val_sets)))]
    X, cluster_labels, _k, ds_idx, gt_indices, mask = collate_sets(items)
    X_dev = X.to(device)
    mask_dev = mask.to(device)
    pad = ~mask_dev

    cpl_orders: list[list[int]] | None = None
    cpl_step_probs: list[list[torch.Tensor]] | None = None
    cpl_W: torch.Tensor | None = None
    hungarian_matches: list[_HungarianMatch] | None = None
    ar_orders: list[list[int]] | None = None
    ar_step_probs: list[list[torch.Tensor]] | None = None

    if cfg.train.method == "cpl":
        theta_full, W, h = model(X_dev, pad, return_h=True)
        N = theta_full.shape[1] - 1
        scores = theta_full[:, :N]
        pred, cpl_trace = cpl_greedy(
            theta_full,
            W,
            cfg.cpl.max_selection_steps,
            mask_dev,
            return_trace=True,
        )
        cpl_orders = cpl_trace.orders
        cpl_step_probs = cpl_trace.step_probs
        cpl_W = W.detach().cpu()
    elif cfg.train.method == "ar":
        # Encode once for both viz panels and the greedy decode trace.
        h = model.encode(X_dev, src_key_padding_mask=pad)
        pred, trace = ar_greedy(
            model,
            X_dev,
            mask_dev,
            cfg.ar.max_selection_steps,
            return_trace=True,
        )
        ar_orders = trace.orders
        ar_step_probs = trace.step_probs
        # Per-token score for histogram / embedding colour: step-0 softmax
        # marginal on each bag slot (same masked logits as greedy argmax).
        N = X_dev.shape[1]
        scores = torch.zeros_like(X_dev[:, :, 0])
        for b in range(X_dev.shape[0]):
            if ar_step_probs[b]:
                scores[b] = ar_step_probs[b][0][:N].to(scores.device)
    else:
        logits, h = model(X_dev, pad, return_h=True)
        scores = torch.sigmoid(logits)
        primary_thr = cfg.hungarian.primary_threshold if cfg.train.method == "hungarian" else cfg.bce.primary_threshold
        pred = bce_threshold(logits, primary_thr, mask_dev)
        if cfg.train.method == "hungarian":
            hungarian_matches = _hungarian_match_per_sample(
                h.detach().cpu(),
                logits.detach().cpu(),
                gt_indices,
                mask,
                cfg,
                X=X,
            )

    gt_mask = batched_indices_to_mask(gt_indices, n=int(X.shape[1]))
    return _VizCache(
        X=X.cpu(),
        cluster_labels=cluster_labels.cpu(),
        ds_idx=ds_idx.cpu(),
        mask=mask.cpu(),
        h=h.detach().cpu(),
        scores=scores.detach().cpu(),
        gt_indices=gt_indices.cpu(),
        gt_mask=gt_mask.cpu(),
        pred_masks_primary=pred.detach().cpu(),
        cpl_W=cpl_W,
        cpl_orders=cpl_orders,
        cpl_step_probs=cpl_step_probs,
        hungarian_matches=hungarian_matches,
        ar_orders=ar_orders,
        ar_step_probs=ar_step_probs,
    )


# ---------------------------------------------------------------------------
# Cluster-id -> CIFAR class-name resolver (used by both embed and grid)
# ---------------------------------------------------------------------------


def _resolve_cluster_class_names(
    cluster_ids: np.ndarray,
    ds_idx_valid: np.ndarray,
    cifar_labels: torch.Tensor | None,
    class_names: list[str] | None,
) -> dict[int, str]:
    """Map each set-local cluster id to its underlying CIFAR class name.

    All tokens within a set-local cluster come from the same CIFAR class
    (see ``SetDataset.__getitem__``), so any representative member is
    enough to look up the dataset label.
    """
    if cifar_labels is None:
        return {}
    out: dict[int, str] = {}
    for cid in np.unique(cluster_ids).astype(int):
        idx = np.flatnonzero(cluster_ids == cid)
        if idx.size == 0:
            continue
        ds_i = int(ds_idx_valid[idx[0]])
        cifar_id = int(cifar_labels[ds_i].item())
        if class_names is not None and 0 <= cifar_id < len(class_names):
            out[int(cid)] = str(class_names[cifar_id])
        else:
            out[int(cid)] = f"class {cifar_id}"
    return out


# ---------------------------------------------------------------------------
# 2D embedding scatter (cluster-coloured)
# ---------------------------------------------------------------------------


def _draw_embedding_panel(
    ax,
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    pred_idx: np.ndarray | None,
    cpl_order: list[int] | None,
    title: str | None = None,
) -> None:
    """One scatter panel where colour and shape both encode the cluster id.

    Predicted tokens get a green ring outside the cluster marker.
    """
    n = coords.shape[0]
    if n == 0:
        if title:
            ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    for cid in np.unique(cluster_ids).astype(int):
        marker = _cluster_marker(int(cid))
        color = _cluster_color(int(cid))
        idx_c = np.flatnonzero(cluster_ids == cid)
        if idx_c.size == 0:
            continue
        ax.scatter(
            coords[idx_c, 0],
            coords[idx_c, 1],
            color=[color] * idx_c.size,
            marker=marker,
            s=85,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            zorder=3,
        )

    if pred_idx is not None and pred_idx.size > 0:
        ax.scatter(
            coords[pred_idx, 0],
            coords[pred_idx, 1],
            facecolors="none",
            edgecolors=PRED_COLOR,
            marker="o",
            s=420,
            linewidths=2.0,
            zorder=7,
        )

    if cpl_order:
        for step, j in enumerate(cpl_order, start=1):
            if j >= n:
                continue
            ax.annotate(
                str(step),
                xy=(coords[j, 0], coords[j, 1]),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=14,
                fontweight="bold",
                color="#c0392b",
                zorder=10,
            )
        if len(cpl_order) >= 2:
            for a, b in zip(cpl_order[:-1], cpl_order[1:]):
                if a >= n or b >= n:
                    continue
                ax.annotate(
                    "",
                    xy=(coords[b, 0], coords[b, 1]),
                    xytext=(coords[a, 0], coords[a, 1]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#c0392b",
                        alpha=0.45,
                        lw=1.2,
                        shrinkA=8,
                        shrinkB=8,
                    ),
                    zorder=4,
                )

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _build_embedding_legend_handles(
    unique_cids: np.ndarray,
    cluster_to_name: dict[int, str] | None,
    method: str,
    has_pred: bool,
) -> list[Line2D]:
    """Legend handles: one shape+colour entry per cluster + optional pred ring."""
    handles: list[Line2D] = []
    for cid in unique_cids.astype(int):
        cid_int = int(cid)
        if cluster_to_name and cid_int in cluster_to_name:
            label = cluster_to_name[cid_int]
        else:
            label = f"cluster {cid_int}"
        handles.append(
            Line2D(
                [],
                [],
                marker=_cluster_marker(cid_int),
                linestyle="",
                markerfacecolor=_cluster_color(cid_int),
                markeredgecolor="black",
                markersize=14,
                label=label,
            )
        )
    if has_pred:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor=PRED_COLOR,
                markersize=18,
                markeredgewidth=2.4,
                label="prediction",
            )
        )
    if method == "cpl":
        handles.append(
            Line2D(
                [],
                [],
                marker=r"$1{\rightarrow}2$",
                linestyle="",
                color="#c0392b",
                markersize=22,
                label="CPL decoder step",
            )
        )
    elif method == "ar":
        handles.append(
            Line2D(
                [],
                [],
                marker=r"$1{\rightarrow}2$",
                linestyle="",
                color="#c0392b",
                markersize=22,
                label="AR decoder step",
            )
        )
    return handles


def _save_single_embedding_figure(
    out_path: str,
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    pred_idx: np.ndarray,
    cpl_order: list[int] | None,
    legend_handles: list[Line2D],
    legend_ncol: int,
) -> None:
    """One embedding scatter panel plus the full shared legend underneath."""
    legend_rows = max(1, math.ceil(len(legend_handles) / legend_ncol))
    legend_height = 0.14 + 0.072 * legend_rows
    fig = plt.figure(figsize=(7.2, 6.4 + 0.52 * (legend_rows - 1)))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, legend_height],
        left=0.07,
        right=0.98,
        top=0.96,
        bottom=0.05,
        hspace=0.10,
    )
    ax_panel = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis("off")
    _draw_embedding_panel(
        ax_panel,
        coords,
        cluster_ids,
        pred_idx,
        cpl_order=cpl_order,
        title=None,
    )
    ax_leg.legend(
        handles=legend_handles,
        loc="center",
        ncol=legend_ncol,
        fontsize=16,
        frameon=False,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _save_embedding_figure(
    out_path_resnet: str,
    out_path_hidden: str,
    cache: _VizCache,
    b: int,
    cfg: Config,
    cifar_labels: torch.Tensor | None = None,
    class_names: list[str] | None = None,
) -> None:
    """Render two single-panel embedding scatters for sample ``b``.

    Raw ResNet features use t-SNE; transformer hidden states use
    ``cfg.viz.embedding``. Each PNG repeats the full legend.
    """
    valid = cache.mask[b].numpy().astype(bool)
    if valid.sum() < 2:
        return
    X_v = cache.X[b][valid].numpy()
    h_v = cache.h[b][valid].numpy()
    cluster_ids = cache.cluster_labels[b][valid].numpy()
    ds_idx_v = cache.ds_idx[b][valid].numpy()
    pred_local = cache.pred_masks_primary[b][valid].numpy()

    coords_raw = _project_2d(X_v, "tsne")
    coords_h = _project_2d(h_v, cfg.viz.embedding)

    pred_idx = np.flatnonzero(pred_local)

    cpl_order_local: list[int] | None = None
    if cfg.train.method == "cpl" and cache.cpl_orders is not None:
        valid_idx = np.flatnonzero(valid)
        global_to_local = {int(g): int(i) for i, g in enumerate(valid_idx)}
        cpl_order_local = [global_to_local[j] for j in cache.cpl_orders[b] if j in global_to_local]
    elif cfg.train.method == "ar" and cache.ar_orders is not None:
        # Reuse the CPL trajectory annotation for the AR step order.
        # ``ar_orders[b]`` may end with the EOS sentinel index N which is not
        # a valid bag position; filter it out via the global_to_local map.
        valid_idx = np.flatnonzero(valid)
        global_to_local = {int(g): int(i) for i, g in enumerate(valid_idx)}
        cpl_order_local = [global_to_local[j] for j in cache.ar_orders[b] if j in global_to_local]

    cluster_to_name = _resolve_cluster_class_names(cluster_ids, ds_idx_v, cifar_labels, class_names)
    unique_cids = np.unique(cluster_ids)
    legend_handles = _build_embedding_legend_handles(
        unique_cids,
        cluster_to_name,
        cfg.train.method,
        has_pred=pred_idx.size > 0,
    )
    legend_ncol = min(3, max(1, len(legend_handles)))

    _save_single_embedding_figure(
        out_path_resnet,
        coords_raw,
        cluster_ids,
        pred_idx,
        cpl_order_local,
        legend_handles,
        legend_ncol,
    )
    _save_single_embedding_figure(
        out_path_hidden,
        coords_h,
        cluster_ids,
        pred_idx,
        cpl_order_local,
        legend_handles,
        legend_ncol,
    )


# ---------------------------------------------------------------------------
# Score histogram
# ---------------------------------------------------------------------------


def _save_histogram_figure(out_path: str, cache: _VizCache, b: int, cfg: Config) -> None:
    """Score distribution + threshold lines + green ticks per random-GT score."""
    valid = cache.mask[b].numpy().astype(bool)
    if valid.sum() == 0:
        return
    scores = cache.scores[b][valid].numpy()
    gt = cache.gt_mask[b][valid].numpy().astype(bool)

    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    ax.hist(scores, bins=40, color="#34495e", alpha=0.75, edgecolor="white")

    method = cfg.train.method
    if method in ("bce", "hungarian"):
        thresholds = cfg.hungarian.thresholds if method == "hungarian" else cfg.bce.thresholds
        primary = cfg.hungarian.primary_threshold if method == "hungarian" else cfg.bce.primary_threshold
        for thr in thresholds:
            ax.axvline(thr, color="#3498db", lw=0.8, alpha=0.5)
        ax.axvline(
            primary,
            color="#e74c3c",
            lw=1.6,
            label=f"primary thr={primary:.2f}",
        )
        ax.set_xlabel("sigmoid(logit)")
        ax.set_xlim(0.0, 1.0)
    elif method == "ar":
        ax.set_xlabel("AR step-1: softmax P(bag slot) from masked logits over N+1 slots")
        ax.set_xlim(0.0, 1.0)
    else:
        ax.set_xlabel("theta")

    if gt.any():
        ymax = ax.get_ylim()[1]
        for s in scores[gt]:
            ax.plot(
                [s, s],
                [0, 0.05 * ymax],
                color=GT_COLOR,
                lw=2.0,
                solid_capstyle="butt",
            )
        ax.plot([], [], color=GT_COLOR, lw=2.0, label="GT sample score")

    ax.set_ylabel("count")
    ax.set_title(f"Score distribution  |  set #{b}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image grid (cluster spine + GT/pred outer border)
# ---------------------------------------------------------------------------


def _tile_indices_sorted(
    valid_indices: np.ndarray,
    cluster_ids: np.ndarray,
    gt_local: np.ndarray,
) -> list[int]:
    """Group tiles by cluster id; random-GT tile first within each block."""
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for k, j in enumerate(valid_indices):
        by_cluster[int(cluster_ids[k])].append(int(j))
    order: list[int] = []
    for cid in sorted(by_cluster.keys()):
        block = by_cluster[cid]
        gts = [j for j in block if bool(gt_local[np.where(valid_indices == j)[0][0]])]
        rest = [j for j in block if j not in set(gts)]
        order.extend(gts + rest)
    return order


def _outer_border_color(is_gt: bool, is_pred: bool) -> str | None:
    """Map ``(is_gt, is_pred)`` to the outer-border colour or ``None``."""
    if is_gt and is_pred:
        return BOTH_COLOR
    if is_gt:
        return GT_COLOR
    if is_pred:
        return PRED_COLOR
    return None


def _save_image_grid_figure(
    out_path: str,
    cache: _VizCache,
    b: int,
    cfg: Config,
    raw_dataset,
    cifar_labels: torch.Tensor,
) -> None:
    """One image per valid token; cluster-spine + GT/prediction outer border."""
    valid = cache.mask[b].numpy().astype(bool)
    if valid.sum() == 0:
        return
    valid_idx_global = np.flatnonzero(valid)
    cluster_ids = cache.cluster_labels[b][valid].numpy()
    gt_local = cache.gt_mask[b][valid].numpy().astype(bool)
    pred_local = cache.pred_masks_primary[b][valid].numpy().astype(bool)

    order = _tile_indices_sorted(valid_idx_global, cluster_ids, gt_local)
    if not order:
        return
    n_tiles = len(order)
    ncols = min(10, n_tiles)
    nrows = max(1, math.ceil(n_tiles / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.5 * ncols, 1.5 * nrows + 0.6),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.05, hspace=0.06)

    for t, j in enumerate(order):
        r, c = divmod(t, ncols)
        ax = axes[r][c]
        ds_i = int(cache.ds_idx[b, j].item())
        img_t, _ = raw_dataset[ds_i]
        img = img_t.permute(1, 2, 0).cpu().numpy()
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_xticks([])
        ax.set_yticks([])

        local_pos = int(np.where(valid_idx_global == j)[0][0])
        cid = int(cluster_ids[local_pos])
        is_gt = bool(gt_local[local_pos])
        is_pred = bool(pred_local[local_pos])

        # Inner spine: cluster-coloured tab20.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.6)
            spine.set_edgecolor(_cluster_color(cid))

        # Outer thick border (drawn outside the axes via a transAxes patch).
        outer = _outer_border_color(is_gt, is_pred)
        if outer is not None:
            ax.add_patch(
                Rectangle(
                    (-0.05, -0.05),
                    1.10,
                    1.10,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor=outer,
                    linewidth=3.4,
                    clip_on=False,
                    zorder=20,
                )
            )

    for t in range(n_tiles, nrows * ncols):
        r, c = divmod(t, ncols)
        axes[r][c].set_visible(False)

    method = cfg.train.method.upper()
    if cfg.train.method == "cpl":
        suffix = "greedy"
    elif cfg.train.method == "hungarian":
        suffix = f"thr={cfg.hungarian.primary_threshold:.2f}"
    else:
        suffix = f"thr={cfg.bce.primary_threshold:.2f}"
    fig.suptitle(
        f"set #{b}  |  inner = cluster colour  |  blue = GT, green = {method}({suffix}), purple = both",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CPL W heatmap
# ---------------------------------------------------------------------------


def _save_w_heatmap_figure(
    out_path: str,
    cache: _VizCache,
    b: int,
    cifar_labels: torch.Tensor | None = None,
    class_names: list[str] | None = None,
) -> None:
    """Restricted W heatmap (valid tokens only), styled for paper figures.

    Tokens are sorted so that all tokens of the same cluster sit in a
    contiguous block; each block is annotated with its CIFAR class name
    on each axis. Thick black separators (outside the diverging
    ``RdBu_r`` colormap) keep the block boundaries crisp.
    """
    if cache.cpl_W is None:
        return
    valid = cache.mask[b].numpy().astype(bool)
    if valid.sum() == 0:
        return
    valid_idx = np.flatnonzero(valid)
    cluster_ids = cache.cluster_labels[b][valid].numpy()
    ds_idx_v = cache.ds_idx[b][valid].numpy()

    order_by_cluster = sorted(range(valid_idx.size), key=lambda k: int(cluster_ids[k]))
    perm = valid_idx[order_by_cluster]
    cluster_ids_sorted = cluster_ids[order_by_cluster]
    ds_idx_sorted = ds_idx_v[order_by_cluster]

    W = cache.cpl_W[b].numpy()
    sub = W[np.ix_(perm, perm)]
    M = sub.shape[0]
    if M == 0:
        return

    cluster_to_name = _resolve_cluster_class_names(cluster_ids_sorted, ds_idx_sorted, cifar_labels, class_names)

    block_starts: list[int] = [0]
    for i in range(1, M):
        if cluster_ids_sorted[i] != cluster_ids_sorted[i - 1]:
            block_starts.append(i)
    block_ends = block_starts[1:] + [M]
    block_boundaries = block_starts[1:]
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for s, e in zip(block_starts, block_ends):
        cid = int(cluster_ids_sorted[s])
        tick_positions.append((s + e - 1) / 2.0)
        tick_labels.append(cluster_to_name.get(cid, f"cluster {cid}"))

    ref_m = 28.0
    fs_scale = max(1.0, math.sqrt(float(M) / ref_m))
    label_fs = max(22.0, min(38.0, 28.0 * fs_scale))
    cbar_fs = max(18.0, min(30.0, 22.0 * fs_scale))
    sep_lw = max(2.2, min(4.0, 2.4 * fs_scale))

    fig_size = max(4.5, 0.11 * M + 2.5)
    fig = plt.figure(figsize=(fig_size * 1.25, fig_size * 1.15))
    ax_w = fig.add_subplot(1, 1, 1)
    vlim = max(float(np.percentile(np.abs(sub), 98.0)), 1e-6)
    im = ax_w.imshow(
        sub,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        aspect="equal",
        interpolation="nearest",
    )

    ax_w.set_xticks(tick_positions)
    ax_w.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=label_fs)
    ax_w.set_yticks(tick_positions)
    ax_w.set_yticklabels(tick_labels, fontsize=label_fs)
    ax_w.tick_params(axis="both", which="major", length=0, pad=10)
    for spine in ax_w.spines.values():
        spine.set_visible(False)

    sep_color = "black"
    for boundary in block_boundaries:
        pos = boundary - 0.5
        ax_w.axhline(pos, color=sep_color, linewidth=sep_lw, zorder=5)
        ax_w.axvline(pos, color=sep_color, linewidth=sep_lw, zorder=5)

    divider = make_axes_locatable(ax_w)
    cax = divider.append_axes("right", size="4.2%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=cbar_fs, length=4, width=0.8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hungarian matching figure (random-GT <-> matched prediction)
# ---------------------------------------------------------------------------


def _save_hungarian_matching_figure(
    out_path: str,
    cache: _VizCache,
    b: int,
    cfg: Config,
    raw_dataset,
) -> None:
    """``GT | matched prediction | column runner-up`` tiles with cost annotations.

    The bottom strip shows ``unmatched predicted positives``: tokens
    above ``cfg.hungarian.primary_threshold`` that were not part of the K
    Hungarian winners (the threshold sweep is what's used for
    ``CluRec / CluPrec``; the matching itself selects exactly K winners).
    """
    if cache.hungarian_matches is None:
        return
    matches = cache.hungarian_matches[b]
    valid = cache.mask[b].numpy().astype(bool)
    if not matches.pred_tokens or valid.sum() == 0:
        return
    cluster_labels = cache.cluster_labels[b].numpy()

    K = len(matches.pred_tokens)
    gt_tokens = matches.gt_tokens
    pred_tokens = matches.pred_tokens
    cls_costs = matches.cls_costs
    dist_costs = matches.dist_costs
    second_pred = matches.second_pred_tokens
    second_cls = matches.second_cls_costs
    second_dist = matches.second_dist_costs

    matched_set = set(pred_tokens)
    pred_local = cache.pred_masks_primary[b].numpy().astype(bool)
    unmatched_pred = [int(j) for j in np.flatnonzero(pred_local & valid).tolist() if int(j) not in matched_set]
    unmatched_pred.sort()

    n_unmatched = len(unmatched_pred)
    has_unmatched = n_unmatched > 0

    rows = K + (1 if has_unmatched else 0)
    fig_h = max(3.6, 1.55 * rows + 0.6)
    fig = plt.figure(figsize=(11.0, fig_h))
    gs = fig.add_gridspec(
        rows,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        left=0.05,
        right=0.95,
        top=0.92,
        bottom=0.04,
        wspace=0.38,
        hspace=0.18,
    )

    a_w = cfg.hungarian.cls_weight
    b_w = cfg.hungarian.dist_weight

    def _put_image(ax, token: int) -> None:
        ds_i = int(cache.ds_idx[b, token].item())
        img_t, _ = raw_dataset[ds_i]
        img = img_t.permute(1, 2, 0).cpu().numpy()
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_xticks([])
        ax.set_yticks([])
        cid = int(cluster_labels[token])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.6)
            spine.set_edgecolor(_cluster_color(cid))

    for k in range(K):
        ax_gt = fig.add_subplot(gs[k, 0])
        ax_pr = fig.add_subplot(gs[k, 1])
        ax_sec = fig.add_subplot(gs[k, 2])
        _put_image(ax_gt, gt_tokens[k])
        _put_image(ax_pr, pred_tokens[k])
        for spine in ax_gt.spines.values():
            spine.set_edgecolor(GT_COLOR)
            spine.set_linewidth(2.4)
        outer = BOTH_COLOR if pred_tokens[k] == gt_tokens[k] else PRED_COLOR
        ax_pr.add_patch(
            Rectangle(
                (-0.05, -0.05),
                1.10,
                1.10,
                transform=ax_pr.transAxes,
                fill=False,
                edgecolor=outer,
                linewidth=2.6,
                clip_on=False,
                zorder=20,
            )
        )
        sec_tok = second_pred[k]
        if sec_tok < 0:
            ax_sec.axis("off")
            ax_sec.text(
                0.5,
                0.5,
                "n/a",
                ha="center",
                va="center",
                transform=ax_sec.transAxes,
                fontsize=10,
                color="#7f8c8d",
            )
        else:
            _put_image(ax_sec, sec_tok)
            sec_outer = BOTH_COLOR if sec_tok == gt_tokens[k] else RUNNER_COLOR
            ax_sec.add_patch(
                Rectangle(
                    (-0.05, -0.05),
                    1.10,
                    1.10,
                    transform=ax_sec.transAxes,
                    fill=False,
                    edgecolor=sec_outer,
                    linewidth=2.2,
                    linestyle=(0, (4, 3)),
                    clip_on=False,
                    zorder=20,
                )
            )
        if k == 0:
            ax_gt.set_title("random GT", fontsize=10, color=GT_COLOR)
            ax_pr.set_title("matched prediction", fontsize=10, color=PRED_COLOR)
            ax_sec.set_title("runner-up (same GT column)", fontsize=9, color=RUNNER_COLOR)

        cls_w = a_w * cls_costs[k]
        dist_w = b_w * dist_costs[k]
        total = cls_w + dist_w
        cost_best = f"best\ncost={total:.2f}\ncls={cls_w:.2f}\ndist={dist_w:.2f}"

        fig.add_artist(
            ConnectionPatch(
                xyA=(1.05, 0.5),
                coordsA=ax_gt.transAxes,
                xyB=(-0.05, 0.5),
                coordsB=ax_pr.transAxes,
                arrowstyle="-|>",
                color="#7f8c8d",
                lw=1.4,
                zorder=2,
            )
        )

        pos_gt = ax_gt.get_position()
        pos_pr = ax_pr.get_position()
        pos_sec = ax_sec.get_position()
        y_row = (pos_gt.y0 + pos_gt.y1) / 2.0
        x_best = (pos_gt.x1 + pos_pr.x0) / 2.0
        fig.text(
            x_best,
            y_row,
            cost_best,
            ha="center",
            va="center",
            fontsize=8,
            color="#34495e",
            bbox=dict(facecolor="white", edgecolor="#bdc3c7", boxstyle="round,pad=0.25"),
        )

        if sec_tok >= 0:
            sc_w = a_w * second_cls[k]
            sd_w = b_w * second_dist[k]
            total2 = sc_w + sd_w
            cost_run = f"runner-up\ncost={total2:.2f}\ncls={sc_w:.2f}\ndist={sd_w:.2f}"
            x_run = (pos_pr.x1 + pos_sec.x0) / 2.0
            fig.text(
                x_run,
                y_row,
                cost_run,
                ha="center",
                va="center",
                fontsize=8,
                color="#5d4d37",
                bbox=dict(facecolor="#fef9e7", edgecolor=RUNNER_COLOR, boxstyle="round,pad=0.25"),
            )
            fig.add_artist(
                ConnectionPatch(
                    xyA=(1.05, 0.35),
                    coordsA=ax_gt.transAxes,
                    xyB=(-0.05, 0.35),
                    coordsB=ax_sec.transAxes,
                    arrowstyle="-|>",
                    color=RUNNER_COLOR,
                    lw=1.2,
                    linestyle="--",
                    zorder=1,
                )
            )

    if has_unmatched:
        gs_u = gs[K, :].subgridspec(1, max(n_unmatched, 1), wspace=0.05)
        for j, tok in enumerate(unmatched_pred):
            ax_u = fig.add_subplot(gs_u[0, j])
            _put_image(ax_u, tok)
            ax_u.add_patch(
                Rectangle(
                    (-0.05, -0.05),
                    1.10,
                    1.10,
                    transform=ax_u.transAxes,
                    fill=False,
                    edgecolor=PRED_COLOR,
                    linewidth=2.0,
                    linestyle=(0, (3, 3)),
                    clip_on=False,
                    zorder=20,
                )
            )
            if j == 0:
                ax_u.set_title(
                    f"unmatched predicted positives (thr={cfg.hungarian.primary_threshold:.2f})",
                    fontsize=9,
                    color=PRED_COLOR,
                    loc="left",
                )

    fig.suptitle(
        f"Hungarian matching  |  set #{b}  |  K={K}  |  dist={cfg.hungarian.distance}  |  dist_features={cfg.hungarian.dist_feature_space}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step-by-step decode figure (AR / CPL: one row per greedy decode step)
# ---------------------------------------------------------------------------


def _save_step_decode_figure(
    out_path: str,
    cache: _VizCache,
    b: int,
    raw_dataset,
    *,
    orders: list[int],
    step_probs: list[torch.Tensor],
    cifar_labels: torch.Tensor | None = None,
    class_names: list[str] | None = None,
) -> None:
    """One row per greedy decode step (shared by AR and CPL visualisations).

    Each row has two panels:

    * **Left**: image tile of the chosen token at that step (cluster-coloured
      spine, green outer border). The EOS step shows an ``EOS`` placard
      centred in the panel.
    * **Right**: bar chart of the per-slot distribution at that step
      (cluster-grouped). The y-range is rescaled per row to its own peak so
      early steps with diffuse mass remain readable while the final EOS spike
      is still on its own scale. Class names sit under each cluster block,
      ``EOS`` is written vertically past the last bar (top / right / bottom
      spines are hidden so neither label is clipped by an axes line). The
      chosen slot is indicated by a black arrow from above. Masked-out bag
      positions have ~0 mass. For CPL and AR the heights are a ``softmax``
      over ``N+1`` slots (bag + EOS).
    """
    if not step_probs:
        return
    valid = cache.mask[b].numpy().astype(bool)
    if valid.sum() == 0:
        return
    valid_idx = np.flatnonzero(valid)
    cluster_ids_full = cache.cluster_labels[b].numpy()

    orders_list = list(orders)

    n_steps = len(step_probs)
    N = int(valid.shape[0])
    N_valid = int(valid.sum())

    # Sort the bag tokens by cluster id then by token index so the bar
    # chart groups bars cluster-by-cluster (matches the W heatmap convention).
    order_local = sorted(range(N_valid), key=lambda k: (int(cluster_ids_full[valid_idx[k]]), int(valid_idx[k])))
    perm = valid_idx[order_local]  # (N_valid,)
    bar_cids = cluster_ids_full[perm]
    ds_idx_perm = cache.ds_idx[b, perm].numpy()
    cluster_to_name = _resolve_cluster_class_names(bar_cids, ds_idx_perm, cifar_labels, class_names)

    # Centres (x in bar index space) and labels under each contiguous cluster block.
    block_centres: list[float] = []
    block_labels: list[str] = []
    i = 0
    while i < N_valid:
        j = i + 1
        cid_i = int(bar_cids[i])
        while j < N_valid and int(bar_cids[j]) == cid_i:
            j += 1
        block_centres.append(0.5 * (i + (j - 1)))
        block_labels.append(cluster_to_name.get(cid_i, f"cluster {cid_i}"))
        i = j

    fig_h = max(2.2, 1.55 * n_steps + 0.6)
    fig = plt.figure(figsize=(11.5, fig_h))
    gs = fig.add_gridspec(
        n_steps,
        2,
        width_ratios=[1.0, 4.5],
        left=0.04,
        right=0.98,
        top=0.98,
        bottom=0.06,
        wspace=0.10,
        hspace=0.30,
    )

    for t in range(n_steps):
        ax_img = fig.add_subplot(gs[t, 0])
        ax_bar = fig.add_subplot(gs[t, 1])

        # Per-row y-range so each step is rescaled to its own peak; this
        # makes early rows (where mass is spread thin) readable while still
        # showing the dominant EOS spike on the final row.
        sp_t = step_probs[t]
        bag_part_t = sp_t[:N].numpy()
        bar_xmax = float(bag_part_t.max()) if bag_part_t.size else 0.0
        if sp_t.shape[0] > N:
            bar_xmax = max(bar_xmax, float(sp_t[N].item()))
        bar_xmax = max(bar_xmax * 1.05, 1e-3)

        chosen = orders_list[t] if t < len(orders_list) else -1
        is_eos = chosen == N

        # Left panel: chosen-image tile or EOS placard.
        if is_eos or chosen < 0:
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.6)
                spine.set_edgecolor("#7f8c8d")
            ax_img.text(
                0.5,
                0.5,
                "EOS",
                ha="center",
                va="center",
                transform=ax_img.transAxes,
                fontsize=18,
                fontweight="bold",
                color="#34495e",
            )
        else:
            ds_i = int(cache.ds_idx[b, chosen].item())
            img_t, _ = raw_dataset[ds_i]
            img = img_t.permute(1, 2, 0).cpu().numpy()
            ax_img.imshow(np.clip(img, 0.0, 1.0))
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            cid = int(cluster_ids_full[chosen])
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.6)
                spine.set_edgecolor(_cluster_color(cid))
            outer = PRED_COLOR
            ax_img.add_patch(
                Rectangle(
                    (-0.05, -0.05),
                    1.10,
                    1.10,
                    transform=ax_img.transAxes,
                    fill=False,
                    edgecolor=outer,
                    linewidth=2.6,
                    clip_on=False,
                    zorder=20,
                )
            )

        ax_img.set_ylabel(
            f"step {t + 1}",
            fontsize=11,
            rotation=0,
            ha="right",
            va="center",
            labelpad=10,
        )

        # Right panel: per-token probability bars + EOS bar at the end.
        probs_full = step_probs[t].numpy()
        bag_probs = probs_full[:N][perm]
        eos_prob = float(probs_full[N]) if probs_full.shape[0] > N else 0.0

        x_positions = np.arange(N_valid + 1)
        bar_colors: list[tuple] = []
        for k in range(N_valid):
            bar_colors.append(_cluster_color(int(bar_cids[k])))
        bar_colors.append((0.4, 0.4, 0.4, 1.0))  # EOS bar grey

        bar_values = np.concatenate([bag_probs, np.array([eos_prob])])
        bars = ax_bar.bar(
            x_positions,
            bar_values,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.4,
            width=0.85,
        )
        # Chosen slot: arrow from above only (points at the top of the bar).
        chosen_idx_arrow: int | None = None
        if is_eos:
            chosen_idx_arrow = N_valid
        elif chosen >= 0:
            local = int(np.where(perm == chosen)[0][0])
            chosen_idx_arrow = local

        # Extra headroom above the tallest bar (arrow + ticks stay in-frame).
        y_top = bar_xmax * 1.26

        if chosen_idx_arrow is not None:
            h_top = float(bar_values[chosen_idx_arrow])
            y_tip = min(h_top + 0.04 * bar_xmax, y_top * 0.96)
            ax_bar.annotate(
                "",
                xy=(float(chosen_idx_arrow), y_tip),
                xytext=(float(chosen_idx_arrow), y_top * 0.91),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="black",
                    lw=2.0,
                    mutation_scale=14,
                    shrinkA=0,
                    shrinkB=2,
                ),
                zorder=25,
                annotation_clip=False,
            )

        # Cluster-block separators: only from y=0 upward so they do not cross
        # class-name labels drawn below the bars.
        for k in range(1, N_valid):
            if int(bar_cids[k]) != int(bar_cids[k - 1]):
                ax_bar.plot(
                    [k - 0.5, k - 0.5],
                    [0.0, bar_xmax],
                    color="black",
                    lw=0.6,
                    alpha=0.5,
                    zorder=2,
                    clip_on=True,
                )
        ax_bar.plot(
            [N_valid - 0.5, N_valid - 0.5],
            [0.0, bar_xmax],
            color="black",
            lw=1.0,
            alpha=0.8,
            zorder=2,
            clip_on=True,
        )

        # Extra right-side breathing room so the rotated "EOS" annotation
        # sits cleanly past the EOS bar rather than abutting any spine /
        # frame line. Bottom padding is generous so cluster class names
        # below the bars do not collide with the (hidden) bottom spine.
        ax_bar.set_xlim(-0.65, N_valid + 1.5)
        label_pad = max(0.22 * bar_xmax, 0.05)
        ax_bar.set_ylim(-label_pad, y_top)
        ax_bar.set_xticks([])
        ax_bar.tick_params(axis="y", labelsize=9)
        # ylim extends below 0 for class labels only; never show negative
        # tick values (probabilities are non-negative).
        tick_loc = MaxNLocator(
            nbins=2,
        )
        y_ticks = np.asarray(tick_loc.tick_values(0.0, float(y_top)), dtype=float)
        y_ticks = y_ticks[(y_ticks >= 0.0) & (y_ticks <= float(y_top) + 1e-12)]
        y_ticks = np.unique(y_ticks)
        if y_ticks.size < 2:
            y_ticks = np.array([0.0, float(y_top)], dtype=float)
        ax_bar.set_yticks(y_ticks)
        # Hide top / right / bottom spines so the rotated EOS label and the
        # class names below the bars never cross an axes border line.
        for side in ("top", "right", "bottom"):
            ax_bar.spines[side].set_visible(False)
        # A subtle baseline at y=0 keeps the bars visually anchored without
        # crossing the labels (which live at y < 0).
        ax_bar.axhline(0.0, color="black", lw=0.6, alpha=0.5, zorder=1, clip_on=True)
        y_text = -0.085 * bar_xmax if bar_xmax > 1e-6 else -0.04
        for xc, lab in zip(block_centres, block_labels):
            ax_bar.text(
                xc,
                y_text,
                lab,
                ha="center",
                va="top",
                fontsize=9,
                rotation=0,
                clip_on=False,
                zorder=30,
            )
        ax_bar.text(
            N_valid + 0.95,
            y_top * 0.48,
            "EOS",
            ha="center",
            va="center",
            rotation=-90,
            fontsize=11,
            fontweight="bold",
            color="#34495e",
            clip_on=False,
            zorder=30,
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_validation_visualizations(
    model: torch.nn.Module,
    val_sets,
    test_ds_raw,
    test_labels: torch.Tensor,
    cfg: Config,
    device: torch.device,
    out_dir: str,
    epoch: int,
    experiment=None,
) -> None:
    """Render all validation visualisations for the current epoch.

    All artefacts go under ``{out_dir}/viz/epoch_{NNN}/`` and (when
    enabled) are uploaded to Comet. ``out_dir`` is typically
    ``cfg.run_dir``.
    """
    if not cfg.viz.enabled:
        return
    n = max(1, min(cfg.viz.num_viz_samples, len(val_sets)))
    epoch_dir = os.path.join(out_dir, "viz", f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    cache = _build_cache(model, val_sets, cfg, device, n_samples=n)
    class_names = getattr(test_ds_raw, "classes", None)

    for b in range(cache.X.shape[0]):
        embed_resnet_path = os.path.join(epoch_dir, f"embed_resnet_tsne_sample{b:02d}.png")
        embed_hidden_path = os.path.join(epoch_dir, f"embed_hidden_sample{b:02d}.png")
        _save_embedding_figure(
            embed_resnet_path,
            embed_hidden_path,
            cache,
            b,
            cfg,
            cifar_labels=test_labels,
            class_names=class_names,
        )
        comet_utils.log_image(
            experiment,
            embed_resnet_path,
            name=f"viz/embed_resnet_tsne/sample_{b:02d}",
            step=epoch,
        )
        comet_utils.log_image(
            experiment,
            embed_hidden_path,
            name=f"viz/embed_hidden/sample_{b:02d}",
            step=epoch,
        )

        if cfg.viz.score_histogram:
            hist_path = os.path.join(epoch_dir, f"hist_sample{b:02d}.png")
            _save_histogram_figure(hist_path, cache, b, cfg)
            comet_utils.log_image(experiment, hist_path, name=f"viz/hist/sample_{b:02d}", step=epoch)

        if cfg.viz.image_grid:
            grid_path = os.path.join(epoch_dir, f"grid_sample{b:02d}.png")
            _save_image_grid_figure(grid_path, cache, b, cfg, test_ds_raw, test_labels)
            comet_utils.log_image(experiment, grid_path, name=f"viz/grid/sample_{b:02d}", step=epoch)

        if cfg.train.method == "cpl" and cfg.viz.w_heatmap:
            w_path = os.path.join(epoch_dir, f"W_sample{b:02d}.png")
            _save_w_heatmap_figure(
                w_path,
                cache,
                b,
                cifar_labels=test_labels,
                class_names=class_names,
            )
            comet_utils.log_image(experiment, w_path, name=f"viz/W/sample_{b:02d}", step=epoch)

        if cfg.train.method == "hungarian":
            hun_path = os.path.join(epoch_dir, f"hungarian_sample{b:02d}.png")
            _save_hungarian_matching_figure(hun_path, cache, b, cfg, test_ds_raw)
            comet_utils.log_image(experiment, hun_path, name=f"viz/hungarian/sample_{b:02d}", step=epoch)

        if cfg.train.method == "cpl" and cache.cpl_step_probs is not None and cache.cpl_orders is not None:
            cpl_path = os.path.join(epoch_dir, f"cpl_sample{b:02d}.png")
            _save_step_decode_figure(
                cpl_path,
                cache,
                b,
                test_ds_raw,
                orders=cache.cpl_orders[b],
                step_probs=cache.cpl_step_probs[b],
                cifar_labels=test_labels,
                class_names=class_names,
            )
            comet_utils.log_image(experiment, cpl_path, name=f"viz/cpl/sample_{b:02d}", step=epoch)

        if cfg.train.method == "ar" and cache.ar_step_probs is not None and cache.ar_orders is not None:
            ar_path = os.path.join(epoch_dir, f"ar_sample{b:02d}.png")
            _save_step_decode_figure(
                ar_path,
                cache,
                b,
                test_ds_raw,
                orders=cache.ar_orders[b],
                step_probs=cache.ar_step_probs[b],
                cifar_labels=test_labels,
                class_names=class_names,
            )
            comet_utils.log_image(experiment, ar_path, name=f"viz/ar/sample_{b:02d}", step=epoch)

    print(f"  viz: wrote {cache.X.shape[0]} samples to {epoch_dir}")


# ---------------------------------------------------------------------------
# Aggregate plot: precision/recall scatter for the current eval table
# ---------------------------------------------------------------------------


_RULE_STYLES: dict[str, dict[str, object]] = {
    "BCE-t": {"marker": "o", "color": "#3498db", "linestyle": "-", "label": "BCE thresholds"},
    "BCE-p": {"marker": "s", "color": "#9b59b6", "linestyle": "--", "label": "BCE percentiles"},
    "HUN-t": {"marker": "o", "color": "#e67e22", "linestyle": "-", "label": "Hungarian thresholds"},
    "HUN-p": {"marker": "s", "color": "#d35400", "linestyle": "--", "label": "Hungarian percentiles"},
}


def _draw_pr_panel(ax, rows) -> None:
    """Render the ``CluPrec / CluRec`` scatter combining every rule.

    Rows whose name starts with one of :data:`_RULE_STYLES` are drawn as
    a sweep curve; the remaining named rules (``CPL``, ``KMeans-oracleK``)
    are plotted as standalone star/triangle markers so they sit on top
    of the sweep without cluttering the legend.
    """
    grouped: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for r in rows:
        for prefix in _RULE_STYLES:
            if r.name.startswith(prefix):
                grouped[prefix].append(
                    (
                        float(r.clu_rec),
                        float(r.clu_prec),
                        r.name[len(prefix) :],
                    )
                )
                break

    for prefix, pts in grouped.items():
        if not pts:
            continue
        style = _RULE_STYLES[prefix]
        xs, ys, labs = zip(*pts)
        ax.plot(
            xs,
            ys,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )
        for x, y, lbl in pts:
            ax.annotate(
                lbl,
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color=style["color"],
            )

    for r in rows:
        rec = float(r.clu_rec)
        prc = float(r.clu_prec)
        if r.name == "CPL":
            ax.plot(rec, prc, "*", color="#e74c3c", markersize=18, label="CPL (greedy)")
        elif r.name == "AR":
            ax.plot(rec, prc, "*", color="#16a085", markersize=18, label="AR (greedy)")
        elif r.name.startswith("KMeans"):
            ax.plot(rec, prc, "^", color="#27ae60", markersize=12, label="KMeans (oracle K)")

    ax.set_xlabel("CluRec")
    ax.set_ylabel("CluPrec")
    ax.set_title("Cluster coverage: CluPrec vs CluRec per inference rule", fontsize=10)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)


def save_pr_scatter(rows, path: str) -> None:
    """Single-panel ``CluPrec / CluRec`` scatter for the current eval rows.

    The medoid-detection panel is gone -- there is no exact-index metric
    in the random-GT task, so cluster-level coverage is the only thing
    worth plotting.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    _draw_pr_panel(ax, rows)
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = ["render_validation_visualizations", "save_pr_scatter"]
