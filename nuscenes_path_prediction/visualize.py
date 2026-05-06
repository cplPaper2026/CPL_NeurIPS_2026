"""Visualisation utility for road-cpl samples (heatmap+offset schema).

Supports two image layouts in the same ``.npz`` schema family:

* ``(2, H, W) uint8`` -- legacy synthetic: channel 0 = road occupancy,
  channel 1 = ego Gaussian heatmap. The two are blended into a grayscale
  road plus a red ego overlay.
* ``(3, H, W) uint8`` -- nuScenes RGB: rendered directly. The ego pose is
  already baked in as a red triangle so no overlay is needed.

Single-sample previews for ``.npz`` files (3 panels):
    1. BEV (road + ego)
    2. BEV + high-resolution GT path scatter
    3. Decision list / metadata text panel

Training/validation previews (built by :func:`render_training_val_sample`)
render a strip of dynamically-selected debug panels in this canonical order:

    1. Road + ego pose
    2. Road + full GT (all valid paths)
    3. Heatmap (sigmoid) + offset arrows on the **selected** cells only
       (omitted when ``learns_heatmap_score=False``, i.e. CPL+grid runs)
    4. Binary mask of the selected cells
    5. Predicted points produced by the selected cells + offsets
    6. Matching panel: GT points + matched-pred mask + matched pred points
    7. CPL selection-order grid (only when CPL is enabled)

Cell-selection rule:
    * supervised (no CPL) -- ``sigmoid(logit) > min_prob``
    * CPL                 -- the cells chosen by ``cpl_greedy_decode``

The visualizer never falls back to a top-k score selection; only cells that
satisfy the selection rule appear in panels 3-5.

Single-sample usage:
    python visualize.py --in data/train/00000000.npz --out preview.png

Web gallery (browse a directory):
    python visualize.py --dir data/<run>/                # train/ + val/ npz
    python visualize.py --dir out/previews/ --no-browser # static images
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import mimetypes
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import matplotlib

matplotlib.use("Agg")  # pyright: ignore[reportUnusedExpression]
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# I/O and tensor helpers
# ---------------------------------------------------------------------------


def _load_sample(npz_path: Path):
    """Return ``(image, points, num_points, valid_paths, meta)``.

    Requires the ``heatmap_offset_v1`` schema (``points`` + ``num_points``).
    Older datasets (``label`` only) raise ``ValueError``.
    """
    with np.load(npz_path) as data:
        if "points" not in data.files or "num_points" not in data.files:
            raise ValueError(f"NPZ {npz_path} is missing 'points'/'num_points' (legacy schema). Regenerate with the current generator.")
        image = data["image"]
        points = data["points"].astype(np.float32, copy=False)
        num_points = data["num_points"].astype(np.int32, copy=False)
        if "valid_paths" in data.files:
            valid_paths = data["valid_paths"].astype(np.uint8, copy=False)
        else:
            valid_paths = (num_points > 0).astype(np.uint8)
    json_path = npz_path.with_suffix(".json")
    meta = {}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
    return image, points, num_points, valid_paths, meta


def _fmt_float(v, spec: str = ".1f") -> str:
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return "?"


def _ego_heatmap_to_alpha(ego: np.ndarray) -> np.ndarray:
    """Ego channel to blend alpha in [0, 1] for an RGBA overlay."""
    a = np.asarray(ego, dtype=np.float32)
    if a.size and float(a.max()) > 1.0 + 1e-6:
        a = a * (1.0 / 255.0)
    return np.clip(a, 0.0, 1.0)


def _image_hw(image: np.ndarray) -> tuple[int, int]:
    """Return ``(H, W)`` for a ``(C, H, W)`` BEV image (any C)."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected (C, H, W), got shape {arr.shape}")
    return int(arr.shape[1]), int(arr.shape[2])


def _image_to_display(image: np.ndarray) -> np.ndarray:
    """Return an ``(H, W, 3) float32 in [0, 1]`` RGB image ready for ``imshow``.

    Channel-count semantics:

    * ``2``: synthetic ``heatmap_offset_v1`` -- channel 0 is rendered as a
      grayscale road, channel 1 is composited on top as a red ego overlay
      with alpha proportional to the heatmap intensity.
    * ``3``: nuScenes ``heatmap_offset_rgb_v1`` -- treated as RGB directly;
      the red ego marker is already baked in by the generator.
    """
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected (C, H, W), got shape {arr.shape}")
    c = int(arr.shape[0])
    if c == 3:
        rgb = arr.astype(np.float32, copy=False)
        if rgb.size and float(rgb.max()) > 1.0 + 1e-6:
            rgb = rgb * (1.0 / 255.0)
        return np.clip(np.transpose(rgb, (1, 2, 0)), 0.0, 1.0)
    if c == 2:
        road = arr[0].astype(np.float32, copy=False)
        if road.size and float(road.max()) > 1.0 + 1e-6:
            road = road * (1.0 / 255.0)
        road = np.clip(road, 0.0, 1.0)
        alpha = _ego_heatmap_to_alpha(arr[1])[..., None]
        base = np.stack([road, road, road], axis=-1)
        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.clip(base * (1.0 - alpha) + red * alpha, 0.0, 1.0)
    raise ValueError(f"Unsupported image channel count {c} (expected 2 or 3).")


def _show_bev(ax, image: np.ndarray, *, zorder: int = 0) -> None:
    """Draw the BEV background (legacy 2-channel or new 3-channel RGB)."""
    ax.imshow(_image_to_display(image), interpolation="nearest", zorder=zorder)


def _to_numpy_f32(x: object) -> np.ndarray:
    """Convert torch tensor or array-like to ``float32`` ``numpy`` array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()  # type: ignore[union-attr]
    return np.asarray(x, dtype=np.float32)


def _to_valid_paths_bool(x: object) -> np.ndarray:
    """Return a 1D boolean array for ``valid_paths`` (numpy or 0/1 tensor)."""
    if hasattr(x, "detach"):
        v = x.detach().cpu().numpy()  # type: ignore[union-attr]
    else:
        v = np.asarray(x)
    v = v.astype(np.int8, copy=False).ravel()
    return v > 0


def _to_int_array(x: object) -> np.ndarray:
    """Coerce a tensor/array of integer counts to ``int64`` 1D ``numpy``."""
    if hasattr(x, "detach"):
        v = x.detach().cpu().numpy()  # type: ignore[union-attr]
    else:
        v = np.asarray(x)
    return np.atleast_1d(v).astype(np.int64, copy=False)


def _scatter_path(ax, pts_xy: np.ndarray, *, color, label: str | None) -> None:
    """Scatter sub-pixel GT points and draw a polyline through them."""
    if pts_xy.shape[0] == 0:
        return
    ax.scatter(
        pts_xy[:, 0],
        pts_xy[:, 1],
        color=color,
        s=14,
        alpha=0.92,
        linewidths=0,
        zorder=5,
        label=label,
    )
    if pts_xy.shape[0] > 1:
        ax.plot(
            pts_xy[:, 0],
            pts_xy[:, 1],
            color=color,
            linewidth=1.0,
            alpha=0.6,
            zorder=4,
        )


# ---------------------------------------------------------------------------
# Cell selection / sub-pixel decoding
#
# These primitives implement the single rule for what counts as a "selected"
# cell: ``sigmoid(logit) > min_prob`` for the supervised path, or "this flat
# index appears in the CPL greedy decode" for the CPL path. The visualizer
# never falls back to a top-k score selection; if no cell qualifies, panels
# 3-5 simply render zero arrows / zero points.
# ---------------------------------------------------------------------------


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically safe sigmoid for ``np.float64`` -> ``np.float32`` output."""
    z = np.clip(logits.astype(np.float64), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _select_cells_above_threshold(probs: np.ndarray, min_prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cell_i, cell_j)`` for cells with ``probs > min_prob``."""
    cell_i, cell_j = np.where(probs > float(min_prob))
    return cell_i.astype(np.int64), cell_j.astype(np.int64)


def _select_cells_from_cpl_indices(
    flat_indices: np.ndarray,
    h_lr: int,
    w_lr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cell_i, cell_j)`` from a 1D array of flat CPL indices."""
    if flat_indices.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    idx = np.asarray(flat_indices, dtype=np.int64).ravel()
    return (idx // w_lr).astype(np.int64), (idx % w_lr).astype(np.int64)


def _pred_points_from_cells(
    cell_i: np.ndarray,
    cell_j: np.ndarray,
    offsets: np.ndarray,
    *,
    D: int,
) -> np.ndarray:
    """Combine selected cells with the offset head into ``(N, 2)`` ``(y, x)`` pixels."""
    if cell_i.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    dx = offsets[0, cell_i, cell_j]
    dy = offsets[1, cell_i, cell_j]
    x = cell_j.astype(np.float32) * float(D) + dx
    y = cell_i.astype(np.float32) * float(D) + dy
    return np.stack([y, x], axis=1).astype(np.float32)


def _cells_to_binary_mask(cell_i: np.ndarray, cell_j: np.ndarray, h_lr: int, w_lr: int) -> np.ndarray:
    """``(h_lr, w_lr)`` ``float32`` mask with ``1.0`` at selected cells."""
    m = np.zeros((h_lr, w_lr), dtype=np.float32)
    if cell_i.size > 0:
        m[cell_i, cell_j] = 1.0
    return m


def _quiver_cells_to_points(
    ax,
    cell_i: np.ndarray,
    cell_j: np.ndarray,
    offsets: np.ndarray,
    *,
    D: int,
    arrow_color: str = "#3aa1ff",
    arrow_alpha: float = 0.85,
) -> None:
    """Draw arrows from each selected cell's center to its sub-pixel point."""
    if cell_i.size == 0:
        return
    cx = cell_j.astype(np.float32) * float(D) + 0.5 * float(D)
    cy = cell_i.astype(np.float32) * float(D) + 0.5 * float(D)
    px = cell_j.astype(np.float32) * float(D) + offsets[0, cell_i, cell_j]
    py = cell_i.astype(np.float32) * float(D) + offsets[1, cell_i, cell_j]
    ax.quiver(
        cx,
        cy,
        px - cx,
        py - cy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        color=arrow_color,
        alpha=arrow_alpha,
        zorder=6,
    )


def _upsample_score_grid_bilinear(probs: np.ndarray, h_lr: int, w_lr: int, H: int, W: int) -> np.ndarray:
    """Bilinearly upsample a low-res probability map to the full image size."""
    out = zoom(probs.astype(np.float64), (H / h_lr, W / w_lr), order=1)
    if out.shape[0] != H or out.shape[1] != W:
        zy, zx = H / max(int(out.shape[0]), 1), W / max(int(out.shape[1]), 1)
        out = zoom(out, (zy, zx), order=1)
    return out[:H, :W].astype(np.float32)


def _draw_lowres_grid(ax, h_lr: int, w_lr: int, *, color: str = "#dddddd", linewidth: float = 0.5) -> None:
    """Overlay the low-resolution grid onto an axis (cosmetic)."""
    ax.set_xticks(np.arange(-0.5, w_lr, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h_lr, 1), minor=True)
    ax.grid(True, which="minor", color=color, linewidth=linewidth, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# .npz single-sample preview (used by the gallery and `--in` mode)
# ---------------------------------------------------------------------------


def _build_sample_figure(
    image: np.ndarray,
    points: np.ndarray,
    num_points: np.ndarray,
    valid_paths: np.ndarray,
    meta: dict,
    stem: str,
):
    """Three-panel preview: BEV+ego, BEV + GT points, metadata text."""
    H, W = _image_hw(image)
    K = int(points.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].set_title("BEV + ego pose")
    _show_bev(axes[0], image)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_title(f"BEV + {int(valid_paths.sum())} GT path(s) (sub-pixel)")
    _show_bev(axes[1], image)
    palette = plt.get_cmap("tab10")
    legend_handles = []
    for k in range(K):
        if not bool(valid_paths[k]):
            continue
        n_k = int(num_points[k])
        if n_k <= 0:
            continue
        pts = points[k, :n_k]
        rgba = palette(k % 10)
        _scatter_path(axes[1], pts, color=rgba, label=f"path {k}")
        legend_handles.append(plt.Line2D([], [], color=rgba, marker="o", linestyle="-", markersize=6, label=f"path {k}"))
    if legend_handles:
        axes[1].legend(handles=legend_handles, loc="lower right", fontsize=8)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlim(-0.5, W - 0.5)
    axes[1].set_ylim(H - 0.5, -0.5)

    axes[2].set_title("Metadata")
    axes[2].axis("off")
    text_lines = _meta_text_lines(stem, meta, valid_paths)
    axes[2].text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", family="monospace", fontsize=9)

    fig.tight_layout()
    return fig


def _meta_text_lines(stem: str, meta: dict, valid_paths: np.ndarray) -> list[str]:
    """Render the metadata panel for either schema (synthetic or nuScenes)."""
    lines: list[str] = [f"sample: {stem}"]
    if not meta:
        lines.append("No metadata available.")
        return lines
    schema = str(meta.get("schema", "?"))
    fmtv = meta.get("format_version", "?")
    kmax = meta.get("config", {}).get("K_MAX", "?")
    lines.append(f"schema: {schema}  format_version: {fmtv}  K_MAX: {kmax}")
    if "ego_global" in meta:
        eg = meta.get("ego_global", {}) or {}
        lines.append(f"ego(global): x={_fmt_float(eg.get('x'))}  y={_fmt_float(eg.get('y'))}  yaw={_fmt_float(eg.get('yaw_rad'), '.2f')}")
        ep = meta.get("ego_pixel")
        if ep:
            lines.append(f"ego(pixel): {_fmt_float(ep[0])}, {_fmt_float(ep[1])}")
        lines.append(f"map: {meta.get('map_name', '?')}  scene: {meta.get('scene_name', '?')}")
        lines.append(f"is_near_junction: {meta.get('is_near_junction', '?')}  n_candidates: {meta.get('n_candidates', '?')}")
    else:
        eg = meta.get("ego", {}) or {}
        lines.append(f"ego: x={_fmt_float(eg.get('x'))}  y={_fmt_float(eg.get('y'))}  theta={_fmt_float(eg.get('theta_rad'), '.2f')}")
        prims = meta.get("primitives_present")
        if prims:
            lines.append("primitives: " + ", ".join(prims))
        lines.append(f"nodes: {meta.get('n_nodes', '?')}  edges: {meta.get('n_edges', '?')}")
    lines.append(f"n_paths: {int(meta.get('n_paths', int(valid_paths.sum())))}")
    for k, pinfo in enumerate(meta.get("paths", []) or []):
        exits_at = pinfo.get("exits_frame_at", ["?", "?"])
        lines.append(f"  path {k}: len={pinfo.get('length_px', '?')}px  n_points={pinfo.get('n_points', '?')}  exits=({exits_at[0]}, {exits_at[1]})")
    return lines


# ---------------------------------------------------------------------------
# Training/validation debug figure
# ---------------------------------------------------------------------------


def _add_cpl_order_subplot(ax, order_np: np.ndarray, stem: str, *, cmap: str = "viridis") -> None:
    """Draw CPL greedy order (1..K) on white background, masked for unselected cells."""
    pos = order_np[order_np > 0]
    base_cmap = copy.copy(plt.get_cmap(cmap))
    if hasattr(base_cmap, "set_bad"):
        base_cmap.set_bad("white")
    data = ma.masked_where(order_np <= 0, order_np)
    ax.set_facecolor("white")
    if pos.size == 0:
        ax.set_title(f"CPL order (no cells before EOS)\n{stem}", fontsize=9)
    else:
        k_max = float(np.max(pos))
        if k_max > 1.0:
            norm: mcolors.Normalize = mcolors.Normalize(vmin=1.0, vmax=k_max)
            t = f"CPL order (1..{int(k_max)})\n{stem}"
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            t = f"CPL order (step 1)\n{stem}"
        im = ax.imshow(data, cmap=base_cmap, norm=norm, interpolation="nearest", aspect="equal", origin="upper")
        ax.set_title(t, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="selection index")
    _draw_lowres_grid(ax, int(order_np.shape[0]), int(order_np.shape[1]))


# ---------------------------------------------------------------------------
# Per-panel renderers for the training/validation debug figure
# ---------------------------------------------------------------------------


# Cosmetic constants shared across panels
_PRED_COLOR = "#3aa1ff"
_GT_COLOR = "#80ff80"


def _panel_road_ego(ax, image: np.ndarray) -> None:
    """Panel 1: BEV with ego pose (RGB image renders the ego marker directly;
    legacy 2-channel inputs go through the gray-road + red-overlay compositor).
    """
    ax.set_title("BEV + ego pose")
    _show_bev(ax, image)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_full_gt(
    ax,
    image: np.ndarray,
    points: np.ndarray,
    num_points: np.ndarray,
    valid_paths: np.ndarray,
) -> None:
    """Panel 2: full GT scatter for every valid path (sub-pixel ``(x, y)``)."""
    H, W = _image_hw(image)
    K = int(points.shape[0])
    n_valid = int(np.count_nonzero(valid_paths))
    ax.set_title(f"Full GT ({n_valid} path(s), sub-pixel)")
    _show_bev(ax, image)

    palette = plt.get_cmap("tab10")
    legend_handles: list[plt.Line2D] = []
    for k in range(K):
        if not bool(valid_paths[k]):
            continue
        n_k = int(num_points[k])
        if n_k <= 0:
            continue
        rgba = palette(k % 10)
        _scatter_path(ax, points[k, :n_k], color=rgba, label=f"path {k}")
        legend_handles.append(plt.Line2D([], [], color=rgba, marker="o", linestyle="-", markersize=6, label=f"path {k}"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def _panel_heatmap(
    ax,
    image: np.ndarray,
    probs: np.ndarray,
    offsets: np.ndarray,
    cell_i: np.ndarray,
    cell_j: np.ndarray,
    pred_pts_yx: np.ndarray,
    *,
    D: int,
    selection_label: str,
) -> None:
    """Panel 3: sigmoid heatmap + offset arrows on the **selected cells only**."""
    H, W = _image_hw(image)
    h_lr, w_lr = probs.shape
    n_sel = int(cell_i.size)
    ax.set_title(f"Heatmap (sigmoid) + arrows on {selection_label} cells (n={n_sel})")
    _show_bev(ax, image, zorder=0)
    pred_up = _upsample_score_grid_bilinear(probs.astype(np.float32), h_lr, w_lr, H, W)
    im_ovl = ax.imshow(pred_up, cmap="hot", vmin=0.0, vmax=1.0, alpha=0.55, interpolation="bilinear", zorder=1)
    _quiver_cells_to_points(ax, cell_i, cell_j, offsets.astype(np.float32), D=D)
    if pred_pts_yx.shape[0] > 0:
        ax.scatter(pred_pts_yx[:, 1], pred_pts_yx[:, 0], s=10, color=_PRED_COLOR, marker=".", zorder=7)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im_ovl, ax=ax, fraction=0.046, pad=0.04)


def _panel_selected_mask(
    ax,
    cell_i: np.ndarray,
    cell_j: np.ndarray,
    h_lr: int,
    w_lr: int,
    *,
    selection_label: str,
) -> None:
    """Panel 4: low-resolution binary mask of the selected cells."""
    n_sel = int(cell_i.size)
    mask = _cells_to_binary_mask(cell_i, cell_j, h_lr, w_lr)
    ax.set_title(f"{selection_label.capitalize()} cells (n={n_sel})")
    ax.imshow(mask, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    _draw_lowres_grid(ax, h_lr, w_lr)


def _panel_pred_points(
    ax,
    image: np.ndarray,
    pred_pts_yx: np.ndarray,
    *,
    selection_label: str,
) -> None:
    """Panel 5: predicted sub-pixel points scattered on the BEV."""
    H, W = _image_hw(image)
    n_pts = int(pred_pts_yx.shape[0])
    ax.set_title(f"Predicted points (n={n_pts}) — {selection_label}")
    _show_bev(ax, image)
    if n_pts > 0:
        ax.scatter(
            pred_pts_yx[:, 1],
            pred_pts_yx[:, 0],
            s=14,
            color=_PRED_COLOR,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def _panel_matching(
    ax,
    image: np.ndarray,
    offsets: np.ndarray,
    gt_yx: np.ndarray,
    matched_pred_mask: np.ndarray,
    *,
    D: int,
    stem: str,
) -> None:
    """Panel 6: GT points + matched-pred mask + sub-pixel pred points at matched cells.

    Only cells flagged by ``matched_pred_mask`` (typed ``{0, 1}``) contribute
    pred points -- per the rule that the matching panel shows the GT and *only*
    the predicted points that received a pair.
    """
    H, W = _image_hw(image)
    h_lr, w_lr = matched_pred_mask.shape
    rep_y, rep_x = H // h_lr, W // w_lr

    _show_bev(ax, image, zorder=0)

    overlay = np.zeros((H, W, 4), dtype=np.float32)
    if matched_pred_mask.any():
        m_full = np.repeat(np.repeat(matched_pred_mask, rep_y, axis=0), rep_x, axis=1)[:H, :W]
        overlay[..., 0] = 1.0
        overlay[..., 1] = 0.2
        overlay[..., 2] = 0.08
        overlay[..., 3] = 0.45 * (m_full > 0.5).astype(np.float32)
    ax.imshow(overlay, interpolation="nearest", zorder=1)

    # Sub-pixel points only for matched cells (not the full prediction set).
    matched_cells_i, matched_cells_j = np.where(matched_pred_mask > 0.5)
    matched_pts_yx = _pred_points_from_cells(
        matched_cells_i.astype(np.int64),
        matched_cells_j.astype(np.int64),
        offsets.astype(np.float32),
        D=D,
    )

    n_match = int(matched_cells_i.size)
    n_gt = int(gt_yx.shape[0])

    if n_gt > 0:
        ax.scatter(gt_yx[:, 1], gt_yx[:, 0], s=22, color=_GT_COLOR, edgecolors="black", linewidths=0.5, zorder=5, label="GT")
        if n_gt > 1:
            ax.plot(gt_yx[:, 1], gt_yx[:, 0], color=_GT_COLOR, linewidth=1.0, alpha=0.6, zorder=4)
    if matched_pts_yx.shape[0] > 0:
        ax.scatter(
            matched_pts_yx[:, 1],
            matched_pts_yx[:, 0],
            s=14,
            color=_PRED_COLOR,
            marker="x",
            linewidths=1.0,
            zorder=6,
            label="matched pred",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Matching: GT (green) + matched cells (red) + matched pred (blue x)\nmatched={n_match}  GT={n_gt}\n{stem}",
        fontsize=8,
    )
    if n_match or n_gt:
        ax.legend(loc="lower right", fontsize=7)


# ---------------------------------------------------------------------------
# Top-level figure assembly
# ---------------------------------------------------------------------------


def _build_training_val_figure(
    image: np.ndarray,
    points: np.ndarray,  # (K, M, 2) float
    num_points: np.ndarray,  # (K,) int
    valid_paths: np.ndarray,  # (K,) bool
    heatmap_logits: np.ndarray,  # (h, w) float (no sigmoid)
    offsets: np.ndarray,  # (2, h, w) float pixels
    stem: str,
    *,
    D: int,
    min_prob: float = 0.5,
    learns_heatmap_score: bool = True,
    cpl_flat_indices: np.ndarray | None = None,
    cpl_order: np.ndarray | None = None,
    match_gt_yx: np.ndarray | None = None,
    match_pred_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Render the canonical training/validation debug strip.

    Panels (always in this order; panels in parentheses are conditional):

    1. road + ego pose
    2. road + full GT
    3. (heatmap + offset arrows on selected cells)
    4. binary mask of selected cells
    5. predicted points produced by selected cells + offsets
    6. matching overlay (GT + matched-pred mask + matched pred points)
    7. (CPL selection-order grid)

    Cell selection rule (single source of truth, identical across panels 3-5):

    * if ``cpl_flat_indices`` is provided -> cells equal those flat indices
      (exactly the CPL greedy-decode picks);
    * otherwise -> ``sigmoid(heatmap_logits) > min_prob``.

    Panel 3 is skipped when ``learns_heatmap_score=False`` (CPL + grid only:
    rendering an unsupervised score map would be misleading).

    Panel 7 is appended whenever ``cpl_flat_indices`` is provided; ``cpl_order``
    is then the corresponding ``(h, w)`` selection-step grid.

    The matching panel is always rendered when both ``match_gt_yx`` and
    ``match_pred_mask`` are provided. Both must come from the training matcher
    (Grid or Hungarian): GT pixels for the seeded path and the matched-pred cell
    mask from ``PathHeadsLoss`` / ``matched_pred_mask``, including under CPL.
    """
    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[0] not in (2, 3):
        raise ValueError(f"image must be (C, H, W) with C in (2, 3), got {image_arr.shape}")
    if heatmap_logits.ndim != 2:
        raise ValueError(f"heatmap_logits must be (h, w), got {heatmap_logits.shape}")
    if offsets.ndim != 3 or offsets.shape[0] != 2:
        raise ValueError(f"offsets must be (2, h, w), got {offsets.shape}")
    h_lr, w_lr = heatmap_logits.shape
    if offsets.shape[1] != h_lr or offsets.shape[2] != w_lr:
        raise ValueError(f"offsets {offsets.shape} must agree with heatmap_logits {heatmap_logits.shape}")

    probs = _sigmoid(heatmap_logits)
    use_cpl = cpl_flat_indices is not None
    show_heatmap_panel = bool(learns_heatmap_score)

    # 1) Resolve the single set of "selected" cells used by panels 3-5.
    if use_cpl:
        cell_i, cell_j = _select_cells_from_cpl_indices(
            np.asarray(cpl_flat_indices, dtype=np.int64),
            h_lr,
            w_lr,
        )
        selection_label = "CPL-selected"
    else:
        cell_i, cell_j = _select_cells_above_threshold(probs, min_prob)
        selection_label = f"prob > {float(min_prob):.2f}"
    pred_pts_yx = _pred_points_from_cells(cell_i, cell_j, offsets.astype(np.float32), D=D)

    # 2) Decide which panels to draw and lay out the figure accordingly.
    has_matching = match_gt_yx is not None and match_pred_mask is not None
    panels: list[str] = ["road_ego", "full_gt"]
    if show_heatmap_panel:
        panels.append("heatmap")
    panels.extend(["selected_mask", "pred_points"])
    if has_matching:
        panels.append("matching")
    if use_cpl:
        panels.append("cpl_order")

    n_panels = len(panels)
    fig, axes_obj = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes = list(np.atleast_1d(axes_obj))

    # 3) Dispatch per-panel renderers.
    for ax, panel in zip(axes, panels, strict=True):
        if panel == "road_ego":
            _panel_road_ego(ax, image_arr)
        elif panel == "full_gt":
            _panel_full_gt(ax, image_arr, points, num_points, valid_paths)
        elif panel == "heatmap":
            _panel_heatmap(
                ax,
                image_arr,
                probs,
                offsets,
                cell_i,
                cell_j,
                pred_pts_yx,
                D=D,
                selection_label=selection_label,
            )
        elif panel == "selected_mask":
            _panel_selected_mask(ax, cell_i, cell_j, h_lr, w_lr, selection_label=selection_label)
        elif panel == "pred_points":
            _panel_pred_points(ax, image_arr, pred_pts_yx, selection_label=selection_label)
        elif panel == "matching":
            assert match_gt_yx is not None and match_pred_mask is not None  # narrow types
            gt_yx = np.asarray(match_gt_yx, dtype=np.float32).reshape(-1, 2)
            mt = np.asarray(match_pred_mask, dtype=np.float32)
            if mt.shape != (h_lr, w_lr):
                raise ValueError(f"match_pred_mask shape {mt.shape} must be ({h_lr}, {w_lr})")
            _panel_matching(ax, image_arr, offsets, gt_yx, mt, D=D, stem=stem)
        elif panel == "cpl_order":
            if cpl_order is None:
                raise ValueError("cpl_order is required when CPL is enabled.")
            order_np = np.asarray(cpl_order, dtype=np.float32)
            if order_np.shape != (h_lr, w_lr):
                raise ValueError(f"cpl_order shape {order_np.shape} must be ({h_lr}, {w_lr})")
            _add_cpl_order_subplot(ax, order_np, stem)

    fig.suptitle(f"val — {stem}", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def render_training_val_sample(
    image: object,
    points: object,
    num_points: object,
    valid_paths: object,
    heatmap_logits: object,
    offsets: object,
    stem: str,
    out_path: str,
    *,
    D: int,
    min_prob: float = 0.5,
    learns_heatmap_score: bool = True,
    cpl_flat_indices: object | None = None,
    cpl_order: object | None = None,
    match_gt_yx: object | None = None,
    match_pred_mask: object | None = None,
) -> str:
    """Save the training/validation debug strip to ``out_path``.

    Accepts ``numpy`` arrays or single-sample CPU torch tensors. Shapes:

    * ``image``           ``(C, H, W)`` with ``C in (2, 3)`` (legacy synthetic
                          road+ego heatmap or nuScenes RGB BEV)
    * ``points``          ``(M, 2)`` (single picked path) or ``(K, M, 2)``
      (eval-mode: all paths drawn in panel 2). ``(x, y)`` full-res pixels.
    * ``num_points``      ``(K,)`` ``int`` (or scalar for single-path mode)
    * ``valid_paths``     ``(K,)`` ``bool``
    * ``heatmap_logits``  ``(h, w)`` raw logits (no sigmoid)
    * ``offsets``         ``(2, h, w)`` raw pixel offsets

    Panel-selection knobs:

    * ``learns_heatmap_score`` -- ``False`` for CPL+grid runs only; the
      heatmap panel is then skipped because the score is not supervised.
    * ``cpl_flat_indices``    -- 1D flat indices from ``cpl_greedy_decode``;
      when present, every prediction-related panel uses these cells (and the
      CPL-order panel is appended).
    * ``cpl_order``           -- ``(h, w)`` selection-step grid; required
      when ``cpl_flat_indices`` is provided.
    * ``match_gt_yx`` /
      ``match_pred_mask``     -- when both are provided, the matching panel
      is drawn (same matcher as training). ``match_gt_yx`` is ``(N, 2)``
      ``(y, x)`` GT pixels and ``match_pred_mask`` is the ``(h, w)``
      ``{0, 1}`` matched-pred cell mask.
    """
    image_np = _to_numpy_f32(image)
    points_np = _to_numpy_f32(points)
    num_points_np = _to_int_array(num_points)
    valid_paths_np = _to_valid_paths_bool(valid_paths)
    heatmap_np = _to_numpy_f32(heatmap_logits)
    offsets_np = _to_numpy_f32(offsets)

    # The training-mode dataset returns a single ``(M, 2)`` for the picked
    # path and a scalar count; promote to ``(1, M, 2)`` / ``(1,)`` so panel 2
    # treats it as a single valid path. Eval-mode tensors are unchanged.
    if points_np.ndim == 2:
        points_np = points_np[None, ...]
        num_points_np = num_points_np.reshape(1)
        valid_paths_np = np.array([True], dtype=bool)
    elif points_np.ndim != 3:
        raise ValueError(f"points must be (M, 2) or (K, M, 2), got {points_np.shape}")

    cpl_order_np: np.ndarray | None = _to_numpy_f32(cpl_order) if cpl_order is not None else None
    match_gt_np: np.ndarray | None = _to_numpy_f32(match_gt_yx) if match_gt_yx is not None else None
    match_pred_np: np.ndarray | None = _to_numpy_f32(match_pred_mask) if match_pred_mask is not None else None

    if image_np.ndim != 3 or image_np.shape[0] not in (2, 3):
        raise ValueError(f"image must be (C, H, W) with C in (2, 3), got {image_np.shape}")
    if heatmap_np.ndim != 2:
        raise ValueError(f"heatmap_logits must be (h, w), got {heatmap_np.shape}")
    if offsets_np.ndim != 3 or offsets_np.shape[0] != 2:
        raise ValueError(f"offsets must be (2, h, w), got {offsets_np.shape}")
    if (match_gt_np is None) != (match_pred_np is None):
        raise ValueError("Pass both match_gt_yx and match_pred_mask, or neither.")

    cpl_idx_np: np.ndarray | None = None
    if cpl_flat_indices is not None:
        if hasattr(cpl_flat_indices, "detach"):
            cpl_idx_np = cpl_flat_indices.detach().cpu().numpy()  # type: ignore[union-attr]
        else:
            cpl_idx_np = np.asarray(cpl_flat_indices, dtype=np.int64)
        if cpl_order_np is None:
            raise ValueError("cpl_order is required whenever cpl_flat_indices is provided.")

    fig = _build_training_val_figure(
        image_np,
        points_np,
        num_points_np,
        valid_paths_np,
        heatmap_np,
        offsets_np,
        stem,
        D=int(D),
        min_prob=float(min_prob),
        learns_heatmap_score=bool(learns_heatmap_score),
        cpl_flat_indices=cpl_idx_np,
        cpl_order=cpl_order_np,
        match_gt_yx=match_gt_np,
        match_pred_mask=match_pred_np,
    )
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_p)


def _normalize_multi_hyp_offsets_np(offsets_all: np.ndarray, n_hyp: int, h_lr: int, w_lr: int) -> np.ndarray:
    """Normalize offsets to ``(K, 2, h, w)`` for multi-hypothesis rendering."""
    arr = np.asarray(offsets_all, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"offsets_all must be 4D, got {arr.shape}")
    if arr.shape[0] != n_hyp:
        raise ValueError(f"offsets_all K mismatch: expected {n_hyp}, got {arr.shape[0]}")
    if arr.shape[1] == 2 and arr.shape[2] == h_lr and arr.shape[3] == w_lr:
        return arr
    if arr.shape[-1] == 2 and arr.shape[1] == h_lr and arr.shape[2] == w_lr:
        return np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)
    raise ValueError(f"offsets_all must be (K,2,h,w) or (K,h,w,2), got {arr.shape}")


def _build_multi_hypothesis_figure(
    image: np.ndarray,
    heatmap_logits_all: np.ndarray,
    offsets_all: np.ndarray,
    hypothesis_probs: np.ndarray,
    selected_hypothesis: int,
    stem: str,
    *,
    D: int,
    min_prob: float,
) -> plt.Figure:
    """Build a debug figure with all K hypotheses + selector distribution."""
    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[0] not in (2, 3):
        raise ValueError(f"image must be (C, H, W) with C in (2, 3), got {image_arr.shape}")
    logits = np.asarray(heatmap_logits_all, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"heatmap_logits_all must be (K, h, w), got {logits.shape}")
    n_hyp, h_lr, w_lr = logits.shape

    probs = np.asarray(hypothesis_probs, dtype=np.float32).reshape(-1)
    if probs.shape[0] != n_hyp:
        raise ValueError(f"hypothesis_probs length mismatch: expected {n_hyp}, got {probs.shape[0]}")
    probs = np.clip(probs, 0.0, 1.0)

    selected = int(selected_hypothesis)
    if selected < 0 or selected >= n_hyp:
        raise ValueError(f"selected_hypothesis {selected} out of range [0, {n_hyp - 1}]")

    offsets = _normalize_multi_hyp_offsets_np(offsets_all, n_hyp, h_lr, w_lr)
    H, W = _image_hw(image_arr)

    n_panels = n_hyp + 1
    n_cols = int(min(4, n_panels))
    n_rows = int(np.ceil(float(n_panels) / float(n_cols)))
    fig, axes_obj = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = list(np.atleast_1d(axes_obj).reshape(-1))

    for k in range(n_hyp):
        ax = axes[k]
        score = _sigmoid(logits[k])
        _show_bev(ax, image_arr, zorder=0)
        pred_up = _upsample_score_grid_bilinear(score, h_lr, w_lr, H, W)
        ax.imshow(pred_up, cmap="hot", vmin=0.0, vmax=1.0, alpha=0.55, interpolation="bilinear", zorder=1)
        cell_i, cell_j = _select_cells_above_threshold(score, min_prob)
        _quiver_cells_to_points(ax, cell_i, cell_j, offsets[k], D=D)
        tag = " [selected]" if k == selected else ""
        ax.set_title(f"Hyp {k}{tag}\np={float(probs[k]):.3f} | active={int(cell_i.size)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == selected:
            for spine in ax.spines.values():
                spine.set_color("#22aa22")
                spine.set_linewidth(2.0)

    dist_ax = axes[n_hyp]
    x = np.arange(n_hyp, dtype=np.int64)
    colors = np.array(["#4c78a8"] * n_hyp, dtype=object)
    colors[selected] = "#22aa22"
    dist_ax.bar(x, probs, color=colors.tolist(), width=0.8)
    dist_ax.set_ylim(0.0, 1.0)
    dist_ax.set_xticks(x)
    dist_ax.set_xlabel("Hypothesis k")
    dist_ax.set_ylabel("p(k)")
    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
    dist_ax.set_title(f"Selector distribution\nselected={selected} | entropy={entropy:.3f}", fontsize=10)
    for k in range(n_hyp):
        dist_ax.text(float(k), float(probs[k]) + 0.02, f"{float(probs[k]):.2f}", ha="center", va="bottom", fontsize=8)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Multi-hypothesis debug — {stem}", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def render_multi_hypothesis_sample(
    image: object,
    heatmap_logits_all: object,
    offsets_all: object,
    hypothesis_probs: object,
    selected_hypothesis: object,
    stem: str,
    out_path: str,
    *,
    D: int,
    min_prob: float = 0.5,
) -> str:
    """Save a dedicated multi-hypothesis debug image for one sample.

    Expected shapes:

    * ``heatmap_logits_all``: ``(K, h, w)``
    * ``offsets_all``: ``(K, 2, h, w)`` (or ``(K, h, w, 2)``)
    * ``hypothesis_probs``: ``(K,)``
    * ``selected_hypothesis``: scalar index
    """
    image_np = _to_numpy_f32(image)
    heatmap_np = _to_numpy_f32(heatmap_logits_all)
    offsets_np = _to_numpy_f32(offsets_all)
    probs_np = _to_numpy_f32(hypothesis_probs)
    sel_np = np.asarray(_to_numpy_f32(selected_hypothesis)).reshape(-1)
    if sel_np.size != 1:
        raise ValueError(f"selected_hypothesis must be scalar, got shape {sel_np.shape}")
    sel = int(sel_np[0])

    fig = _build_multi_hypothesis_figure(
        image_np,
        heatmap_np,
        offsets_np,
        probs_np,
        sel,
        stem,
        D=int(D),
        min_prob=float(min_prob),
    )
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_p)


def render_cpl_selection_order(
    order_grid: object,
    stem: str,
    out_path: str,
    *,
    cmap: str = "viridis",
) -> str:
    """Save a stand-alone low-res image of CPL greedy selection order (1..K)."""
    order_np = _to_numpy_f32(order_grid)
    if order_np.ndim != 2:
        raise ValueError(f"order_grid must be (h, w), got {order_np.shape}")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="white")
    _add_cpl_order_subplot(ax, order_np, stem, cmap=cmap)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_p, dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return str(out_p)


def render_sample(npz_path: str, out_path: str | None = None, show: bool = False) -> str | None:
    """Render the single-sample preview for an ``.npz`` (gallery / ``--in`` mode)."""
    npz = Path(npz_path)
    image, points, num_points, valid_paths, meta = _load_sample(npz)
    fig = _build_sample_figure(image, points, num_points, valid_paths, meta, npz.stem)

    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out_p)

    if show:
        plt.show()
    plt.close(fig)
    return None


# ---------------------------------------------------------------------------
# In-memory LRU cache for rendered .npz samples (used by the web gallery).
# ---------------------------------------------------------------------------

_RENDER_CACHE: OrderedDict[tuple[str, float], bytes] = OrderedDict()
_RENDER_CACHE_LIMIT = 128
_RENDER_CACHE_LOCK = threading.Lock()
# Serialise matplotlib figure work; pyplot's global state is not thread-safe.
_RENDER_FIGURE_LOCK = threading.Lock()


def _render_npz_to_png_bytes(npz_path: Path, stem_label: str) -> bytes:
    try:
        mtime = npz_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = (str(npz_path), mtime)
    with _RENDER_CACHE_LOCK:
        cached = _RENDER_CACHE.get(key)
        if cached is not None:
            _RENDER_CACHE.move_to_end(key)
            return cached

    image, points, num_points, valid_paths, meta = _load_sample(npz_path)
    buf = io.BytesIO()
    with _RENDER_FIGURE_LOCK:
        fig = _build_sample_figure(image, points, num_points, valid_paths, meta, stem_label)
        try:
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        finally:
            plt.close(fig)
    data = buf.getvalue()

    with _RENDER_CACHE_LOCK:
        _RENDER_CACHE[key] = data
        _RENDER_CACHE.move_to_end(key)
        while len(_RENDER_CACHE) > _RENDER_CACHE_LIMIT:
            _RENDER_CACHE.popitem(last=False)
    return data


# ---------------------------------------------------------------------------
# Web gallery: browse a directory of images on a local HTTP port
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tiff", ".tif"})
_SAMPLE_EXTENSIONS = frozenset({".npz"})
_GALLERY_EXTENSIONS = _IMAGE_EXTENSIONS | _SAMPLE_EXTENSIONS

# Cap recursion to avoid pathological listings; a typical road-cpl run is a
# couple of levels deep (`<run>/train/00000000.npz`).
_GALLERY_MAX_DEPTH = 4

_GALLERY_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>road-cpl gallery</title>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100vh; background: #111; color: #e0e0e0; font-family: system-ui, sans-serif; }
    body { display: flex; flex-direction: column; }
    .bar { flex: 0 0 auto; display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: #1a1a1a; border-bottom: 1px solid #333; flex-wrap: wrap; gap: 8px; }
    .bar h1 { margin: 0; font-size: 14px; font-weight: 600; word-break: break-all; display: flex; align-items: center; gap: 8px; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
    .badge.train { background: #1f4d2b; color: #b8e6c4; }
    .badge.val   { background: #4a3520; color: #f0c896; }
    .badge.image { background: #2a3a55; color: #b8d0f0; }
    .badge.other { background: #3a3a3a; color: #ccc; }
    .meta { font-size: 12px; color: #888; }
    .controls { display: flex; gap: 6px; align-items: center; }
    button { background: #333; color: #fff; border: 1px solid #555; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; }
    button:hover { background: #444; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    .jump { display: flex; align-items: center; gap: 4px; font-size: 12px; color: #aaa; }
    .jump input { width: 80px; background: #222; color: #fff; border: 1px solid #444; border-radius: 3px; padding: 4px 6px; font-family: monospace; font-size: 12px; }
    .jump input:focus { outline: 1px solid #4a8acc; }
    select { background: #222; color: #fff; border: 1px solid #444; border-radius: 3px; padding: 4px 6px; font-size: 12px; }
    .hint { flex: 0 0 auto; font-size: 11px; color: #666; margin: 6px 12px 0; }
    .viewport { flex: 1 1 auto; min-height: 240px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 8px; overflow: auto; }
    .frame { flex: 0 1 auto; max-width: 100%; max-height: min(82vh, calc(100vh - 160px)); min-height: 80px; display: flex; align-items: center; justify-content: center; position: relative; }
    #view { max-width: 100%; max-height: min(82vh, calc(100vh - 160px)); width: auto; height: auto; object-fit: contain; display: block; }
    .spinner { position: absolute; color: #888; font-size: 12px; }
    .err { color: #f88; padding: 8px; max-width: 100%; word-break: break-all; }
  </style>
</head>
<body>
  <div class="bar">
    <h1 id="title"><span id="kind-badge"></span><span id="title-text">road-cpl gallery</span></h1>
    <div class="meta" id="counter"></div>
    <div class="controls">
      <select id="filter" title="Filter">
        <option value="all">All</option>
        <option value="sample">Samples (.npz)</option>
        <option value="image">Images</option>
        <option value="train">train/</option>
        <option value="val">val/</option>
      </select>
      <button type="button" id="first" title="First (Home)">&#9198;</button>
      <button type="button" id="prev10" title="-10 (PageUp)">-10</button>
      <button type="button" id="prev" title="Previous (Left)">&#9664;</button>
      <button type="button" id="next" title="Next (Right)">&#9654;</button>
      <button type="button" id="next10" title="+10 (PageDown)">+10</button>
      <button type="button" id="last" title="Last (End)">&#9197;</button>
      <span class="jump">jump <input id="jump" type="number" min="1" /> </span>
    </div>
  </div>
  <p class="hint">&larr; / &rarr; navigate &nbsp; PageUp / PageDown +-10 &nbsp; Home / End first/last &nbsp; type a number into the jump box and press Enter</p>
  <div class="viewport">
    <div id="msg" class="err" style="display:none"></div>
    <div class="frame" id="frame" style="display:none">
      <img id="view" alt="preview" />
      <div id="spinner" class="spinner" style="display:none">rendering...</div>
    </div>
  </div>
  <script>
    let allItems = [];
    let items = [];
    let idx = 0;
    let filter = 'all';
    const view = document.getElementById('view');
    const frame = document.getElementById('frame');
    const spinner = document.getElementById('spinner');
    const msg = document.getElementById('msg');
    const counter = document.getElementById('counter');
    const titleText = document.getElementById('title-text');
    const kindBadge = document.getElementById('kind-badge');
    const jumpInp = document.getElementById('jump');
    const filterSel = document.getElementById('filter');

    function imageUrlForName(name) {
      const enc = name.split('/').map(encodeURIComponent).join('/');
      return '/i/' + enc + '?t=' + Date.now();
    }
    function classifyPath(name) {
      if (name.startsWith('train/')) return 'train';
      if (name.startsWith('val/')) return 'val';
      return null;
    }
    function applyFilter() {
      if (filter === 'all') { items = allItems.slice(); return; }
      if (filter === 'sample' || filter === 'image' || filter === 'other') {
        items = allItems.filter(it => it.kind === filter);
      } else if (filter === 'train' || filter === 'val') {
        items = allItems.filter(it => it.name.startsWith(filter + '/'));
      } else {
        items = allItems.slice();
      }
    }
    function show(i) {
      if (items.length === 0) {
        frame.style.display = 'none';
        msg.style.display = 'block';
        msg.textContent = 'No matching items in this directory.';
        counter.textContent = '0 / 0';
        titleText.textContent = '(empty)';
        kindBadge.style.display = 'none';
        jumpInp.value = '';
        return;
      }
      i = ((i % items.length) + items.length) % items.length;
      idx = i;
      const it = items[i];
      titleText.textContent = it.name;
      view.alt = it.name;
      msg.style.display = 'none';
      frame.style.display = 'flex';
      const isSample = it.kind === 'sample';
      spinner.style.display = isSample ? 'block' : 'none';
      const split = classifyPath(it.name);
      const badgeKind = split || it.kind || 'other';
      kindBadge.className = 'badge ' + badgeKind;
      kindBadge.style.display = 'inline-block';
      kindBadge.textContent = badgeKind;
      const url = imageUrlForName(it.name);
      view.onerror = async function() {
        spinner.style.display = 'none';
        let detail = '';
        try {
          const r = await fetch(url, { cache: 'no-store' });
          detail = ' (HTTP ' + r.status + ')';
          if (!r.ok) {
            try { detail += ' - ' + (await r.text()).slice(0, 200); } catch (e) {}
          }
        } catch (e) {}
        let extra = '';
        if (location.hostname === '0.0.0.0') {
          extra = ' Open http://127.0.0.1:' + location.port + '/ instead of http://0.0.0.0.';
        }
        msg.style.display = 'block';
        msg.textContent = 'Failed to load' + detail + ': ' + url + '.' + extra;
        frame.style.display = 'none';
      };
      view.onload = function() {
        spinner.style.display = 'none';
        msg.style.display = 'none';
        frame.style.display = 'flex';
      };
      view.src = url;
      counter.textContent = (i + 1) + ' / ' + items.length;
      jumpInp.value = String(i + 1);
      jumpInp.max = String(items.length);
      if (window.history && window.history.replaceState) {
        try {
          const params = new URLSearchParams();
          params.set('i', String(i));
          if (filter !== 'all') params.set('f', filter);
          window.history.replaceState(null, '', '/?' + params.toString());
        } catch (e) {}
      }
    }
    function prev() { show(idx - 1); }
    function next() { show(idx + 1); }
    function step(d) { show(idx + d); }
    document.getElementById('prev').addEventListener('click', prev);
    document.getElementById('next').addEventListener('click', next);
    document.getElementById('prev10').addEventListener('click', () => step(-10));
    document.getElementById('next10').addEventListener('click', () => step(10));
    document.getElementById('first').addEventListener('click', () => show(0));
    document.getElementById('last').addEventListener('click', () => show(items.length - 1));
    view.addEventListener('click', (e) => { if (e.clientX < window.innerWidth/2) prev(); else next(); });
    jumpInp.addEventListener('change', () => {
      const n = parseInt(jumpInp.value, 10);
      if (!isNaN(n)) show(n - 1);
    });
    jumpInp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); jumpInp.dispatchEvent(new Event('change')); } });
    filterSel.addEventListener('change', () => {
      filter = filterSel.value;
      applyFilter();
      show(0);
    });
    document.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
      if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); next(); }
      else if (e.key === 'Home') { e.preventDefault(); show(0); }
      else if (e.key === 'End') { e.preventDefault(); show(items.length - 1); }
      else if (e.key === 'PageUp') { e.preventDefault(); step(-10); }
      else if (e.key === 'PageDown') { e.preventDefault(); step(10); }
    });
    fetch('/api/images').then(r => r.json()).then(data => {
      if (Array.isArray(data.items)) {
        allItems = data.items;
      } else if (Array.isArray(data.images)) {
        allItems = data.images.map(n => ({ name: n, kind: n.endsWith('.npz') ? 'sample' : 'image' }));
      } else {
        allItems = [];
      }
      const p = new URLSearchParams(window.location.search);
      const f = p.get('f');
      if (f && ['all','sample','image','train','val','other'].indexOf(f) >= 0) {
        filter = f;
        filterSel.value = f;
      }
      applyFilter();
      const n = parseInt(p.get('i') || '0', 10);
      show(!isNaN(n) && n >= 0 && n < items.length ? n : 0);
    }).catch(() => {
      msg.style.display = 'block';
      msg.textContent = 'Failed to load gallery list.';
    });
  </script>
</body>
</html>
"""


def _list_gallery_files(root: Path) -> list[str]:
    """List image and ``.npz`` sample files under ``root`` (recursive, sorted)."""
    if not root.is_dir():
        return []
    found: list[str] = []
    root_depth = len(root.parts)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _GALLERY_EXTENSIONS:
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        if len(p.parts) - root_depth > _GALLERY_MAX_DEPTH:
            continue
        found.append(rel.as_posix())
    found.sort()
    return found


def _safe_gallery_path(root: Path, name: str) -> Path | None:
    """Return the file path under ``root`` for the relative ``name``, or ``None``."""
    if not name or name.startswith(("/", "\\")):
        return None
    posix = name.replace("\\", "/")
    parts = [seg for seg in posix.split("/") if seg]
    if not parts or any(seg in ("..", ".") for seg in parts):
        return None
    try:
        root_r = root.resolve()
        candidate = (root.joinpath(*parts)).resolve()
    except (OSError, ValueError):
        return None
    try:
        candidate.relative_to(root_r)
    except ValueError:
        return None
    if not candidate.is_file() or candidate.suffix.lower() not in _GALLERY_EXTENSIONS:
        return None
    return candidate


def _send_json(handler: BaseHTTPRequestHandler, obj: object, code: int = 200) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _send_bytes(handler: BaseHTTPRequestHandler, body: bytes, content_type: str, code: int = 200) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _classify_item(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in _SAMPLE_EXTENSIONS:
        return "sample"
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    return "other"


def _items_payload(root: Path) -> dict[str, object]:
    names = _list_gallery_files(root)
    items = [{"name": n, "kind": _classify_item(n)} for n in names]
    counts: dict[str, int] = {}
    for it in items:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1  # type: ignore[index]
    return {
        "items": items,
        "counts": counts,
        "root": str(root),
        "images": names,
    }


def _gallery_handler_class(root: Path) -> type[BaseHTTPRequestHandler]:
    root_resolved = root.resolve()

    class GalleryRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args) -> None:
            if self.path.startswith("/i/") or self.path in ("/", "/api/images"):
                return
            super().log_message(fmt, *log_args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"

            if path == "/":
                body = _GALLERY_INDEX_HTML.encode("utf-8")
                _send_bytes(self, body, "text/html; charset=utf-8")
                return

            if path == "/api/images":
                _send_json(self, _items_payload(root_resolved))
                return

            if path.startswith("/i/"):
                raw = path[len("/i/") :]
                if not raw:
                    self.send_error(404, "Not found")
                    return
                name = unquote(raw, encoding="utf-8")
                full = _safe_gallery_path(root_resolved, name)
                if full is None:
                    self.send_error(404, "Not found")
                    return

                if full.suffix.lower() in _SAMPLE_EXTENSIONS:
                    try:
                        rel_stem = str(full.relative_to(root_resolved).with_suffix(""))
                    except ValueError:
                        rel_stem = full.stem
                    try:
                        png = _render_npz_to_png_bytes(full, rel_stem)
                    except Exception as exc:  # pragma: no cover - surfaced to client
                        self.send_error(500, f"Render error: {exc}")
                        return
                    _send_bytes(self, png, "image/png")
                    return

                try:
                    blob = full.read_bytes()
                except OSError:
                    self.send_error(500, "Read error")
                    return
                mt, _ = mimetypes.guess_type(str(full))
                if not mt:
                    mt = "application/octet-stream"
                _send_bytes(self, blob, mt)
                return

            self.send_error(404, "Not found")

    return GalleryRequestHandler


def run_gallery(image_dir: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start a local HTTP server that serves a simple image browser UI."""
    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    handler = _gallery_handler_class(root)
    server = ThreadingHTTPServer((host, int(port)), handler)  # type: ignore[misc]
    base = f"http://{host}:{int(port)}/"
    items = _list_gallery_files(root)
    n_samples = sum(1 for n in items if Path(n).suffix.lower() in _SAMPLE_EXTENSIONS)
    n_images = len(items) - n_samples
    print(f"Gallery: {len(items)} item(s) in {root} ({n_samples} sample(s), {n_images} image(s))")
    print(f"Serving at {base} (Ctrl+C to stop)")

    if open_browser:
        try:
            webbrowser.open(base)
        except (OSError, webbrowser.Error):
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=("Visualise a single road-cpl .npz sample, or browse a directory of images and/or .npz dataset samples in a web gallery."))
    parser.add_argument("--in", dest="inp", default=None, help="Path to a single <id>.npz (omit if using --dir)")
    parser.add_argument("--out", dest="out", default=None, help="Output PNG path for single .npz mode")
    parser.add_argument("--show", action="store_true", help="Open matplotlib view for single .npz")
    parser.add_argument("--dir", default=None, help="Directory to browse in the web gallery (recursive).")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for --dir (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser when using --dir")
    args = parser.parse_args()

    if args.dir is not None:
        run_gallery(args.dir, host=args.host, port=args.port, open_browser=not args.no_browser)
        return

    if not args.inp:
        parser.error("Provide --in <path.npz> for a single sample, or --dir for web gallery")
    render_sample(args.inp, args.out, args.show)


if __name__ == "__main__":
    main()
