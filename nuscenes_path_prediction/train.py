"""Entry point for training and evaluating the road path forecaster.

This module wires together the dataset, model, losses, optimizer, validation
metrics, visualization, and Comet logging. It supports two run modes:

* **Training**: optimize the model for ``run.num_epochs`` and save two best
  checkpoints under ``run.resolved_save_dir``: one at the lowest validation
  ``val_min_ade`` (``best_min_ade_model.pth``) and one at the lowest
  validation ``val_min_hd`` (``best_min_hd_model.pth``).
* **Test-only** (``--test-only``): load a checkpoint, run a single validation
  pass (loss + path metrics + viz), and exit without training.

The configuration is a :class:`~train_config.RoadCplConfig` dataclass tree.
Defaults come from :func:`~train_config.get_default_road_cpl_config`; an
optional YAML file (``--config``) is merged on top of the defaults, and CLI
flags then override the merged configuration.

Artifacts produced under ``run.resolved_save_dir``:

* ``best_min_ade_model.pth`` -- best-by-``val_min_ade`` checkpoint with model,
  optimizer, epoch, and the full validation ``metrics`` dict at save time.
* ``best_min_hd_model.pth`` -- best-by-``val_min_hd`` checkpoint with model,
  optimizer, epoch, and the full validation ``metrics`` dict at save time.
* ``config_resolved.json`` -- the fully resolved configuration used.
* ``viz/<subdir>/sample_NN.png`` -- per-pass validation previews
  (``subdir`` is ``test`` for test-only runs, otherwise ``epoch_NNN``).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from comet_ml import Experiment
from in_memory_dataset import RoadCplInMemoryDataset
from lazy_dataset import RoadCplLazyDataset
from metrics import PathForecastingMetrics
from model import GenerativePathForecaster
from path_heads_loss import (
    GridMatcher,
    HungarianMatcher,
    MultiHypothesisPathLoss,
    PathHeadsLoss,
    decode_subpixel_points,
)
from runtime_profile import (
    CudaTimerStats,
    aggregate_inference_timing,
    attach_inference_hooks,
    detach_forward_hooks,
    print_inference_timing_report,
    run_inference_loop,
    save_inference_timing_report,
    warmup_inference,
)
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_config import (
    DataConfig,
    EvalMetricsConfig,
    GeneratorConfig,
    ModelConfig,
    OptimizerConfig,
    RoadCplConfig,
    get_default_road_cpl_config,
    load_road_cpl_config_from_path,
    resolve_run_work_dirs,
    road_cpl_config_to_comet_params,
    save_road_cpl_config_json,
)
from visualize import render_multi_hypothesis_sample, render_training_val_sample

# ---------------------------------------------------------------------------
# Constants & logger
# ---------------------------------------------------------------------------

BEST_MIN_ADE_CHECKPOINT_FILENAME = "best_min_ade_model.pth"
BEST_MIN_HD_CHECKPOINT_FILENAME = "best_min_hd_model.pth"
RESOLVED_CONFIG_FILENAME = "config_resolved.json"
INFERENCE_TIMING_FILENAME = "inference_timing.json"
VIZ_ROOT_SUBDIR = "viz"
TEST_VIZ_SUBDIR = "test"

# Replace non-finite or implausibly large metric values with this sentinel
# before logging them, so that one bad sample does not poison the average.
DEFAULT_METRIC_REPLACE_INF = 1e3
METRIC_OVERFLOW_THRESHOLD = 1.0e7

# Train-only image/label augmentation probabilities.
TRAIN_FLIP_LEFT_RIGHT_PROB = 0.5
TRAIN_FLIP_UP_DOWN_PROB = 0.5

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A collated batch produced by :func:`collate_fn`. Tensor fields are stacked
#: along dimension 0; non-tensor fields (e.g. ``meta``, ``path``) are lists.
BatchDict = dict[str, Any]

#: Dictionary returned by :meth:`GenerativePathForecaster.forward`.
ModelOutputs = dict[str, torch.Tensor | dict[str, torch.Tensor] | None]

#: Heatmap+offset loss container. Either:
#: - :class:`PathHeadsLoss` for baseline / CPL / AR runs (always present, even
#:   under CPL it computes the offset L1 term with the heatmap term disabled), or
#: - :class:`MultiHypothesisPathLoss` for the K-way multi-hypothesis baseline.
LossFn = PathHeadsLoss | MultiHypothesisPathLoss

#: Either backend exposes the same ``__getitem__`` schema and ``__len__``,
#: so callers (viz, metrics) can use them interchangeably.
RoadCplDataset = RoadCplInMemoryDataset | RoadCplLazyDataset


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict[str, Any]], *, downsample_factor: int = 8) -> BatchDict:
    """Stack fields and sparsify GT points to unique downsample cells."""
    result = _stack_batch_fields(batch)
    _preprocess_gt_points_in_batch(result, downsample_factor=downsample_factor)
    return result


def _stack_batch_fields(batch: list[dict[str, Any]]) -> BatchDict:
    """Stack tensor fields along dim 0; collect non-tensor fields into lists."""
    result: BatchDict = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result


def _resample_polyline_uniform_arclength(poly_xy: np.ndarray, *, step_px: float) -> np.ndarray:
    """Resample ``(x, y)`` points at roughly uniform arc-length spacing."""
    poly = np.ascontiguousarray(poly_xy, dtype=np.float32)
    if poly.shape[0] <= 1 or step_px <= 0.0:
        return poly
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arclen = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(seg).astype(np.float32)], axis=0)
    total = float(arclen[-1])
    if total <= 1e-6:
        return poly[:1].copy()
    samples = np.arange(0.0, total, float(step_px), dtype=np.float32)
    if samples.size == 0 or samples[-1] < total:
        samples = np.concatenate([samples, np.array([total], dtype=np.float32)], axis=0)
    x = np.interp(samples, arclen, poly[:, 0]).astype(np.float32)
    y = np.interp(samples, arclen, poly[:, 1]).astype(np.float32)
    return np.ascontiguousarray(np.stack([x, y], axis=1), dtype=np.float32)


def _keep_first_point_per_downsample_cell(
    poly_xy: np.ndarray,
    *,
    downsample_factor: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Keep only the first point that lands in each low-res cell."""
    poly = np.ascontiguousarray(poly_xy, dtype=np.float32)
    if poly.shape[0] <= 1:
        return poly
    d = max(1, int(downsample_factor))
    h = max(1, int(grid_h))
    w = max(1, int(grid_w))
    cells = np.floor(poly / float(d)).astype(np.int64)
    cell_j = np.clip(cells[:, 0], 0, w - 1)
    cell_i = np.clip(cells[:, 1], 0, h - 1)
    keep = np.zeros(poly.shape[0], dtype=np.bool_)
    seen: set[tuple[int, int]] = set()
    for idx in range(poly.shape[0]):
        key = (int(cell_i[idx]), int(cell_j[idx]))
        if key in seen:
            continue
        seen.add(key)
        keep[idx] = True
    out = poly[keep]
    if out.shape[0] == 0:
        out = poly[:1]
    return np.ascontiguousarray(out, dtype=np.float32)


def _sparsify_gt_path_points(
    poly_xy: np.ndarray,
    *,
    downsample_factor: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Resample a GT path, then enforce unique cells in the ``/downsample`` grid."""
    if poly_xy.shape[0] <= 1:
        return np.ascontiguousarray(poly_xy, dtype=np.float32)
    step_px = float(max(1, int(downsample_factor))) * math.sqrt(2.0)
    resampled = _resample_polyline_uniform_arclength(poly_xy, step_px=step_px)
    return _keep_first_point_per_downsample_cell(
        resampled,
        downsample_factor=downsample_factor,
        grid_h=grid_h,
        grid_w=grid_w,
    )


def _sparsify_single_gt_path_tensor(
    gt_points_path: torch.Tensor,
    *,
    n_valid: int,
    downsample_factor: int,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Sparsify one padded GT path tensor and return just its valid prefix."""
    n = max(0, min(int(n_valid), int(gt_points_path.shape[0])))
    if n <= 0:
        return gt_points_path.new_zeros((0, 2))
    poly_np = gt_points_path[:n].detach().to(device=torch.device("cpu"), dtype=torch.float32).numpy()
    sparse_np = _sparsify_gt_path_points(
        poly_np,
        downsample_factor=downsample_factor,
        grid_h=grid_h,
        grid_w=grid_w,
    )
    if sparse_np.shape[0] > gt_points_path.shape[0]:
        sparse_np = sparse_np[: gt_points_path.shape[0]]
    return torch.from_numpy(sparse_np).to(device=gt_points_path.device, dtype=gt_points_path.dtype)


def _preprocess_gt_points_in_batch(result: BatchDict, *, downsample_factor: int) -> None:
    """In-place GT sparsification so no two points share a downsampled cell."""
    images = result.get("image")
    gt_points = result.get("gt_points")
    num_points = result.get("num_points")
    if not (isinstance(images, torch.Tensor) and isinstance(gt_points, torch.Tensor) and isinstance(num_points, torch.Tensor)):
        return
    if images.ndim != 4 or downsample_factor < 1:
        return

    grid_h = max(1, int(images.shape[-2]) // int(downsample_factor))
    grid_w = max(1, int(images.shape[-1]) // int(downsample_factor))

    if gt_points.ndim == 3 and num_points.ndim == 1:
        new_points = gt_points.clone()
        new_num_points = num_points.to(torch.long).clone()
        for b in range(new_points.shape[0]):
            sparse = _sparsify_single_gt_path_tensor(
                new_points[b],
                n_valid=int(new_num_points[b].item()),
                downsample_factor=downsample_factor,
                grid_h=grid_h,
                grid_w=grid_w,
            )
            n_new = min(int(new_points.shape[1]), int(sparse.shape[0]))
            new_points[b].zero_()
            if n_new > 0:
                new_points[b, :n_new] = sparse[:n_new]
            new_num_points[b] = n_new
        result["gt_points"] = new_points
        result["num_points"] = new_num_points
        return

    if gt_points.ndim == 4 and num_points.ndim == 2:
        new_points = gt_points.clone()
        new_num_points = num_points.to(torch.long).clone()
        valid_paths = result.get("valid_paths")
        new_valid_paths = valid_paths.clone() if isinstance(valid_paths, torch.Tensor) and valid_paths.shape == new_num_points.shape else None
        for b in range(new_points.shape[0]):
            for k in range(new_points.shape[1]):
                sparse = _sparsify_single_gt_path_tensor(
                    new_points[b, k],
                    n_valid=int(new_num_points[b, k].item()),
                    downsample_factor=downsample_factor,
                    grid_h=grid_h,
                    grid_w=grid_w,
                )
                n_new = min(int(new_points.shape[2]), int(sparse.shape[0]))
                new_points[b, k].zero_()
                if n_new > 0:
                    new_points[b, k, :n_new] = sparse[:n_new]
                new_num_points[b, k] = n_new
                if new_valid_paths is not None:
                    new_valid_paths[b, k] = n_new >= 2
        result["gt_points"] = new_points
        result["num_points"] = new_num_points
        if new_valid_paths is not None:
            result["valid_paths"] = new_valid_paths


def train_collate_fn(
    batch: list[dict[str, Any]],
    *,
    use_flip_augmentation: bool,
    downsample_factor: int,
) -> BatchDict:
    """Collate train samples, optionally flip, then sparsify GT points."""
    result = _stack_batch_fields(batch)
    if use_flip_augmentation:
        images = result.get("image")
        gt_points = result.get("gt_points")
        num_points = result.get("num_points")
        if isinstance(images, torch.Tensor) and isinstance(gt_points, torch.Tensor) and isinstance(num_points, torch.Tensor):
            aug_images, aug_gt_points = _apply_random_flip_augmentations(images, gt_points, num_points)
            result["image"] = aug_images
            result["gt_points"] = aug_gt_points
    _preprocess_gt_points_in_batch(result, downsample_factor=downsample_factor)
    return result


def _pin_memory_for(device: torch.device | None) -> bool:
    """Whether ``pin_memory=True`` is appropriate for the given device."""
    return device is not None and device.type == "cuda"


def _build_dataset(
    *,
    data_cfg: DataConfig,
    root: str,
    split: str,
    mode: str,
    rng_seed: int | None = None,
) -> RoadCplDataset:
    """Pick the lazy or in-memory dataset backend based on ``data_cfg.lazy``."""
    cls: type[RoadCplDataset] = RoadCplLazyDataset if data_cfg.lazy else RoadCplInMemoryDataset
    return cls(root=root, split=split, mode=mode, rng_seed=rng_seed)


def _make_loader(
    dataset: RoadCplDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    collate: Callable[[list[dict[str, Any]]], BatchDict] = collate_fn,
) -> DataLoader:
    """Build a :class:`DataLoader` with the project's collate function.

    ``persistent_workers`` and ``prefetch_factor`` only take effect when
    ``num_workers > 0`` (PyTorch raises otherwise).
    """
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def build_train_loader(
    data_root: str,
    *,
    batch_size: int,
    data_cfg: DataConfig,
    device: torch.device | None,
    use_flip_augmentation: bool,
    downsample_factor: int,
) -> DataLoader:
    """Train-mode loader: one randomly chosen valid GT per sample, shuffled.

    When ``use_flip_augmentation`` is true, train batches are randomly flipped
    left/right and up/down in the collate function, with GT points mirrored
    consistently.
    """
    dataset = _build_dataset(data_cfg=data_cfg, root=data_root, split="train", mode="train")
    return _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.train_num_workers,
        pin_memory=_pin_memory_for(device),
        persistent_workers=data_cfg.persistent_workers,
        prefetch_factor=data_cfg.prefetch_factor,
        collate=partial(
            train_collate_fn,
            use_flip_augmentation=use_flip_augmentation,
            downsample_factor=downsample_factor,
        ),
    )


def build_val_loader(
    data_root: str,
    *,
    batch_size: int,
    data_cfg: DataConfig,
    device: torch.device | None,
    downsample_factor: int,
) -> DataLoader:
    """Val loader using train-mode sampling (single GT per sample), unshuffled."""
    dataset = _build_dataset(data_cfg=data_cfg, root=data_root, split="val", mode="train")
    return _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.val_num_workers,
        pin_memory=_pin_memory_for(device),
        persistent_workers=data_cfg.persistent_workers,
        prefetch_factor=data_cfg.prefetch_factor,
        collate=partial(collate_fn, downsample_factor=downsample_factor),
    )


def build_eval_loader_and_dataset(
    data_root: str,
    *,
    batch_size: int,
    data_cfg: DataConfig,
    device: torch.device | None,
    downsample_factor: int,
) -> tuple[DataLoader, RoadCplDataset]:
    """Build the eval-mode val dataset (full label stack per sample) and a loader.

    The dataset is reused for visualization (random-access by index), and the
    loader is iterated for multi-GT path metrics.
    """
    dataset = _build_dataset(
        data_cfg=data_cfg,
        root=data_root,
        split="val",
        mode="eval",
        rng_seed=0,
    )
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.val_num_workers,
        pin_memory=_pin_memory_for(device),
        persistent_workers=data_cfg.persistent_workers,
        prefetch_factor=data_cfg.prefetch_factor,
        collate=partial(collate_fn, downsample_factor=downsample_factor),
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# 2. Model / loss / optimizer factories
# ---------------------------------------------------------------------------


def build_model(
    model_cfg: ModelConfig,
    *,
    use_cpl: bool,
    use_ar: bool,
    use_multi_hypothesis: bool,
    device: torch.device,
) -> GenerativePathForecaster:
    """Instantiate the path forecaster and move it to ``device``."""
    return GenerativePathForecaster(
        model_cfg,
        use_cpl=use_cpl,
        use_ar=use_ar,
        use_multi_hypothesis=use_multi_hypothesis,
    ).to(device)


def build_loss_fn(rcfg: RoadCplConfig) -> LossFn:
    """Build the configured loss orchestrator.

    For CPL / AR / supervised baseline runs this returns a
    :class:`PathHeadsLoss` (heatmap BCE is enabled when
    :attr:`~train_config.RoadCplConfig.learns_heatmap_score` is true).

    For the K-way multi-hypothesis baseline (``model.multi_hypothesis.enabled``)
    this returns a :class:`MultiHypothesisPathLoss` with winner-take-all path
    supervision and a hypothesis-selection cross-entropy term.
    """
    matcher = _build_matcher(rcfg)
    if rcfg.use_multi_hypothesis:
        return MultiHypothesisPathLoss(
            matcher=matcher,
            heatmap_weight=rcfg.run.heatmap_weight,
            offset_weight=rcfg.run.offset_weight,
            offset_loss=rcfg.run.offset_loss,  # type: ignore[arg-type]
            matched_bce_weight=rcfg.heatmap_bce.matched_bce_weight,
            unmatched_bce_weight=rcfg.heatmap_bce.unmatched_bce_weight,
            selection_weight=rcfg.model.multi_hypothesis.ce_weight,
            classification=rcfg.learns_heatmap_score,
        )
    return PathHeadsLoss(
        matcher=matcher,
        heatmap_weight=rcfg.run.heatmap_weight,
        offset_weight=rcfg.run.offset_weight,
        offset_loss=rcfg.run.offset_loss,  # type: ignore[arg-type]
        matched_bce_weight=rcfg.heatmap_bce.matched_bce_weight,
        unmatched_bce_weight=rcfg.heatmap_bce.unmatched_bce_weight,
        classification=rcfg.learns_heatmap_score,
    )


def _build_matcher(rcfg: RoadCplConfig) -> GridMatcher | HungarianMatcher:
    """Construct the GridMatcher or HungarianMatcher implied by ``rcfg.run.matching``."""
    g = rcfg.generator
    if rcfg.run.matching == "hungarian":
        h = rcfg.hungarian
        return HungarianMatcher(
            grid_size=g.label_H,
            D=g.D,
            dist_weight=h.dist_weight,
            prob_weight=h.prob_weight,
        )
    return GridMatcher(grid_size=g.label_H, D=g.D)


def build_optimizer(
    model: GenerativePathForecaster,
    *,
    lr: float,
    cfg: OptimizerConfig,
) -> optim.Optimizer:
    """Build the AdamW optimizer for the model from :class:`OptimizerConfig`."""
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.weight_decay,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
    )


# ---------------------------------------------------------------------------
# 3. Checkpoint I/O
# ---------------------------------------------------------------------------


@dataclass
class CheckpointInfo:
    """Bookkeeping returned by :func:`load_checkpoint`.

    ``best_val_min_ade`` and ``best_val_min_hd`` are the per-metric best
    values that the training loop has observed; both default to ``+inf`` so
    the very next validation pass always improves and saves a checkpoint.
    """

    start_epoch: int = 0
    best_val_min_ade: float = float("inf")
    best_val_min_hd: float = float("inf")
    raw_metrics: dict[str, float] | None = None


def load_checkpoint(
    path: str,
    model: GenerativePathForecaster,
    *,
    optimizer: optim.Optimizer | None = None,
    device: torch.device,
    for_resume: bool,
) -> CheckpointInfo:
    """Load weights into ``model`` and optionally restore optimizer/epoch.

    Args:
        path: Path to a ``.pth`` file. May be either a raw ``state_dict`` or a
            dictionary with at least a ``model_state_dict`` key (and optional
            ``optimizer_state_dict``, ``epoch``, ``metrics``).
        model: Target model. Loaded in-place.
        optimizer: Optional optimizer to restore when ``for_resume=True`` and
            the checkpoint contains an ``optimizer_state_dict``.
        device: Map location for :func:`torch.load`.
        for_resume: If ``True``, populate ``start_epoch`` and the per-metric
            best trackers from the checkpoint's ``metrics`` payload so
            training resumes from the next epoch with the correct baseline.
            If ``False`` (test-only), return a default-initialized
            :class:`CheckpointInfo` with only ``raw_metrics`` filled when
            present in the file.

    Returns:
        A :class:`CheckpointInfo` describing where to resume.
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    info = CheckpointInfo()
    if not isinstance(ckpt, dict):
        return info
    logger.info("Loaded weights from %s at epoch %s", path, ckpt.get("epoch"))

    raw_metrics = ckpt.get("metrics")
    if isinstance(raw_metrics, dict):
        info.raw_metrics = {str(k): float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
    if for_resume:
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if info.raw_metrics is not None:
            ade = info.raw_metrics.get("val_min_ade")
            hd = info.raw_metrics.get("val_min_hd")
            if ade is not None and math.isfinite(ade):
                info.best_val_min_ade = float(ade)
            if hd is not None and math.isfinite(hd):
                info.best_val_min_hd = float(hd)
        if "epoch" in ckpt:
            info.start_epoch = int(ckpt["epoch"]) + 1
    return info


def save_best_checkpoint(
    out_dir: str,
    *,
    filename: str,
    epoch: int,
    model: GenerativePathForecaster,
    optimizer: optim.Optimizer,
    metrics: dict[str, float],
    rcfg: RoadCplConfig,
) -> str:
    """Persist a best-by-metric checkpoint and refresh the resolved config.

    The full validation ``metrics`` dict is embedded in the checkpoint so
    consumers can later read whichever score triggered the save (typically
    ``val_min_ade`` or ``val_min_hd``) along with all the side metrics.

    Returns the absolute path of the saved checkpoint.
    """
    ckpt_path = os.path.join(out_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        },
        ckpt_path,
    )
    save_road_cpl_config_json(os.path.join(out_dir, RESOLVED_CONFIG_FILENAME), rcfg)
    logger.info("Saved new best checkpoint to %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# 4. Path-metric helpers
# ---------------------------------------------------------------------------


def _drivable_area_full_res(image_bchw: torch.Tensor, ev: EvalMetricsConfig) -> torch.Tensor:
    """Binarize the road channel at full resolution for off-road computation.

    Validation metrics now operate in full-res pixel coordinates, so the
    drivable map is the original ``(H, W)`` road channel binarized at
    ``ev.road_binarize_threshold`` -- no downsampling.

    Returns ``(B, H, W)`` float in ``{0, 1}``.
    """
    drivable_area_map = (image_bchw < 1).any(dim=1).to(torch.float32)
    return drivable_area_map


def _gt_points_pixels(gt_points: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
    """Slice a padded path to its valid prefix and convert ``(x, y) -> (y, x)``.

    Args:
        gt_points: ``(M_MAX, 2)`` zero-padded ``(x, y)`` points in pixels.
        num_points: scalar ``long`` valid count for this path.

    Returns:
        ``(N, 2)`` ``(y, x)`` pixel-coordinate tensor (``N <= M_MAX``).
    """
    n = int(num_points.item())
    if n <= 0:
        return torch.empty(0, 2, device=gt_points.device, dtype=torch.float32)
    pts = gt_points[:n]
    return torch.stack([pts[:, 1], pts[:, 0]], dim=1).to(torch.float32)


def _pred_points_from_outputs(
    heatmap_logits_2d: torch.Tensor,
    offsets_2d: torch.Tensor,
    *,
    D: int,
    min_prob: float,
) -> torch.Tensor:
    """Decode one sample's heatmap+offsets to ``(N, 2)`` ``(y, x)`` pixel points."""
    return decode_subpixel_points(heatmap_logits_2d, offsets_2d, D=D, min_prob=min_prob)


def _pred_points_from_cpl(
    chosen_indices: torch.Tensor,
    offsets_2d: torch.Tensor,
    *,
    D: int,
    grid_size: int,
) -> torch.Tensor:
    """Combine CPL chosen flat cell indices with the offset head into ``(y, x)`` pixels."""
    if chosen_indices.numel() == 0:
        return torch.empty(0, 2, device=chosen_indices.device, dtype=torch.float32)
    cell_i = chosen_indices // grid_size
    cell_j = chosen_indices % grid_size
    dx = offsets_2d[0, cell_i, cell_j]
    dy = offsets_2d[1, cell_i, cell_j]
    x = cell_j.to(dx.dtype) * float(D) + dx
    y = cell_i.to(dy.dtype) * float(D) + dy
    return torch.stack([y, x], dim=1).to(torch.float32)


def _decode_sequence_indices(
    model: GenerativePathForecaster,
    outputs: ModelOutputs,
    *,
    use_cpl: bool,
    use_ar: bool,
) -> dict[str, Any] | None:
    """Decode ordered cell indices for sequence heads (CPL or AR)."""
    if use_cpl:
        return model.cpl_greedy_decode(outputs)
    if use_ar:
        return model.ar_greedy_decode(outputs)
    return None


def _safe_metric_float(
    value: torch.Tensor | float,
    *,
    replace_inf: float = DEFAULT_METRIC_REPLACE_INF,
) -> float:
    """Coerce a metric to a finite Python ``float`` for aggregation/logging.

    Non-finite or implausibly large values are replaced by ``replace_inf``
    instead of poisoning downstream averages. Multi-element tensors are
    rejected explicitly to surface bugs early rather than crashing.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar tensor, got numel={value.numel()}")
        v = float(value.item())
    else:
        v = float(value)
    if not math.isfinite(v) or v > METRIC_OVERFLOW_THRESHOLD:
        return float(replace_inf)
    return v


def _mean_or(default: float, xs: list[float]) -> float:
    """Mean of ``xs``, or ``default`` if the list is empty."""
    return float(sum(xs) / len(xs)) if xs else default


# Keys aligned with ``_per_sample_path_metrics`` outputs (internal short names).
_PATH_METRIC_INTERNAL_KEYS: tuple[str, ...] = (
    "min_ade",
    "off_road",
    "min_hd",
    "min_hd_p90",
    "min_hd_precision_max_err",
    "min_hd_recall_max_err",
)


def _empty_path_metric_accum() -> dict[str, list[float]]:
    return {k: [] for k in _PATH_METRIC_INTERNAL_KEYS}


def _path_metric_val_key(internal: str) -> str:
    """Flat ``val_*`` name used for global metrics and Comet (off-road exception)."""
    if internal == "off_road":
        return "val_off_road_rate"
    return f"val_{internal}"


def _path_metric_sweep_suffix(internal: str) -> str:
    """Suffix after ``val_p{{tag}}_`` for threshold-sweep metrics."""
    if internal == "off_road":
        return "off_road_rate"
    return internal


def _prob_tag_for_keys(p: float) -> str:
    """Encode probability for metric keys, e.g. 0.3 -> ``0p30``."""
    return f"{float(p):.2f}".replace(".", "p")


def _gt_paths_from_eval_batch(
    gt_points_all_b: torch.Tensor,
    num_points_all_b: torch.Tensor,
    valid_paths_b: torch.Tensor,
) -> list[torch.Tensor]:
    """Extract a list of GT point tensors for one eval-mode sample.

    Args:
        gt_points_all_b: ``(K_MAX, M_MAX, 2)`` zero-padded ``(x, y)`` points.
        num_points_all_b: ``(K_MAX,) long`` valid count per path slot.
        valid_paths_b: ``(K_MAX,) bool`` validity flag per path slot.

    Returns:
        List of ``(N_k, 2)`` ``(y, x)`` pixel-coordinate tensors. Slots with
        zero valid points are skipped.
    """
    gts: list[torch.Tensor] = []
    for k in range(gt_points_all_b.shape[0]):
        if not bool(valid_paths_b[k].item()):
            continue
        pts = _gt_points_pixels(gt_points_all_b[k], num_points_all_b[k])
        if pts.shape[0] > 0:
            gts.append(pts)
    return gts


def _per_sample_path_metrics(
    pred_pts: torch.Tensor,
    gts: list[torch.Tensor],
    drivable: torch.Tensor,
) -> dict[str, float]:
    """Compute min-ADE, min-HD (+ directional errors), min-HD-p90, off-road for one sample."""
    pfm = PathForecastingMetrics
    min_hd, min_hd_prec, min_hd_rec = pfm.min_hd_components(pred_pts, gts)
    return {
        "min_ade": _safe_metric_float(pfm.min_ade(pred_pts, gts)),
        "off_road": _safe_metric_float(pfm.off_road_rate(pred_pts, drivable), replace_inf=1.0),
        "min_hd": _safe_metric_float(min_hd),
        "min_hd_p90": _safe_metric_float(pfm.min_hd_p90(pred_pts, gts)),
        "min_hd_precision_max_err": _safe_metric_float(min_hd_prec),
        "min_hd_recall_max_err": _safe_metric_float(min_hd_rec),
    }


def _pred_points_for_sample(
    outputs: ModelOutputs,
    *,
    use_cpl: bool,
    use_ar: bool,
    sample_idx: int,
    seq_decoded: dict[str, Any] | None,
    ev: EvalMetricsConfig,
    gen_cfg: GeneratorConfig,
    path_score_min_prob: float | None = None,
) -> torch.Tensor:
    """Get the predicted ``(N, 2)`` ``(y, x)`` pixel point set for one sample."""
    heatmap_logits = outputs["heatmap_logits"]
    offsets = outputs["offsets"]
    assert isinstance(heatmap_logits, torch.Tensor) and isinstance(offsets, torch.Tensor)
    if use_cpl or use_ar:
        if seq_decoded is None:
            raise RuntimeError("Sequence path requested but seq_decoded is None.")
        chosen = seq_decoded["indices"][sample_idx]
        return _pred_points_from_cpl(
            chosen,
            offsets[sample_idx],
            D=gen_cfg.D,
            grid_size=gen_cfg.label_H,
        )
    min_prob = float(ev.path_score_min_prob if path_score_min_prob is None else path_score_min_prob)
    return _pred_points_from_outputs(
        heatmap_logits[sample_idx],
        offsets[sample_idx],
        D=gen_cfg.D,
        min_prob=min_prob,
    )


def _accumulate_path_metrics_pass(
    model: GenerativePathForecaster,
    val_eval_loader: DataLoader,
    *,
    use_cpl: bool,
    use_ar: bool,
    device: torch.device,
    ev: EvalMetricsConfig,
    gen_cfg: GeneratorConfig,
    max_batches: int | None,
    path_score_min_prob: float | None,
    stratify_by_num_paths: bool,
) -> tuple[dict[str, list[float]], dict[int, dict[str, list[float]]], dict[int, int]]:
    """One forward pass over ``val_eval_loader``; optionally stratify by ``len(gts)``."""
    accum = _empty_path_metric_accum()
    accum_by_k: dict[int, dict[str, list[float]]] = {}
    counts_by_k: dict[int, int] = {}
    n_batches = 0
    with torch.no_grad():
        for batch in val_eval_loader:
            images = batch["image"].to(device)
            gt_points_all = batch["gt_points"].to(device)
            num_points_all = batch["num_points"].to(device)
            valid_paths = batch["valid_paths"].to(device)
            outputs = model(images)
            seq_decoded = _decode_sequence_indices(
                model,
                outputs,
                use_cpl=use_cpl,
                use_ar=use_ar,
            )
            drivable = _drivable_area_full_res(images, ev).float()

            for b in range(gt_points_all.shape[0]):
                gts = _gt_paths_from_eval_batch(gt_points_all[b], num_points_all[b], valid_paths[b])
                if not gts:
                    continue
                pred_pts = _pred_points_for_sample(
                    outputs,
                    use_cpl=use_cpl,
                    use_ar=use_ar,
                    sample_idx=b,
                    seq_decoded=seq_decoded,
                    ev=ev,
                    gen_cfg=gen_cfg,
                    path_score_min_prob=path_score_min_prob,
                )
                row = _per_sample_path_metrics(pred_pts, gts, drivable[b])
                for k in _PATH_METRIC_INTERNAL_KEYS:
                    accum[k].append(row[k])
                if stratify_by_num_paths:
                    n_paths = len(gts)
                    if n_paths not in accum_by_k:
                        accum_by_k[n_paths] = _empty_path_metric_accum()
                    counts_by_k[n_paths] = counts_by_k.get(n_paths, 0) + 1
                    for k in _PATH_METRIC_INTERNAL_KEYS:
                        accum_by_k[n_paths][k].append(row[k])

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

    return accum, accum_by_k, counts_by_k


def _finalize_path_metric_accumulators(
    accum: dict[str, list[float]],
    accum_by_k: dict[int, dict[str, list[float]]],
    counts_by_k: dict[int, int],
) -> dict[str, float]:
    """Global means, per-``n_paths`` means, and per-bucket sample counts."""
    out: dict[str, float] = {}
    for internal in _PATH_METRIC_INTERNAL_KEYS:
        out[_path_metric_val_key(internal)] = _mean_or(float("nan"), accum[internal])
    for k in sorted(accum_by_k.keys()):
        out[f"val_n_samples_npaths_{k}"] = float(counts_by_k[k])
        for internal in _PATH_METRIC_INTERNAL_KEYS:
            sk = f"{_path_metric_val_key(internal)}_npaths_{k}"
            out[sk] = _mean_or(float("nan"), accum_by_k[k][internal])
    return out


def compute_validation_path_metrics(
    model: GenerativePathForecaster,
    val_eval_loader: DataLoader,
    *,
    use_cpl: bool,
    use_ar: bool,
    device: torch.device,
    ev: EvalMetricsConfig,
    gen_cfg: GeneratorConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run the multi-GT validation metrics over the eval-mode loader.

    Predictions and GT are expressed in full-resolution ``(y, x)`` pixel
    coordinates so that the drivable-area map (the binarized full-res road
    channel) and the metric values share the same units.

    Global and per-``n_paths`` statistics use ``ev.path_score_min_prob``. When
    sequence heads are off and ``ev.path_score_min_prob_grid`` is set, additional passes
    log ``val_p{{tag}}_*`` headline metrics at each grid threshold (Option A).
    """
    model.eval()
    primary_prob = float(ev.path_score_min_prob)

    accum, accum_by_k, counts_by_k = _accumulate_path_metrics_pass(
        model,
        val_eval_loader,
        use_cpl=use_cpl,
        use_ar=use_ar,
        device=device,
        ev=ev,
        gen_cfg=gen_cfg,
        max_batches=max_batches,
        path_score_min_prob=None,
        stratify_by_num_paths=True,
    )
    out = _finalize_path_metric_accumulators(accum, accum_by_k, counts_by_k)

    grid = ()
    if not (use_cpl or use_ar) and ev.path_score_min_prob_grid:
        grid = tuple(sorted({float(p) for p in ev.path_score_min_prob_grid}))

    for p in grid:
        tag = _prob_tag_for_keys(p)
        sweep_prefix = f"val_p{tag}_"
        if math.isclose(p, primary_prob, rel_tol=0.0, abs_tol=1e-9):
            for internal in _PATH_METRIC_INTERNAL_KEYS:
                suf = _path_metric_sweep_suffix(internal)
                out[f"{sweep_prefix}{suf}"] = float(out[_path_metric_val_key(internal)])
            continue

        s_accum, _, _ = _accumulate_path_metrics_pass(
            model,
            val_eval_loader,
            use_cpl=use_cpl,
            use_ar=use_ar,
            device=device,
            ev=ev,
            gen_cfg=gen_cfg,
            max_batches=max_batches,
            path_score_min_prob=p,
            stratify_by_num_paths=False,
        )
        for internal in _PATH_METRIC_INTERNAL_KEYS:
            suf = _path_metric_sweep_suffix(internal)
            out[f"{sweep_prefix}{suf}"] = _mean_or(float("nan"), s_accum[internal])

    return out


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------


def _deterministic_chosen_path_index(valid_paths: torch.Tensor, seed: int) -> int:
    """Mirror the dataset's train-mode pick: uniform over valid K, seeded.

    Used by Hungarian visualization to reproduce a single GT per sample so the
    matched-pred overlay reflects what the train-time loss would see.
    """
    idx = (valid_paths > 0).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise RuntimeError("valid_paths has no true entries.")
    rng = np.random.default_rng(int(seed) % 2**32)
    j = int(rng.integers(0, int(idx.numel())))
    return int(idx[j].item())


def _pick_viz_indices(
    dataset_size: int,
    *,
    n_viz: int,
    base_seed: int | None,
    seed_offset: int,
) -> list[int]:
    """Return up to ``n_viz`` shuffled indices into the eval val dataset."""
    n = min(int(n_viz), int(dataset_size))
    if n <= 0:
        return []
    gen: torch.Generator | None = None
    if base_seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(base_seed) + int(seed_offset))
    perm = torch.randperm(dataset_size, generator=gen)
    return [int(perm[j].item()) for j in range(n)]


def _predict_viz_batch(
    model: GenerativePathForecaster,
    samples: list[dict[str, Any]],
    *,
    use_cpl: bool,
    use_ar: bool,
    device: torch.device,
) -> tuple[ModelOutputs, dict[str, Any] | None]:
    """Run the model on a small batch of viz samples and optional sequence decode."""
    images = torch.stack([s["image"] for s in samples], dim=0).to(device)
    with torch.no_grad():
        outputs = model(images)
        seq_dec = _decode_sequence_indices(
            model,
            outputs,
            use_cpl=use_cpl,
            use_ar=use_ar,
        )
    return outputs, seq_dec


def _chosen_gt_points_yx(sample: dict[str, Any], seed: int) -> tuple[torch.Tensor, int]:
    """Return ``((N, 2) (y, x), chosen_path_idx)`` for the seeded chosen GT path."""
    chosen = _deterministic_chosen_path_index(sample["valid_paths"], seed)
    gt_pts_all = sample["gt_points"]
    num_pts_all = sample["num_points"]
    if gt_pts_all.dim() == 3:
        gt_xy = gt_pts_all[chosen]
        n = num_pts_all[chosen]
    else:
        gt_xy = gt_pts_all
        n = num_pts_all
    n_long = n if isinstance(n, torch.Tensor) else torch.as_tensor(int(n))
    gt_yx = _gt_points_pixels(gt_xy.detach().cpu(), n_long.detach().cpu())
    return gt_yx, chosen


def _matching_overlay(
    sample: dict[str, Any],
    heatmap_logits_2d: torch.Tensor,
    offsets_2d: torch.Tensor,
    *,
    loss_fn: LossFn,
    seed: int,
    m_max: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build ``(gt_yx, matched_pred_mask_2d)`` for the supervised matching panel.

    Uses the configured matcher (Grid or Hungarian) to mark which predicted
    cells were paired with GT for the deterministic chosen path. Returns
    ``(None, None)`` when the loss function is not available.
    """
    if loss_fn is None:
        return None, None
    if "gt_points" not in sample:
        return None, None
    gt_yx, chosen = _chosen_gt_points_yx(sample, seed)
    n_int = int(gt_yx.shape[0])
    # Pad GT to (M_MAX, 2) (x, y) for the matcher's fixed-length API.
    padded = torch.zeros(m_max, 2, dtype=torch.float32)
    if n_int > 0:
        gt_pts_all = sample["gt_points"]
        gt_xy = gt_pts_all[chosen] if gt_pts_all.dim() == 3 else gt_pts_all
        padded[:n_int] = gt_xy[:n_int].detach().cpu().to(torch.float32)
    matched = loss_fn.matched_pred_mask_2d(
        heatmap_logits_2d.detach().cpu(),
        offsets_2d.detach().cpu(),
        padded,
        torch.tensor(n_int, dtype=torch.long),
    )
    return gt_yx, matched


def _viz_sample_seed(base_seed: int, seed_offset: int, sample_idx: int) -> int:
    """Stable per-sample seed combining a base, an epoch/run offset, and an index."""
    return base_seed * 1_000_000 + int(seed_offset) * 10_000 + int(sample_idx)


def _clone_and_preprocess_viz_samples(
    samples: list[dict[str, Any]],
    *,
    downsample_factor: int,
) -> list[dict[str, Any]]:
    """Clone viz samples and run the same GT sparsifier used by dataloaders."""
    processed: list[dict[str, Any]] = []
    for sample in samples:
        copied = dict(sample)
        for key in ("image", "gt_points", "num_points", "valid_paths"):
            val = copied.get(key)
            if isinstance(val, torch.Tensor):
                copied[key] = val.clone()

        image = copied.get("image")
        gt_points = copied.get("gt_points")
        num_points = copied.get("num_points")
        if not isinstance(image, torch.Tensor) or not isinstance(gt_points, torch.Tensor) or not isinstance(num_points, torch.Tensor):
            processed.append(copied)
            continue

        batched: BatchDict = {
            "image": image.unsqueeze(0),
            "gt_points": gt_points.unsqueeze(0),
            "num_points": num_points.unsqueeze(0),
        }
        valid_paths = copied.get("valid_paths")
        if isinstance(valid_paths, torch.Tensor):
            batched["valid_paths"] = valid_paths.unsqueeze(0)
        _preprocess_gt_points_in_batch(batched, downsample_factor=downsample_factor)

        copied["gt_points"] = batched["gt_points"][0]
        copied["num_points"] = batched["num_points"][0]
        if "valid_paths" in batched and isinstance(copied.get("valid_paths"), torch.Tensor):
            copied["valid_paths"] = batched["valid_paths"][0]
        processed.append(copied)
    return processed


def render_validation_visualizations(
    rcfg: RoadCplConfig,
    model: GenerativePathForecaster,
    val_viz_dataset: RoadCplDataset,
    *,
    use_cpl: bool,
    use_ar: bool,
    loss_fn: LossFn,
    device: torch.device,
    viz_subdir: str,
    seed_offset: int,
    comet_experiment: object | None = None,
    comet_step: int = 0,
) -> None:
    """Render ``viz.num_viz_val_samples`` previews and (optionally) log to Comet.

    Files are written to ``{run.resolved_save_dir}/viz/{viz_subdir}/sample_NN.png``.
    """
    vz = rcfg.viz
    indices = _pick_viz_indices(
        len(val_viz_dataset),
        n_viz=vz.num_viz_val_samples,
        base_seed=vz.viz_seed,
        seed_offset=seed_offset,
    )
    if not indices:
        return

    samples = _clone_and_preprocess_viz_samples(
        [val_viz_dataset[i] for i in indices],
        downsample_factor=int(rcfg.generator.D),
    )
    outputs, seq_dec = _predict_viz_batch(model, samples, use_cpl=use_cpl, use_ar=use_ar, device=device)
    heatmap_logits = outputs["heatmap_logits"]
    offsets = outputs["offsets"]
    assert isinstance(heatmap_logits, torch.Tensor) and isinstance(offsets, torch.Tensor)
    mh_tensors = _multi_hypothesis_outputs(outputs)

    viz_dir = os.path.join(rcfg.run.resolved_save_dir, VIZ_ROOT_SUBDIR, viz_subdir)
    os.makedirs(viz_dir, exist_ok=True)

    base_seed = int(vz.viz_seed) if vz.viz_seed is not None else 0
    for i, sample in enumerate(samples):
        seed = _viz_sample_seed(base_seed, seed_offset, i)

        cpl_order_i: torch.Tensor | None = None
        cpl_idx_i: torch.Tensor | None = None
        if (use_cpl or use_ar) and seq_dec is not None:
            cpl_order_i = seq_dec["order"][i].detach().cpu()
            cpl_idx_i = seq_dec["indices"][i].detach().cpu()

        # Matching overlay: same Grid/Hungarian behaviour as training (via loss_fn).
        match_gt, match_pred = _matching_overlay(
            sample,
            heatmap_logits[i],
            offsets[i],
            loss_fn=loss_fn,
            seed=seed,
            m_max=int(rcfg.generator.M_MAX),
        )

        out_png = os.path.join(viz_dir, f"sample_{i:02d}.png")
        render_training_val_sample(
            sample["image"],
            sample["gt_points"],
            sample["num_points"],
            sample["valid_paths"],
            heatmap_logits[i].detach().cpu(),
            offsets[i].detach().cpu(),
            stem=Path(sample["path"]).stem,
            out_path=out_png,
            D=int(rcfg.generator.D),
            min_prob=float(rcfg.eval.path_score_min_prob),
            learns_heatmap_score=rcfg.learns_heatmap_score,
            cpl_flat_indices=cpl_idx_i,
            cpl_order=cpl_order_i,
            match_gt_yx=match_gt,
            match_pred_mask=match_pred,
        )
        if comet_experiment is not None:
            comet_experiment.log_image(  # type: ignore[union-attr]
                out_png,
                name=f"val/{viz_subdir}/sample_{i:02d}",
                step=comet_step,
            )
        if mh_tensors is not None:
            _render_and_log_multi_hypothesis_sample(
                rcfg,
                sample,
                mh_tensors,
                sample_idx=i,
                viz_dir=viz_dir,
                viz_subdir=viz_subdir,
                comet_experiment=comet_experiment,
                comet_step=comet_step,
            )

    logger.info("Wrote %d val preview(s) to %s", len(samples), viz_dir)


def _multi_hypothesis_outputs(
    outputs: ModelOutputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return ``(heatmap_logits_all, offsets_all, hypothesis_probs, selected_hypothesis)``
    if all four tensors are present, else ``None``."""
    heatmap_logits_all = outputs.get("heatmap_logits_all")
    offsets_all = outputs.get("offsets_all")
    hypothesis_probs = outputs.get("hypothesis_probs")
    selected_hypothesis = outputs.get("selected_hypothesis")
    if (
        isinstance(heatmap_logits_all, torch.Tensor)
        and isinstance(offsets_all, torch.Tensor)
        and isinstance(hypothesis_probs, torch.Tensor)
        and isinstance(selected_hypothesis, torch.Tensor)
    ):
        return heatmap_logits_all, offsets_all, hypothesis_probs, selected_hypothesis
    return None


def _render_and_log_multi_hypothesis_sample(
    rcfg: RoadCplConfig,
    sample: dict[str, Any],
    mh_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    sample_idx: int,
    viz_dir: str,
    viz_subdir: str,
    comet_experiment: object | None,
    comet_step: int,
) -> None:
    """Save the per-sample multi-hypothesis debug image and (optionally) log it to Comet."""
    heatmap_logits_all, offsets_all, hypothesis_probs, selected_hypothesis = mh_tensors
    out_multi_png = os.path.join(viz_dir, f"sample_{sample_idx:02d}_multi_hyp.png")
    render_multi_hypothesis_sample(
        sample["image"],
        heatmap_logits_all[sample_idx].detach().cpu(),
        offsets_all[sample_idx].detach().cpu(),
        hypothesis_probs[sample_idx].detach().cpu(),
        selected_hypothesis[sample_idx].detach().cpu(),
        stem=Path(sample["path"]).stem,
        out_path=out_multi_png,
        D=int(rcfg.generator.D),
        min_prob=float(rcfg.eval.path_score_min_prob),
    )
    if comet_experiment is not None:
        comet_experiment.log_image(  # type: ignore[union-attr]
            out_multi_png,
            name=f"val/{viz_subdir}/sample_{sample_idx:02d}_multi_hyp",
            step=comet_step,
        )


# ---------------------------------------------------------------------------
# 6. Comet logging
# ---------------------------------------------------------------------------


def log_metrics_to_comet(
    comet_experiment: object | None,
    *,
    step: int,
    metrics: dict[str, float],
) -> None:
    """Log finite numeric metrics to Comet, dropping NaN/non-numeric entries."""
    if comet_experiment is None or not metrics:
        return
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if math.isnan(fv):
            continue
        clean[k] = fv
    if clean:
        comet_experiment.log_metrics(clean, step=step)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 7. Train / val step helpers
# ---------------------------------------------------------------------------


def _valid_point_mask(num_points: torch.Tensor, *, max_points: int) -> torch.Tensor:
    """Build ``(B, M_MAX)`` mask where ``True`` marks valid GT points."""
    idx = torch.arange(max_points, device=num_points.device).unsqueeze(0)
    return idx < num_points.to(torch.long).view(-1, 1)


def _apply_random_flip_augmentations(
    images: torch.Tensor,
    gt_points: torch.Tensor,
    num_points: torch.Tensor,
    *,
    flip_left_right_prob: float = TRAIN_FLIP_LEFT_RIGHT_PROB,
    flip_up_down_prob: float = TRAIN_FLIP_UP_DOWN_PROB,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply train-time random flips to images and GT points.

    ``gt_points`` are expected as ``(B, M_MAX, 2)`` in ``(x, y)`` pixels.
    Only valid points (as given by ``num_points``) are transformed.
    """
    if images.ndim != 4 or gt_points.ndim != 3:
        return images, gt_points
    b, _, h, w = images.shape
    if b == 0:
        return images, gt_points

    out_images = images.clone()
    out_gt_points = gt_points.clone()
    valid_mask = _valid_point_mask(num_points, max_points=gt_points.shape[1])

    flip_lr = torch.rand(b, device=images.device) < float(flip_left_right_prob)
    if bool(flip_lr.any().item()):
        out_images[flip_lr] = torch.flip(out_images[flip_lr], dims=[-1])
        x = out_gt_points[..., 0]
        x_flipped = float(w - 1) - x
        out_gt_points[..., 0] = torch.where(flip_lr[:, None] & valid_mask, x_flipped, x)

    flip_ud = torch.rand(b, device=images.device) < float(flip_up_down_prob)
    if bool(flip_ud.any().item()):
        out_images[flip_ud] = torch.flip(out_images[flip_ud], dims=[-2])
        y = out_gt_points[..., 1]
        y_flipped = float(h - 1) - y
        out_gt_points[..., 1] = torch.where(flip_ud[:, None] & valid_mask, y_flipped, y)

    return out_images, out_gt_points


def compute_step_loss(
    model: GenerativePathForecaster,
    batch: BatchDict,
    *,
    use_cpl: bool,
    use_ar: bool,
    loss_fn: LossFn,
    cpl_weight: float,
    ar_weight: float,
    gen_cfg: GeneratorConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run a forward pass and compute heatmap+offset(+sequence) losses.

    Returns:
        ``(total_loss, breakdown)`` where ``breakdown`` carries scalar floats
        for ordering terms (CPL/AR when enabled), BCE-with-logits heatmap term,
        and offset regression term so callers can log them separately.
    """
    images = batch["image"].to(device)
    gt_points = batch["gt_points"].to(device)
    num_points = batch["num_points"].to(device)
    outputs = model(images)
    terms = loss_fn(outputs, gt_points, num_points)
    if "total" not in terms:
        raise RuntimeError("Loss module must return a 'total' term.")
    total = terms["total"]
    # Forward every reported term except `total` so MH-only metrics
    # (selection, hypothesis_entropy, ...) flow through to the logger.
    breakdown: dict[str, float] = {k: float(v.detach().item()) for k, v in terms.items() if k != "total"}
    if use_cpl:
        cpl_term = model.cpl_loss(
            outputs,
            gt_points,
            num_points,
            meta=batch.get("meta", [{} for _ in range(images.shape[0])]),
            D=gen_cfg.D,
        )
        total = total + float(cpl_weight) * cpl_term
        breakdown["cpl"] = float(cpl_term.detach().item())
    if use_ar:
        ar_term = model.ar_loss(
            outputs,
            gt_points,
            num_points,
            meta=batch.get("meta", [{} for _ in range(images.shape[0])]),
            D=gen_cfg.D,
        )
        total = total + float(ar_weight) * ar_term
        breakdown["ar"] = float(ar_term.detach().item())
    return total, breakdown


def _accumulate_breakdown(accum: dict[str, float], terms: dict[str, float]) -> None:
    """Accumulate per-batch term floats into a running sum."""
    for k, v in terms.items():
        accum[k] = accum.get(k, 0.0) + float(v)


def train_one_epoch(
    model: GenerativePathForecaster,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    *,
    use_cpl: bool,
    use_ar: bool,
    loss_fn: LossFn,
    cpl_weight: float,
    ar_weight: float,
    gen_cfg: GeneratorConfig,
    device: torch.device,
    epoch_idx: int,
    num_epochs: int,
    comet_experiment: object | None,
) -> dict[str, float]:
    """Run one full training epoch; return mean total loss and term breakdown."""
    model.train()
    accum_total = 0.0
    accum_terms: dict[str, float] = {}
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx + 1}/{num_epochs} [Train]")
    for batch in pbar:
        optimizer.zero_grad()
        loss, terms = compute_step_loss(
            model,
            batch,
            use_cpl=use_cpl,
            use_ar=use_ar,
            loss_fn=loss_fn,
            cpl_weight=cpl_weight,
            ar_weight=ar_weight,
            gen_cfg=gen_cfg,
            device=device,
        )
        loss.backward()
        optimizer.step()
        accum_total += float(loss.item())
        _accumulate_breakdown(accum_terms, terms)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    n = max(len(loader), 1)
    out = {"total": accum_total / n}
    for k, v in accum_terms.items():
        out[k] = v / n
    return out


def run_validation_loss(
    model: GenerativePathForecaster,
    loader: DataLoader,
    *,
    use_cpl: bool,
    use_ar: bool,
    loss_fn: LossFn,
    cpl_weight: float,
    ar_weight: float,
    gen_cfg: GeneratorConfig,
    device: torch.device,
    desc: str = "[Val]",
) -> dict[str, float]:
    """Run a no-grad pass over ``loader``; return mean total + per-term losses."""
    model.eval()
    accum_total = 0.0
    accum_terms: dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            loss, terms = compute_step_loss(
                model,
                batch,
                use_cpl=use_cpl,
                use_ar=use_ar,
                loss_fn=loss_fn,
                cpl_weight=cpl_weight,
                ar_weight=ar_weight,
                gen_cfg=gen_cfg,
                device=device,
            )
            accum_total += float(loss.item())
            _accumulate_breakdown(accum_terms, terms)
    n = max(len(loader), 1)
    out = {"total": accum_total / n}
    for k, v in accum_terms.items():
        out[k] = v / n
    return out


# ---------------------------------------------------------------------------
# 8. High-level orchestration
# ---------------------------------------------------------------------------


def _prob_tag_to_display(tag: str) -> str:
    """Inverse of ``_prob_tag_for_keys`` for logs (``0p30`` -> ``0.30``)."""
    if "p" not in tag:
        return tag
    left, right = tag.split("p", 1)
    return f"{left}.{right}"


def _collect_sweep_tags(path_m: dict[str, float]) -> list[str]:
    """Distinct threshold tags from keys like ``val_p0p30_min_ade``."""
    suf = "_" + _path_metric_sweep_suffix("min_ade")
    tags: set[str] = set()
    for key in path_m:
        if key.startswith("val_p") and key.endswith(suf):
            tags.add(key[len("val_p") : -len(suf)])
    return sorted(tags)


def _collect_npaths_strata(path_m: dict[str, float]) -> list[tuple[str, list[int]]]:
    """Display rows for ``n_paths`` as ``1``, ``2``, ``3``, and ``>=4``."""
    ks = sorted({int(key.replace("val_n_samples_npaths_", "")) for key in path_m if key.startswith("val_n_samples_npaths_")})
    strata: list[tuple[str, list[int]]] = [(f"n_paths={k}", [k]) for k in (1, 2, 3) if k in ks]
    ks_ge_4 = [k for k in ks if k >= 4]
    if ks_ge_4:
        strata.append(("n_paths>=4", ks_ge_4))
    return strata


def _aggregate_npaths_bucket(path_m: dict[str, float], bucket_ks: list[int]) -> tuple[int, dict[str, float]]:
    """Aggregate per-``n_paths`` means using sample-count-weighted averages."""
    total_samples = sum(float(path_m.get(f"val_n_samples_npaths_{k}", 0.0)) for k in bucket_ks)
    bucket_metrics: dict[str, float] = {}
    for internal in _PATH_METRIC_INTERNAL_KEYS:
        numerator = 0.0
        denominator = 0.0
        metric_key_base = _path_metric_val_key(internal)
        for k in bucket_ks:
            sample_count = float(path_m.get(f"val_n_samples_npaths_{k}", 0.0))
            metric_key = f"{metric_key_base}_npaths_{k}"
            if sample_count <= 0.0 or metric_key not in path_m:
                continue
            numerator += float(path_m[metric_key]) * sample_count
            denominator += sample_count
        bucket_metrics[internal] = float(numerator / denominator) if denominator > 0.0 else float("nan")
    return round(total_samples), bucket_metrics


def _format_npaths_bucket_line(label: str, n_samples: int, bucket_metrics: dict[str, float]) -> str:
    """Render one ``n_paths`` row in the validation log block."""
    return (
        "    "
        + f"{label} (n_samples={n_samples}) | "
        + " | ".join(
            [
                f"min_ade={bucket_metrics['min_ade']:.4f}",
                f"min_hd={bucket_metrics['min_hd']:.4f}",
                f"min_hd_p90={bucket_metrics['min_hd_p90']:.4f}",
                f"hd_prec={bucket_metrics['min_hd_precision_max_err']:.4f}",
                f"hd_rec={bucket_metrics['min_hd_recall_max_err']:.4f}",
                f"off_road={bucket_metrics['off_road']:.2%}",
            ]
        )
    )


def _format_path_metrics_block(
    prefix: str,
    val_loss: float,
    path_m: dict[str, float],
    *,
    val_loss_breakdown: dict[str, float] | None = None,
) -> str:
    """Multi-line validation summary: global metrics, ambiguity strata, optional threshold sweep."""
    lines: list[str] = [prefix, f"  Val loss (mean batch): {val_loss:.6f}"]
    if val_loss_breakdown:
        vb = " | ".join(f"{k}={float(v):.4f}" for k, v in sorted(val_loss_breakdown.items()))
        lines.append(f"  Loss breakdown: {vb}")

    lines.append("  Global path metrics:")
    lines.append(
        "    "
        + " | ".join(
            [
                f"min_ade={path_m['val_min_ade']:.4f}",
                f"min_hd={path_m['val_min_hd']:.4f}",
                f"min_hd_p90={path_m['val_min_hd_p90']:.4f}",
                f"min_hd_precision_max_err={path_m['val_min_hd_precision_max_err']:.4f}",
                f"min_hd_recall_max_err={path_m['val_min_hd_recall_max_err']:.4f}",
                f"off_road_rate={path_m['val_off_road_rate']:.2%}",
            ]
        )
    )

    npaths_strata = _collect_npaths_strata(path_m)
    if npaths_strata:
        lines.append("  By number of valid GT paths (primary heatmap threshold):")
        for label, bucket_ks in npaths_strata:
            n_samples, bucket_metrics = _aggregate_npaths_bucket(path_m, bucket_ks)
            lines.append(_format_npaths_bucket_line(label, n_samples, bucket_metrics))

    sweep_tags = _collect_sweep_tags(path_m)
    if sweep_tags:
        lines.append("  Heatmap threshold sweep (global means):")
        for tag in sweep_tags:
            pdisp = _prob_tag_to_display(tag)
            sweep_parts: list[str] = []
            for internal in _PATH_METRIC_INTERNAL_KEYS:
                suf = _path_metric_sweep_suffix(internal)
                key = f"val_p{tag}_{suf}"
                val = path_m[key]
                if internal == "off_road":
                    sweep_parts.append(f"{suf}={val:.2%}")
                else:
                    sweep_parts.append(f"{suf}={val:.4f}")
            lines.append(f"    path_score_min_prob={pdisp} | " + " | ".join(sweep_parts))

    return "\n".join(lines)


def run_validation_pass(
    model: GenerativePathForecaster,
    *,
    val_loader: DataLoader,
    val_eval_loader: DataLoader,
    val_viz_dataset: RoadCplDataset,
    rcfg: RoadCplConfig,
    use_cpl: bool,
    use_ar: bool,
    loss_fn: LossFn,
    device: torch.device,
    summary_prefix: str,
    val_loss_desc: str,
    viz_subdir: str,
    seed_offset: int,
    comet_experiment: object | None,
    comet_step: int,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Single end-to-end validation pass: loss + path metrics + viz + Comet log.

    Returns the merged validation metrics dict (including ``val_loss``,
    ``val_min_ade``, ``val_min_hd``, all stratified path metrics, and any
    ``extra_metrics`` injected by the caller). Callers use this dict for
    best-checkpoint tracking.
    """
    val_terms = run_validation_loss(
        model,
        val_loader,
        use_cpl=use_cpl,
        use_ar=use_ar,
        loss_fn=loss_fn,
        cpl_weight=rcfg.run.cpl_weight,
        ar_weight=rcfg.run.ar_weight,
        gen_cfg=rcfg.generator,
        device=device,
        desc=val_loss_desc,
    )
    avg_val_loss = float(val_terms["total"])
    path_m = compute_validation_path_metrics(
        model,
        val_eval_loader,
        use_cpl=use_cpl,
        use_ar=use_ar,
        device=device,
        ev=rcfg.eval,
        gen_cfg=rcfg.generator,
        max_batches=rcfg.viz.val_path_metrics_max_batches,
    )
    loss_breakdown = {k: float(v) for k, v in val_terms.items() if k != "total"}
    logger.info(
        "%s",
        _format_path_metrics_block(
            summary_prefix,
            avg_val_loss,
            path_m,
            val_loss_breakdown=loss_breakdown if loss_breakdown else None,
        ),
    )

    metrics: dict[str, float] = {"val_loss": avg_val_loss, **path_m}
    for k, v in val_terms.items():
        if k != "total":
            metrics[f"val_loss_{k}"] = float(v)
    if extra_metrics:
        metrics.update({k: float(v) for k, v in extra_metrics.items()})
    log_metrics_to_comet(comet_experiment, step=comet_step, metrics=metrics)

    render_validation_visualizations(
        rcfg,
        model,
        val_viz_dataset,
        use_cpl=use_cpl,
        use_ar=use_ar,
        loss_fn=loss_fn,
        device=device,
        viz_subdir=viz_subdir,
        seed_offset=seed_offset,
        comet_experiment=comet_experiment,
        comet_step=comet_step,
    )
    return metrics


def _select_device() -> torch.device:
    """Pick CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_inference_timing(
    *,
    model: GenerativePathForecaster,
    val_eval_loader: DataLoader,
    rcfg: RoadCplConfig,
    use_cpl: bool,
    use_ar: bool,
    device: torch.device,
) -> None:
    """Dedicated inference-timing pass over ``val_eval_loader``.

    Runs a warmup loop (no hooks), then attaches encoder / heads / decode
    hooks and re-iterates the loader as a pure forward (+ greedy-decode) loop
    so per-example timings are not diluted by other passes (loss eval, viz).
    Writes ``inference_timing.json`` under ``rcfg.run.resolved_save_dir``.
    """
    stats = CudaTimerStats()
    warmup_inference(
        model,
        val_eval_loader,
        device,
        n_batches=int(rcfg.eval.inference_timing_warmup_batches),
        use_cpl=use_cpl,
        use_ar=use_ar,
    )
    handles = attach_inference_hooks(model, stats)
    try:
        run_inference_loop(
            model,
            val_eval_loader,
            device,
            use_cpl=use_cpl,
            use_ar=use_ar,
            desc="  timing",
        )
    finally:
        detach_forward_hooks(handles)
    report = aggregate_inference_timing(stats)
    print_inference_timing_report(report)
    timing_path = os.path.join(rcfg.run.resolved_save_dir, INFERENCE_TIMING_FILENAME)
    save_inference_timing_report(report, timing_path)
    logger.info("Wrote inference timing report to %s", timing_path)


def _run_test_only(
    *,
    model: GenerativePathForecaster,
    val_loader: DataLoader,
    val_eval_loader: DataLoader,
    val_viz_dataset: RoadCplDataset,
    rcfg: RoadCplConfig,
    use_cpl: bool,
    use_ar: bool,
    loss_fn: LossFn,
    device: torch.device,
    comet_experiment: object | None,
) -> None:
    """Single validation pass + viz; no training, no checkpointing.

    When ``rcfg.eval.log_inference_timing`` is true, an additional dedicated
    inference-timing pass runs after the validation pass (see
    :func:`_run_inference_timing`).
    """
    run_validation_pass(
        model,
        val_loader=val_loader,
        val_eval_loader=val_eval_loader,
        val_viz_dataset=val_viz_dataset,
        rcfg=rcfg,
        use_cpl=use_cpl,
        use_ar=use_ar,
        loss_fn=loss_fn,
        device=device,
        summary_prefix="Test-only",
        val_loss_desc="[Test / Val]",
        viz_subdir=TEST_VIZ_SUBDIR,
        seed_offset=0,
        comet_experiment=comet_experiment,
        comet_step=0,
    )
    if rcfg.eval.log_inference_timing:
        _run_inference_timing(
            model=model,
            val_eval_loader=val_eval_loader,
            rcfg=rcfg,
            use_cpl=use_cpl,
            use_ar=use_ar,
            device=device,
        )


def _maybe_save_best_for_metric(
    *,
    metric_name: str,
    metric_value: float,
    current_best: float,
    out_dir: str,
    filename: str,
    epoch: int,
    model: GenerativePathForecaster,
    optimizer: optim.Optimizer,
    metrics: dict[str, float],
    rcfg: RoadCplConfig,
) -> float:
    """Save a checkpoint if ``metric_value`` improves on ``current_best``.

    Returns the (possibly updated) best value. Non-finite metric values are
    ignored so a single corrupt validation pass cannot pin the tracker.
    """
    if not math.isfinite(metric_value):
        return current_best
    if metric_value >= current_best:
        return current_best
    logger.info(
        "Improved %s: %.6f -> %.6f (epoch %d)",
        metric_name,
        current_best,
        metric_value,
        epoch + 1,
    )
    save_best_checkpoint(
        out_dir,
        filename=filename,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        rcfg=rcfg,
    )
    return metric_value


def _run_training_loop(
    *,
    model: GenerativePathForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_eval_loader: DataLoader,
    val_viz_dataset: RoadCplDataset,
    optimizer: optim.Optimizer,
    rcfg: RoadCplConfig,
    use_cpl: bool,
    use_ar: bool,
    loss_fn: LossFn,
    device: torch.device,
    start_epoch: int,
    best_val_min_ade: float,
    best_val_min_hd: float,
    out_dir: str,
    comet_experiment: object | None,
) -> None:
    """Run epochs ``[start_epoch, num_epochs)``; save best-by-metric checkpoints.

    Two checkpoints are tracked independently: ``best_min_ade_model.pth`` is
    refreshed whenever ``val_min_ade`` reaches a new minimum, and
    ``best_min_hd_model.pth`` whenever ``val_min_hd`` does.
    """
    r = rcfg.run
    if start_epoch >= r.num_epochs:
        logger.warning(
            "Nothing to train: start_epoch (%d) >= num_epochs (%d).",
            start_epoch,
            r.num_epochs,
        )
        return

    for epoch in range(start_epoch, r.num_epochs):
        train_terms = train_one_epoch(
            model,
            train_loader,
            optimizer,
            use_cpl=use_cpl,
            use_ar=use_ar,
            loss_fn=loss_fn,
            cpl_weight=r.cpl_weight,
            ar_weight=r.ar_weight,
            gen_cfg=rcfg.generator,
            device=device,
            epoch_idx=epoch,
            num_epochs=r.num_epochs,
            comet_experiment=comet_experiment,
        )
        avg_train_loss = float(train_terms["total"])
        extra: dict[str, float] = {"train_loss": avg_train_loss}
        for k, v in train_terms.items():
            if k != "total":
                extra[f"train_loss_{k}"] = float(v)
        val_metrics = run_validation_pass(
            model,
            val_loader=val_loader,
            val_eval_loader=val_eval_loader,
            val_viz_dataset=val_viz_dataset,
            rcfg=rcfg,
            use_cpl=use_cpl,
            use_ar=use_ar,
            loss_fn=loss_fn,
            device=device,
            summary_prefix=f"Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f}",
            val_loss_desc=f"Epoch {epoch + 1}/{r.num_epochs} [Val]",
            viz_subdir=f"epoch_{epoch + 1:03d}",
            seed_offset=epoch,
            comet_experiment=comet_experiment,
            comet_step=epoch + 1,
            extra_metrics=extra,
        )

        val_min_ade = float(val_metrics.get("val_min_ade", float("nan")))
        val_min_hd = float(val_metrics.get("val_min_hd", float("nan")))

        best_val_min_ade = _maybe_save_best_for_metric(
            metric_name="val_min_ade",
            metric_value=val_min_ade,
            current_best=best_val_min_ade,
            out_dir=out_dir,
            filename=BEST_MIN_ADE_CHECKPOINT_FILENAME,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics=val_metrics,
            rcfg=rcfg,
        )
        best_val_min_hd = _maybe_save_best_for_metric(
            metric_name="val_min_hd",
            metric_value=val_min_hd,
            current_best=best_val_min_hd,
            out_dir=out_dir,
            filename=BEST_MIN_HD_CHECKPOINT_FILENAME,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics=val_metrics,
            rcfg=rcfg,
        )


def train_model(rcfg: RoadCplConfig, comet_experiment: object | None = None) -> None:
    """End-to-end training/eval entry point.

    Routes to either a single test-only validation pass (when
    ``run.test_only`` is set) or a full multi-epoch training loop. In both
    cases, the resolved configuration is dumped to disk for reproducibility.
    """
    r = rcfg.run
    device = _select_device()
    out_dir = r.resolved_save_dir
    os.makedirs(out_dir, exist_ok=True)
    save_road_cpl_config_json(os.path.join(out_dir, RESOLVED_CONFIG_FILENAME), rcfg)

    if r.test_only and not r.checkpoint_path:
        raise ValueError("--test-only requires --checkpoint to load model weights.")
    if r.test_only:
        logger.info("Test-only on device: %s (no training)", device)
    else:
        logger.info("Training on device: %s", device)
    logger.info("Checkpoints and outputs: %s", out_dir)

    val_loader = build_val_loader(
        r.data_root,
        batch_size=r.batch_size,
        data_cfg=rcfg.data,
        device=device,
        downsample_factor=int(rcfg.generator.D),
    )
    val_eval_loader, val_viz_dataset = build_eval_loader_and_dataset(
        r.data_root,
        batch_size=r.batch_size,
        data_cfg=rcfg.data,
        device=device,
        downsample_factor=int(rcfg.generator.D),
    )
    train_loader: DataLoader | None = None
    if not r.test_only:
        train_loader = build_train_loader(
            r.data_root,
            batch_size=r.batch_size,
            data_cfg=rcfg.data,
            device=device,
            use_flip_augmentation=r.use_flip_augmentation,
            downsample_factor=int(rcfg.generator.D),
        )

    use_cpl = rcfg.use_cpl
    use_ar = rcfg.use_ar
    use_multi_hypothesis = rcfg.use_multi_hypothesis
    model = build_model(
        rcfg.model,
        use_cpl=use_cpl,
        use_ar=use_ar,
        use_multi_hypothesis=use_multi_hypothesis,
        device=device,
    )
    loss_fn = build_loss_fn(rcfg)
    optimizer = build_optimizer(model, lr=r.learning_rate, cfg=rcfg.optimizer)

    ckpt_info = CheckpointInfo()
    if r.checkpoint_path is not None:
        ckpt_info = load_checkpoint(
            r.checkpoint_path,
            model,
            optimizer=None if r.test_only else optimizer,
            device=device,
            for_resume=not r.test_only,
        )
        if r.test_only and ckpt_info.raw_metrics is not None:
            logger.info("Checkpoint metrics (at save): %s", ckpt_info.raw_metrics)

    if r.test_only:
        _run_test_only(
            model=model,
            val_loader=val_loader,
            val_eval_loader=val_eval_loader,
            val_viz_dataset=val_viz_dataset,
            rcfg=rcfg,
            use_cpl=use_cpl,
            use_ar=use_ar,
            loss_fn=loss_fn,
            device=device,
            comet_experiment=comet_experiment,
        )
        return

    assert train_loader is not None
    _run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_eval_loader=val_eval_loader,
        val_viz_dataset=val_viz_dataset,
        optimizer=optimizer,
        rcfg=rcfg,
        use_cpl=use_cpl,
        use_ar=use_ar,
        loss_fn=loss_fn,
        device=device,
        start_epoch=ckpt_info.start_epoch,
        best_val_min_ade=ckpt_info.best_val_min_ade,
        best_val_min_hd=ckpt_info.best_val_min_hd,
        out_dir=out_dir,
        comet_experiment=comet_experiment,
    )


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ``python train.py``."""
    p = argparse.ArgumentParser(description="Train or evaluate the road path forecaster.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML file merged on top of defaults; CLI then overrides the merged config.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pth file to load. With training, resumes from the next epoch if the file contains epoch/optimizer.",
    )
    p.add_argument(
        "--test-only",
        action="store_true",
        help="Run a single val pass and viz; no training (requires --checkpoint).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for viz and checkpoints. Default: timestamped under train/...; in test-only, defaults to the checkpoint's directory.",
    )
    p.add_argument(
        "--name",
        type=str,
        default=argparse.SUPPRESS,
        help="Run folder name (train/<loss>/NAME) and Comet experiment name. No path separators. Overrides timestamp default.",
    )
    p.add_argument(
        "--matching",
        type=str,
        choices=["grid", "hungarian"],
        default=argparse.SUPPRESS,
        help="Matching strategy for heatmap (and offset) targets.",
    )
    p.add_argument(
        "--use-cpl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable CPL ordering loss in place of the heatmap classification term.",
    )
    p.add_argument(
        "--use-ar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable autoregressive next-token ordering loss in place of the heatmap classification term.",
    )
    p.add_argument(
        "--use-multi-hypothesis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the K-way multi-hypothesis baseline head (winner-take-all path supervision + selector CE).",
    )
    p.add_argument(
        "--num-hypotheses",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of parallel hypotheses when --use-multi-hypothesis is set.",
    )
    p.add_argument(
        "--hypothesis-ce-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight on the hypothesis-selection cross-entropy term (multi-hypothesis only).",
    )
    p.add_argument(
        "--offset-loss",
        type=str,
        choices=["l1", "smooth_l1", "mse", "none"],
        default=argparse.SUPPRESS,
        help="Offset regression criterion.",
    )
    p.add_argument(
        "--offset-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight on the offset regression term.",
    )
    p.add_argument(
        "--heatmap-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight on the heatmap BCE term (ignored by sequence heads under grid matching).",
    )
    p.add_argument(
        "--cpl-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight on the CPL ordering term (only used when --use-cpl is set).",
    )
    p.add_argument(
        "--ar-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight on the AR ordering term (only used when --use-ar is set).",
    )
    p.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Override run.batch_size")
    p.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS, help="Override run.learning_rate")
    p.add_argument("--num-epochs", type=int, default=argparse.SUPPRESS, help="Override run.num_epochs")
    p.add_argument("--data-root", type=str, default=argparse.SUPPRESS, help="Override run.data_root")
    p.add_argument(
        "--flip-augmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable train-time random left/right and up/down flip augmentation.",
    )
    p.add_argument(
        "--comet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Comet (default: from comet.enabled in the config).",
    )
    p.add_argument(
        "--lazy-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open NPZ samples on demand (data.lazy=True, default) vs. preload everything into RAM (--no-lazy-data).",
    )
    p.add_argument(
        "--profile-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Honoured only with --test-only: run a dedicated warmup pass and then time encoder / heads / decode during the validation pass; writes inference_timing.json.",
    )
    p.add_argument(
        "--profile-warmup-batches",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of warmup batches before the timed inference pass (default: eval.inference_timing_warmup_batches).",
    )
    return p.parse_args()


# Mapping of CLI attribute name -> ``run`` field name. Only attributes that are
# actually present on ``args`` (because the user passed the flag) are applied.
_RUN_CLI_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("matching", "matching"),
    ("offset_loss", "offset_loss"),
    ("offset_weight", "offset_weight"),
    ("heatmap_weight", "heatmap_weight"),
    ("cpl_weight", "cpl_weight"),
    ("ar_weight", "ar_weight"),
    ("batch_size", "batch_size"),
    ("learning_rate", "learning_rate"),
    ("num_epochs", "num_epochs"),
    ("data_root", "data_root"),
    ("flip_augmentation", "use_flip_augmentation"),
    ("name", "run_stem"),
)


def apply_cli_to_road_cpl_config(rcfg: RoadCplConfig, args: argparse.Namespace) -> None:
    """Override ``rcfg`` in-place with only the CLI flags the user provided."""
    for arg_attr, run_attr in _RUN_CLI_OVERRIDES:
        if hasattr(args, arg_attr) and getattr(args, arg_attr) is not None:
            setattr(rcfg.run, run_attr, getattr(args, arg_attr))
    if getattr(args, "use_cpl", None) is not None:
        rcfg.run.use_cpl = bool(args.use_cpl)
    if getattr(args, "use_ar", None) is not None:
        rcfg.run.use_ar = bool(args.use_ar)
    if getattr(args, "use_multi_hypothesis", None) is not None:
        rcfg.model.multi_hypothesis.enabled = bool(args.use_multi_hypothesis)
    if hasattr(args, "num_hypotheses"):
        rcfg.model.multi_hypothesis.num_hypotheses = int(args.num_hypotheses)
    if hasattr(args, "hypothesis_ce_weight"):
        rcfg.model.multi_hypothesis.ce_weight = float(args.hypothesis_ce_weight)
    if getattr(args, "comet", None) is not None:
        rcfg.comet.enabled = bool(args.comet)
    if getattr(args, "lazy_data", None) is not None:
        rcfg.data.lazy = bool(args.lazy_data)
    if getattr(args, "profile_inference", None) is not None:
        rcfg.eval.log_inference_timing = bool(args.profile_inference)
    if hasattr(args, "profile_warmup_batches"):
        rcfg.eval.inference_timing_warmup_batches = int(args.profile_warmup_batches)
    # Re-validate after CLI overrides: per-section invariants and the cross-section
    # head-mode mutual exclusion enforced by RoadCplConfig.__post_init__.
    rcfg.run.__post_init__()
    rcfg.model.multi_hypothesis.__post_init__()
    rcfg.eval.__post_init__()
    rcfg.__post_init__()


def prepare_run(rcfg: RoadCplConfig, args: argparse.Namespace) -> None:
    """Apply CLI overrides and resolve work directories on ``rcfg`` in-place.

    Order matters:

    1. CLI overrides on user-supplied fields (``--loss-type`` etc.).
    2. ``--checkpoint`` / ``--test-only`` / ``--out-dir`` are normalized so
       that :func:`resolve_run_work_dirs` sees the final state.
    3. ``resolve_run_work_dirs`` finalizes ``resolved_save_dir`` and
       ``comet_name``.
    """
    apply_cli_to_road_cpl_config(rcfg, args)
    r = rcfg.run
    if args.checkpoint is not None:
        r.checkpoint_path = os.path.abspath(args.checkpoint)
    if args.test_only:
        r.test_only = True
    if args.out_dir is not None:
        r.save_dir = os.path.abspath(args.out_dir)
    elif args.test_only and args.checkpoint is not None:
        r.save_dir = str(Path(args.checkpoint).resolve().parent)
    resolve_run_work_dirs(rcfg)


def _build_hparams(rcfg: RoadCplConfig) -> dict[str, Any]:
    """Flatten ``rcfg`` for Comet and add user/host context fields."""
    hparams: dict[str, Any] = dict(road_cpl_config_to_comet_params(rcfg))
    hparams["user"] = os.environ.get("USER", "default_user")
    hparams["machine"] = platform.node()
    hparams["resolved_save_dir"] = rcfg.run.resolved_save_dir
    return hparams


def build_comet_experiment(
    rcfg: RoadCplConfig,
    hparams: dict[str, Any],
) -> object | None:
    """Create the Comet experiment and log hparams, or return ``None`` if disabled."""
    if not rcfg.comet.enabled:
        logger.info("Comet logging disabled (comet.enabled=false)")
        return None
    experiment = Experiment(
        workspace=rcfg.comet.workspace,
        project_name=rcfg.comet.project_name,
    )
    experiment.set_name(rcfg.run.name)  # type: ignore[union-attr]
    experiment.log_parameters(hparams)  # type: ignore[union-attr]
    return experiment


def _configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once. No-op if handlers are already installed."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_root_config(config_path: str | None) -> RoadCplConfig:
    """Load defaults + optional YAML override into a :class:`RoadCplConfig`."""
    if config_path is None:
        return get_default_road_cpl_config()
    return load_road_cpl_config_from_path(Path(config_path))


def main() -> None:
    """CLI entry point: parse args, resolve config, build comet, and train."""
    _configure_logging()
    args = parse_args()
    rcfg = _load_root_config(args.config)
    prepare_run(rcfg, args)
    hparams = _build_hparams(rcfg)
    comet_experiment = build_comet_experiment(rcfg, hparams)
    train_model(rcfg, comet_experiment=comet_experiment)


if __name__ == "__main__":
    main()
