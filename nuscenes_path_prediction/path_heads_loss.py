"""Heatmap + offset losses for the path forecaster.

Two matching strategies build per-cell targets from a list of full-resolution
GT path points:

* ``GridMatcher``: each GT point is assigned to the cell it falls into
  (``cell = (floor(y/D), floor(x/D))``). Heatmap target is the binary
  occupancy mask; offset target at every GT cell is ``(x - j*D, y - i*D)``.
* ``HungarianMatcher``: bipartite matching between the model's predicted
  sub-pixel points (``cell_top_left + predicted_offset``) and the GT points,
  using a cost that combines geometric distance and ``-log_sigmoid(logit)``.

Both matchers expose:

* :meth:`heatmap_target` / :meth:`build_targets` -- per-sample target tensors
  used by :meth:`PathHeadsLoss.forward`.
* :meth:`assign_offset_targets` -- offset targets and the cell mask only,
  intended for the CPL path which replaces the heatmap classification term
  with the CPL ordering loss but still wants offset supervision.
* :meth:`matched_pred_mask` -- a ``(H, W) {0, 1}`` mask of cells matched to
  GT, for visualization.

Top-level :class:`PathHeadsLoss` orchestrates: it weights ``heatmap_weight *
BCEWithLogits + offset_weight * (L1 | SmoothL1 | MSE)`` and returns a dictionary of
named terms plus the total loss for ``backward()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gt_cells(gt_points_b: torch.Tensor, D: int, grid_size: int) -> torch.Tensor:
    """Map ``(M, 2)`` ``(x, y)`` pixel points to clamped ``(M, 2)`` ``(i, j)`` cell indices."""
    if gt_points_b.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=gt_points_b.device)
    cells = torch.floor(gt_points_b / float(D)).to(torch.long)
    j = cells[:, 0].clamp_(0, grid_size - 1)
    i = cells[:, 1].clamp_(0, grid_size - 1)
    return torch.stack([i, j], dim=1)


def _flatten_valid_points(gt_points_b: torch.Tensor, num_points_b: torch.Tensor) -> torch.Tensor:
    """Slice the ``(M_MAX, 2)`` padded array down to the valid prefix."""
    n = int(num_points_b.item())
    if n <= 0:
        return torch.empty(0, 2, device=gt_points_b.device, dtype=gt_points_b.dtype)
    return gt_points_b[:n].contiguous()


def _grid_centers_topleft(grid_size: int, D: int, device: torch.device) -> torch.Tensor:
    """Return ``(N, 2)`` ``(x_topleft, y_topleft)`` of every cell, in pixels."""
    j = torch.arange(grid_size, device=device)
    i = torch.arange(grid_size, device=device)
    iy, jx = torch.meshgrid(i, j, indexing="ij")
    coords = torch.stack([jx, iy], dim=-1).reshape(-1, 2).to(torch.float32) * float(D)
    return coords


# ---------------------------------------------------------------------------
# Target containers
# ---------------------------------------------------------------------------


@dataclass
class HeatmapOffsetTargets:
    """Per-batch dense targets used by both matchers and CPL composition.

    Attributes:
        heatmap: ``(B, H, W)`` float in ``{0, 1}``. ``1`` marks positive cells
            for ``BCEWithLogitsLoss``.
        offsets: ``(B, 2, H, W)`` float. Offset targets in pixels relative to
            each cell's top-left corner. Only the entries selected by ``mask``
            are valid; the rest are zero-filled and ignored downstream.
        mask: ``(B, H, W)`` bool. Cells that have an offset target.
        matched_pred_mask: ``(B, H, W)`` float in ``{0, 1}`` -- the cells the
            matcher considers positive (same as ``heatmap`` for ``GridMatcher``,
            possibly different for ``HungarianMatcher`` if the heatmap target
            is set differently than the offset mask). For viz/diagnostics.
    """

    heatmap: torch.Tensor
    offsets: torch.Tensor
    mask: torch.Tensor
    matched_pred_mask: torch.Tensor


# ---------------------------------------------------------------------------
# Grid matching
# ---------------------------------------------------------------------------


class GridMatcher(nn.Module):
    """Direct cell-occupancy assignment.

    Heatmap positives = the cells that any GT point falls into. Offset
    targets at those cells = ``(gt_x - j*D, gt_y - i*D)`` for the GT point
    that landed in the cell. If multiple GT points fall in the same cell
    (cannot happen given the generator's sparsifier, but kept defensive),
    only the last one wins.
    """

    def __init__(self, grid_size: int, D: int) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.D = int(D)

    @torch.no_grad()
    def build_targets(self, gt_points: torch.Tensor, num_points: torch.Tensor, *, device: torch.device | None = None) -> HeatmapOffsetTargets:
        """Build dense per-cell targets for a whole batch.

        Args:
            gt_points: ``(B, M_MAX, 2) float`` zero-padded GT points (pixels).
            num_points: ``(B,) long`` valid count per sample.
            device: target device for the produced tensors.
        """
        B = gt_points.shape[0]
        H = self.grid_size
        dev = device if device is not None else gt_points.device
        D = self.D
        heat = torch.zeros((B, H, H), dtype=torch.float32, device=dev)
        off = torch.zeros((B, 2, H, H), dtype=torch.float32, device=dev)
        msk = torch.zeros((B, H, H), dtype=torch.bool, device=dev)
        for b in range(B):
            pts = _flatten_valid_points(gt_points[b].to(dev), num_points[b].to(dev))
            if pts.numel() == 0:
                continue
            cells = _gt_cells(pts, D, H)
            i = cells[:, 0]
            j = cells[:, 1]
            heat[b, i, j] = 1.0
            msk[b, i, j] = True
            off[b, 0, i, j] = pts[:, 0] - j.to(pts.dtype) * float(D)
            off[b, 1, i, j] = pts[:, 1] - i.to(pts.dtype) * float(D)
        return HeatmapOffsetTargets(heatmap=heat, offsets=off, mask=msk, matched_pred_mask=heat.clone())

    @torch.no_grad()
    def assign_offset_targets(
        self,
        outputs: dict[str, torch.Tensor],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(offsets, mask, matched_pred_mask)`` -- offset-only path."""
        t = self.build_targets(gt_points, num_points, device=outputs["heatmap_logits"].device)
        return t.offsets, t.mask, t.matched_pred_mask

    @torch.no_grad()
    def matched_pred_mask(
        self,
        heatmap_logits_2d: torch.Tensor,
        offsets_2d: torch.Tensor,  # noqa: ARG002
        gt_points_b: torch.Tensor,
        num_points_b: torch.Tensor,
    ) -> torch.Tensor:
        """``(H, W) {0, 1}`` mask of cells GT lands in (independent of preds)."""
        H = self.grid_size
        out = torch.zeros((H, H), dtype=torch.float32, device=heatmap_logits_2d.device)
        pts = _flatten_valid_points(gt_points_b, num_points_b)
        if pts.numel() == 0:
            return out
        cells = _gt_cells(pts, self.D, H)
        out[cells[:, 0], cells[:, 1]] = 1.0
        return out


# ---------------------------------------------------------------------------
# Hungarian matching (predicted-point cost)
# ---------------------------------------------------------------------------


class HungarianMatcher(nn.Module):
    """Bipartite matching using predicted sub-pixel points.

    For each batch element, predicted points
    ``P[n] = (j*D + dx[i, j], i*D + dy[i, j])`` (one per cell) are matched
    against the valid GT points using

    ``cost(n, m) = dist_weight * ||P[n] - gt[m]|| + prob_weight * (-log_sigmoid(logit[n]))``

    via :func:`scipy.optimize.linear_sum_assignment`. The matched cells become
    heatmap positives; their offset target equals ``(gt_x - j*D, gt_y - i*D)``
    using each matched cell's own ``(i, j)``.
    """

    def __init__(
        self,
        grid_size: int,
        D: int,
        *,
        dist_weight: float = 1.0,
        prob_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.D = int(D)
        self.dist_weight = float(dist_weight)
        self.prob_weight = float(prob_weight)

    @staticmethod
    def _predicted_points_flat(offsets_2d: torch.Tensor, grid_size: int, D: int) -> torch.Tensor:
        """``(2, H, W)`` offsets -> ``(N, 2)`` predicted ``(x, y)`` points in pixels."""
        device = offsets_2d.device
        topleft = _grid_centers_topleft(grid_size, D, device)  # (N, 2) (x, y)
        dx = offsets_2d[0].reshape(-1)  # (N,)
        dy = offsets_2d[1].reshape(-1)  # (N,)
        return topleft + torch.stack([dx, dy], dim=1)

    @torch.no_grad()
    def _match_one(
        self,
        heatmap_logits_2d: torch.Tensor,
        offsets_2d: torch.Tensor,
        gt_pts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(matched_pred_idx_long, matched_gt_idx_long)`` for one sample."""
        H = self.grid_size
        device = heatmap_logits_2d.device
        if gt_pts.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty
        pred_pts = self._predicted_points_flat(offsets_2d, H, self.D)  # (N, 2)
        dist = torch.cdist(pred_pts, gt_pts.to(pred_pts.dtype), p=2.0)  # (N, M)
        prob_cost = -F.logsigmoid(heatmap_logits_2d.reshape(-1)).unsqueeze(1)  # (N, 1)
        cost = self.dist_weight * dist + self.prob_weight * prob_cost  # (N, M)
        cost_np = cost.detach().cpu().numpy()
        pi, gi = linear_sum_assignment(cost_np)
        return (
            torch.as_tensor(pi, dtype=torch.long, device=device),
            torch.as_tensor(gi, dtype=torch.long, device=device),
        )

    @torch.no_grad()
    def build_targets(
        self,
        outputs: dict[str, torch.Tensor],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
    ) -> HeatmapOffsetTargets:
        """Build dense per-cell targets using Hungarian matching."""
        H = self.grid_size
        D = self.D
        logits = outputs["heatmap_logits"]
        offsets = outputs["offsets"]
        device = logits.device
        B = logits.shape[0]
        heat = torch.zeros((B, H, H), dtype=torch.float32, device=device)
        off = torch.zeros((B, 2, H, H), dtype=torch.float32, device=device)
        msk = torch.zeros((B, H, H), dtype=torch.bool, device=device)
        for b in range(B):
            pts = _flatten_valid_points(gt_points[b].to(device), num_points[b].to(device))
            if pts.numel() == 0:
                continue
            pi, gi = self._match_one(logits[b], offsets[b], pts)
            if pi.numel() == 0:
                continue
            cell_i = pi // H
            cell_j = pi % H
            heat[b, cell_i, cell_j] = 1.0
            msk[b, cell_i, cell_j] = True
            matched_gt = pts[gi]
            off[b, 0, cell_i, cell_j] = matched_gt[:, 0] - cell_j.to(matched_gt.dtype) * float(D)
            off[b, 1, cell_i, cell_j] = matched_gt[:, 1] - cell_i.to(matched_gt.dtype) * float(D)
        return HeatmapOffsetTargets(heatmap=heat, offsets=off, mask=msk, matched_pred_mask=heat.clone())

    @torch.no_grad()
    def assign_offset_targets(
        self,
        outputs: dict[str, torch.Tensor],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(offsets, mask, matched_pred_mask)`` -- offset-only path."""
        t = self.build_targets(outputs, gt_points, num_points)
        return t.offsets, t.mask, t.matched_pred_mask

    @torch.no_grad()
    def matched_pred_mask(
        self,
        heatmap_logits_2d: torch.Tensor,
        offsets_2d: torch.Tensor,
        gt_points_b: torch.Tensor,
        num_points_b: torch.Tensor,
    ) -> torch.Tensor:
        """``(H, W) {0, 1}`` mask of cells matched to GT for one sample."""
        H = self.grid_size
        out = torch.zeros((H, H), dtype=torch.float32, device=heatmap_logits_2d.device)
        pts = _flatten_valid_points(gt_points_b, num_points_b)
        if pts.numel() == 0:
            return out
        pi, _ = self._match_one(heatmap_logits_2d, offsets_2d, pts)
        if pi.numel() == 0:
            return out
        out[pi // H, pi % H] = 1.0
        return out


# ---------------------------------------------------------------------------
# Top-level loss orchestrator
# ---------------------------------------------------------------------------


def _make_offset_loss(name: str) -> nn.Module | None:
    """Build the per-cell offset regression criterion (or ``None`` to disable)."""
    n = name.lower()
    if n == "none":
        return None
    if n == "l1":
        return nn.L1Loss(reduction="none")
    if n == "smooth_l1":
        return nn.SmoothL1Loss(reduction="none")
    if n == "mse":
        return nn.MSELoss(reduction="none")
    raise ValueError(f"offset_loss must be 'l1', 'smooth_l1', 'mse', or 'none'; got {name!r}")


class PathHeadsLoss(nn.Module):
    """Heatmap (BCEWithLogits) + offset (L1/SmoothL1/MSE) loss orchestrator.

    Use ``classification=True`` (default) to include the heatmap term; CPL
    sets ``classification=False`` to plug in CPL's ordering loss in its place
    while still using this module to compute the offset L1 term.
    """

    def __init__(
        self,
        *,
        matcher: GridMatcher | HungarianMatcher,
        heatmap_weight: float = 1.0,
        offset_weight: float = 1.0,
        offset_loss: Literal["l1", "smooth_l1", "mse", "none"] = "l1",
        matched_bce_weight: float = 1.0,
        unmatched_bce_weight: float = 1.0,
        classification: bool = True,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.heatmap_weight = float(heatmap_weight)
        self.offset_weight = float(offset_weight)
        self.matched_bce_weight = float(matched_bce_weight)
        self.unmatched_bce_weight = float(unmatched_bce_weight)
        self.classification = bool(classification)
        self.offset_criterion = _make_offset_loss(offset_loss)

    def _heatmap_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-cell weighted ``BCEWithLogits`` reduced to a scalar."""
        weight = target * self.matched_bce_weight + (1.0 - target) * self.unmatched_bce_weight
        return F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="mean")

    def _offset_loss(
        self,
        offsets: torch.Tensor,
        offset_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """L1/SmoothL1 over masked cells (mean per (cell, channel))."""
        if self.offset_criterion is None:
            return offsets.new_zeros(())
        per = self.offset_criterion(offsets, offset_target)  # (B, 2, H, W)
        m = mask.unsqueeze(1).expand_as(per)  # (B, 2, H, W) bool
        denom = m.sum().clamp_min(1).to(per.dtype)
        return (per * m.to(per.dtype)).sum() / denom

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute heatmap and/or offset losses; return ``{heatmap, offset, total}``.

        Args:
            outputs: model output dict with at least ``heatmap_logits``
                ``(B, H, W)`` and ``offsets`` ``(B, 2, H, W)``.
            gt_points: ``(B, M_MAX, 2)`` zero-padded ``(x, y)`` GT points.
            num_points: ``(B,) long`` valid point count per sample.
        """
        if isinstance(self.matcher, HungarianMatcher):
            tgt = self.matcher.build_targets(outputs, gt_points, num_points)
        else:
            tgt = self.matcher.build_targets(gt_points, num_points, device=outputs["heatmap_logits"].device)

        zero = outputs["heatmap_logits"].new_zeros(())
        heatmap_term = self._heatmap_loss(outputs["heatmap_logits"], tgt.heatmap) if self.classification else zero
        offset_term = self._offset_loss(outputs["offsets"], tgt.offsets, tgt.mask)
        total = self.heatmap_weight * heatmap_term + self.offset_weight * offset_term
        return {"heatmap": heatmap_term, "offset": offset_term, "total": total}

    @torch.no_grad()
    def matched_pred_mask_2d(
        self,
        heatmap_logits_2d: torch.Tensor,
        offsets_2d: torch.Tensor,
        gt_points_b: torch.Tensor,
        num_points_b: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper for visualization / debugging."""
        return self.matcher.matched_pred_mask(heatmap_logits_2d, offsets_2d, gt_points_b, num_points_b)


class MultiHypothesisPathLoss(nn.Module):
    """Winner-take-all path loss for the baseline K-hypothesis head.

    Inputs are expected to include:

    * ``heatmap_logits_all``: ``(B, K, H, W)``
    * ``offsets_all``: ``(B, K, 2, H, W)`` (or ``(B, K, H, W, 2)``)
    * ``hypothesis_logits``: ``(B, K)``

    For each sample we evaluate the per-hypothesis path loss, pick
    ``k* = argmin_k path_loss_k``, supervise only ``k*`` for path terms, and
    train the selector with ``CE(hypothesis_logits, k*)``.
    """

    def __init__(
        self,
        *,
        matcher: GridMatcher | HungarianMatcher,
        heatmap_weight: float = 1.0,
        offset_weight: float = 1.0,
        offset_loss: Literal["l1", "smooth_l1", "mse", "none"] = "l1",
        matched_bce_weight: float = 1.0,
        unmatched_bce_weight: float = 1.0,
        selection_weight: float = 1.0,
        classification: bool = True,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.heatmap_weight = float(heatmap_weight)
        self.offset_weight = float(offset_weight)
        self.matched_bce_weight = float(matched_bce_weight)
        self.unmatched_bce_weight = float(unmatched_bce_weight)
        self.selection_weight = float(selection_weight)
        self.classification = bool(classification)
        self.offset_criterion = _make_offset_loss(offset_loss)

    def forward(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits_all, offsets_all, hypothesis_logits = self._extract_mh_tensors(outputs)
        bsz, num_hyp = self._validate_shapes(logits_all, offsets_all, hypothesis_logits)
        offsets_all = self._normalize_offsets_shape(offsets_all, num_hyp)

        heat_per_hyp, off_per_hyp, path_per_hyp = self._per_hypothesis_path_losses(logits_all, offsets_all, gt_points, num_points, bsz, num_hyp)
        winner_idx = torch.argmin(path_per_hyp, dim=1)  # (B,)
        gather_idx = winner_idx.unsqueeze(1)
        winner_heat = torch.gather(heat_per_hyp, dim=1, index=gather_idx).squeeze(1)
        winner_off = torch.gather(off_per_hyp, dim=1, index=gather_idx).squeeze(1)

        heatmap_term = winner_heat.mean()
        offset_term = winner_off.mean()
        path_term = self.heatmap_weight * heatmap_term + self.offset_weight * offset_term
        selection_term = F.cross_entropy(hypothesis_logits, winner_idx)

        probs = torch.softmax(hypothesis_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        max_prob = probs.max(dim=1).values.mean()
        selected_mean = winner_idx.to(dtype=probs.dtype).mean()

        total = path_term + self.selection_weight * selection_term
        return {
            "heatmap": heatmap_term,
            "offset": offset_term,
            "selection": selection_term,
            "selected_hypothesis_mean": selected_mean,
            "hypothesis_entropy": entropy,
            "hypothesis_max_prob": max_prob,
            "total": total,
        }

    @torch.no_grad()
    def matched_pred_mask_2d(
        self,
        heatmap_logits_2d: torch.Tensor,
        offsets_2d: torch.Tensor,
        gt_points_b: torch.Tensor,
        num_points_b: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper for visualization / debugging."""
        return self.matcher.matched_pred_mask(heatmap_logits_2d, offsets_2d, gt_points_b, num_points_b)

    @staticmethod
    def _extract_mh_tensors(
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_all = outputs.get("heatmap_logits_all")
        offsets_all = outputs.get("offsets_all")
        hypothesis_logits = outputs.get("hypothesis_logits")
        if not isinstance(logits_all, torch.Tensor):
            raise RuntimeError("MultiHypothesisPathLoss requires outputs['heatmap_logits_all'] tensor.")
        if not isinstance(offsets_all, torch.Tensor):
            raise RuntimeError("MultiHypothesisPathLoss requires outputs['offsets_all'] tensor.")
        if not isinstance(hypothesis_logits, torch.Tensor):
            raise RuntimeError("MultiHypothesisPathLoss requires outputs['hypothesis_logits'] tensor.")
        return logits_all, offsets_all, hypothesis_logits

    @staticmethod
    def _validate_shapes(
        logits_all: torch.Tensor,
        offsets_all: torch.Tensor,
        hypothesis_logits: torch.Tensor,
    ) -> tuple[int, int]:
        if logits_all.ndim != 4:
            raise ValueError(f"heatmap_logits_all must be (B, K, H, W), got {tuple(logits_all.shape)}")
        if hypothesis_logits.ndim != 2:
            raise ValueError(f"hypothesis_logits must be (B, K), got {tuple(hypothesis_logits.shape)}")
        bsz, num_hyp = int(logits_all.shape[0]), int(logits_all.shape[1])
        if hypothesis_logits.shape[0] != bsz or hypothesis_logits.shape[1] != num_hyp:
            raise ValueError(f"hypothesis_logits shape must match heatmap_logits_all batch and K dims, got {tuple(hypothesis_logits.shape)} vs {(bsz, num_hyp)}")
        if offsets_all.shape[0] != bsz:
            raise ValueError(f"offsets_all batch dim mismatch: expected {bsz}, got {offsets_all.shape[0]}")
        return bsz, num_hyp

    @staticmethod
    def _normalize_offsets_shape(offsets_all: torch.Tensor, k: int) -> torch.Tensor:
        # Accept both (B, K, 2, H, W) and (B, K, H, W, 2) for convenience.
        if offsets_all.ndim != 5:
            raise ValueError(f"offsets_all must be 5D, got shape {tuple(offsets_all.shape)}")
        if offsets_all.shape[1] != k:
            raise ValueError(f"offsets_all K dim mismatch: expected {k}, got {offsets_all.shape[1]}")
        if offsets_all.shape[2] == 2:
            return offsets_all
        if offsets_all.shape[-1] == 2:
            return offsets_all.permute(0, 1, 4, 2, 3).contiguous()
        raise ValueError(f"offsets_all must have a 2-channel offset axis, got shape {tuple(offsets_all.shape)}")

    def _per_hypothesis_path_losses(
        self,
        logits_all: torch.Tensor,
        offsets_all: torch.Tensor,
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
        bsz: int,
        num_hyp: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ``(B, K)`` heatmap, offset, and combined path losses."""
        heatmap_losses: list[torch.Tensor] = []
        offset_losses: list[torch.Tensor] = []
        path_losses: list[torch.Tensor] = []
        for k in range(num_hyp):
            logits_k = logits_all[:, k]  # (B, H, W)
            offsets_k = offsets_all[:, k]  # (B, 2, H, W)
            tgt = self._build_targets_for_hypothesis(logits_k, offsets_k, gt_points, num_points)
            heat_k = self._heatmap_loss_per_sample(logits_k, tgt.heatmap) if self.classification else logits_k.new_zeros((bsz,))
            off_k = self._offset_loss_per_sample(offsets_k, tgt.offsets, tgt.mask)
            heatmap_losses.append(heat_k)
            offset_losses.append(off_k)
            path_losses.append(self.heatmap_weight * heat_k + self.offset_weight * off_k)
        heat_per_hyp = torch.stack(heatmap_losses, dim=1)
        off_per_hyp = torch.stack(offset_losses, dim=1)
        path_per_hyp = torch.stack(path_losses, dim=1)
        return heat_per_hyp, off_per_hyp, path_per_hyp

    def _build_targets_for_hypothesis(
        self,
        heatmap_logits: torch.Tensor,
        offsets: torch.Tensor,
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
    ) -> HeatmapOffsetTargets:
        if isinstance(self.matcher, HungarianMatcher):
            return self.matcher.build_targets(
                {"heatmap_logits": heatmap_logits, "offsets": offsets},
                gt_points,
                num_points,
            )
        return self.matcher.build_targets(gt_points, num_points, device=heatmap_logits.device)

    def _heatmap_loss_per_sample(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = target * self.matched_bce_weight + (1.0 - target) * self.unmatched_bce_weight
        per_cell = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="none")
        return per_cell.mean(dim=(1, 2))

    def _offset_loss_per_sample(
        self,
        offsets: torch.Tensor,
        offset_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.offset_criterion is None:
            return offsets.new_zeros((offsets.shape[0],))
        per = self.offset_criterion(offsets, offset_target)  # (B, 2, H, W)
        m = mask.unsqueeze(1).expand_as(per)
        numer = (per * m.to(per.dtype)).sum(dim=(1, 2, 3))
        denom = m.sum(dim=(1, 2, 3)).clamp_min(1).to(per.dtype)
        return numer / denom


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def decode_subpixel_points(
    heatmap_logits_2d: torch.Tensor,
    offsets_2d: torch.Tensor,
    *,
    D: int,
    min_prob: float,
) -> torch.Tensor:
    """Convert one sample's heatmap+offset output into a ``(N, 2)`` ``(y, x)`` point set.

    Active cells are ``sigmoid(logit) > min_prob``; if none clear the threshold,
    an empty tensor is returned. The returned points are full-resolution pixel
    coordinates ``(y, x)`` -- ordering compatible with the rest of the metric
    code that uses ``(y, x)``.
    """
    H = heatmap_logits_2d.shape[0]
    device = heatmap_logits_2d.device
    probs = torch.sigmoid(heatmap_logits_2d)
    above = probs > float(min_prob)
    if not above.any():
        return torch.empty(0, 2, device=device, dtype=torch.float32)
    idx_flat = above.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    cell_i = idx_flat // H
    cell_j = idx_flat % H
    dx = offsets_2d[0, cell_i, cell_j]
    dy = offsets_2d[1, cell_i, cell_j]
    x = cell_j.to(dx.dtype) * float(D) + dx
    y = cell_i.to(dy.dtype) * float(D) + dy
    return torch.stack([y, x], dim=1).to(torch.float32)


__all__ = [
    "GridMatcher",
    "HeatmapOffsetTargets",
    "HungarianMatcher",
    "MultiHypothesisPathLoss",
    "PathHeadsLoss",
    "decode_subpixel_points",
]
