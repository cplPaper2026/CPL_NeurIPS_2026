"""Random ground-truth sampling for cluster-selection bags.

Each bag has ``K`` clusters. The ground truth is exactly one token per
cluster chosen uniformly at random from that cluster's valid members.

Returning *indices* (rather than a boolean mask) keeps the Hungarian
matcher, the consistency loss, and the visualisation decoupled from
``N``: the mask is a derived artefact for callers that need one
(``cpl_loss``, ``bce_loss``).
"""

from __future__ import annotations

import torch


def sample_random_gt_indices(
    cluster_labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return one random index per cluster id for a single bag.

    Shapes:
        ``cluster_labels`` : ``(N,)`` long, negative values treated as pad.
        ``valid_mask``     : ``(N,)`` bool (optional; True = real token).

    Returns a ``(K,)`` LongTensor with one valid index per distinct
    non-negative cluster id, sorted by cluster id ascending. Padding /
    invalid entries are never picked; if a cluster has no valid members
    it is skipped (so the output length is <= the number of unique ids).
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(cluster_labels, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(cluster_labels.device)

    unique = cluster_labels.unique(sorted=True)
    picks: list[int] = []
    for cid in unique.tolist():
        if cid < 0:
            continue
        members = ((cluster_labels == cid) & valid_mask).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        r = torch.randint(0, members.numel(), (1,), generator=generator, device=members.device).item()
        picks.append(int(members[r].item()))
    return torch.tensor(picks, dtype=torch.long, device=cluster_labels.device)


def indices_to_mask(indices: torch.Tensor, n: int) -> torch.Tensor:
    """Scatter a ``(K,)`` index vector to a ``(N,)`` boolean mask.

    Negative indices (padding slots) are ignored. Useful for turning the
    Hungarian / random-GT representation into the boolean mask consumed
    by :func:`losses.cpl_loss` and :func:`losses.bce_loss`.
    """
    mask = torch.zeros(n, dtype=torch.bool, device=indices.device)
    if indices.numel() == 0:
        return mask
    valid = indices >= 0
    if valid.any():
        mask[indices[valid]] = True
    return mask


def batched_indices_to_mask(gt_indices_padded: torch.Tensor, n: int) -> torch.Tensor:
    """Batched version of :func:`indices_to_mask`.

    ``gt_indices_padded``: ``(B, K_max)`` with ``-1`` in pad slots.
    Returns ``(B, N)`` bool on the same device.
    """
    B = gt_indices_padded.shape[0]
    out = torch.zeros(B, n, dtype=torch.bool, device=gt_indices_padded.device)
    for b in range(B):
        row = gt_indices_padded[b]
        valid = row >= 0
        if valid.any():
            out[b, row[valid]] = True
    return out


def batched_sample_random_gt(
    cluster_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply :func:`sample_random_gt_indices` row-wise.

    Shapes:
        ``cluster_labels`` : ``(B, N)`` long.
        ``valid_mask``     : ``(B, N)`` bool.

    Returns ``(gt_indices_padded, k_batch)`` where

    * ``gt_indices_padded`` : ``(B, K_max)`` LongTensor, ``-1`` in pad slots.
    * ``k_batch``           : ``(B,)`` LongTensor of per-row cluster counts.
    """
    B = cluster_labels.shape[0]
    picks_per_row: list[torch.Tensor] = []
    k_per_row: list[int] = []
    for b in range(B):
        idxs = sample_random_gt_indices(cluster_labels[b], valid_mask[b], generator=generator)
        picks_per_row.append(idxs)
        k_per_row.append(int(idxs.numel()))
    k_max = max(k_per_row) if k_per_row else 0
    out = torch.full((B, max(k_max, 1)), -1, dtype=torch.long, device=cluster_labels.device)
    for b, idxs in enumerate(picks_per_row):
        if idxs.numel() > 0:
            out[b, : idxs.numel()] = idxs
    k_batch = torch.tensor(k_per_row, dtype=torch.long, device=cluster_labels.device)
    return out, k_batch


def random_gt_permutations(
    gt_indices_padded: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one random permutation of the valid GT entries per row.

    ``gt_indices_padded`` is ``(B, K_max)`` with token indices in
    ``[0, N)`` for valid entries and ``-1`` in pad slots. For each row we
    pick a uniformly-random permutation of its valid entries and place
    them in ``perm_padded[b, :K_b]``; pad slots remain ``-1``.

    Returns ``(perm_padded, perm_valid_mask)`` both shaped ``(B, K_max)``.
    ``perm_valid_mask[b, t] = True`` iff position ``t`` holds a real GT
    index (i.e. ``t < K_b``).
    """
    B, K_max = gt_indices_padded.shape
    device = gt_indices_padded.device
    perm_padded = torch.full((B, K_max), -1, dtype=gt_indices_padded.dtype, device=device)
    perm_valid_mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)
    if K_max == 0:
        return perm_padded, perm_valid_mask
    for b in range(B):
        row = gt_indices_padded[b]
        valid = (row >= 0).nonzero(as_tuple=True)[0]
        K_b = int(valid.numel())
        if K_b == 0:
            continue
        perm = torch.randperm(K_b, generator=generator, device=device)
        perm_padded[b, :K_b] = row[valid[perm]]
        perm_valid_mask[b, :K_b] = True
    return perm_padded, perm_valid_mask


__all__ = [
    "batched_indices_to_mask",
    "batched_sample_random_gt",
    "indices_to_mask",
    "random_gt_permutations",
    "sample_random_gt_indices",
]
