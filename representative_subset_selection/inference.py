"""Inference rules: CPL greedy decode, AR greedy decode, BCE thresholding, K-means oracle-K."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from runtime_profile import CudaTimerStats, cuda_timing_section

from config import BIG_NEG

# ---------------------------------------------------------------------------
# CPL greedy decode (with optional decoder-step trace for visualisation)
# ---------------------------------------------------------------------------


@dataclass
class CPLTrace:
    """Per-batch trace of a CPL greedy-decode run, for visualisation.

    All tensors are on CPU. ``orders[b]`` lists bag token indices in
    selection order; when greedy picks EOS the sentinel ``N`` is appended
    (same convention as :class:`ARTrace`). ``step_probs[b][t]`` is the
    softmax distribution over ``N + 1`` slots at step ``t`` for sample
    ``b`` (already-selected and padded bag positions have ~0 mass).
    """

    orders: list[list[int]]
    step_probs: list[list[torch.Tensor]]


@torch.no_grad()
def cpl_greedy(
    theta: torch.Tensor,
    W: torch.Tensor,
    max_steps: int,
    valid_mask: torch.Tensor,
    return_order: bool = False,
    return_trace: bool = False,
):
    """Sequential greedy decode over the CPL distribution.

    Stops as soon as the EOS slot is selected (or the per-batch step
    budget is exhausted). When ``return_order`` is True, also returns a
    list ``orders`` of length B where ``orders[b]`` is a LongTensor of
    selected token indices in selection order (excluding EOS).

    ``return_trace`` is mutually exclusive with ``return_order``. When
    True, returns a :class:`CPLTrace` with per-step softmax probabilities
    over ``N + 1`` slots and ``orders`` that may end with the EOS index
    ``N``.
    """
    if return_order and return_trace:
        raise ValueError("cpl_greedy: return_order and return_trace are mutually exclusive")

    B, Np1 = theta.shape
    N = Np1 - 1
    device = theta.device

    masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_mask = torch.zeros(B, Np1, dtype=torch.bool, device=device)
    context = torch.zeros(B, Np1, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    pad_bias = torch.zeros(B, Np1, device=device)
    pad_bias[:, :N] = BIG_NEG * (~valid_mask).float()

    orders: list[list[int]] = [[] for _ in range(B)]
    step_probs: list[list[torch.Tensor]] = [[] for _ in range(B)]

    for _ in range(max_steps):
        if done.all():
            break
        logits = theta + context + BIG_NEG * selected_mask.float() + pad_bias

        if return_trace:
            probs = torch.softmax(logits, dim=1).detach().cpu()
            for br in range(B):
                if not bool(done[br].item()):
                    step_probs[br].append(probs[br].clone())

        j = logits.argmax(dim=1)

        for b in range(B):
            if done[b]:
                continue
            idx = int(j[b].item())
            if idx == N:
                done[b] = True
                if return_trace:
                    orders[b].append(N)
                continue
            if selected_mask[b, idx]:
                done[b] = True
                continue
            masks[b, idx] = True
            selected_mask[b, idx] = True
            context[b] += W[b, :, idx]
            orders[b].append(idx)

    if return_trace:
        return masks, CPLTrace(orders=orders, step_probs=step_probs)
    if return_order:
        order_tensors = [torch.tensor(o, dtype=torch.long, device=device) for o in orders]
        return masks, order_tensors
    return masks


# ---------------------------------------------------------------------------
# AR greedy decode
# ---------------------------------------------------------------------------


@dataclass
class ARTrace:
    """Per-batch trace of an AR greedy-decode run, for visualisation.

    All tensors are on CPU. ``orders[b]`` lists the bag token indices
    selected (in order) before EOS. ``step_probs[b][t]`` is the
    ``softmax`` distribution over the ``N + 1`` slots at decoder query
    position ``t`` for sample ``b`` (``N`` bag tokens + 1 EOS slot),
    computed from the same masked logits as greedy ``argmax``:
    padded and already-selected bag positions receive ``BIG_NEG`` before
    ``softmax``, so they carry ~0 mass (same convention as
    :class:`CPLTrace`).
    """

    orders: list[list[int]]
    step_probs: list[list[torch.Tensor]]


@torch.no_grad()
def ar_greedy(
    model,
    X: torch.Tensor,
    valid_mask: torch.Tensor,
    max_steps: int,
    return_trace: bool = False,
    timing_stats: CudaTimerStats | None = None,
):
    """Greedy auto-regressive decode for the AR mode.

    The bag is encoded once into ``h``; the decoder is then re-run from
    scratch with a growing query sequence. At step ``t`` we pick the
    argmax over ``N + 1`` slots after applying ``BIG_NEG`` to padded and
    already-selected bag positions; if the EOS slot wins (or the per-row
    step budget is exhausted), the row is marked done.

    Re-running the decoder with a growing query each step is correct and
    simple; for the small query lengths used here (``T = K + 1`` typically
    <= 11) the overhead vs. KV-cached decoding is negligible.

    Returns ``pred_mask: (B, N)`` on the same device as ``X``. When
    ``return_trace`` is True, also returns an :class:`ARTrace` with the
    per-step orders and per-step ``softmax`` distributions over ``N+1``
    slots (CPU tensors), matching the training loss readout.

    Optional ``timing_stats`` (:class:`~runtime_profile.CudaTimerStats`)
    accumulates GPU time for the autoregressive decode loop under key
    ``ar_autoreg_ms``.
    """
    device = X.device
    B, N, _ = X.shape
    pad = ~valid_mask

    h = model.encode(X, src_key_padding_mask=pad)  # (B, N, D)

    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    history = torch.full((B, max_steps), -1, dtype=torch.long, device=device)
    history_valid = torch.zeros(B, max_steps, dtype=torch.bool, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    pad_bag_bias = torch.zeros(B, N, device=device, dtype=h.dtype)
    pad_bag_bias[~valid_mask] = BIG_NEG

    orders: list[list[int]] = [[] for _ in range(B)]
    step_probs: list[list[torch.Tensor]] = [[] for _ in range(B)]

    with cuda_timing_section(timing_stats, "ar_autoreg_ms"):
        for t in range(max_steps):
            if bool(done.all()):
                break

            # Build dec_in for query positions 0..t (length t+1). We reuse the
            # model's helper by passing the (B, t) "previous selections" view.
            prev_perm = history[:, :t] if t > 0 else history[:, :0]
            prev_valid = history_valid[:, :t] if t > 0 else history_valid[:, :0]
            dec_in = model.build_decoder_input(h, prev_perm, prev_valid)  # (B, t+1, D)

            bos_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
            tgt_key_padding_mask = ~torch.cat([bos_valid, prev_valid], dim=1)

            q = model.decode(
                h=h,
                dec_in=dec_in,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=pad,
            )  # (B, t+1, D)
            logits = model.pointer_logits(q, h)  # (B, t+1, N+1)
            last = logits[:, -1, :].clone()  # (B, N+1)

            # Mask padded and already-selected bag positions; EOS slot stays open.
            last[:, :N] = last[:, :N] + pad_bag_bias
            last[:, :N] = last[:, :N] + BIG_NEG * selected.to(last.dtype)

            if return_trace:
                probs = torch.softmax(last, dim=1).detach().cpu()  # (B, N+1)
                for b in range(B):
                    if not bool(done[b].item()):
                        step_probs[b].append(probs[b].clone())

            winners = last.argmax(dim=1)  # (B,)
            for b in range(B):
                if bool(done[b].item()):
                    continue
                j = int(winners[b].item())
                if j == N:
                    done[b] = True
                    if return_trace:
                        orders[b].append(N)  # marker for EOS in the trace
                    continue
                if selected[b, j]:
                    # Defensive: the mask should make this impossible; treat as EOS.
                    done[b] = True
                    continue
                selected[b, j] = True
                history[b, t] = j
                history_valid[b, t] = True
                orders[b].append(j)

    pred_mask = selected & valid_mask
    if return_trace:
        return pred_mask, ARTrace(orders=orders, step_probs=step_probs)
    return pred_mask


# ---------------------------------------------------------------------------
# BCE thresholding
# ---------------------------------------------------------------------------


@torch.no_grad()
def bce_threshold(logits: torch.Tensor, threshold: float, valid_mask: torch.Tensor) -> torch.Tensor:
    """Predict ``sigmoid(logits) >= threshold`` (intersected with valid_mask)."""
    probs = torch.sigmoid(logits)
    return (probs >= threshold) & valid_mask


@torch.no_grad()
def unary_threshold(scores: torch.Tensor, threshold: float, valid_mask: torch.Tensor) -> torch.Tensor:
    """Generic raw-score threshold (used for percentile sweeps on CPL theta)."""
    return (scores >= threshold) & valid_mask


# ---------------------------------------------------------------------------
# K-means oracle-K reference baseline
# ---------------------------------------------------------------------------


def _lloyd_kmeans_numpy(feats: np.ndarray, K: int, rng: np.random.Generator, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd K-means on rows of ``feats``; Forgy init from K random points."""
    n = feats.shape[0]
    feats = np.asarray(feats, dtype=np.float64)
    if K <= 0:
        raise ValueError("K must be positive")
    if K == 1:
        centers = feats.mean(axis=0, keepdims=True)
        return np.zeros(n, dtype=np.int32), centers
    if n < K:
        raise ValueError("K cannot exceed n")
    init_idx = rng.choice(n, size=K, replace=False)
    centers = feats[init_idx].copy()
    d2 = ((feats[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1).astype(np.int32)
    for _ in range(max_iter):
        new_centers = np.zeros_like(centers)
        for j in range(K):
            m = labels == j
            if np.any(m):
                new_centers[j] = feats[m].mean(axis=0)
            else:
                new_centers[j] = centers[j]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
        d2 = ((feats[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels, centers


def kmeans_oracle_k_pred_mask(
    X: torch.Tensor,
    mask: torch.Tensor,
    cluster_labels: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """K-means on valid feature rows, ``K`` = number of GT semantic clusters.

    After clustering, picks one point per K-means cluster: the member
    closest to that cluster's centroid (same medoid rule used for the GT
    mask). Returns ``(B, N)`` bool predictions on CPU.
    """
    B, N, _D = X.shape
    out = torch.zeros(B, N, dtype=torch.bool)
    Xf = X.detach().float().cpu().numpy()
    mask_np = mask.cpu().numpy()
    cl_np = cluster_labels.cpu().numpy()

    for b in range(B):
        idxs = np.flatnonzero(mask_np[b])
        if idxs.size == 0:
            continue
        feats = Xf[b, idxs]
        labs = cl_np[b, idxs]
        valid_sem = np.unique(labs[labs >= 0])
        K = int(valid_sem.size)
        if K < 1:
            continue
        n = feats.shape[0]
        if n < K:
            continue
        try:
            km_labels, centers = _lloyd_kmeans_numpy(feats, K, rng)
        except ValueError:
            continue
        for j in range(K):
            member_rows = np.flatnonzero(km_labels == j)
            if member_rows.size == 0:
                continue
            pts = feats[member_rows]
            c = centers[j]
            dists = np.sum((pts - c) ** 2, axis=1)
            rbest = int(member_rows[int(np.argmin(dists))])
            tok = int(idxs[rbest])
            out[b, tok] = True
    return out


__all__ = [
    "ARTrace",
    "CPLTrace",
    "ar_greedy",
    "bce_threshold",
    "cpl_greedy",
    "kmeans_oracle_k_pred_mask",
    "unary_threshold",
]
