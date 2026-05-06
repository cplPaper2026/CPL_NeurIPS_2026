"""Training losses for the four supported methods.

* :func:`cpl_loss` -- Contextual Plackett-Luce ordering loss. Per random
  GT permutation we iterate over all ``K_b + 1`` prefixes and sum the
  EOS-weighted cross-entropy term on each.
* :func:`ar_loss` -- Auto-regressive set selection. Same EOS-weighted
  cross-entropy as CPL, evaluated on the pointer-network logits produced
  by the AR decoder for one random GT permutation per call.
* :func:`bce_loss` -- per-token binary cross-entropy with optional
  per-batch automatic ``pos_weight`` (since the GT mask is sparse).
* :func:`hungarian_loss` -- DETR-style bipartite matching between
  valid tokens and random GT indices, followed by BCE on matched /
  unmatched targets and a SmoothL1 consistency term on matched pairs
  in transformer ``h``. The distance term in the matching cost can use
  either ``h`` or frozen ResNet bag features ``X`` (see
  ``HungarianConfig.dist_feature_space``). Optional
  ``HungarianConfig.exclude_identity_match`` forbids assigning a GT
  column to the same token index so matching picks the next-cheapest row.
  Optional ``HungarianConfig.entropy_weight`` adds a mean Shannon entropy
  penalty on normalized ``sigmoid(logits)`` over valid tokens (disabled when
  the weight is 0).

``cpl_loss`` and ``ar_loss`` share :func:`_eos_weighted_ce`, the
asymmetric EOS-weighted cross-entropy primitive: at non-EOS prefix steps
it adds ``-mean(log p[remaining]) + pre_eos_weight * p[EOS]``; at the
EOS prefix step it adds ``post_eos_weight * (-log p[EOS])``. The two
losses differ only in how they obtain the per-step ``(N+1)`` log-probs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from config import BIG_NEG, ArConfig, HungarianConfig

# Large finite cost so scipy's linear_sum_assignment avoids identity (i==j) pairs.
_HUNGARIAN_IDENTITY_BLOCK = 1e12


def _apply_hungarian_identity_column_mask(
    C: torch.Tensor,
    valid_idx: torch.Tensor,
    gt_slots: torch.Tensor,
    exclude_identity_match: bool,
) -> None:
    """In-place: set a huge cost for (row, col) where valid_idx[row] == gt_slots[col].

    When ``exclude_identity_match`` is False or there is only one valid
    token, this is a no-op (no alternative assignment exists).
    """
    if not exclude_identity_match:
        return
    if int(valid_idx.numel()) <= 1:
        return
    K = int(gt_slots.numel())
    for c in range(K):
        gt_tok = int(gt_slots[c].item())
        hit = (valid_idx == gt_tok).nonzero(as_tuple=True)[0]
        if hit.numel() > 0:
            C[int(hit[0].item()), c] = _HUNGARIAN_IDENTITY_BLOCK


def _eos_weighted_ce(
    log_probs: torch.Tensor,
    remaining_mask: torch.Tensor,
    eos_step: torch.Tensor,
    valid_step: torch.Tensor,
    pre_eos_weight: float,
    post_eos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric EOS-weighted cross-entropy term shared by CPL and AR.

    Shapes (``...`` is any leading "step" axes, e.g. ``(B, T)`` for AR
    or ``(K_b + 1,)`` for CPL):

    * ``log_probs``      : ``(..., N + 1)`` log-softmax over slots; slot
                           ``N`` is the EOS log-probability.
    * ``remaining_mask`` : ``(..., N)`` bool, ``True`` at GT tokens that
                           are still unselected at this prefix step.
    * ``eos_step``       : ``(...,)`` bool, ``True`` iff this prefix step
                           is the "all-covered, predict EOS" step.
    * ``valid_step``     : ``(...,)`` bool, ``True`` iff the step should
                           contribute to the loss (False on padded steps).
    * ``pre_eos_weight`` : weight on ``p[EOS]`` added at non-EOS steps
                           (penalises early EOS).
    * ``post_eos_weight``: weight on ``-log p[EOS]`` at the EOS step.

    Per-step term (matching the original CPL formulation):

    * ``eos_step``      -> ``post_eos_weight * (-log_probs[..., N])``.
    * non-EOS, valid    -> ``-mean(log_probs[..., r] for r in remaining)``
                           ``+ pre_eos_weight * exp(log_probs[..., N])``.
    * ``~valid_step``   -> 0 (excluded from both sum and count).

    Returns ``(loss_sum, count)`` so callers can normalise with their
    preferred denominator (typically ``count`` itself).
    """
    log_p_eos = log_probs[..., -1]  # (...,)
    log_p_bag = log_probs[..., :-1]  # (..., N)

    # Non-EOS step term: uniform target over the remaining GTs.
    rem_f = remaining_mask.to(log_probs.dtype)
    rem_count = rem_f.sum(dim=-1).clamp_min(1.0)  # (...,)
    neg_loglik = -(rem_f * log_p_bag).sum(dim=-1) / rem_count  # (...,)
    pre_eos_pen = pre_eos_weight * torch.exp(log_p_eos)  # (...,)
    non_eos_term = neg_loglik + pre_eos_pen

    # EOS step term: standard CE on the EOS slot, scaled by post_eos_weight.
    eos_term = post_eos_weight * (-log_p_eos)

    eos_step_f = eos_step.to(log_probs.dtype)
    valid_step_f = valid_step.to(log_probs.dtype)
    per_step = (eos_step_f * eos_term + (1.0 - eos_step_f) * non_eos_term) * valid_step_f

    loss_sum = per_step.sum()
    count = valid_step_f.sum()
    return loss_sum, count


def cpl_loss(
    theta: torch.Tensor,
    W: torch.Tensor,
    gt_masks: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    pre_eos_weight: float,
    post_eos_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CPL ordering loss with full prefix unrolling per sampled permutation.

    Shapes:
        ``theta``     : ``(B, N+1)``
        ``W``         : ``(B, N+1, N+1)``
        ``gt_masks``  : ``(B, N)`` ground-truth medoid mask
        ``valid_mask``: ``(B, N)`` ``True`` = real token (padding excluded)

    For each batch element ``b`` and each of ``iterations`` random
    permutations of its ``K_b`` GT indices, we evaluate the loss at all
    ``K_b + 1`` prefix steps (size 0 through ``K_b``):

    * For prefix size ``s < K_b``: target is uniform over the remaining
      ``K_b - s`` GTs and an additional ``pre_eos_weight * p[EOS]``
      penalty discourages early EOS.
    * For prefix size ``s = K_b``: target is EOS with
      ``post_eos_weight * (-log p[EOS])``.

    Logits at prefix step ``s`` are
    ``theta[b] + W[b] @ S_mask_s + BIG_NEG * (S_mask_s | pad_ext)``
    where ``S_mask_s`` marks the first ``s`` permutation entries. The
    per-prefix passes are vectorised across ``s`` (matrix multiply once
    against an ``(N+1, K_b+1)`` mask).

    Returns ``(loss, stats)`` with ``loss`` echoed in ``stats`` for
    Comet logging.
    """
    B, Np1 = theta.shape
    N = Np1 - 1
    device = theta.device
    dtype = theta.dtype

    loss_sum = torch.zeros((), device=device, dtype=dtype)
    total_terms = torch.zeros((), device=device, dtype=dtype)

    for b in range(B):
        valid = valid_mask[b]
        gt_indices = (gt_masks[b] & valid).nonzero(as_tuple=True)[0]
        K_b = int(gt_indices.numel())
        if K_b == 0:
            continue

        pad_ext = torch.cat([~valid, torch.zeros(1, dtype=torch.bool, device=device)])  # (N+1,)
        pad_ext_f = pad_ext.to(dtype).unsqueeze(0)  # (1, N+1)

        # gt_full: bag-only membership mask for this row's GTs (length N).
        gt_full = torch.zeros(N, dtype=torch.bool, device=device)
        gt_full[gt_indices] = True

        for _ in range(iterations):
            perm = gt_indices[torch.randperm(K_b, device=device)]  # (K_b,)

            # one_hot[s, n] = 1 iff perm[s] == n. Inclusive cumsum gives
            # "tokens in perm[:s+1]"; prepend a zero row for prefix size 0.
            one_hot_bag = torch.zeros(K_b, N, dtype=dtype, device=device)
            one_hot_bag.scatter_(1, perm.unsqueeze(-1), 1.0)
            cumsum_bag = one_hot_bag.cumsum(dim=0)  # (K_b, N)
            zero_row = torch.zeros(1, N, dtype=dtype, device=device)
            seen_bag = torch.cat([zero_row, cumsum_bag], dim=0)  # (K_b+1, N)

            # Extend to (K_b+1, N+1) by appending a zero column for EOS.
            seen_full = torch.cat(
                [seen_bag, torch.zeros(K_b + 1, 1, dtype=dtype, device=device)],
                dim=1,
            )  # (K_b+1, N+1)

            # contexts[s] = W[b] @ seen_full[s]; vectorised via matmul.
            contexts = seen_full @ W[b].transpose(0, 1)  # (K_b+1, N+1)

            big_neg_mask = BIG_NEG * (seen_full + pad_ext_f).clamp_max(1.0)
            logits = theta[b].unsqueeze(0) + contexts + big_neg_mask  # (K_b+1, N+1)
            log_probs = F.log_softmax(logits, dim=-1)  # (K_b+1, N+1)

            # Remaining = GTs - seen_bag (boolean).
            remaining_mask = gt_full.unsqueeze(0) & (seen_bag < 0.5)  # (K_b+1, N)

            # Step indicators: prefix step s ∈ {0, .., K_b}; EOS at s == K_b.
            steps = torch.arange(K_b + 1, device=device)
            eos_step = steps == K_b  # (K_b+1,)
            valid_step = torch.ones(K_b + 1, dtype=torch.bool, device=device)

            term_sum, term_count = _eos_weighted_ce(
                log_probs,
                remaining_mask,
                eos_step,
                valid_step,
                pre_eos_weight,
                post_eos_weight,
            )
            loss_sum = loss_sum + term_sum
            total_terms = total_terms + term_count

    loss = loss_sum / total_terms.clamp_min(1.0)
    stats = {"loss": float(loss.detach().item())}
    return loss, stats


def _auto_pos_weight(gt_masks: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Per-batch pos_weight = (# valid negatives) / (# valid positives).

    Falls back to 1.0 when there are no positives (avoids divide-by-zero
    and keeps the loss well-defined even on degenerate batches).
    """
    valid_f = valid_mask.float()
    pos = (gt_masks.float() * valid_f).sum()
    neg = ((~gt_masks).float() * valid_f).sum()
    if pos.item() <= 0:
        return torch.tensor(1.0, device=gt_masks.device)
    return (neg / pos).clamp_min(1.0)


def bce_loss(
    logits: torch.Tensor,
    gt_masks: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Binary cross-entropy on the per-token medoid classifier.

    Shapes:
        ``logits``    : ``(B, N)``
        ``gt_masks``  : ``(B, N)`` bool
        ``valid_mask``: ``(B, N)`` bool

    Padding positions are masked out of the loss. ``pos_weight`` can be
    ``None`` (auto = ``n_neg / n_pos`` per batch) or a fixed scalar.

    Returns ``(loss, stats)`` for Comet logging (``bce_loss``, ``pos_weight``,
    ``n_pos``, ``n_neg``).
    """
    target = gt_masks.to(logits.dtype)
    pw = _auto_pos_weight(gt_masks, valid_mask) if pos_weight is None else torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    valid_f = valid_mask.to(raw.dtype)
    n_valid = valid_f.sum().clamp_min(1.0)
    loss = (raw * valid_f).sum() / n_valid
    pw_f = float(pw.item()) if isinstance(pw, torch.Tensor) else float(pw)
    pos_ct = float((gt_masks & valid_mask).sum().item())
    neg_ct = float(((~gt_masks) & valid_mask).sum().item())
    stats = {
        "bce_loss": float(loss.detach().item()),
        "pos_weight": pw_f,
        "n_pos": pos_ct,
        "n_neg": neg_ct,
    }
    return loss, stats


def _pairwise_distance(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
    """Pairwise distance between row vectors of ``a`` and ``b``.

    Shapes: ``a`` is ``(N, D)``, ``b`` is ``(K, D)``. Returns ``(N, K)``.
    """
    if kind == "l2":
        # Squared L2: (N, K) via (a - b).pow(2).sum(-1) without materialising
        # the full (N, K, D) difference tensor.
        a2 = (a * a).sum(dim=-1, keepdim=True)  # (N, 1)
        b2 = (b * b).sum(dim=-1, keepdim=True).T  # (1, K)
        ab = a @ b.T  # (N, K)
        return (a2 + b2 - 2.0 * ab).clamp_min(0.0)
    if kind == "cosine":
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        return 1.0 - a_n @ b_n.T
    raise ValueError(f"unknown distance kind {kind!r}")


_SIGMOID_ENTROPY_EPS = 1e-8


def _batch_valid_sigmoid_entropy(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = _SIGMOID_ENTROPY_EPS,
) -> torch.Tensor:
    """Mean Shannon entropy of row-wise normalized sigmoid mass on valid tokens.

    For each batch row, ``p = sigmoid(logits) * valid``,
    ``q = p / (sum(p) + eps)``, then ``H = -sum q log q``. Rows with fewer
    than two valid tokens are omitted from the average (gradient 0 there).

    Returns a scalar tensor differentiable in ``logits``.
    """
    dtype = logits.dtype
    device = logits.device
    valid_f = valid_mask.to(dtype=dtype, device=device)
    p = torch.sigmoid(logits) * valid_f
    denom = p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = p / denom
    q_safe = q.clamp_min(eps)
    H = -(q * q_safe.log()).sum(dim=-1)
    n_valid = valid_mask.sum(dim=-1)
    eligible = (n_valid > 1).to(dtype=dtype, device=device)
    denom_elig = eligible.sum().clamp_min(1.0)
    return (H * eligible).sum() / denom_elig


def hungarian_loss(
    logits: torch.Tensor,
    h: torch.Tensor,
    X: torch.Tensor | None,
    gt_indices_padded: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: HungarianConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DETR-style Hungarian matching + BCE + consistency on matched pairs.

    Shapes:
        ``logits``            : ``(B, N)``
        ``h``                 : ``(B, N, D)`` transformer hidden states
        ``X``                 : ``(B, N, D)`` bag features (ResNet); required
                                when ``cfg.dist_feature_space == "resnet"``.
        ``gt_indices_padded`` : ``(B, K_max)`` token indices in ``[0, N)``,
                                ``-1`` in padding slots.
        ``valid_mask``        : ``(B, N)`` bool (``True`` = real token).

    Steps per batch element:

    1. Build cost matrix ``C[i, j]`` over valid tokens ``i`` and GTs ``j``
       as ``cls_w * (-log sigmoid(logit_i)) + dist_w * d(f_i, f_gt_j)``.
       Features ``f`` are ``h.detach()`` or ``X.detach()`` according to
       ``cfg.dist_feature_space`` (``transformer`` vs ``resnet``); the
       matching is not differentiable (scipy on CPU numpy). If
       ``cfg.exclude_identity_match`` is True and there are at least two
       valid tokens, the identity cell for each GT column is blocked so
       Hungarian picks the lowest-cost *other* token for that column.
    2. Run ``scipy.optimize.linear_sum_assignment`` to obtain the
       assignment ``sigma`` of exactly ``K`` winners.
    3. BCE: target is 1 at matched token indices, 0 at all other valid
       tokens. ``pos_weight`` auto = ``n_neg / n_pos`` per batch unless
       overridden by ``cfg.pos_weight``.
    4. Consistency (autograd ON): ``SmoothL1`` (or cosine surrogate) on
       ``h[match_i]`` vs ``h[gt_j]`` — always transformer ``h``, independent
       of ``dist_feature_space``.
    5. Optional entropy sharpening (autograd ON): when
       ``cfg.entropy_weight > 0``, adds that weight times the mean Shannon
       entropy of normalized ``sigmoid(logits)`` over valid positions per
       row (rows with ``N_valid < 2`` skipped). Minimizing encourages peaked
       mass over the bag; combine with BCE/consistency so representations
       stay diverse.

    Total loss is
    ``bce + cfg.consistency_weight * consistency +
    cfg.entropy_weight * entropy``.

    Returns ``(loss, stats)`` where ``stats`` is a logging-friendly dict
    with ``bce_loss``, ``cons_loss``, ``weighted_consistency``,
    ``entropy_loss``, ``weighted_entropy``,
    ``mean_cls_cost``, ``mean_dist_cost``, ``num_matches``.
    """
    B, N = logits.shape
    device = logits.device
    dtype = logits.dtype

    if cfg.dist_feature_space == "resnet" and X is None:
        raise ValueError("hungarian_loss: dist_feature_space='resnet' requires bag features X")

    # --------------------------------------------------------------
    # 1-2. Build target mask via Hungarian; collect matched pairs.
    # --------------------------------------------------------------
    target = torch.zeros(B, N, dtype=dtype, device=device)
    match_token_rows: list[int] = []  # b-indices (flat, for gather)
    match_token_cols: list[int] = []  # token indices in [0, N)
    match_gt_cols: list[int] = []  # GT token indices in [0, N)
    cls_cost_sum = 0.0
    dist_cost_sum = 0.0
    num_matches = 0

    neg_logsig = F.logsigmoid(logits).detach().neg().clamp_max(50.0)  # (B, N)
    h_det = h.detach()

    for b in range(B):
        vm = valid_mask[b]
        gt_row = gt_indices_padded[b]
        gt_slots = gt_row[gt_row >= 0]
        K = int(gt_slots.numel())
        if K == 0:
            continue
        valid_idx = vm.nonzero(as_tuple=True)[0]
        N_valid = int(valid_idx.numel())
        if N_valid < K:
            # Pathological bag: not enough valid tokens. Skip matching;
            # target stays all-zero (BCE-negative everywhere).
            continue

        if cfg.dist_feature_space == "resnet":
            feats_valid = X[b, valid_idx].detach()
            feats_gt = X[b, gt_slots].detach()
        else:
            feats_valid = h_det[b, valid_idx]
            feats_gt = h_det[b, gt_slots]

        dist_cost = _pairwise_distance(feats_valid, feats_gt, cfg.distance)  # (N_valid, K)
        cls_cost = neg_logsig[b, valid_idx].unsqueeze(1).expand(-1, K)  # (N_valid, K)

        C = cfg.cls_weight * cls_cost + cfg.dist_weight * dist_cost
        _apply_hungarian_identity_column_mask(C, valid_idx, gt_slots, cfg.exclude_identity_match)
        cost_np = C.detach().cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost_np)
        # row_ind ranges over valid tokens (0..N_valid-1); col_ind over GTs.
        for r, c in zip(row_ind, col_ind):
            tok = int(valid_idx[r].item())
            gt_tok = int(gt_slots[c].item())
            target[b, tok] = 1.0
            match_token_rows.append(b)
            match_token_cols.append(tok)
            match_gt_cols.append(gt_tok)
            cls_cost_sum += float(cls_cost[r, c].item())
            dist_cost_sum += float(dist_cost[r, c].item())
            num_matches += 1

    # --------------------------------------------------------------
    # 3. BCE on matched/unmatched valid tokens.
    # --------------------------------------------------------------
    valid_f = valid_mask.to(dtype)
    n_pos = (target * valid_f).sum()
    n_neg = ((1.0 - target) * valid_f).sum()
    if cfg.pos_weight is None:
        pw = (n_neg / n_pos).clamp_min(1.0) if n_pos.item() > 0 else torch.tensor(1.0, device=device, dtype=dtype)
    else:
        pw = torch.tensor(float(cfg.pos_weight), device=device, dtype=dtype)
    raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    n_valid = valid_f.sum().clamp_min(1.0)
    bce_term = (raw * valid_f).sum() / n_valid

    # --------------------------------------------------------------
    # 4. Consistency loss (autograd ON) on matched pairs.
    # --------------------------------------------------------------
    if num_matches > 0:
        rows_t = torch.tensor(match_token_rows, dtype=torch.long, device=device)
        match_tok_t = torch.tensor(match_token_cols, dtype=torch.long, device=device)
        match_gt_t = torch.tensor(match_gt_cols, dtype=torch.long, device=device)
        h_match = h[rows_t, match_tok_t]  # (M, D)
        h_gt = h[rows_t, match_gt_t]  # (M, D)
        if cfg.distance == "cosine":
            # Cosine surrogate on transformer h (independent of dist_feature_space).
            cons_term = (1.0 - F.cosine_similarity(h_match, h_gt, dim=-1)).mean()
        else:
            cons_term = F.smooth_l1_loss(h_match, h_gt, reduction="mean")
    else:
        cons_term = torch.zeros((), device=device, dtype=dtype)

    loss = bce_term + cfg.consistency_weight * cons_term

    ent_w = float(cfg.entropy_weight)
    if ent_w > 0.0:
        entropy_term = _batch_valid_sigmoid_entropy(logits, valid_mask)
        loss = loss + ent_w * entropy_term
        ent_raw = float(entropy_term.detach().item())
    else:
        ent_raw = 0.0

    cons_w = float(cfg.consistency_weight)
    stats = {
        "bce_loss": float(bce_term.detach().item()),
        "cons_loss": float(cons_term.detach().item()),
        "weighted_consistency": cons_w * float(cons_term.detach().item()),
        "entropy_loss": ent_raw,
        "weighted_entropy": ent_w * ent_raw,
        "mean_cls_cost": float(cls_cost_sum / max(num_matches, 1)),
        "mean_dist_cost": float(dist_cost_sum / max(num_matches, 1)),
        "num_matches": float(num_matches),
    }
    return loss, stats


def ar_loss(
    logits: torch.Tensor,
    gt_indices_padded: torch.Tensor,
    perm_padded: torch.Tensor,
    perm_valid_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: ArConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """EOS-weighted cross-entropy on the AR pointer-network logits.

    Shapes:
        ``logits``            : ``(B, T, N + 1)``, ``T = K_max + 1``;
                                slot ``N`` is the EOS logit per query.
        ``gt_indices_padded`` : ``(B, K_max)`` token indices in ``[0, N)``,
                                ``-1`` in pad slots.
        ``perm_padded``       : ``(B, K_max)`` random permutation of the
                                row's GT indices, ``-1`` in pad slots.
                                Must agree with ``gt_indices_padded`` as
                                a *set* per row.
        ``perm_valid_mask``   : ``(B, K_max)`` bool, ``True`` at valid
                                permutation slots (i.e. ``slot < K_b``).
        ``valid_mask``        : ``(B, N)`` bool, ``True`` at real bag tokens.
        ``cfg``               : :class:`config.ArConfig`.

    Per element ``b`` with ``K_b`` clusters, the decoder query at prefix
    step ``t in [0, K_b - 1]`` is asked to spread mass uniformly over the
    still-unselected GTs ``remaining_t`` (and is penalised for placing
    mass on EOS via ``pre_eos_weight * p[EOS]``). The query at step
    ``K_b`` is asked to pick EOS via ``post_eos_weight * (-log p[EOS])``.
    Padded query positions (``t > K_b``) are masked out.

    Already-selected bag positions and bag padding are masked to
    ``BIG_NEG`` before the per-step ``log_softmax`` so they receive zero
    probability mass (matching the CPL formulation).

    The remaining-set / step indicators are built vectorised from a
    triangular cumsum of one-hot encodings of ``perm_padded``; the loss
    math is delegated to :func:`_eos_weighted_ce`.
    """
    B, T, slots = logits.shape
    N = slots - 1
    device = logits.device
    dtype = logits.dtype

    if perm_valid_mask.shape != (B, T - 1):
        raise ValueError(f"perm_valid_mask must be (B, T-1) = ({B}, {T - 1}); got {tuple(perm_valid_mask.shape)}")

    K_max = T - 1
    k_per_row = perm_valid_mask.sum(dim=1).long()  # (B,) = K_b

    # ------------------------------------------------------------------
    # "Seen so far" mask per query position via cumsum of one-hots.
    # ------------------------------------------------------------------
    perm_safe = perm_padded.clamp_min(0)  # (B, K_max)
    perm_one_hot = torch.zeros(B, K_max, N, dtype=dtype, device=device)
    perm_one_hot.scatter_(2, perm_safe.unsqueeze(-1), 1.0)
    perm_one_hot = perm_one_hot * perm_valid_mask.unsqueeze(-1).to(dtype)

    cumsum_inclusive = perm_one_hot.cumsum(dim=1)  # (B, K_max, N)

    # Seen-before-step-t: t == 0 -> nothing; t >= 1 -> cumsum at t - 1.
    seen_zero = torch.zeros(B, 1, N, dtype=dtype, device=device)
    seen_mask_per_step = torch.cat([seen_zero, cumsum_inclusive], dim=1)  # (B, T, N)
    seen_bool = seen_mask_per_step > 0.5  # (B, T, N)

    # Full GT membership mask (B, N).
    gt_full = torch.zeros(B, N, dtype=torch.bool, device=device)
    gt_valid = gt_indices_padded >= 0
    if gt_valid.any():
        rows = torch.arange(B, device=device).unsqueeze(1).expand_as(gt_indices_padded)
        gt_full[rows[gt_valid], gt_indices_padded[gt_valid]] = True

    remaining_mask = gt_full.unsqueeze(1) & ~seen_bool  # (B, T, N)

    # Step validity / EOS step indicator.
    arange_T = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    query_valid_mask = arange_T <= k_per_row.unsqueeze(1)  # (B, T)
    eos_step = (arange_T == k_per_row.unsqueeze(1)) & query_valid_mask  # (B, T)

    # ------------------------------------------------------------------
    # Apply BIG_NEG to selected and padded bag slots before log_softmax,
    # then delegate the EOS-weighted CE to the shared helper.
    # ------------------------------------------------------------------
    pad_bag = ~valid_mask  # (B, N)
    bag_block = (seen_bool | pad_bag.unsqueeze(1)).to(dtype)  # (B, T, N)
    eos_block_zero = torch.zeros(B, T, 1, dtype=dtype, device=device)
    block = torch.cat([bag_block, eos_block_zero], dim=-1)  # (B, T, N+1)

    masked_logits = logits + BIG_NEG * block
    log_probs = F.log_softmax(masked_logits, dim=-1)  # (B, T, N+1)

    loss_sum, count = _eos_weighted_ce(
        log_probs,
        remaining_mask,
        eos_step,
        query_valid_mask,
        cfg.pre_eos_weight,
        cfg.post_eos_weight,
    )
    loss = loss_sum / count.clamp_min(1.0)
    stats = {"loss": float(loss.detach().item())}
    return loss, stats


__all__ = ["ar_loss", "bce_loss", "cpl_loss", "hungarian_loss"]
