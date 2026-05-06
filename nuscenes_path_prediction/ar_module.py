"""Autoregressive decoder head with a pointer-network readout.

The AR head consumes the model's per-cell embedding grid and predicts an
ordered sequence of grid-cell indices via a small ``TransformerDecoder``
followed by a pointer-network dot-product readout against the same cell
embeddings.

Design notes:

* The next-cell scores come from a content-based dot-product
  ``query_proj(q) . key_proj(memory)^T / sqrt(d)`` plus a learnable EOS key.
  This makes the AR scoring spatially translation-aware in the same way the
  CPL head's ``W_dense`` is, instead of using a fixed
  ``Linear(hidden_dim, num_cells + 1)`` indexed by absolute cell ID.
* The decoder input at step ``t >= 1`` is ``memory[idx_{t-1}]`` -- the
  encoder's representation of the previously selected cell -- rather than a
  separate ``nn.Embedding(num_cells + 1)`` table. The encoder's contextual
  representation is therefore directly fed back into the decoder.
* Already-selected bag positions are masked with ``BIG_NEG`` before the
  per-step softmax in both training and inference. This matches what the
  CPL loss does and removes a training/inference mismatch.

The external surface (``forward``, ``loss``, ``greedy_decode``) and the
shapes consumed by the rest of the trainer are unchanged from the previous
classification-head version.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from train_config import ArHeadConfig

BIG_NEG = -1e9


class _MLP(nn.Module):
    """Small two-layer MLP used by the pointer-net key/query projections."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        last_in = hidden_dim if num_layers > 1 else input_dim
        layers.append(nn.Linear(last_in, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ARDecoderHead(nn.Module):
    """Autoregressive transformer decoder + pointer-network head over a cell grid."""

    def __init__(self, ar: ArHeadConfig, *, hidden_dim: int, grid_size: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.eos_index = self.num_cells
        self.max_steps = int(ar.max_steps)
        self.loss_type = str(ar.loss_type)

        if hidden_dim % int(ar.num_heads) != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by ar_head.num_heads ({ar.num_heads}).")

        ffn_dim = hidden_dim * int(ar.ffn_dim_multiplier)
        act = (ar.activation or "gelu").lower()
        norm_first = bool(ar.norm_first)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=int(ar.num_heads),
            dim_feedforward=ffn_dim,
            dropout=float(ar.dropout),
            activation=act,
            batch_first=True,
            norm_first=norm_first,
        )
        final_norm = nn.LayerNorm(hidden_dim) if norm_first else None
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(ar.num_layers), norm=final_norm)

        # Pointer-net key/query projections operating on the encoder cell embeddings.
        self.key_proj = _MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)
        self.query_proj = _MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)

        # Learnable BOS query and EOS key (single vectors, not per-cell-index tables).
        self.bos = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.bos, std=float(ar.bos_init_std))
        self.eos_key = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.eos_key, std=float(ar.eos_init_std))

        # Per-step query positional embedding, used at every decoder slot 0..max_steps + 1.
        self.step_pos = nn.Embedding(self.max_steps + 2, hidden_dim)
        nn.init.trunc_normal_(self.step_pos.weight, std=float(ar.step_pos_init_std))

    def forward(self, embeddings_grid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flatten the encoder grid into ``memory`` tokens for the decoder."""
        if embeddings_grid.ndim != 4:
            raise ValueError(f"Expected embeddings_grid shape (B, C, H, W), got {tuple(embeddings_grid.shape)}")
        bsz, c, h, w = embeddings_grid.shape
        if c != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {c}.")
        if h != self.grid_size or w != self.grid_size:
            raise ValueError(f"Expected ({self.grid_size}, {self.grid_size}), got ({h}, {w}).")
        memory = embeddings_grid.view(bsz, c, -1).permute(0, 2, 1).contiguous()  # (B, N, C)
        return {"memory": memory}

    # ------------------------------------------------------------------
    # Building blocks (used by both loss and greedy_decode)
    # ------------------------------------------------------------------

    def _build_decoder_input(
        self,
        memory: torch.Tensor,
        prev_idx: torch.Tensor,
        prev_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble decoder input tokens for query positions ``0..K``.

        ``prev_idx`` has shape ``(B, K)``. Slot 0 of the returned tensor is BOS.
        Slot ``t >= 1`` is ``memory[prev_idx[:, t - 1]]`` for valid permutation
        entries; padded slots fall back to BOS. Each slot then receives the
        per-step learnable positional embedding ``step_pos[t]``.
        """
        bsz, k_max = prev_idx.shape
        device = memory.device
        d = self.hidden_dim

        if k_max > 0:
            safe_idx = prev_idx.clamp_min(0).unsqueeze(-1).expand(-1, -1, d)
            prev_selected = torch.gather(memory, 1, safe_idx)  # (B, K, D)
            bos_row = self.bos.view(1, 1, d).expand(bsz, k_max, d)
            prev_selected = torch.where(prev_valid.unsqueeze(-1), prev_selected, bos_row)
        else:
            prev_selected = memory.new_zeros((bsz, 0, d))

        bos_first = self.bos.view(1, 1, d).expand(bsz, 1, d)
        dec_in = torch.cat([bos_first, prev_selected], dim=1)  # (B, K + 1, D)

        T = dec_in.shape[1]
        if self.step_pos.num_embeddings < T:
            raise ValueError(f"AR decoder query length {T} exceeds step_pos capacity ({self.step_pos.num_embeddings}); raise ar_head.max_steps.")
        steps = torch.arange(T, device=device)
        dec_in = dec_in + self.step_pos(steps).unsqueeze(0)
        return dec_in

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def _decode(
        self,
        memory: torch.Tensor,
        dec_in: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the transformer decoder with a causal self-attention mask."""
        return self.decoder(
            tgt=dec_in,
            memory=memory,
            tgt_mask=self._causal_mask(dec_in.shape[1], dec_in.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def _pointer_logits(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Pointer-network logits ``(B, T, N + 1)`` over (memory cells + EOS).

        ``logits[..., :N] = query_proj(q) . key_proj(memory)^T / sqrt(d)``
        ``logits[..., N]  = query_proj(q) . eos_key / sqrt(d)``
        """
        scale = math.sqrt(float(self.hidden_dim))
        q_p = self.query_proj(q)  # (B, T, D)
        keys = self.key_proj(memory)  # (B, N, D)
        bag_logits = q_p @ keys.transpose(1, 2) / scale  # (B, T, N)
        eos_logits = (q_p @ self.eos_key) / scale  # (B, T)
        return torch.cat([bag_logits, eos_logits.unsqueeze(-1)], dim=-1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
        meta: list[dict[str, Any]],  # noqa: ARG002 -- kept for API parity with CPL head.
        *,
        D: int,
    ) -> torch.Tensor:
        """Single-pass teacher-forced next-token cross-entropy.

        Per sample with path length ``K_b`` the decoder query at slot ``t in
        [0, K_b - 1]`` is asked to predict ``idx_t`` (the t-th cell on the
        path); the query at slot ``K_b`` is asked to predict EOS. Slots
        beyond ``K_b`` are masked out. Already-selected bag cells are blocked
        with ``BIG_NEG`` before the softmax, mirroring the CPL training-time
        mask.
        """
        memory = outputs["memory"]
        if memory.ndim != 3:
            raise ValueError(f"Expected memory shape (B, N, C), got {tuple(memory.shape)}")

        gt_idx_padded, valid_mask = self._build_target_batch(gt_points, num_points, D=D, device=memory.device)
        bsz, k_max = gt_idx_padded.shape
        n = self.num_cells
        T = k_max + 1
        device = memory.device
        dtype = memory.dtype

        # Decoder input: BOS at slot 0; h[idx_t] (or BOS for padded) at slot t + 1.
        dec_in = self._build_decoder_input(memory, gt_idx_padded, valid_mask)
        bos_valid = torch.ones(bsz, 1, dtype=torch.bool, device=device)
        tgt_key_padding_mask = ~torch.cat([bos_valid, valid_mask], dim=1)
        q = self._decode(memory, dec_in, tgt_key_padding_mask)
        logits = self._pointer_logits(q, memory)  # (B, T, N + 1)

        # Per-slot "seen so far" bag mask, vectorised via cumsum of one-hots.
        # seen[b, 0] is empty; seen[b, t] for t >= 1 is the union of idx_0..idx_{t-1}.
        if k_max > 0:
            perm_safe = gt_idx_padded.clamp_min(0)
            one_hot = torch.zeros(bsz, k_max, n, dtype=dtype, device=device)
            one_hot.scatter_(2, perm_safe.unsqueeze(-1), 1.0)
            one_hot = one_hot * valid_mask.unsqueeze(-1).to(dtype)
            cumsum_inclusive = one_hot.cumsum(dim=1)
            seen_per_step = torch.cat([torch.zeros(bsz, 1, n, dtype=dtype, device=device), cumsum_inclusive], dim=1)
        else:
            seen_per_step = torch.zeros(bsz, T, n, dtype=dtype, device=device)
        seen_bag = (seen_per_step > 0.5).to(dtype)

        eos_zero = torch.zeros(bsz, T, 1, dtype=dtype, device=device)
        block = torch.cat([seen_bag, eos_zero], dim=-1)
        masked_logits = logits + BIG_NEG * block

        # Targets: idx_t at slot t < K_b; EOS at slot K_b; ignored beyond.
        k_per_row = valid_mask.sum(dim=1).long()
        arange_T = torch.arange(T, device=device).unsqueeze(0)
        query_valid = arange_T <= k_per_row.unsqueeze(1)  # (B, T)
        targets = torch.full((bsz, T), self.eos_index, dtype=torch.long, device=device)
        if k_max > 0:
            targets[:, :k_max] = torch.where(
                valid_mask,
                gt_idx_padded.clamp_min(0),
                torch.full_like(gt_idx_padded, self.eos_index),
            )

        flat_logits = masked_logits.reshape(-1, masked_logits.shape[-1])
        flat_targets = targets.reshape(-1)
        flat_valid = query_valid.reshape(-1)
        if not bool(flat_valid.any().item()):
            return torch.zeros((), device=device, dtype=dtype)
        valid_logits = flat_logits[flat_valid]
        valid_targets = flat_targets[flat_valid]

        if self.loss_type == "one_hot_prob":
            return self._one_hot_probability_loss(valid_logits, valid_targets)
        return self._index_cross_entropy_loss(valid_logits, valid_targets)

    @staticmethod
    def _one_hot_probability_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Categorical NLL from softmax probabilities and one-hot targets."""
        probs = F.softmax(logits, dim=-1)
        target_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(dtype=probs.dtype)
        log_probs = torch.log(probs.clamp_min(1e-9))
        per_step_nll = -(target_one_hot * log_probs).sum(dim=-1)
        return per_step_nll.mean()

    @staticmethod
    def _index_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, reduction="mean")

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def greedy_decode(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Batched autoregressive greedy decode masking already-selected cells.

        At step ``t`` we re-run the decoder on the growing query sequence (BOS
        plus the encoder embeddings of all previously selected cells), take the
        last query's pointer-network logits, mask already-selected bag slots
        with ``BIG_NEG`` (EOS slot stays open), and pick the argmax. Stops
        when EOS wins or the per-row step budget is exhausted.
        """
        memory = outputs["memory"]
        if memory.ndim != 3:
            raise ValueError(f"Expected memory shape (B, N, C), got {tuple(memory.shape)}")
        bsz, n, _ = memory.shape
        device = memory.device

        selected = torch.zeros(bsz, n, dtype=torch.bool, device=device)
        history = torch.zeros(bsz, self.max_steps, dtype=torch.long, device=device)
        history_valid = torch.zeros(bsz, self.max_steps, dtype=torch.bool, device=device)
        done = torch.zeros(bsz, dtype=torch.bool, device=device)

        pred_flat = torch.zeros(bsz, n, dtype=torch.float32, device=device)
        order_flat = torch.zeros(bsz, n, dtype=torch.float32, device=device)
        chosen: list[list[int]] = [[] for _ in range(bsz)]

        for t in range(self.max_steps):
            if bool(done.all().item()):
                break
            prev_idx = history[:, :t]
            prev_valid = history_valid[:, :t]
            dec_in = self._build_decoder_input(memory, prev_idx, prev_valid)
            bos_valid = torch.ones(bsz, 1, dtype=torch.bool, device=device)
            tgt_key_padding_mask = ~torch.cat([bos_valid, prev_valid], dim=1)
            q = self._decode(memory, dec_in, tgt_key_padding_mask)
            last_logits = self._pointer_logits(q[:, -1:, :], memory).squeeze(1).clone()  # (B, N + 1)
            last_logits[:, :n] = last_logits[:, :n] + BIG_NEG * selected.to(last_logits.dtype)

            winners = last_logits.argmax(dim=1)
            for b in range(bsz):
                if bool(done[b].item()):
                    continue
                j = int(winners[b].item())
                if j == self.eos_index or bool(selected[b, j].item()):
                    done[b] = True
                    continue
                selected[b, j] = True
                history[b, t] = j
                history_valid[b, t] = True
                step = len(chosen[b]) + 1
                pred_flat[b, j] = 1.0
                order_flat[b, j] = float(step)
                chosen[b].append(j)

        indices = [torch.tensor(o, dtype=torch.long, device=device) for o in chosen]
        return {
            "mask": pred_flat.view(bsz, self.grid_size, self.grid_size),
            "order": order_flat.view(bsz, self.grid_size, self.grid_size),
            "indices": indices,
        }

    # ------------------------------------------------------------------
    # GT extraction
    # ------------------------------------------------------------------

    def _build_target_batch(
        self,
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
        *,
        D: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ``(B, M_MAX, 2)`` GT points into ``(B, K_max)`` cell-index permutations.

        Returns ``(gt_idx_padded, valid_mask)`` where ``gt_idx_padded[b, k]`` is
        the k-th cell on path ``b`` (clamped to the grid) and
        ``valid_mask[b, k] = (k < K_b)``.
        """
        bsz = gt_points.shape[0]
        seqs: list[torch.Tensor] = []
        max_len = 0
        for b in range(bsz):
            idx = self._gt_indices_from_points(gt_points[b], num_points[b], D=D)
            if idx.numel() > self.max_steps:
                idx = idx[: self.max_steps]
            seqs.append(idx.to(device=device))
            max_len = max(max_len, int(idx.numel()))

        gt_idx_padded = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
        valid_mask = torch.zeros(bsz, max_len, dtype=torch.bool, device=device)
        for b, idx in enumerate(seqs):
            n_b = int(idx.numel())
            if n_b > 0:
                gt_idx_padded[b, :n_b] = idx
                valid_mask[b, :n_b] = True
        return gt_idx_padded, valid_mask

    def _gt_indices_from_points(
        self,
        gt_points_b: torch.Tensor,
        num_points_b: torch.Tensor,
        *,
        D: int,
    ) -> torch.Tensor:
        n = int(num_points_b.item())
        if n <= 0:
            return torch.empty(0, dtype=torch.long, device=gt_points_b.device)
        pts = gt_points_b[:n]
        cells = torch.floor(pts / float(D)).to(torch.long)
        j = cells[:, 0].clamp_(0, self.grid_size - 1)
        i = cells[:, 1].clamp_(0, self.grid_size - 1)
        flat = i * self.grid_size + j

        # Order-preserving deduplication (defensive; the sparsifier already enforces it).
        seen: set[int] = set()
        keep = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
        for k, idx in enumerate(flat.detach().cpu().tolist()):
            if idx not in seen:
                seen.add(idx)
                keep[k] = True
        return flat[keep]
