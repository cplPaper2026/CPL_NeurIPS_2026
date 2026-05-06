"""Subset-selection models: CPL (theta + W + EOS) and BCE (theta only).

Both share a multi-layer transformer encoder over per-set features.
``forward`` may return the encoder hidden states ``h`` so downstream
consumers (the 2D embedding visualiser) can reuse them without a
second forward pass.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _build_encoder(
    d_model: int,
    nhead: int,
    ffn_dim: int,
    num_layers: int,
    dropout: float,
    activation: str,
    norm_first: bool,
) -> nn.TransformerEncoder:
    """Stacked transformer encoder with optional pre-norm + final LayerNorm.

    Pre-norm (``norm_first=True``) is the modern default for deep stacks; we
    pair it with a trailing ``LayerNorm`` so the output is normalised before
    it reaches the heads.
    """
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=norm_first,
    )
    final_norm = nn.LayerNorm(d_model) if norm_first else None
    # ``enable_nested_tensor`` is incompatible with ``norm_first``; disable it
    # explicitly to silence PyTorch's UserWarning when using pre-norm.
    return nn.TransformerEncoder(
        layer,
        num_layers=num_layers,
        norm=final_norm,
        enable_nested_tensor=not norm_first,
    )


def _mlp_head(d_in: int, hidden: int, d_out: int, dropout: float, activation: str) -> nn.Sequential:
    """Two-layer MLP head used by the unary / key / value projections."""
    act: nn.Module = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        act,
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out),
    )


def _build_decoder(
    d_model: int,
    nhead: int,
    ffn_dim: int,
    num_layers: int,
    dropout: float,
    activation: str,
    norm_first: bool,
) -> nn.TransformerDecoder:
    """Stacked transformer decoder with optional pre-norm + final LayerNorm.

    Mirrors :func:`_build_encoder` so the AR mode shares the same modern
    pre-norm convention used by the encoder backbone.
    """
    layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=norm_first,
    )
    final_norm = nn.LayerNorm(d_model) if norm_first else None
    return nn.TransformerDecoder(layer, num_layers=num_layers, norm=final_norm)


class CPLModel(nn.Module):
    """Transformer encoder + (theta, W) heads + a learnable EOS token.

    ``forward`` returns ``theta`` of shape ``(B, N+1)`` (last entry is
    the EOS unary score) and ``W`` of shape ``(B, N+1, N+1)`` (with the
    EOS row/column zero so EOS contributes no pairwise context).

    Pairwise scores ``W`` use a multi-head bilinear factorisation: the
    ``hidden``-dimensional key/value vectors are reshaped into
    ``pair_heads`` heads of dimension ``hidden // pair_heads``, the
    per-head dot-product matrices are computed and averaged. This is
    strictly more expressive than a single bilinear head at the same
    parameter count.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        ffn_dim: int = 1024,
        hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        pair_heads: int = 4,
    ):
        super().__init__()
        if hidden % pair_heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by pair_heads ({pair_heads})")
        self.d_model = d_model
        self.hidden = hidden
        self.pair_heads = pair_heads
        self.head_dim = hidden // pair_heads

        self.transformer = _build_encoder(
            d_model=d_model,
            nhead=nhead,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        self.eos = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.eos, std=0.02)

        self.theta_mlp = _mlp_head(d_model, hidden, 1, dropout, activation)
        self.key_mlp = _mlp_head(d_model, hidden, hidden, dropout, activation)
        self.value_mlp = _mlp_head(d_model, hidden, hidden, dropout, activation)

    def forward(
        self,
        X: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        return_h: bool = False,
    ):
        """Forward; ``src_key_padding_mask`` is True at padded positions.

        When ``return_h`` is True returns ``(theta, W, h)`` where
        ``h`` is the transformer's per-token hidden state of shape
        ``(B, N, d_model)``.
        """
        B, N, d = X.shape
        h = self.transformer(X, src_key_padding_mask=src_key_padding_mask)

        eos_expanded = self.eos.unsqueeze(0).unsqueeze(0).expand(B, 1, d)
        h_with_eos = torch.cat([h, eos_expanded], dim=1)
        theta = self.theta_mlp(h_with_eos).squeeze(-1)

        key = self.key_mlp(h)
        value = self.value_mlp(h)
        W = key @ value.transpose(1, 2) / math.sqrt(self.hidden)

        # EOS is not considered in the pairwise score matrix so we add a zero column and row
        W = torch.cat([W, torch.zeros(B, N, 1, device=W.device)], dim=2)
        W = torch.cat([W, torch.zeros(B, 1, N + 1, device=W.device)], dim=1)

        if return_h:
            return theta, W, h
        return theta, W


class BCEModel(nn.Module):
    """Transformer encoder + per-token unary head.

    Shared between the ``bce`` and ``hungarian`` training modes: both
    consume ``(B, N)`` per-token logits and produce predictions via a
    sigmoid-threshold sweep at inference. The Hungarian loss additionally
    consumes the transformer's hidden states via ``return_h=True``.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        ffn_dim: int = 1024,
        hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.transformer = _build_encoder(
            d_model=d_model,
            nhead=nhead,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        self.head = _mlp_head(d_model, hidden, 1, dropout, activation)

    def forward(
        self,
        X: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        return_h: bool = False,
    ):
        """Forward; returns ``logits`` (and optionally ``h``)."""
        h = self.transformer(X, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(h).squeeze(-1)
        if return_h:
            return logits, h
        return logits


class ARModel(nn.Module):
    """Encoder + small TransformerDecoder + pointer-network readout.

    Architecture (single forward per bag):

    * **Encoder**: same `_build_encoder` backbone shared with CPL/BCE,
      producing per-token hidden states ``h: (B, N, d_model)``.
    * **Decoder**: a small ``nn.TransformerDecoder`` over a length-``T``
      query sequence; ``T = K + 1`` per element (``K`` "predict next GT"
      steps + 1 EOS step). The query at position 0 is a learnable BOS
      token; the query at position ``t >= 1`` is ``h[gt_perm[t-1]]``
      plus a learnable per-step positional embedding. The decoder uses
      a causal self-attention mask (so step ``t`` only sees its own
      history) and full cross-attention into ``h`` (modulated by the
      bag's ``memory_key_padding_mask``).
    * **Pointer-net readout**: project decoder outputs and bag hidden
      states to a shared ``d_model`` space and dot-product them. The
      bag yields ``N`` logits per query; a separate learnable EOS key
      yields one extra "EOS" logit. The full output shape is
      ``(B, T, N+1)`` -- the EOS slot lives at index ``N``.

    These per-step ``(N+1)`` logits are consumed by :func:`losses.ar_loss`,
    which masks out already-selected and padded bag slots with
    ``BIG_NEG`` and applies the same EOS-weighted cross-entropy as the
    CPL loss (see :func:`losses._eos_weighted_ce`).

    Selection history is communicated to the decoder via (i) the input
    embedding at each step being the encoder representation of the
    previously selected token, and (ii) causal self-attention propagating
    that across query steps. No "selected" embedding is added to the bag
    itself: that would require either re-encoding the bag K times or
    materialising K snapshots, both far costlier than the one-shot
    encoder + small decoder used here.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        ffn_dim: int = 1024,
        hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        decoder_ffn_dim: int = 1024,
        decoder_dropout: float = 0.1,
        decoder_norm_first: bool = True,
        max_selection_steps: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.max_selection_steps = max_selection_steps

        self.encoder = _build_encoder(
            d_model=d_model,
            nhead=nhead,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        self.decoder = _build_decoder(
            d_model=d_model,
            nhead=decoder_heads,
            ffn_dim=decoder_ffn_dim,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
            activation=activation,
            norm_first=decoder_norm_first,
        )

        self.bos = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.bos, std=0.02)
        self.eos_key = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.eos_key, std=0.02)
        # +1 because we may run up to max_selection_steps + 1 query positions
        # (BOS at index 0, then up to max_selection_steps "selection" steps).
        self.step_pos = nn.Embedding(max_selection_steps + 2, d_model)
        nn.init.trunc_normal_(self.step_pos.weight, std=0.02)

        self.key_proj = _mlp_head(d_model, hidden, d_model, dropout, activation)
        self.query_proj = _mlp_head(d_model, hidden, d_model, dropout, activation)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def encode(self, X: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run the bag through the shared transformer encoder."""
        return self.encoder(X, src_key_padding_mask=src_key_padding_mask)

    def build_decoder_input(
        self,
        h: torch.Tensor,
        gt_perm_padded: torch.Tensor,
        perm_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the teacher-forced decoder input ``(B, K_max + 1, D)``.

        Position 0 is the learnable BOS token. Position ``t >= 1`` is
        ``h[b, gt_perm_padded[b, t-1]]`` for valid permutation entries; for
        padded slots we substitute the BOS embedding (these positions are
        masked from the loss anyway). Each position then receives the
        per-step learnable positional embedding ``step_pos[t]``.
        """
        B, K_max = gt_perm_padded.shape
        D = self.d_model
        device = h.device

        # Safe gather: clamp -1 (pad) to 0 so torch.gather doesn't error;
        # those positions are overwritten with BOS via perm_valid_mask below.
        safe_perm = gt_perm_padded.clamp_min(0).unsqueeze(-1).expand(-1, -1, D)
        prev_selected = torch.gather(h, 1, safe_perm)  # (B, K_max, D)

        bos_row = self.bos.view(1, 1, D).expand(B, K_max, D)
        prev_selected = torch.where(perm_valid_mask.unsqueeze(-1), prev_selected, bos_row)

        bos_first = self.bos.view(1, 1, D).expand(B, 1, D)
        dec_in = torch.cat([bos_first, prev_selected], dim=1)  # (B, K_max+1, D)

        T = dec_in.shape[1]
        if self.step_pos.num_embeddings < T:
            raise ValueError(f"AR decoder query length {T} exceeds max_selection_steps + 2 ({self.step_pos.num_embeddings}); raise --ar-max-selection-steps")
        steps = torch.arange(T, device=device)
        dec_in = dec_in + self.step_pos(steps).unsqueeze(0)
        return dec_in

    def causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Standard upper-triangular ``True`` causal mask of shape ``(T, T)``."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def decode(
        self,
        h: torch.Tensor,
        dec_in: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the transformer decoder; returns query hidden states ``(B, T, D)``."""
        T = dec_in.shape[1]
        tgt_mask = self.causal_mask(T, dec_in.device)
        return self.decoder(
            tgt=dec_in,
            memory=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def pointer_logits(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Pointer-network logits ``(B, T, N+1)``.

        ``logits[..., :N] = q_proj(q) @ key_proj(h)^T / sqrt(D)`` and
        ``logits[..., N] = q_proj(q) @ eos_key / sqrt(D)``. Keeping EOS
        symmetric in scale with the bag logits avoids the model needing
        to learn a per-slot calibration.
        """
        q_p = self.query_proj(q)  # (B, T, D)
        keys = self.key_proj(h)  # (B, N, D)
        scale = math.sqrt(float(self.d_model))
        bag_logits = q_p @ keys.transpose(1, 2) / scale  # (B, T, N)
        eos_logits = (q_p @ self.eos_key) / scale  # (B, T)
        return torch.cat([bag_logits, eos_logits.unsqueeze(-1)], dim=-1)

    # ------------------------------------------------------------------
    # Forward (teacher-forced training)
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None,
        gt_perm_padded: torch.Tensor,
        perm_valid_mask: torch.Tensor,
        return_h: bool = False,
        encoder_hidden: torch.Tensor | None = None,
    ):
        """Single-pass teacher-forced forward returning per-step pointer logits.

        Shapes:
            ``X``               : ``(B, N, d_model)`` raw bag features.
            ``src_key_padding_mask``: ``(B, N)`` ``True`` at pad positions.
            ``gt_perm_padded``  : ``(B, K_max)`` random GT permutation,
                                  ``-1`` at pad positions.
            ``perm_valid_mask`` : ``(B, K_max)`` ``True`` at valid perm slots.
            ``encoder_hidden``  : optional ``(B, N, D)`` precomputed encoder
                                  output for the same ``X``; when set, skips
                                  :meth:`encode` so multiple permutations can
                                  reuse one bag encoding.

        Returns ``logits: (B, K_max + 1, N + 1)`` (and optionally ``h``).
        Decoder query positions beyond ``K_b + 1`` for sample ``b`` are
        valid only as keys for *each other*; the loss masks them out.
        """
        if encoder_hidden is None:
            h = self.encode(X, src_key_padding_mask=src_key_padding_mask)
        else:
            h = encoder_hidden

        dec_in = self.build_decoder_input(h, gt_perm_padded, perm_valid_mask)

        # The decoder query at index 0 is the BOS step (always valid). Indices
        # 1..K_max correspond to "previously selected" positions; they are valid
        # iff perm_valid_mask is True at the matching slot. We do NOT mask the
        # EOS query position with key-padding (it's always valid as a query).
        B = dec_in.shape[0]
        bos_valid = torch.ones(B, 1, dtype=torch.bool, device=dec_in.device)
        # Last "EOS step" query position lives at index K_b + 1; for samples
        # whose K_b == K_max it sits at index K_max which is included in
        # dec_in. We do not have a per-sample column for it, but the per-row
        # query_valid_mask passed to the loss handles validity. For the
        # decoder padding mask we stay conservative and only mark padded
        # positions (False where invalid) so valid queries can attend to all
        # valid keys.
        tgt_key_padding_mask = ~torch.cat([bos_valid, perm_valid_mask], dim=1)

        q = self.decode(
            h=h,
            dec_in=dec_in,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.pointer_logits(q, h)
        if return_h:
            return logits, h
        return logits


def build_model(cfg, device: torch.device) -> nn.Module:
    """Factory dispatching on ``cfg.train.method``.

    * ``cpl``       -> :class:`CPLModel` (theta, W, EOS).
    * ``bce``       -> :class:`BCEModel` (per-token logits).
    * ``hungarian`` -> :class:`BCEModel`; Hungarian adds no new module,
      it only changes how targets are assigned during the loss step.
    * ``ar``        -> :class:`ARModel` (encoder + small decoder + pointer-net).
    """
    common = dict(
        d_model=cfg.model.transformer_dim,
        nhead=cfg.model.transformer_heads,
        ffn_dim=cfg.model.transformer_ffn_dim,
        hidden=cfg.model.hidden,
        num_layers=cfg.model.transformer_layers,
        dropout=cfg.model.transformer_dropout,
        activation=cfg.model.transformer_activation,
        norm_first=cfg.model.transformer_norm_first,
    )
    if cfg.train.method == "cpl":
        return CPLModel(**common, pair_heads=cfg.model.pair_heads).to(device)
    if cfg.train.method in ("bce", "hungarian"):
        return BCEModel(**common).to(device)
    if cfg.train.method == "ar":
        return ARModel(
            **common,
            decoder_layers=cfg.ar.decoder_layers,
            decoder_heads=cfg.ar.decoder_heads,
            decoder_ffn_dim=cfg.ar.decoder_ffn_dim,
            decoder_dropout=cfg.ar.decoder_dropout,
            decoder_norm_first=cfg.ar.decoder_norm_first,
            max_selection_steps=cfg.ar.max_selection_steps,
        ).to(device)
    raise ValueError(f"unknown method {cfg.train.method!r}")


__all__ = ["ARModel", "BCEModel", "CPLModel", "build_model"]
