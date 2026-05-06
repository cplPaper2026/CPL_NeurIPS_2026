import math
from typing import Any

import torch
from ar_module import ARDecoderHead
from cpl_module import CPLHead
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from train_config import ModelConfig


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding, as commonly used in research
    (e.g., see "Attention Is All You Need" and Vision Transformer papers).

    Encodes each (y, x) coordinate as a vector of sines and cosines at multiple frequencies.
    """

    def __init__(self, hidden_dim: int, grid_h: int, grid_w: int, **kwargs) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4 for 2D sinusoidal encoding.")
        self.hidden_dim = hidden_dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Precompute the positional encodings and register as a buffer
        pos_enc = self._build_encoding(grid_h, grid_w, hidden_dim)
        # (1, C, H, W) shape to broadcast along batch
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_embed', pos_enc, persistent=False)

    @staticmethod
    def _build_encoding(H, W, C):
        # H, W = spatial grid
        # C = hidden_dim, C must be divisible by 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c_each = C // 2  # Divide embedding evenly between y and x
        div_term = torch.exp(torch.arange(0, c_each, 2, dtype=torch.float32) * (-math.log(10000.0) / c_each))  # (c_each // 2,)
        div_term = div_term.to(device)

        # Create (H, W) coordinate meshgrid
        y_pos = torch.arange(H, dtype=torch.float32, device=device)
        x_pos = torch.arange(W, dtype=torch.float32, device=device)
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')  # each (H, W)

        # Y encoding
        pe_y = torch.zeros(H, W, c_each)
        pe_y[..., 0::2] = torch.sin(y_grid.unsqueeze(-1) * div_term)
        pe_y[..., 1::2] = torch.cos(y_grid.unsqueeze(-1) * div_term)

        # X encoding
        pe_x = torch.zeros(H, W, c_each)
        pe_x[..., 0::2] = torch.sin(x_grid.unsqueeze(-1) * div_term)
        pe_x[..., 1::2] = torch.cos(x_grid.unsqueeze(-1) * div_term)

        # Concatenate along channel
        pos_emb = torch.cat([pe_y, pe_x], dim=-1)  # (H, W, C)
        pos_emb = pos_emb.permute(2, 0, 1).contiguous()  # (C, H, W)
        return pos_emb.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return x + self.pos_embed


class GenerativePathForecaster(nn.Module):
    """
    Core Vision Transformer architecture for path prediction.

    Takes a 2-channel road map + ego pose and outputs a 3-channel per-cell
    target on the 32x32 grid:

    * Channel 0 (heatmap): a single logit per cell (no sigmoid) to be supervised
      with ``BCEWithLogitsLoss`` against the binary occupancy of the GT path.
    * Channels 1-2 (offsets): raw linear pixel offsets ``(dx, dy)`` from the
      cell's top-left corner to the predicted sub-pixel point. The full-res
      predicted point at cell ``(i, j)`` is
      ``(j*D + dx[i, j], i*D + dy[i, j])``.

    A shared embedding grid is also exposed for the optional sequence heads
    (CPL or AR decoder).
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        *,
        use_cpl: bool,
        use_ar: bool = False,
        use_multi_hypothesis: bool = False,
    ) -> None:
        super().__init__()
        if sum(int(x) for x in (use_cpl, use_ar, use_multi_hypothesis)) > 1:
            raise ValueError("use_cpl, use_ar, and use_multi_hypothesis are mutually exclusive.")

        self.cfg = model_cfg
        hidden_dim = model_cfg.hidden_dim
        grid_size = model_cfg.grid_size
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.use_cpl = use_cpl
        self.use_ar = use_ar
        self.use_multi_hypothesis = use_multi_hypothesis
        self.num_hypotheses = int(model_cfg.multi_hypothesis.num_hypotheses)

        # 1. Feature Extractor
        backbone_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone_resnet.children())[:-4])

        # 2. Spatial Positional Encoding
        self.pos_encoding = PositionalEncoding2D(hidden_dim, grid_size, grid_size, init_std=model_cfg.pos_init_std)

        # 3. Transformer Encoder
        ffn = hidden_dim * model_cfg.ffn_dim_multiplier
        act = (model_cfg.transformer_activation or "gelu").lower()
        if act not in ("gelu", "relu"):
            msg = f"transformer_activation must be gelu or relu, got {model_cfg.transformer_activation!r}"
            raise ValueError(msg)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=model_cfg.num_heads,
            dim_feedforward=ffn,
            dropout=model_cfg.transformer_dropout,
            activation=act,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=model_cfg.num_layers)

        # 4. Output heads (per token).
        # Heatmap: 1 raw logit per cell (no activation; the loss applies sigmoid).
        # Offset:  2 raw linear values per cell (dx, dy in full-res pixels).
        h_hidden = int(model_cfg.heatmap_head.heatmap_hidden)
        o_hidden = int(model_cfg.heatmap_head.offset_hidden)
        self.heatmap_head = nn.Sequential(
            nn.Linear(hidden_dim, h_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(h_hidden, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, o_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(o_hidden, 2),
        )

        self.cpl_head = (
            CPLHead(
                model_cfg.cpl_head,
                hidden_dim=hidden_dim,
                grid_size=grid_size,
            )
            if use_cpl
            else None
        )
        self.ar_head = (
            ARDecoderHead(
                model_cfg.ar_head,
                hidden_dim=hidden_dim,
                grid_size=grid_size,
            )
            if use_ar
            else None
        )

        # 5. Multi-hypothesis modules (only when MH is enabled).
        # See ``forward`` for how these compose with the shared heatmap/offset heads.
        self._build_multi_hypothesis_modules(hidden_dim, model_cfg.pos_init_std)

    def _build_multi_hypothesis_modules(self, hidden_dim: int, init_std: float) -> None:
        """Instantiate the K-way summary token, queries, and fuse layers for MH."""
        if not self.use_multi_hypothesis:
            self.hypothesis_summary_token = None
            self.hypothesis_queries = None
            self.hypothesis_summary_fuse = None
            self.spatial_hypothesis_fuse = None
            self.hypothesis_logit_proj = None
            return
        # Single learnable token appended to spatial tokens (fed into the encoder).
        self.hypothesis_summary_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.hypothesis_summary_token, mean=0.0, std=float(init_std))
        # K learnable vectors used only after the encoder (never attend in encoder).
        self.hypothesis_queries = nn.Parameter(torch.zeros(1, self.num_hypotheses, hidden_dim))
        nn.init.normal_(self.hypothesis_queries, mean=0.0, std=float(init_std))
        # Encoder summary output concat each hypothesis query -> z_k.
        self.hypothesis_summary_fuse = nn.Linear(2 * hidden_dim, hidden_dim)
        # Each spatial token concat z_k -> fused token for shared heads.
        self.spatial_hypothesis_fuse = nn.Linear(2 * hidden_dim, hidden_dim)
        # One scalar selector logit per hypothesis from z_k.
        self.hypothesis_logit_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        """Run the backbone, transformer, and per-cell heads.

        Args:
            x: Tensor of shape ``(B, 2, H, W)``. Channel 0 is the road map,
               channel 1 is the ego heatmap. With ``H = W = 256`` and
               backbone stride 8, the per-cell grid is 32x32.

        Returns:
            Dictionary with:

            * ``heatmap_logits``: ``(B, 32, 32)`` raw logits (no sigmoid).
            * ``offsets``: ``(B, 2, 32, 32)`` -- channel 0 is dx, channel 1 is dy.
            * ``embeddings``: ``(B, hidden_dim, 32, 32)``.
            * ``cpl``: Optional dict with CPL components (``theta``, ``W_dense``).
            * ``ar``: Optional dict with AR decoder memory tokens.
            * Multi-hypothesis extras (always present, ``None`` when MH is off):
              ``heatmap_logits_all`` ``(B, K, 32, 32)``,
              ``offsets_all`` ``(B, K, 2, 32, 32)``,
              ``hypothesis_logits`` / ``hypothesis_probs`` ``(B, K)``,
              ``selected_hypothesis`` ``(B,)``.
        """
        B = x.size(0)

        # images, directions = x.split(1, dim=1)
        # x_3ch = torch.concat([images, images, directions], dim=1)
        x_3ch = x
        features = self.backbone(x_3ch)
        features = self.pos_encoding(features)

        spatial_tokens = features.view(B, self.hidden_dim, -1).permute(0, 2, 1)  # (B, N, C)

        if self.use_multi_hypothesis:
            (
                spatial_embeddings,
                heatmap_logits,
                offsets,
                mh_extras,
            ) = self._forward_multi_hypothesis(spatial_tokens)
        else:
            spatial_embeddings = self.transformer(spatial_tokens)
            heatmap_logits, offsets = self._heads_from_tokens(spatial_embeddings, B)
            mh_extras = {
                "heatmap_logits_all": None,
                "offsets_all": None,
                "hypothesis_logits": None,
                "hypothesis_probs": None,
                "selected_hypothesis": None,
            }

        embeddings_grid = spatial_embeddings.permute(0, 2, 1).contiguous().view(B, self.hidden_dim, self.grid_size, self.grid_size)

        cpl_outputs = self.cpl_head(embeddings_grid) if self.cpl_head is not None else None
        ar_outputs = self.ar_head(embeddings_grid) if self.ar_head is not None else None
        return {
            "heatmap_logits": heatmap_logits,
            "offsets": offsets,
            "embeddings": embeddings_grid,
            "cpl": cpl_outputs,
            "ar": ar_outputs,
            **mh_extras,
        }

    def _heads_from_tokens(
        self,
        embeddings: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the shared heatmap/offset heads to ``(B, N, C)`` tokens."""
        heatmap_logits_tok = self.heatmap_head(embeddings).squeeze(-1)  # (B, N)
        offsets_tok = self.offset_head(embeddings)  # (B, N, 2)
        heatmap_logits = heatmap_logits_tok.view(batch_size, self.grid_size, self.grid_size)
        offsets = offsets_tok.permute(0, 2, 1).contiguous().view(batch_size, 2, self.grid_size, self.grid_size)
        return heatmap_logits, offsets

    def _forward_multi_hypothesis(
        self,
        spatial_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        """K-way multi-hypothesis path: encoder + per-hypothesis fusion + selection.

        Returns ``(spatial_embeddings, selected_heatmap, selected_offsets, mh_extras)``
        where ``mh_extras`` carries the full ``(B, K, ...)`` tensors plus selector
        logits/probs and the chosen index per sample.
        """
        if (
            self.hypothesis_summary_token is None
            or self.hypothesis_queries is None
            or self.hypothesis_summary_fuse is None
            or self.spatial_hypothesis_fuse is None
            or self.hypothesis_logit_proj is None
        ):
            raise RuntimeError("Multi-hypothesis mode requires summary token, queries, and fusion modules.")

        B, n_spatial, _ = spatial_tokens.shape
        K = self.num_hypotheses
        summary_in = self.hypothesis_summary_token.expand(B, 1, -1)  # (B, 1, C)
        encoder_input = torch.cat([spatial_tokens, summary_in], dim=1)  # (B, N+1, C)
        encoded = self.transformer(encoder_input)
        spatial_embeddings = encoded[:, :-1, :]  # (B, N, C)
        # Keep length-1 seq dim so we can broadcast to K without unsqueeze.
        summary_seq = encoded[:, -1:, :]  # (B, 1, C)

        hyp_queries = self.hypothesis_queries.expand(B, K, -1)  # (B, K, C)
        summary_k = summary_seq.expand(-1, K, -1)  # (B, K, C)
        z_k = self.hypothesis_summary_fuse(torch.cat([summary_k, hyp_queries], dim=-1))  # (B, K, C)

        hypothesis_logits = self.hypothesis_logit_proj(z_k).squeeze(-1)  # (B, K)
        hypothesis_probs = torch.softmax(hypothesis_logits, dim=-1)
        selected_hypothesis = hypothesis_probs.argmax(dim=-1)  # (B,)

        # For each k: concat spatial token with z_k, project, shared heads
        # -> (B, K, N) heatmap logits and (B, K, N, 2) offsets.
        zk_n = z_k.unsqueeze(2).expand(-1, -1, n_spatial, -1)  # (B, K, N, C)
        spat_k = spatial_embeddings.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, N, C)
        tok_fused = self.spatial_hypothesis_fuse(torch.cat([spat_k, zk_n], dim=-1))  # (B, K, N, C)
        # Linear layers only touch the last dim; no need to merge batch/K/N axes.
        heatmap_logits_tok = self.heatmap_head(tok_fused).squeeze(-1)  # (B, K, N)
        offsets_tok = self.offset_head(tok_fused)  # (B, K, N, 2)

        heatmap_logits_all = heatmap_logits_tok.view(B, K, self.grid_size, self.grid_size)
        offsets_all = offsets_tok.permute(0, 1, 3, 2).contiguous().view(B, K, 2, self.grid_size, self.grid_size)

        batch_idx = torch.arange(B, device=spatial_tokens.device)
        heatmap_logits = heatmap_logits_all[batch_idx, selected_hypothesis]  # (B, H, W)
        offsets = offsets_all[batch_idx, selected_hypothesis]  # (B, 2, H, W)

        mh_extras: dict[str, torch.Tensor | None] = {
            "heatmap_logits_all": heatmap_logits_all,
            "offsets_all": offsets_all,
            "hypothesis_logits": hypothesis_logits,
            "hypothesis_probs": hypothesis_probs,
            "selected_hypothesis": selected_hypothesis,
        }
        return spatial_embeddings, heatmap_logits, offsets, mh_extras

    def cpl_loss(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
        meta: list[dict[str, Any]],
        *,
        D: int,
    ) -> torch.Tensor:
        """Compute the CPL ordering loss from per-sample GT point lists.

        Args:
            outputs: Output of :meth:`forward`.
            gt_points: ``(B, M_MAX, 2)`` zero-padded ``(x, y)`` GT points in pixels.
            num_points: ``(B,) long`` valid point count per sample.
            meta: per-sample metadata dictionaries.
            D: full-res-to-grid downsample factor (cell size in pixels).
        """
        if self.cpl_head is None:
            raise RuntimeError("CPL loss requested but CPL head is disabled.")
        cpl_outputs = outputs["cpl"]
        if cpl_outputs is None:
            raise RuntimeError("Model outputs do not contain CPL components.")
        return self.cpl_head.loss(cpl_outputs, gt_points, num_points, meta, D=D)

    def ar_loss(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
        meta: list[dict[str, Any]],
        *,
        D: int,
    ) -> torch.Tensor:
        """Compute the AR next-token loss from per-sample GT point lists."""
        if self.ar_head is None:
            raise RuntimeError("AR loss requested but AR head is disabled.")
        ar_outputs = outputs["ar"]
        if ar_outputs is None:
            raise RuntimeError("Model outputs do not contain AR components.")
        return self.ar_head.loss(ar_outputs, gt_points, num_points, meta, D=D)

    @torch.no_grad()
    def cpl_greedy_decode(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
    ) -> dict[str, torch.Tensor]:
        """CPL greedy decode (cell-level).

        Returns a dict with:

        * ``mask``: ``(B, H, W)`` binary cell occupancy.
        * ``order``: ``(B, H, W)`` selection step 1..K (0 = unselected).
        * ``indices``: list (length ``B``) of 1D tensors with chosen flat cell
          indices in selection order; useful for combining with offsets.
        """
        if self.cpl_head is None:
            raise RuntimeError("CPL decode requested but CPL head is disabled.")
        cpl_outputs = outputs["cpl"]
        if cpl_outputs is None:
            raise RuntimeError("Model outputs do not contain CPL components.")
        return self.cpl_head.greedy_decode(cpl_outputs)

    @torch.no_grad()
    def ar_greedy_decode(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
    ) -> dict[str, torch.Tensor]:
        """AR greedy decode (cell-level) with the CPL-compatible output format."""
        if self.ar_head is None:
            raise RuntimeError("AR decode requested but AR head is disabled.")
        ar_outputs = outputs["ar"]
        if ar_outputs is None:
            raise RuntimeError("Model outputs do not contain AR components.")
        return self.ar_head.greedy_decode(ar_outputs)


if __name__ == "__main__":
    model = GenerativePathForecaster(ModelConfig(), use_cpl=True, use_ar=False)
    dummy_input = torch.zeros(4, 2, 256, 256)
    outputs = model(dummy_input)

    print(f"Heatmap logits: {outputs['heatmap_logits'].shape}")  # [4, 32, 32]
    print(f"Offsets: {outputs['offsets'].shape}")  # [4, 2, 32, 32]
    print(f"Embeddings: {outputs['embeddings'].shape}")  # [4, 128, 32, 32]
    if outputs["cpl"] is not None:
        print(f"CPL theta: {outputs['cpl']['theta'].shape}")

    mh_cfg = ModelConfig()
    mh_cfg.multi_hypothesis.enabled = True
    mh_model = GenerativePathForecaster(mh_cfg, use_cpl=False, use_multi_hypothesis=True)
    mh_outputs = mh_model(dummy_input)
    print(f"[MH] heatmap_logits_all: {mh_outputs['heatmap_logits_all'].shape}")  # [4, K, 32, 32]
    print(f"[MH] offsets_all: {mh_outputs['offsets_all'].shape}")  # [4, K, 2, 32, 32]
    print(f"[MH] hypothesis_probs: {mh_outputs['hypothesis_probs'].shape}")  # [4, K]
    print(f"[MH] selected_hypothesis: {mh_outputs['selected_hypothesis'].shape}")  # [4]
