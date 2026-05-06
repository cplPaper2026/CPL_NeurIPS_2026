import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from train_config import CplHeadConfig

BIG_NEG = -1e9


class MLP(nn.Module):
    """Small helper MLP used by the CPL heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        last_in = hidden_dim if num_layers > 1 else input_dim
        layers.append(nn.Linear(last_in, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CPLHead(nn.Module):
    """
    Contextual Plackett-Luce (CPL) head operating on an existing embedding grid.

    This module does not extract visual features. It receives the shared model
    embeddings and provides:
    - CPL logits components (theta, W_dense)
    - permutation-invariant CPL loss
    - greedy autoregressive decoding
    """

    def __init__(self, cpl: CplHeadConfig, hidden_dim: int, grid_size: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.max_steps = cpl.max_steps
        self.eos_in_graph = cpl.eos_in_graph
        self.loss_type = str(cpl.loss_type)

        n_lay = cpl.mlp_num_layers
        self.eos_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * cpl.eos_init_std)
        self.f_theta = MLP(hidden_dim, hidden_dim, 1, num_layers=n_lay)
        self.key_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=n_lay)
        self.value_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=n_lay)

    def forward(self, embeddings_grid: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Build CPL components from a shared embedding grid.

        Args:
            embeddings_grid: Tensor of shape (B, hidden_dim, grid_size, grid_size).

        Returns:
            Dictionary with:
              - theta: shape (B, N+1)
              - W_dense: shape (B, N+1, N+1)
        """
        B, C, H, W = embeddings_grid.shape
        if self.hidden_dim != C:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {C}.")
        if self.grid_size != H or self.grid_size != W:
            raise ValueError(f"Expected ({self.grid_size}, {self.grid_size}), got ({H}, {W}).")

        tokens = embeddings_grid.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        eos = self.eos_token.expand(B, -1, -1)
        memory = torch.cat([tokens, eos], dim=1)  # (B, N+1, C)

        theta = self.f_theta(memory).squeeze(-1)  # (B, N+1)
        keys = self.key_mlp(memory)
        values = self.value_mlp(memory)
        W_dense = torch.bmm(keys, values.transpose(1, 2)) / math.sqrt(self.hidden_dim)

        if not self.eos_in_graph:
            # EOS is not in the graph, so we need to set the W_dense values to 0 for the EOS token.
            W_dense[:, -1, :] = 0.0
            W_dense[:, :, -1] = 0.0

        return {"theta": theta, "W_dense": W_dense}

    def _gt_indices_from_points(self, gt_points_b: torch.Tensor, num_points_b: torch.Tensor, *, D: int) -> torch.Tensor:
        """``(M_MAX, 2)`` ``(x, y)`` points -> 1D flat cell indices in selection order.

        The order is preserved as stored in ``gt_points`` (path order from the
        ego), which is more meaningful than reordering by Euclidean distance
        when paths re-enter the ego's neighborhood. Duplicate cell indices (if
        any -- the generator's sparsifier guarantees there are none) are
        dropped while preserving order.
        """
        n = int(num_points_b.item())
        if n <= 0:
            return torch.empty(0, dtype=torch.long, device=gt_points_b.device)
        pts = gt_points_b[:n]
        cells = torch.floor(pts / float(D)).to(torch.long)
        j = cells[:, 0].clamp_(0, self.grid_size - 1)
        i = cells[:, 1].clamp_(0, self.grid_size - 1)
        flat = i * self.grid_size + j
        # Order-preserving deduplication (defensive; sparsifier already enforces it).
        seen: set[int] = set()
        keep = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
        flat_cpu = flat.detach().cpu().tolist()
        for k, idx in enumerate(flat_cpu):
            if idx not in seen:
                seen.add(idx)
                keep[k] = True
        return flat[keep]

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
        meta: list[dict[str, Any]],  # noqa: ARG002
        *,
        D: int,
        BIG_NEG: float = -1e9,
    ) -> torch.Tensor:
        """
        Optimized vectorized CPL loss handling variable path lengths within a batch.

        Args:
            outputs: {"theta": (B, N+1), "W_dense": (B, N+1, N+1)}
            gt_points: (B, M_MAX, 2) zero-padded GT points.
            num_points: (B,) valid point count per sample.
            D: downsample factor.
        """
        theta = outputs["theta"]  # (B, N+1)
        W_dense = outputs["W_dense"]  # (B, N+1, N+1)
        B, n_plus_1 = theta.shape
        n = n_plus_1 - 1
        device = theta.device

        # 1. Extract GT indices for all samples in batch
        # We still need a small loop here to handle variable lengths from the helper
        all_gt_idx = []
        max_path_len = 0
        for b in range(B):
            idx = self._gt_indices_from_points(gt_points[b], num_points[b], D=D)
            all_gt_idx.append(idx)
            max_path_len = max(max_path_len, len(idx))

        # K is the maximum number of decision steps in this batch (max_path_len + 1 for the end token)
        K = max_path_len + 1

        # 2. Prepare Tensors for batch processing
        # padded_targets: (B, K) - The ground truth index for each step
        # valid_mask: (B, K) - Boolean mask to ignore padding steps in loss calculation
        # subset_masks: (B, K, N+1) - Binary mask of points already visited at each step
        padded_targets = torch.full((B, K), n, dtype=torch.long, device=device)
        valid_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        subset_masks = torch.zeros((B, K, n_plus_1), device=device)

        for b, idx in enumerate(all_gt_idx):
            path_len = len(idx)
            # Set targets: the sequence is [idx_0, idx_1, ..., idx_L-1, n]
            if path_len > 0:
                padded_targets[b, :path_len] = idx
            # valid_mask marks where we actually have a GT step to supervise
            valid_mask[b, : path_len + 1] = True

            # Build subset masks: for step k, mask contains all points from idx[:k]
            if path_len > 0:
                # We use a running prefix to fill the mask efficiently
                running_mask = torch.zeros(n_plus_1, device=device)
                for k in range(1, path_len + 1):
                    running_mask[idx[k - 1]] = 1.0
                    subset_masks[b, k] = running_mask

        # 3. Vectorized Context Calculation
        # context = W_dense @ subset_mask for all B and K simultaneously
        # (B, K, N+1) = (B, N+1, N+1) @ (B, K, N+1).transpose
        # Using bmm (batch matrix multiplication)
        contexts = torch.bmm(subset_masks, W_dense.transpose(1, 2))

        # 4. Compute Logits
        # Add theta (B, 1, N+1) + context (B, K, N+1) + mask penalty
        logits = theta.unsqueeze(1) + contexts + (subset_masks * BIG_NEG)

        # 5. Masked Cross Entropy
        # Flatten everything to (B*K, N+1) and (B*K)
        logits_flat = logits.reshape(-1, n_plus_1)
        targets_flat = padded_targets.reshape(-1)
        valid_flat = valid_mask.reshape(-1)

        # Filter out the padding terms
        valid_logits = logits_flat[valid_flat]
        valid_targets = targets_flat[valid_flat]
        if self.loss_type == "one_hot_prob":
            final_loss = self._one_hot_probability_loss(valid_logits, valid_targets)
        else:
            final_loss = self._index_cross_entropy_loss(valid_logits, valid_targets)

        return final_loss

    def _one_hot_probability_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Categorical NLL from softmax probabilities and one-hot targets."""
        probs = F.softmax(logits, dim=-1)
        target_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(dtype=probs.dtype)
        log_probs = torch.log(probs.clamp_min(1e-9))
        per_step_nll = -(target_one_hot * log_probs).sum(dim=-1)
        return per_step_nll.mean()

    def _index_cross_entropy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard CE with class-index targets."""
        return F.cross_entropy(logits, targets, reduction="mean")

    def old_loss(
        self,
        outputs: dict[str, torch.Tensor],
        gt_points: torch.Tensor,
        num_points: torch.Tensor,
        meta: list[dict[str, Any]],  # noqa: ARG002 -- kept for API parity / future use
        *,
        D: int,
    ) -> torch.Tensor:
        """Permutation-aware CPL loss from per-sample GT point lists.

        Args:
            outputs: ``{"theta": (B, N+1), "W_dense": (B, N+1, N+1)}``.
            gt_points: ``(B, M_MAX, 2)`` zero-padded ``(x, y)`` GT points.
            num_points: ``(B,) long`` valid point count per sample.
            meta: kept for compatibility; not used now that ego pose is implicit
                in the path ordering.
            D: full-res-to-grid downsample factor.
        """
        theta = outputs["theta"]
        W_dense = outputs["W_dense"]
        B, n_plus_1 = theta.shape
        n = n_plus_1 - 1
        device = theta.device

        total_loss = torch.tensor(0.0, device=device)
        n_terms = 0
        for b in range(B):
            gt_idx = self._gt_indices_from_points(gt_points[b].to(device), num_points[b].to(device), D=D)
            if gt_idx.numel() == 0:
                total_loss = total_loss + F.cross_entropy(theta[b].unsqueeze(0), torch.tensor([n], device=device))
                n_terms += 1
                continue

            for k in range(len(gt_idx) + 1):
                s_sub = gt_idx[:k]
                next_indice = int(gt_idx[k].item()) if k < len(gt_idx) else n

                s_mask = torch.zeros(n_plus_1, dtype=torch.bool, device=device)
                if s_sub.numel() > 0:
                    s_mask[s_sub] = True

                context = W_dense[b] @ s_mask.to(W_dense.dtype)
                logits = theta[b] + context + BIG_NEG * s_mask.to(theta.dtype)
                target = torch.tensor([next_indice], device=device)
                loss_term = F.cross_entropy(logits.unsqueeze(0), target)
                total_loss = total_loss + loss_term
                n_terms += 1

        return total_loss / max(n_terms, 1)

    @torch.no_grad()
    def greedy_decode(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Autoregressive decode -> ``(mask, order, indices)``.

        * ``mask``:    ``(B, H, W)`` binary cell occupancy.
        * ``order``:   ``(B, H, W)`` selection step ``1..K`` (``0`` = unselected).
        * ``indices``: list of length ``B`` with 1D ``long`` tensors of the
          chosen flat cell indices in selection order. Useful for combining
          with the offset head to recover sub-pixel ``(x, y)`` predictions.
        """
        theta = outputs["theta"]
        W_dense = outputs["W_dense"]
        B, n_plus_1 = theta.shape
        n = n_plus_1 - 1
        device = theta.device

        pred_flat = torch.zeros((B, n), dtype=torch.float32, device=device)
        order_flat = torch.zeros((B, n), dtype=torch.float32, device=device)
        chosen_indices: list[torch.Tensor] = []
        for b in range(B):
            mask = torch.zeros(n_plus_1, dtype=torch.bool, device=device)
            context = torch.zeros(n_plus_1, dtype=torch.float32, device=device)
            step = 0
            picks: list[int] = []

            for _ in range(self.max_steps):
                logits = theta[b] + context + BIG_NEG * mask.float()
                chosen_idx = int(torch.argmax(logits).item())
                if chosen_idx == n:
                    break
                step += 1
                pred_flat[b, chosen_idx] = 1.0
                order_flat[b, chosen_idx] = float(step)
                mask[chosen_idx] = True
                context = context + W_dense[b, :, chosen_idx]
                picks.append(chosen_idx)

            chosen_indices.append(torch.tensor(picks, dtype=torch.long, device=device))

        return {
            "mask": pred_flat.view(B, self.grid_size, self.grid_size),
            "order": order_flat.view(B, self.grid_size, self.grid_size),
            "indices": chosen_indices,
        }
