"""Lazy (on-demand) PyTorch dataset for road-cpl synthetic samples.

Same API and per-item output as :class:`in_memory_dataset.RoadCplInMemoryDataset`,
but each ``.npz`` (and its sidecar ``.json``) is opened inside ``__getitem__``
instead of being eagerly loaded at construction time. Use this when the dataset
does not fit comfortably in RAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from in_memory_dataset import (
    _VALID_MODES,
    _load_npz_arrays,
    _pad_or_truncate_paths,
    _to_float01,
)
from torch.utils.data import Dataset
from train_config import DEFAULT_CONFIG


class RoadCplLazyDataset(Dataset):
    """On-demand counterpart of :class:`RoadCplInMemoryDataset`.

    Indexes the split directory once (just to list ``*.npz`` paths), then loads
    one sample per ``__getitem__`` call. Returns the exact same dict schema:
    ``image``, ``gt_points``, ``num_points``, ``valid_paths``, ``path``,
    plus ``chosen_k`` (train mode) and ``meta`` (when ``return_meta=True``).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "train",
        return_meta: bool = True,
        transform: Any | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.return_meta = return_meta
        self.transform = transform
        self._rng = np.random.default_rng(rng_seed)
        self.k_max = int(DEFAULT_CONFIG.K_MAX)
        self.m_max = int(DEFAULT_CONFIG.M_MAX)

        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")
        self._npz_paths: list[Path] = sorted(self.split_dir.glob("*.npz"))
        if not self._npz_paths:
            raise RuntimeError(f"No .npz samples found under {self.split_dir}")

    @property
    def num_samples(self) -> int:
        return len(self._npz_paths)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"index {idx} out of range for {self.num_samples} samples")
        npz_path = self._npz_paths[idx]
        image_t, points_all, num_points_all, valid_bool = self._load_sample_tensors(npz_path)

        sample: dict[str, Any] = {
            "image": image_t,
            "valid_paths": valid_bool,
            "path": str(npz_path),
        }

        valid_idx = valid_bool.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            raise RuntimeError(f"Sample {npz_path} has no valid paths.")

        if self.mode == "train":
            pick = int(self._rng.integers(0, valid_idx.numel()))
            chosen_k = int(valid_idx[pick].item())
            sample["gt_points"] = points_all[chosen_k].contiguous()  # (M_MAX, 2)
            sample["num_points"] = num_points_all[chosen_k].clone()  # scalar long
            sample["chosen_k"] = torch.tensor(chosen_k, dtype=torch.long)
        else:
            sample["gt_points"] = points_all.contiguous()  # (K_MAX, M_MAX, 2)
            sample["num_points"] = num_points_all.contiguous()  # (K_MAX,)

        if self.return_meta:
            sample["meta"] = self._load_meta(npz_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _load_sample_tensors(self, npz_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load a single ``.npz`` and convert to the in-memory tensor layout."""
        image, pts, npp, valid_paths = _load_npz_arrays(npz_path)
        if image.ndim != 3 or image.shape[0] not in (2, 3):
            raise ValueError(f"Expected image (C, H, W) with C in (2, 3) in {npz_path}, got {image.shape}")
        pts, npp, valid_paths = _pad_or_truncate_paths(pts, npp, valid_paths, k_max=self.k_max, m_max=self.m_max)
        if pts.shape != (self.k_max, self.m_max, 2):
            raise ValueError(f"Unexpected points shape after pad in {npz_path}: {pts.shape} (want ({self.k_max}, {self.m_max}, 2))")
        img_f = _to_float01(image)
        image_t = torch.from_numpy(np.ascontiguousarray(img_f))
        points_t = torch.from_numpy(np.ascontiguousarray(pts))
        num_points_t = torch.from_numpy(np.ascontiguousarray(npp.astype(np.int64)))
        valid_t = torch.from_numpy(np.ascontiguousarray(valid_paths.astype(np.bool_)))
        return image_t, points_t, num_points_t, valid_t

    def _load_meta(self, npz_path: Path) -> dict[str, Any]:
        jp = npz_path.with_suffix(".json")
        if not jp.exists():
            return {}
        with open(jp, encoding="utf-8") as f:
            return json.load(f)
