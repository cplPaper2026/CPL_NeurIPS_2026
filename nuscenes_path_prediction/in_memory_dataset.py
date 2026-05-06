"""In-memory PyTorch dataset for road-cpl synthetic samples (heatmap+offset schema).

Loads every ``.npz`` in ``<root>/<split>/`` once at construction time, then
``__getitem__`` only indexes pre-stacked tensors. Each NPZ is expected to
follow the ``heatmap_offset_v1`` schema:

* ``image``      ``(2, H, W) uint8``
* ``points``     ``(K_MAX, M_MAX, 2) float32`` -- full-resolution ``(x, y)``
                 path points (zero-padded), starting at the ego.
* ``num_points`` ``(K_MAX,) int32`` -- number of valid points per path.
* ``valid_paths`` ``(K_MAX,) uint8``.

Legacy datasets (``label`` field, no ``points``) are not supported; regenerate
with :mod:`generate_data`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from train_config import DEFAULT_CONFIG

_VALID_MODES = ("train", "eval")


def _load_npz_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ``(image, points, num_points, valid_paths)`` from a sample NPZ."""
    with np.load(npz_path) as data:
        files = set(data.files)
        if "points" not in files or "num_points" not in files:
            raise ValueError(f"NPZ {npz_path} is missing 'points'/'num_points' (legacy schema). Regenerate the dataset with the current generator.")
        image = data["image"]
        points = data["points"].astype(np.float32, copy=False)
        num_points = data["num_points"].astype(np.int32, copy=False)
        if "valid_paths" in files:
            valid_paths = data["valid_paths"].astype(np.uint8, copy=False)
        else:
            valid_paths = (num_points > 0).astype(np.uint8)
    return image, points, num_points, valid_paths


def _to_float01(arr: np.ndarray) -> np.ndarray:
    """Cast ``arr`` to ``float32`` in [0, 1], scaling 0..255 inputs."""
    a = arr.astype(np.float32, copy=True)
    if a.size and float(a.max()) > 1.0:
        a = a * (1.0 / 255.0)
    return a


def _pad_or_truncate_paths(
    points: np.ndarray,
    num_points: np.ndarray,
    valid_paths: np.ndarray,
    *,
    k_max: int,
    m_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conform a sample's per-path arrays to ``(K_MAX, M_MAX)``.

    Truncates extra paths/points and zero-pads short ones. Defensive only:
    a freshly generated dataset already matches the configured shapes.
    """
    K = points.shape[0]
    if k_max < K:
        points = points[:k_max]
        num_points = num_points[:k_max]
        valid_paths = valid_paths[:k_max]
        K = k_max
    M = points.shape[1]
    if m_max < M:
        points = points[:, :m_max, :]
        num_points = np.minimum(num_points, m_max)
        M = m_max
    pad_k = k_max - K
    pad_m = m_max - M
    if pad_k > 0 or pad_m > 0:
        points = np.pad(points, ((0, pad_k), (0, pad_m), (0, 0)), mode="constant")
        if pad_k > 0:
            num_points = np.concatenate([num_points, np.zeros(pad_k, dtype=num_points.dtype)])
            valid_paths = np.concatenate([valid_paths, np.zeros(pad_k, dtype=valid_paths.dtype)])
    return points, num_points, valid_paths


class RoadCplInMemoryDataset(Dataset):
    """Eager in-memory dataset returning ``(image, points, num_points, valid_paths)``.

    In ``mode="train"`` ``__getitem__`` chooses a single valid GT path and
    returns ``gt_points`` ``(M_MAX, 2)``, ``num_points`` ``scalar long``, and
    ``chosen_k``. In ``mode="eval"`` it returns the full ``(K_MAX, M_MAX, 2)``
    point stack so that multi-GT validation metrics can iterate over all
    plausible alternatives.
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
        npz_paths: list[Path] = sorted(self.split_dir.glob("*.npz"))
        if not npz_paths:
            raise RuntimeError(f"No .npz samples found under {self.split_dir}")

        images: list[torch.Tensor] = []
        points_list: list[torch.Tensor] = []
        num_points_list: list[torch.Tensor] = []
        valids: list[torch.Tensor] = []
        metas: list[dict[str, Any]] = [] if return_meta else []

        for p in npz_paths:
            image, pts, npp, valid_paths = _load_npz_arrays(p)
            if image.ndim != 3 or image.shape[0] not in (2, 3):
                raise ValueError(f"Expected image (C, H, W) with C in (2, 3) in {p}, got {image.shape}")
            pts, npp, valid_paths = _pad_or_truncate_paths(pts, npp, valid_paths, k_max=self.k_max, m_max=self.m_max)
            if pts.shape != (self.k_max, self.m_max, 2):
                raise ValueError(f"Unexpected points shape after pad in {p}: {pts.shape} (want ({self.k_max}, {self.m_max}, 2))")
            img_f = _to_float01(image)
            images.append(torch.from_numpy(np.ascontiguousarray(img_f)))
            points_list.append(torch.from_numpy(np.ascontiguousarray(pts)))
            num_points_list.append(torch.from_numpy(np.ascontiguousarray(npp.astype(np.int64))))
            valids.append(torch.from_numpy(np.ascontiguousarray(valid_paths.astype(np.bool_))))
            if return_meta:
                jp = p.with_suffix(".json")
                if jp.exists():
                    with open(jp, encoding="utf-8") as f:
                        metas.append(json.load(f))
                else:
                    metas.append({})

        self._images = torch.stack(images, dim=0)  # (N, 2, H, W)
        self._points = torch.stack(points_list, dim=0)  # (N, K_MAX, M_MAX, 2)
        self._num_points = torch.stack(num_points_list, dim=0)  # (N, K_MAX)
        self._valid_paths = torch.stack(valids, dim=0)  # (N, K_MAX)
        self._metas = metas
        self._npz_paths: list[Path] = list(npz_paths)

    @property
    def num_samples(self) -> int:
        return int(self._images.shape[0])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"index {idx} out of range for {self.num_samples} samples")
        image_t = self._images[idx].contiguous()
        valid_bool = self._valid_paths[idx]
        points_all = self._points[idx]  # (K_MAX, M_MAX, 2)
        num_points_all = self._num_points[idx]  # (K_MAX,)

        sample: dict[str, Any] = {
            "image": image_t,
            "valid_paths": valid_bool,
            "path": str(self._npz_paths[idx]),
        }

        valid_idx = valid_bool.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            raise RuntimeError(f"Sample {self._npz_paths[idx]} has no valid paths.")

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
            sample["meta"] = self._metas[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
