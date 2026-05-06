"""CIFAR feature extraction, set-bag dataset, and DataLoader builders."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from config import CIFAR_DISPLAY_NAME, PADDING_CLUSTER, Config

CIFAR_VARIANTS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


def build_feature_extractor(device: torch.device) -> nn.Module:
    """Frozen, ImageNet-pretrained ResNet-18 with the FC layer stripped."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    return resnet.to(device)


@torch.no_grad()
def extract_features(dataset: Dataset, model: nn.Module, device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` over ``dataset`` and concatenate the feature/label tensors."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for imgs, labels in tqdm(loader, desc="Extracting features"):
        feats = model(imgs.to(device))
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Set bag dataset + collate
# ---------------------------------------------------------------------------


def collate_sets(batch: list[tuple]) -> tuple[torch.Tensor, ...]:
    """Pad variable-length set bags to the batch max length.

    Returns ``(X, cluster_labels, k_batch, dataset_indices, gt_indices, mask)``:

    * ``X``                : ``(B, N_max, D)`` features
    * ``cluster_labels``   : ``(B, N_max)``, ``PADDING_CLUSTER`` in pad slots
    * ``k_batch``          : ``(B,)`` ground-truth cluster count per set
    * ``dataset_indices``  : ``(B, N_max)``; 0 for pad slots (unused there)
    * ``gt_indices``       : ``(B, K_max)`` random-GT token indices per cluster,
                             with ``-1`` in pad slots.
    * ``mask``             : ``(B, N_max)`` bool, ``True`` = real token
    """
    lengths = [item[0].shape[0] for item in batch]
    n_max = max(lengths)
    bsz = len(batch)
    d = batch[0][0].shape[1]
    k_max = max(int(item[4].numel()) for item in batch)
    k_max = max(k_max, 1)
    X = torch.zeros(bsz, n_max, d)
    cluster_labels = torch.full((bsz, n_max), PADDING_CLUSTER, dtype=torch.long)
    dataset_indices = torch.zeros(bsz, n_max, dtype=torch.long)
    gt_indices = torch.full((bsz, k_max), -1, dtype=torch.long)
    mask = torch.zeros(bsz, n_max, dtype=torch.bool)
    k_batch = torch.empty(bsz, dtype=torch.long)
    for i, (xi, ci, k_i, di, gi) in enumerate(batch):
        n = xi.shape[0]
        X[i, :n] = xi
        cluster_labels[i, :n] = ci
        dataset_indices[i, :n] = di
        mask[i, :n] = True
        k_batch[i] = k_i
        if gi.numel() > 0:
            gt_indices[i, : gi.numel()] = gi
    return X, cluster_labels, k_batch, dataset_indices, gt_indices, mask


class SetDataset(Dataset):
    """Random set bags built by sampling K classes and points per class.

    Each ``__getitem__`` returns
    ``(X, cluster_ids, k_tensor, dataset_indices, gt_indices)`` where

    * ``cluster_ids``    : contiguous ``[0, K)`` mapping, independent of
                           the raw CIFAR label space.
    * ``gt_indices``     : ``(K,)`` LongTensor of one randomly-sampled
                           token index per cluster (the task's ground
                           truth), sampled fresh every time the item is
                           drawn from the dataset.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        max_set_size: int,
        min_k: int,
        max_k: int,
        min_samples_per_cluster: int,
        max_samples_per_cluster: int,
        num_sets: int,
        num_classes: int,
    ):
        self.features = features
        self.labels = labels
        self.max_set_size = max_set_size
        self.min_k = min_k
        self.max_k = max_k
        self.min_spc = min_samples_per_cluster
        self.max_spc = max_samples_per_cluster
        self.num_sets = num_sets
        self.num_classes = num_classes

        self.class_indices: dict[int, torch.Tensor] = {c: torch.where(labels == c)[0] for c in range(num_classes)}

    def __len__(self) -> int:
        return self.num_sets

    def __getitem__(self, _idx: int):
        k = random.randint(self.min_k, self.max_k)
        chosen_classes = random.sample(range(self.num_classes), k)

        indices: list[torch.Tensor] = []
        cluster_ids: list[torch.Tensor] = []
        for cid, cls in enumerate(chosen_classes):
            n_pts = random.randint(self.min_spc, self.max_spc)
            pool = self.class_indices[cls]
            n_take = min(int(n_pts), int(pool.numel()))
            if n_take == 0:
                raise RuntimeError(f"SetDataset: class {cls} has an empty index pool.")
            perm = torch.randperm(pool.numel(), device=pool.device)
            sel = pool[perm[:n_take]]
            indices.append(sel)
            cluster_ids.append(torch.full((n_take,), cid, dtype=torch.long))

        idx_cat = torch.cat(indices)
        cid_cat = torch.cat(cluster_ids)

        total = len(idx_cat)
        if total > self.max_set_size:
            perm = torch.randperm(total)[: self.max_set_size]
            idx_cat = idx_cat[perm]
            cid_cat = cid_cat[perm]

        X = self.features[idx_cat]
        dataset_indices = idx_cat.long().clone()
        k_tensor = torch.tensor(k, dtype=torch.long)

        # Sample one random token per surviving cluster as the GT.
        # A cluster may have been emptied by the max-set-size subsample
        # above; we only keep clusters that still have members.
        gt_picks: list[int] = []
        present = sorted(int(c) for c in cid_cat.unique().tolist() if int(c) >= 0)
        for cid_val in present:
            members = (cid_cat == cid_val).nonzero(as_tuple=True)[0]
            if members.numel() == 0:
                continue
            r = int(torch.randint(0, members.numel(), (1,)).item())
            gt_picks.append(int(members[r].item()))
        gt_indices = torch.tensor(gt_picks, dtype=torch.long)
        return X, cid_cat, k_tensor, dataset_indices, gt_indices


# ---------------------------------------------------------------------------
# Top-level loader builder used by training / evaluation
# ---------------------------------------------------------------------------


@dataclass
class Loaders:
    """Bundle of artefacts produced by :func:`build_loaders`.

    ``test_ds_raw`` is a separately-constructed CIFAR dataset that returns
    the *unnormalised* image tensor; the visualiser uses it to draw image
    grids without having to re-undo the ImageNet normalisation.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    val_sets: SetDataset
    test_ds_raw: Dataset
    test_labels: torch.Tensor


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _build_raw_transform() -> transforms.Compose:
    return transforms.Compose([transforms.Resize(224), transforms.ToTensor()])


def build_loaders(cfg: Config, device: torch.device) -> Loaders:
    """Build train + validation set loaders plus the raw test dataset for viz."""
    cifar_cls = CIFAR_VARIANTS[cfg.data.dataset]
    transform = _build_transform()

    print(f"Loading {CIFAR_DISPLAY_NAME[cfg.data.dataset]}...")
    train_ds = cifar_cls(cfg.data.data_root, train=True, download=True, transform=transform)
    test_ds = cifar_cls(cfg.data.data_root, train=False, download=True, transform=transform)

    extractor = build_feature_extractor(device)
    print("Extracting train features...")
    train_feats, train_labels = extract_features(train_ds, extractor, device, cfg.data.feature_batch_size)
    print("Extracting test features...")
    test_feats, test_labels = extract_features(test_ds, extractor, device, cfg.data.feature_batch_size)
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_sets = SetDataset(
        train_feats,
        train_labels,
        cfg.data.max_set_size,
        cfg.data.min_k,
        cfg.data.max_k,
        cfg.data.min_samples_per_cluster,
        cfg.data.max_samples_per_cluster,
        cfg.data.num_train_sets,
        num_classes=cfg.num_classes,
    )
    val_sets = SetDataset(
        test_feats,
        test_labels,
        cfg.data.max_set_size,
        cfg.data.min_k,
        cfg.data.max_k,
        cfg.data.min_samples_per_cluster,
        cfg.data.max_samples_per_cluster,
        cfg.data.num_val_sets,
        num_classes=cfg.num_classes,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_sets,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
        collate_fn=collate_sets,
    )
    val_loader = DataLoader(
        val_sets,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
        collate_fn=collate_sets,
    )
    test_ds_raw = cifar_cls(cfg.data.data_root, train=False, download=True, transform=_build_raw_transform())
    return Loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        val_sets=val_sets,
        test_ds_raw=test_ds_raw,
        test_labels=test_labels,
    )


__all__ = [
    "CIFAR_VARIANTS",
    "Loaders",
    "SetDataset",
    "build_feature_extractor",
    "build_loaders",
    "collate_sets",
    "extract_features",
]
