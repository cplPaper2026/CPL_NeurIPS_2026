#!/usr/bin/env python3
"""Select a subset of (npz, json) pairs from an input folder, stratified by
``n_paths``, and split it into train/val while keeping the same per-n_paths
distribution in both splits.

Group atomicity
---------------
All augmentations of the same base sample share a filename prefix (everything
before the last ``_``), e.g. ``00000208_00 / 00000208_01 / 00000208_02`` are
the same base under different transformations. The script treats each base
as an indivisible **group**: every augmentation of a base ends up entirely in
train or entirely in val.

Spatial dedup (optional)
------------------------
With ``--min-distance METERS > 0`` the script also drops groups that are too
close to an already-kept group within the same ``map_name`` (greedy filter
on ``ego_global`` (x, y)).

Modes (mutually exclusive, exactly one required)
------------------------------------------------
``--distribution P1 P2 ...``  fractions for n_paths=1,2,...; auto-normalized.
``--from-largest``            greedy: take whole buckets from the largest
                              n_paths down (rarest first) until ``--total``
                              is reached.

Example::

    python split_nuscenes_train_val.py \\
        --input-dir   data/nuscenes/all \\
        --train-dir   data/nuscenes/train \\
        --val-dir     data/nuscenes/val \\
        --distribution 0.1 0.2 0.3 0.4 \\
        --total       10000 \\
        --train-ratio 0.8 \\
        --min-distance 5 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Group:
    """A base sample bundled with all of its augmentation stems."""

    prefix: str
    stems: list[str]
    n_paths: int  # representative bucket for stratification
    map_name: str
    x: float
    y: float

    @property
    def n_samples(self) -> int:
        return len(self.stems)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_dir, train_dir, val_dir = (
        args.input_dir.resolve(),
        args.train_dir.resolve(),
        args.val_dir.resolve(),
    )
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir does not exist or is not a directory: {input_dir}")

    groups = scan_input_dir(
        input_dir,
        no_aug_up_to=args.no_aug_up_to,
        max_stems_low_paths=args.max_stems_low_paths,
    )
    report_availability("Initial", groups)

    if args.min_distance > 0:
        groups = spatial_dedup(rng, groups, args.min_distance)
        report_availability(f"After spatial dedup (>= {args.min_distance} m)", groups)

    groups_by_k = group_by_n_paths(groups)
    counts_by_k = decide_counts(args, groups_by_k)
    selected_by_k = sample_groups_per_bucket(rng, groups_by_k, counts_by_k)

    train_stems, val_stems, train_groups, val_groups = split_train_val(rng, selected_by_k, args.train_ratio)

    if args.dry_run:
        print("\n[dry-run] skipping file transfer")
    else:
        transfer_pairs(train_stems, input_dir, train_dir, move=args.move)
        transfer_pairs(val_stems, input_dir, val_dir, move=args.move)

    print_split_stats("Train", train_dir, train_groups, dry_run=args.dry_run)
    print_split_stats("Val", val_dir, val_groups, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="Folder with (.npz, .json) pairs")
    p.add_argument("--train-dir", type=Path, required=True, help="Output train folder")
    p.add_argument("--val-dir", type=Path, required=True, help="Output val folder")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--distribution",
        type=float,
        nargs="+",
        help="Fractions for n_paths=1,2,...; auto-normalized to sum 1. Mutually exclusive with --from-largest.",
    )
    mode.add_argument(
        "--from-largest",
        action="store_true",
        help="Greedy mode: take whole buckets starting from the largest n_paths down (rarest first) until --total is reached. Mutually exclusive with --distribution.",
    )
    p.add_argument(
        "--total",
        type=int,
        default=None,
        help="Approximate total samples to select (rounded to whole groups). Default: max feasible given the chosen mode and availability.",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of selected samples that go to train (rest to val).",
    )
    p.add_argument(
        "--min-distance",
        type=float,
        default=0.0,
        help="Per-map minimum Euclidean distance (meters) between selected groups' ego_global (x, y). 0 disables spatial filtering.",
    )
    p.add_argument(
        "--no-aug-up-to",
        type=int,
        default=1,
        help="For groups whose representative n_paths is <= this value, cap the "
        "number of stems to --max-stems-low-paths (lowest aug indices kept, e.g. "
        "``_00``, ``_01``); drop the rest. 0 disables this trimming. Default: 1.",
    )
    p.add_argument(
        "--max-stems-low-paths",
        type=int,
        default=3,
        help="When --no-aug-up-to triggers trimming, keep at most this many stems per group (lowest aug indices). Default: 2.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy, leaves input dir untouched).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full selection/split logic and print stats, but skip copying/moving files.",
    )
    args = p.parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be in (0, 1)")
    if args.min_distance < 0:
        raise SystemExit("--min-distance must be non-negative")
    if args.no_aug_up_to < 0:
        raise SystemExit("--no-aug-up-to must be non-negative")
    if args.max_stems_low_paths < 1:
        raise SystemExit("--max-stems-low-paths must be >= 1")
    return args


# ---------------------------------------------------------------------------
# Input scanning -> Group objects
# ---------------------------------------------------------------------------


def scan_input_dir(
    input_dir: Path,
    *,
    no_aug_up_to: int = 1,
    max_stems_low_paths: int = 2,
) -> list[Group]:
    """Discover (npz, json) pairs and bundle stems with the same prefix.

    For groups whose representative n_paths is <= ``no_aug_up_to`` we cap the
    number of stems to ``max_stems_low_paths`` (keeping the lowest aug indices
    and dropping the rest); augmenting low-path samples adds little variety.
    """
    stems = find_npz_json_pairs(input_dir)
    if not stems:
        raise SystemExit("No (npz, json) pairs found in input dir")
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        by_prefix[group_prefix(stem)].append(stem)
    groups: list[Group] = []
    dropped_augs = 0
    for prefix, raw_stems in by_prefix.items():
        raw_stems.sort()
        meta = read_sample_meta(input_dir / f"{raw_stems[0]}.json")
        is_low = no_aug_up_to > 0 and meta["n_paths"] <= no_aug_up_to
        if is_low and len(raw_stems) > max_stems_low_paths:
            dropped_augs += len(raw_stems) - max_stems_low_paths
            kept_stems = raw_stems[:max_stems_low_paths]
        else:
            kept_stems = raw_stems
        groups.append(
            Group(
                prefix=prefix,
                stems=kept_stems,
                n_paths=meta["n_paths"],
                map_name=meta["map_name"],
                x=meta["x"],
                y=meta["y"],
            )
        )
    if dropped_augs:
        print(f"Dropped {dropped_augs} augmentation stems from groups with n_paths <= {no_aug_up_to} (kept up to {max_stems_low_paths} lowest-aug stems per group)")
    groups.sort(key=lambda g: g.prefix)
    return groups


def find_npz_json_pairs(input_dir: Path) -> list[str]:
    npz_stems = {p.stem for p in input_dir.glob("*.npz")}
    json_stems = {p.stem for p in input_dir.glob("*.json")}
    only_npz = npz_stems - json_stems
    only_json = json_stems - npz_stems
    if only_npz:
        print(f"Warning: {len(only_npz)} .npz without matching .json (skipped)")
    if only_json:
        print(f"Warning: {len(only_json)} .json without matching .npz (skipped)")
    return sorted(npz_stems & json_stems)


def group_prefix(stem: str) -> str:
    """Everything before the last ``_`` (e.g. ``00000208_03`` -> ``00000208``)."""
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def read_sample_meta(json_path: Path) -> dict:
    with json_path.open("r") as f:
        meta = json.load(f)
    ego = meta["ego_global"]
    return {
        "n_paths": int(meta["n_paths"]),
        "map_name": str(meta["map_name"]),
        "x": float(ego["x"]),
        "y": float(ego["y"]),
    }


def report_availability(label: str, groups: list[Group]) -> None:
    if not groups:
        raise SystemExit(f"{label}: no groups available")
    n_samples = sum(g.n_samples for g in groups)
    by_k: dict[int, list[Group]] = defaultdict(list)
    for g in groups:
        by_k[g.n_paths].append(g)
    max_k = max(by_k)
    print(f"{label}: {len(groups)} groups, {n_samples} samples; max n_paths in data = {max_k}")
    print("  per-bucket availability (groups / samples):")
    for k in sorted(by_k):
        bucket = by_k[k]
        print(f"    n_paths={k:>2}: groups={len(bucket):>5}  samples={sum(g.n_samples for g in bucket):>6}")


# ---------------------------------------------------------------------------
# Spatial dedup
# ---------------------------------------------------------------------------


def spatial_dedup(rng: random.Random, groups: list[Group], min_distance: float) -> list[Group]:
    """Greedy per-map filter: drop groups within ``min_distance`` of a kept one."""
    by_map: dict[str, list[Group]] = defaultdict(list)
    for g in groups:
        by_map[g.map_name].append(g)
    kept: list[Group] = []
    min_dist_sq = min_distance * min_distance
    for map_name in sorted(by_map):
        candidates = list(by_map[map_name])
        rng.shuffle(candidates)
        accepted: list[Group] = []
        for g in candidates:
            if all((g.x - a.x) ** 2 + (g.y - a.y) ** 2 >= min_dist_sq for a in accepted):
                accepted.append(g)
        kept.extend(accepted)
    kept.sort(key=lambda g: g.prefix)
    return kept


# ---------------------------------------------------------------------------
# Per-bucket allocation (samples target -> per-bucket sample target)
# ---------------------------------------------------------------------------


def group_by_n_paths(groups: list[Group]) -> dict[int, list[Group]]:
    by_k: dict[int, list[Group]] = defaultdict(list)
    for g in groups:
        by_k[g.n_paths].append(g)
    return dict(by_k)


def decide_counts(args: argparse.Namespace, groups_by_k: dict[int, list[Group]]) -> dict[int, int]:
    """Return target *sample* counts per n_paths bucket for the chosen mode."""
    available_samples = {k: sum(g.n_samples for g in groups) for k, groups in groups_by_k.items()}
    if args.from_largest:
        counts = decide_counts_largest_first(available_samples, args.total)
        mode_label = "from-largest (rarest first)"
    else:
        distribution = normalize_distribution(args.distribution)
        counts = decide_counts_by_distribution(distribution, available_samples, args.total)
        mode_label = "distribution"
    target_total = sum(counts.values())
    print(f"\nMode: {mode_label}; target ~{target_total} samples; per-bucket sample targets:")
    for k in sorted(counts):
        if counts[k] > 0:
            print(f"  n_paths={k:>2}: {counts[k]:>6}")
    return counts


def normalize_distribution(values: list[float]) -> list[float]:
    if any(v < 0 for v in values):
        raise SystemExit("--distribution values must be non-negative")
    total = sum(values)
    if total <= 0:
        raise SystemExit("--distribution must contain at least one positive value")
    return [v / total for v in values]


def decide_counts_by_distribution(
    distribution: list[float],
    available_samples: dict[int, int],
    total: int | None,
) -> dict[int, int]:
    if total is None:
        total = max_feasible_total(distribution, available_samples)
        print(f"--total not provided; using max feasible total = {total}")
    counts = largest_remainder_round(distribution, total)
    validate_counts_against_availability(counts, available_samples)
    return counts


def max_feasible_total(distribution: list[float], available_samples: dict[int, int]) -> int:
    bounds: list[int] = []
    for idx, p_k in enumerate(distribution):
        if p_k <= 0:
            continue
        k = idx + 1
        bounds.append(int(available_samples.get(k, 0) / p_k))
    if not bounds:
        raise SystemExit("--distribution has no positive entries")
    return min(bounds)


def largest_remainder_round(distribution: list[float], total: int) -> dict[int, int]:
    raw = [(idx + 1, distribution[idx] * total) for idx in range(len(distribution))]
    counts = {k: int(x) for k, x in raw}
    leftover = total - sum(counts.values())
    remainders = sorted(((x - int(x), k) for k, x in raw), reverse=True)
    for _, k in remainders:
        if leftover <= 0:
            break
        counts[k] += 1
        leftover -= 1
    return counts


def validate_counts_against_availability(counts: dict[int, int], available_samples: dict[int, int]) -> None:
    for k, c in counts.items():
        if c > available_samples.get(k, 0):
            raise SystemExit(f"Not enough samples with n_paths={k}: requested {c}, available {available_samples.get(k, 0)}")


def decide_counts_largest_first(available_samples: dict[int, int], total: int | None) -> dict[int, int]:
    capacity = sum(available_samples.values())
    if total is None:
        total = capacity
        print(f"--total not provided; using all available = {total}")
    if total > capacity:
        raise SystemExit(f"Requested --total {total} exceeds total available {capacity}")
    counts: dict[int, int] = dict.fromkeys(available_samples, 0)
    remaining = total
    for k in sorted(available_samples, reverse=True):
        if remaining <= 0:
            break
        take = min(available_samples[k], remaining)
        counts[k] = take
        remaining -= take
    return counts


# ---------------------------------------------------------------------------
# Group selection (whole groups added until per-bucket sample target is met)
# ---------------------------------------------------------------------------


def sample_groups_per_bucket(
    rng: random.Random,
    groups_by_k: dict[int, list[Group]],
    sample_targets: dict[int, int],
) -> dict[int, list[Group]]:
    selected: dict[int, list[Group]] = {}
    for k, target in sample_targets.items():
        if target <= 0:
            continue
        pool = list(groups_by_k.get(k, []))
        rng.shuffle(pool)
        selected[k] = pick_groups_for_target(pool, target)
    return selected


def pick_groups_for_target(pool: list[Group], target_samples: int) -> list[Group]:
    """Take whole groups until reaching ~``target_samples``.

    Stop at the index that minimizes ``|cumulative_samples - target|`` so we
    don't badly overshoot when adding the next group would push us further
    past the target than stopping short of it.
    """
    chosen: list[Group] = []
    running = 0
    for g in pool:
        next_running = running + g.n_samples
        if next_running >= target_samples:
            if next_running - target_samples <= target_samples - running:
                chosen.append(g)
                running = next_running
            break
        chosen.append(g)
        running = next_running
    return chosen


# ---------------------------------------------------------------------------
# Train / val split (group-level, sample-aware ratio)
# ---------------------------------------------------------------------------


def split_train_val(
    rng: random.Random,
    selected_by_k: dict[int, list[Group]],
    train_ratio: float,
) -> tuple[list[str], list[str], list[Group], list[Group]]:
    train_stems: list[str] = []
    val_stems: list[str] = []
    train_groups: list[Group] = []
    val_groups: list[Group] = []
    for k in sorted(selected_by_k):
        groups = list(selected_by_k[k])
        rng.shuffle(groups)
        train_part, val_part = split_groups_by_sample_ratio(groups, train_ratio)
        train_groups.extend(train_part)
        val_groups.extend(val_part)
        train_stems.extend(s for g in train_part for s in g.stems)
        val_stems.extend(s for g in val_part for s in g.stems)
    return train_stems, val_stems, train_groups, val_groups


def split_groups_by_sample_ratio(groups: list[Group], train_ratio: float) -> tuple[list[Group], list[Group]]:
    total_samples = sum(g.n_samples for g in groups)
    target_train = round(train_ratio * total_samples)
    running = 0
    split_idx = len(groups)
    for i, g in enumerate(groups):
        next_running = running + g.n_samples
        if next_running >= target_train:
            split_idx = i + 1 if (next_running - target_train) <= (target_train - running) else i
            break
        running = next_running
    return groups[:split_idx], groups[split_idx:]


# ---------------------------------------------------------------------------
# File I/O + reporting
# ---------------------------------------------------------------------------


def transfer_pairs(stems: list[str], src_dir: Path, dst_dir: Path, *, move: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    op = shutil.move if move else shutil.copy2
    for stem in stems:
        for suffix in (".npz", ".json"):
            src = src_dir / f"{stem}{suffix}"
            dst = dst_dir / f"{stem}{suffix}"
            if not src.is_file():
                raise SystemExit(f"Expected file missing: {src}")
            if dst.exists():
                raise SystemExit(f"Refusing to overwrite existing destination: {dst}")
            op(str(src), str(dst))


def print_split_stats(name: str, dir_: Path, groups: list[Group], *, dry_run: bool) -> None:
    n_groups = len(groups)
    n_samples = sum(g.n_samples for g in groups)
    total_paths = sum(g.n_paths * g.n_samples for g in groups)
    arrow = "would go to" if dry_run else "->"
    print(f"\n{name}: {n_groups} groups, {n_samples} samples, {total_paths} total paths {arrow} {dir_}")
    by_k: dict[int, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for g in groups:
        gc, sc = by_k[g.n_paths]
        by_k[g.n_paths] = (gc + 1, sc + g.n_samples)
    for k in sorted(by_k):
        gc, sc = by_k[k]
        share = sc / n_samples if n_samples else 0.0
        print(f"  n_paths={k:>2}: groups={gc:>5}  samples={sc:>6}  ({share:>6.1%})")


if __name__ == "__main__":
    main()
