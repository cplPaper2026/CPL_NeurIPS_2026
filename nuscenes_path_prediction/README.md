# Multi-Modal Path Prediction on nuScenes Maps

Code for the **multi-modal path prediction** experiment in the
*Contextual Plackett-Luce* (CPL) paper. See the top-level
[`README.md`](../README.md) for the abstract and a description of the
overall project.

The task: given a 256x256 BEV RGB rendering of a local road graph (with
the ego pose baked in), predict one geometrically valid driving path.
The training supervision is one of several valid ground-truth paths
sampled at random per draw, so the model must commit to a single
coherent continuation rather than averaging across modes.

## Architecture

```
image (3, 256, 256)               # BEV RGB: drivable area + lane dividers + ego marker
  -> ResNet-18 backbone (truncated to layer2)
  -> 2D positional encoding
  -> Transformer encoder
  -> per-cell MLP heads:
       heatmap_head -> 1 logit per 8x8 cell  (no sigmoid)
       offset_head  -> (dx, dy) per cell     (raw pixels)
  -> embeddings_grid (B, hidden_dim, 32, 32)
  -> optional sequence head: CPLHead | ARDecoderHead | MultiHypothesisHead
```

A predicted point at cell `(i, j)` is reconstructed as
`(x, y) = (j*D + dx[i, j], i*D + dy[i, j])` in full-resolution pixels
(`D = 8` by default).

## 1. Install

```bash
cd nuscenes_path_prediction
pip install -r requirements.txt
```

The required dependencies are listed in [`requirements.txt`](requirements.txt):
`numpy`, `opencv-python`, `shapely`, `scipy`, `h5py`, `torch`,
`matplotlib`, `tqdm`, `pyyaml`, plus `nuscenes-devkit` (needed for the
map-based generator) and `comet-ml` (only used when Comet logging is
enabled).

## 2. Data preparation

The benchmark used in the paper is built from the **nuScenes maps
expansion**. A purely procedural fallback is also included for users
without nuScenes access (Step D below).

### Step A. Download the nuScenes maps expansion

Download the **Maps expansion** archive from the nuScenes website
(https://www.nuscenes.org/download — requires a free account) and
extract it. Only the four map JSON files are needed (no images, no
LiDAR), which gives a final layout like

```
<dataroot>/
  maps/
    expansion/
      boston-seaport.json
      singapore-onenorth.json
      singapore-hollandvillage.json
      singapore-queenstown.json
```

### Step B. Render BEV samples

```bash
python generate_nuscenes_paths.py \
    --dataroot /path/to/nuscenes/v1.0/ \
    --n-samples 10000 \
    --out data/nuscenes/ \
    --workers 4 \
    --seed 0
```

This writes 10k `(.npz, .json)` pairs under `data/nuscenes/train/` and
`data/nuscenes/val/` (deterministic 80/20 split by sample index, set
via `--train-frac`). Each `.npz` follows the
`heatmap_offset_rgb_v1` schema:

- `image`        `uint8 (3, H, W)`           — RGB BEV with the ego marker baked in.
- `points`       `float32 (K_MAX, M_MAX, 2)` — zero-padded full-resolution `(x, y)` GT points starting at the ego, sparsified so no two points fall in the same `D x D` grid cell.
- `num_points`   `int32 (K_MAX,)`            — number of valid points per path slot.
- `valid_paths`  `uint8 (K_MAX,)`            — 1 if the slot is a real GT path.

Defaults: BEV `256x256` at `0.5 m/px`, `K_MAX=6` paths per sample,
`M_MAX=64` points per path. Per-path sparsification is arc-length
resampling at `D * sqrt(2)` step + greedy per-cell deduplication so
that the `(K_MAX, M_MAX, 2)` arrays carry at most one point per
`D x D` cell, which is what the heatmap+offset training expects.

### Step C. (Optional) Stratified re-split by number of paths

`generate_nuscenes_paths.py` already writes a deterministic 80/20 split,
but its split is *index-based* and not balanced across the number of
valid paths per sample. If you want a split that preserves the
`n_paths` distribution across train and val, regenerate everything
into a single folder (`--train-frac 1.0`) and then call:

```bash
python split_nuscenes_train_val.py \
    --input-dir   data/nuscenes/train \
    --train-dir   data/nuscenes_balanced/train \
    --val-dir     data/nuscenes_balanced/val \
    --distribution 0.1 0.2 0.3 0.4 \
    --total       10000 \
    --train-ratio 0.8 \
    --min-distance 5 \
    --seed 42
```

`--distribution` is the desired fraction per `n_paths` bucket
(positions are `n_paths=1, 2, 3, ...`). `--from-largest` is an
alternative greedy mode that keeps whole rare buckets first. Add
`--dry-run` to see the planned counts without copying any files.

### Step D. (Optional) Procedural-only synthetic fallback

If you do not have access to the nuScenes maps, the procedural
generator [`generate_data.py`](generate_data.py) produces a similar BEV
schema (with a 2-channel `(road, ego_heatmap)` `image` instead of
3-channel RGB) without any external data:

```bash
python generate_data.py --n 10000 --out data/synthetic/ --workers 8 --seed 0
```

Use this only for smoke tests; the paper numbers use the nuScenes-derived
benchmark from Steps A and B.

## 3. Training

Run all commands from inside `nuscenes_path_prediction/`. The
`--data-root` flag points at the directory that contains the
`train/` and `val/` subfolders produced in Step B.

The loss has two orthogonal axes: matching strategy and head mode.

```bash
# Grid-supervised baseline (default: matching=grid, no sequence head)
python train.py --data-root data/nuscenes/

# Hungarian matched heatmap baseline (DETR-style)
python train.py --config configs/hungarian_mean_path.yaml \
    --data-root data/nuscenes/

# Multi-hypothesis baseline (winner-take-all path + selector CE)
python train.py --config configs/multi_hypothesis.yaml \
    --data-root data/nuscenes/

# CPL ordering head (replaces heatmap classification under grid matching)
python train.py --use-cpl --data-root data/nuscenes/

# Autoregressive pointer head (transformer decoder)
python train.py --use-ar --data-root data/nuscenes/
```

Paper-aligned defaults (same as the dataclass defaults in
[`train_config.py`](train_config.py)): AdamW with `lr=3e-4`, weight
decay `1e-4`, `batch_size=256`, `num_epochs=300`, horizontal-flip
augmentation enabled. Heatmap BCE weight 10, offset L1 weight 5.

Hungarian matcher costs (`HungarianCostConfig`) default to
`prob_weight=1` and `dist_weight=1`; [`configs/hungarian_mean_path.yaml`](configs/hungarian_mean_path.yaml)
overrides them to `prob_weight=10` and `dist_weight=0.1` for the mean-path baseline.

CPL and AR sequence heads minimise per-step cross-entropy over grid
cells (including the EOS slot); see [`cpl_module.py`](cpl_module.py) and
[`ar_module.py`](ar_module.py). This codebase does **not** use the
CIFAR-style `pre_eos_weight` / `post_eos_weight` knobs from the subset-selection experiment.

Useful overrides:

```bash
# Override individual loss term weights and the offset criterion
python train.py --offset-loss smooth_l1 --offset-weight 0.5 \
    --heatmap-weight 1.0 --ar-weight 1.0

# Open NPZ samples on demand (use when the dataset doesn't fit in RAM)
python train.py --lazy-data --data-root data/nuscenes/

# Disable Comet logging for the run
python train.py --no-comet --data-root data/nuscenes/
```

Training writes two best-by-metric checkpoints under
`run.resolved_save_dir` (default
`train/<head>_<matching>/<stem>/`):

- `best_min_ade_model.pth` — lowest validation `val_min_ade`.
- `best_min_hd_model.pth`  — lowest validation `val_min_hd`.

Each checkpoint embeds the full validation `metrics` dict at save
time. The fully resolved configuration is dumped alongside as
`config_resolved.json`.

### YAML configuration overlays

Training YAML files merge as **patches** onto
`get_default_road_cpl_config()`; unknown keys raise `KeyError`. Prefer
listing only fields that differ from the Python defaults. Two reference
overlays ship in [`configs/`](configs/):

- [`configs/hungarian_mean_path.yaml`](configs/hungarian_mean_path.yaml) — Hungarian matching with the stable mean-path baseline.
- [`configs/multi_hypothesis.yaml`](configs/multi_hypothesis.yaml) — K-way multi-hypothesis baseline.

Mutually exclusive head modes (`use_cpl`, `use_ar`,
`model.multi_hypothesis.enabled`) are enforced at config time.

## 4. Validation / test-only

```bash
python train.py --test-only \
    --checkpoint train/cpl_grid/<stem>/best_min_ade_model.pth
```

This loads the checkpoint, runs the standard validation pass (loss +
path metrics + visualisations), and exits. The validation metrics
(reported in full-resolution pixel coordinates) are:

| Metric        | Meaning                                                                                  |
|---------------|------------------------------------------------------------------------------------------|
| `min-ADE`     | Average displacement from the closest valid GT path (per-scene min over the K modes).    |
| `min-HD`      | Hausdorff distance to the closest valid GT path.                                         |
| `min-HD-p90`  | 90th percentile of `min-HD` across the val set.                                          |
| `off-road`    | Fraction of predicted points outside the binarised road channel.                         |

`min-ADE` is the headline metric (used to pick `best_min_ade_model.pth`).

## 5. Inference runtime profiling

For paper-style runtime comparisons across model variants (supervised,
multi-hypothesis, CPL, AR), add `--profile-inference` to a
`--test-only` run with `--batch-size 1`. The trainer first runs the
standard validation pass, then a dedicated inference-timing pass over
the eval loader: a few warmup batches (no measurement), followed by a
hook-driven, three-bucket timed loop. The report is printed and saved
as `inference_timing.json` next to the checkpoint.

```bash
python train.py --test-only \
    --checkpoint train/cpl_grid/<stem>/best_min_ade_model.pth \
    --profile-inference \
    --batch-size 1
# Optional: tweak warmup length (default 5).
# --profile-warmup-batches 10
```

Reported keys (milliseconds):

| Key                       | Meaning                                                                  |
|---------------------------|--------------------------------------------------------------------------|
| `*_backbone_ms`           | ResNet feature extractor forward.                                        |
| `*_encoder_ms`            | Transformer encoder forward.                                             |
| `*_seq_head_forward_ms`   | CPL / AR head forward (only when a sequence head is enabled).            |
| `*_heads_ms`              | Heatmap + offset MLPs (and any multi-hypothesis fusion).                 |
| `*_decode_ms`             | `cpl_head.greedy_decode` / `ar_head.greedy_decode` body.                 |
| `*_forward_total_ms`      | Top-level `model.forward`.                                               |
| `*_total_inference_ms`    | `forward_total + decode`.                                                |

Each metric is reported as both `avg_batch_*` (mean over batches) and
`per_example_*` (sum / total examples seen). The profiler is
implemented as forward hooks on `model.backbone`, `model.transformer`,
the optional sequence head, and a non-invasive monkey-patch around
`head.greedy_decode`; the existing inference loop in `train.py` is not
touched. Paper numbers were measured on an NVIDIA A100 with
`--batch-size 1`.

## 6. Comet logging (optional)

Comet logging is opt-in: it activates only when `COMET_API_KEY` is set
in the environment and `comet.enabled` is True (the default). Pass
`--no-comet` to disable it for a single run.

The defaults in [`train_config.py`](train_config.py) use the placeholder
workspace `your-comet-workspace` and project `cpl-road`. Override both
in a YAML overlay (`comet.workspace`, `comet.project_name`) before
publishing your runs.

## Output format

```
data/nuscenes/                         # produced by Step B
├─ train/
│  ├─ 00000000.npz   # image (3, H, W) uint8, points (K_MAX, M_MAX, 2) float32, ...
│  ├─ 00000000.json  # metadata sidecar (schema heatmap_offset_rgb_v1)
│  └─ ...
├─ val/
│  └─ ...
└─ generation_summary.json
```

## Implementation notes

- The model emits raw heatmap logits (no sigmoid); BCE-with-logits is
  applied inside `PathHeadsLoss`.
- Offsets are unconstrained pixel deltas; `L1Loss` (or `SmoothL1Loss`)
  is applied only on cells that the matcher considers positive.
- The Hungarian matcher uses predicted sub-pixel points for the cost,
  tightly coupling the heatmap and offset heads during matching.
- Validation metrics (`min-ADE`, `min-HD`, `min-HD-p90`, `off-road`)
  are computed in full-resolution pixel coordinates so the binarised
  road channel can be queried directly for the off-road rate.
- Generator knobs include `H`, `W` (default `256`), `D` (default `8`
  -> 32x32 grid), `K_MAX` (max paths per sample, default 6), `M_MAX`
  (max points per path after sparsification, default 64), and
  `lane_width_px`.

## Project layout

| File                                                             | Purpose                                                                                          |
| ---                                                              | ---                                                                                              |
| [`train_config.py`](train_config.py)                             | Centralised dataclass configuration and YAML I/O.                                                |
| [`generator.py`](generator.py)                                   | Procedural engine: primitives, planar graph, collision, polylines, polyline sparsifier.          |
| [`generate_data.py`](generate_data.py)                           | CLI entry point for the procedural-only synthetic dataset (Step D fallback).                     |
| [`generate_nuscenes_paths.py`](generate_nuscenes_paths.py)       | CLI entry point for the nuScenes-map BEV generator (Step B).                                     |
| [`split_nuscenes_train_val.py`](split_nuscenes_train_val.py)     | Optional stratified re-split utility (Step C).                                                   |
| [`in_memory_dataset.py`](in_memory_dataset.py)                   | Eager in-memory `Dataset` returning `gt_points`, `num_points`, `valid_paths`.                    |
| [`lazy_dataset.py`](lazy_dataset.py)                             | On-demand counterpart for datasets that don't fit in RAM (`--lazy-data`).                        |
| [`model.py`](model.py)                                           | Backbone + transformer + heatmap/offset/CPL/AR/MH heads.                                         |
| [`path_heads_loss.py`](path_heads_loss.py)                       | `GridMatcher`, `HungarianMatcher`, `PathHeadsLoss`, `MultiHypothesisPathLoss`, `decode_subpixel_points`. |
| [`cpl_module.py`](cpl_module.py)                                 | Plackett-Luce ordering loss + greedy decode.                                                     |
| [`ar_module.py`](ar_module.py)                                   | Autoregressive transformer decoder + greedy decode.                                              |
| [`train.py`](train.py)                                           | Training / eval orchestration, CLI.                                                              |
| [`metrics.py`](metrics.py)                                       | min-ADE / min-HD / min-HD-p90 / off-road, in full-res pixels.                                    |
| [`runtime_profile.py`](runtime_profile.py)                       | Hook-driven inference timing (`--profile-inference`).                                            |
| [`visualize.py`](visualize.py)                                   | Single-sample preview, six-panel training/val debug figure, web gallery.                         |
| [`path_segments.py`](path_segments.py)                           | Polyline / segment helpers used by the generator.                                                |
| [`configs/`](configs/)                                           | YAML overlays: `hungarian_mean_path.yaml`, `multi_hypothesis.yaml`.                              |
