# Contextual Plackett-Luce: An Efficient Neural Model for Probabilistic Sequence Selection under Ambiguity

Anonymous code release accompanying the NeurIPS 2026 submission.
Author identities, affiliations, and external links are intentionally
omitted for double-blind review.

## Abstract

Selecting a coherent sequence or subset of elements is a fundamental
problem in structured prediction, arising in tasks such as detection,
trajectory forecasting, and representative subset selection. In many
such settings, the target is inherently ambiguous: each input admits
multiple valid outputs, while supervision provides only a single
sampled instance. This induces a mismatch between the underlying
multi-modal target distribution and the observed training signal. We
propose **Contextual Plackett-Luce (CPL)**, a structured probabilistic
model for sequence selection that extends the classical Plackett-Luce
model to a context-dependent setting following an Ising-style
parameterisation with unary and pairwise interaction terms. CPL can be
viewed as a hybrid between fully autoregressive prediction and
parallel sequence selection: autoregressive models effectively capture
uncertainty but are computationally expensive on modern parallel
hardware such as GPUs, while parallel methods are efficient but
struggle to represent multi-modal dependencies. CPL combines the
strengths of both by constructing the parameters of a probabilistic
selection model in a fully parallel manner, followed by a lightweight
autoregressive selection process in which each step applies
incremental updates to contextual logits. This decoupling of parallel
scoring and sequential selection enables efficient computation without
sacrificing expressivity. We evaluate CPL on two structured selection
tasks: multi-modal path prediction and representative subset
selection. CPL achieves improved structural consistency and robustness
under ambiguous supervision compared to strong parallel baselines.

## Tasks

This repository contains the full implementation of both experiments
reported in the paper, as two self-contained subprojects:

| Task                                                                                                            | Setting                                                                                                       | Folder                                                                                                                |
| ---                                                                                                             | ---                                                                                                           | ---                                                                                                                   |
| [Multi-modal path prediction](nuscenes_path_prediction/README.md)                                               | Ordered selection: predict one geometrically valid driving path on a BEV rendering of nuScenes road graphs.   | [`nuscenes_path_prediction/`](nuscenes_path_prediction/)                                                              |
| [Representative subset selection](representative_subset_selection/README.md)                                    | Unordered selection: pick one representative per latent cluster from a bag of CIFAR-10 / CIFAR-100 features.  | [`representative_subset_selection/`](representative_subset_selection/)                                                |

Each subfolder ships with its own `README.md`, `requirements.txt`, and
end-to-end install / data / train / eval / runtime instructions.

## Repository layout

```
cpl-to-publish/
├─ README.md                               # this file
├─ .gitignore
├─ nuscenes_path_prediction/               # multi-modal path prediction (Section 4.1 + Appendix C.1)
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ configs/                             # YAML overlays for the matching/head variants
│  ├─ train.py                             # training + --test-only eval + --profile-inference
│  ├─ generate_nuscenes_paths.py           # BEV renderer using the nuScenes maps expansion
│  ├─ split_nuscenes_train_val.py          # optional stratified re-split utility
│  ├─ generate_data.py                     # procedural-only synthetic fallback
│  └─ ...                                  # model, losses, datasets, metrics, runtime profiler, viz
└─ representative_subset_selection/        # representative subset selection (Section 4.2 + Appendix C.2)
   ├─ README.md
   ├─ requirements.txt
   ├─ main.py                              # CLI entry point
   ├─ train.py                             # training + --eval-only + --log-inference-timing
   ├─ config.py                            # dataclass-based configuration + YAML overlay
   └─ ...                                  # data, models, losses, inference rules, metrics, runtime profiler, viz
```

## Quick start

Per-task minimal commands. See each subfolder's `README.md` for the
full set of options.

### Multi-modal path prediction

```bash
cd nuscenes_path_prediction
pip install -r requirements.txt

# 1. Download the nuScenes Maps expansion (free account required) into
#    /path/to/nuscenes/v1.0/maps/expansion/, then render the BEV samples:
python generate_nuscenes_paths.py \
    --dataroot /path/to/nuscenes/v1.0/ \
    --n-samples 10000 --out data/nuscenes/ --workers 4

# 2. Train CPL (or substitute --use-ar / --config configs/hungarian_mean_path.yaml /
#    --config configs/multi_hypothesis.yaml / no flag for the grid baseline)
python train.py --use-cpl --data-root data/nuscenes/

# 3. Evaluate a checkpoint and (optionally) profile per-example inference
python train.py --test-only \
    --checkpoint train/cpl_grid/<stem>/best_min_ade_model.pth \
    --profile-inference --batch-size 1
```

### Representative subset selection

```bash
cd representative_subset_selection
pip install -r requirements.txt

# 1. Train CPL on CIFAR-10 (or substitute --method bce/hungarian/ar)
#    CIFAR-10/100 is downloaded automatically by torchvision on first run.
python main.py --method cpl --dataset cifar10 --epochs 300

# 2. Evaluate a checkpoint and (optionally) profile per-example inference
python main.py --method cpl --dataset cifar10 \
    --eval-only --batch-size 1 --log-inference-timing \
    --checkpoint-path runs/cpl_cifar10/best.pth
```

## Reproducing the paper numbers

Both subprojects ship with the paper's hyperparameters as the dataclass
defaults:

- **Path prediction** (Appendix C.1): AdamW, `lr=3e-4`, weight decay
  `1e-4`, `batch_size=256`, `num_epochs=300`, horizontal-flip
  augmentation, heatmap BCE weight 10, offset L1 weight 5. Hungarian
  matcher defaults: `prob_weight=1`, `dist_weight=1`; the bundled
  [`configs/hungarian_mean_path.yaml`](nuscenes_path_prediction/configs/hungarian_mean_path.yaml)
  overrides to `prob_weight=10`, `dist_weight=0.1`. CPL and AR heads use
  per-step cross-entropy over grid cells (see the nuScenes task README);
  they do not use the CIFAR subset-selection `pre_eos_weight` /
  `post_eos_weight` hyperparameters.
- **Subset selection** (Appendix C.2): Adam, `lr=5e-4`,
  `batch_size=256`, `num_epochs=300`. Bags sample 3-10 CIFAR classes
  with 12-40 images per class, capped at 160 elements per bag (5,000
  training bags, 1,000 validation bags by default). Transformer
  encoder: hidden 512, 3 layers, 4 heads, FFN 1024, GELU pre-norm,
  dropout 0.1. CPL uses 4 bilinear heads for the pairwise `W` matrix.
  Hungarian uses `cls_weight=10`, `dist_weight=0.01` on frozen
  ResNet features, identity matches excluded.

The exact CLI invocations for each baseline (Hungarian set prediction,
multi-hypothesis, AR pointer, CPL) are listed in the per-task READMEs.

## Hardware

All training runs and runtime measurements reported in the paper were
performed on a single **NVIDIA A100** GPU. Inference timings (`Runtime
(ms)` columns in the paper tables) are mean per-example values
collected with `--batch-size 1` and a CUDA / cuDNN warm-up pass before
measurement; see the **Inference runtime profiling** section in each
subfolder's README for the exact invocation.

## License and citation

License and citation information will be added with the de-anonymised
release.
