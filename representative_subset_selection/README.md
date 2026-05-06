# Representative Subset Selection on CIFAR-10/100

Code for the **Representative Subset Selection** experiment in the
*Contextual Plackett-Luce* (CPL) paper. See the top-level
[`README.md`](../README.md) for the abstract and a description of the
overall project.

The task: given a "bag" of `N` ImageNet-pretrained ResNet-18 features
sampled from `K` CIFAR classes, predict a subset that **covers every
cluster** while avoiding redundant picks. The ground truth is one
randomly-sampled token per cluster, refreshed every time the bag is
drawn from the dataset, so the model cannot memorise any specific
exemplar — only the cluster structure.

## Methods

Four learned methods plus one non-trained reference baseline:

| `--method`   | Training signal                                                                                                                                                                                                                                                                                                                                                | Inference                                                              |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `cpl`        | Contextual Plackett-Luce ordering loss with a learnable EOS token (variable-size selection).                                                                                                                                                                                                                                                                   | Sequential greedy decode that stops when EOS is selected.              |
| `bce`        | Per-token binary cross-entropy against the random-GT mask, with auto `pos_weight = n_neg / n_pos`.                                                                                                                                                                                                                                                             | Threshold sweep over `sigmoid(logit)` + percentile-based thresholds.   |
| `hungarian`  | DETR-style bipartite matching between valid tokens and the `K` random GT indices (cost: `cls_w * (-log sigmoid(logit_i)) + dist_w * d(f_i, f_gt_j)`) followed by **(a)** BCE on matched / unmatched targets, **(b)** an optional SmoothL1 consistency loss on matched pairs in transformer space, and **(c)** an optional entropy sharpening term. The matching itself is non-differentiable (scipy `linear_sum_assignment`). | Same threshold sweep as `bce` (matching is only a training-time signal). |
| `ar`         | Auto-regressive set-coverage. The bag is encoded once; a small `nn.TransformerDecoder` consumes a teacher-forced length-`K+1` sequence (BOS + a random permutation of the `K` GT indices) under a causal mask. A pointer-network readout produces per-step logits over `N+1` slots; the loss is the same EOS-weighted cross-entropy used by CPL.               | Greedy AR decode: argmax over `N+1` after masking padded and already-selected positions; stops at EOS or step budget. |

K-means with oracle K (number of GT semantic clusters) is reported as a
non-trained reference baseline alongside every method.

## 1. Install

```bash
cd representative_subset_selection
pip install -r requirements.txt
```

The required dependencies are `torch`, `torchvision`, `numpy`, `scipy`,
`scikit-learn`, `matplotlib`, `tqdm`, and `pyyaml`. Two extras are
optional:

- `comet-ml` — only used when Comet logging is enabled (see step 6).
- `umap-learn` — alternative 2D embedding for the visualiser.
  When missing, `--viz-embedding umap` falls back to PCA.

If `scikit-learn` is missing or t-SNE fails, the visualiser also falls
back to PCA.

## 2. Data preparation

CIFAR-10 / CIFAR-100 is downloaded automatically by `torchvision` on the
first run into `--data-root` (default `./data/`). No manual step is
required.

ImageNet-pretrained ResNet-18 features are extracted on the first epoch
and cached in RAM (not on disk), so subsequent epochs reuse them
without recomputation.

## 3. Training

Run all commands from inside `representative_subset_selection/`.

```bash
# CPL on CIFAR-10
python main.py --method cpl --dataset cifar10 --epochs 300

# BCE baseline on CIFAR-10
python main.py --method bce --dataset cifar10 --epochs 300

# Hungarian on CIFAR-10
python main.py --method hungarian --dataset cifar10 --epochs 300

# AR pointer baseline on CIFAR-10
python main.py --method ar --dataset cifar10 --epochs 300
```

To reproduce the paper's CIFAR-100 numbers, swap `--dataset cifar10` for
`--dataset cifar100`. The paper defaults are also the dataclass defaults
in [`config.py`](config.py): `epochs=300`, `batch_size=256`,
`learning_rate=5e-4`, hidden size 512, 3 transformer layers, 4 heads,
GELU pre-norm, dropout 0.1.

Artefacts land under `runs/<run_name>/` (default name
`<method>_<dataset>`):

```
runs/<run_name>/
  config.json
  best.pth                      # best CluF1 across the eval rules
  epoch_NNNN.pth                # snapshot per eval epoch
  pr_epoch_NNNN.png             # CluPrec/CluRec scatter combining all rules
  viz/epoch_NNNN/
    embed_sampleNN.png          # cluster-coloured 2D scatter (raw + transformer h)
    hist_sampleNN.png           # score distribution + thresholds + GT-sample ticks
    grid_sampleNN.png           # image grid: cluster spine + GT/pred outer border
    W_sampleNN.png              # CPL only: W matrix on valid tokens, cluster-sorted
    hungarian_sampleNN.png      # Hungarian only: GT <-> matched-pred tile pairs + costs
    ar_sampleNN.png             # AR only: per-step chosen tile + softmax bar chart (N+1)
```

### YAML configuration overlays

Any field in [`config.py`](config.py) can be overridden from a YAML
file. The YAML overlay is applied first, then CLI flags override the
merged result.

```bash
python main.py --config path/to/overlay.yaml --method cpl
```

## 4. Validation / eval-only

To re-run evaluation and visualisation on a saved checkpoint without
training:

```bash
python main.py --method bce --dataset cifar10 \
    --eval-only --checkpoint-path runs/bce_cifar10/best.pth
```

The evaluation pass reports four cluster-level metrics per inference
rule. The exact ground-truth token index is **not** consumed by any of
these metrics; only the per-token `cluster_labels` are needed, which
matches the random-instance task definition (any token in the right
cluster counts as a hit).

| Metric     | Meaning                                                                                              |
|------------|------------------------------------------------------------------------------------------------------|
| `CluRec`   | Fraction of GT clusters covered by at least one predicted token.                                     |
| `CluPrec`  | (# distinct cluster ids hit by predictions) / (# predictions). Penalises duplicate intra-cluster picks. |
| `CluF1`    | Per-sample harmonic mean of `CluRec` and `CluPrec`, then averaged over the batch.                    |
| `CardErr`  | `mean(\|pred_count - gt_count\|)` — diagnoses under- or over-selection.                              |

`CluF1` is the primary metric used to pick `best.pth`. For BCE and
Hungarian the row of the threshold/percentile sweep with the highest
`CluF1` is selected; for CPL and AR the matching greedy-decode row
(`CPL` / `AR`) is used directly.

## 5. Inference runtime profiling

To reproduce the per-example inference times reported in the paper, run
eval-only with batch size 1 and the timing flag:

```bash
python main.py --method cpl --dataset cifar10 \
    --eval-only --batch-size 1 \
    --log-inference-timing --inference-timing-warmup-batches 100 \
    --checkpoint-path runs/cpl_cifar10/best.pth
```

The trainer first runs `--inference-timing-warmup-batches` batches with
no measurement (CUDA / cuDNN warm-up), then attaches forward hooks and
logs per-region times during the val pass.

| Reported key                       | Meaning                                                                  |
|------------------------------------|--------------------------------------------------------------------------|
| `*_encoder_ms`                     | Transformer encoder forward.                                             |
| `*_forward_total_ms`               | Top-level model forward (encoder + heads).                               |
| `*_forward_heads_ms`               | Heads only (derived as `forward_total - encoder`).                       |
| `*_decoder_ms`                     | AR only: transformer decoder forward.                                    |
| `*_cpl_greedy_ms`                  | CPL only: greedy decode body.                                            |
| `*_ar_autoreg_ms`                  | AR only: full greedy-decode loop body.                                   |
| `*_ar_inference_total_ms`          | AR only: forward + greedy decode total.                                  |

Each metric appears as both `avg_batch_*` and `per_example_*`. Paper
numbers were measured on an NVIDIA A100 with `--batch-size 1`.

## 6. Comet logging (optional)

Comet logging is opt-in: it activates only when `COMET_API_KEY` is set
in the environment. When the key is unset (or `--no-comet` is passed),
training proceeds with no external logging. With the key set, the run
logs:

- Full configuration (flattened `Config` dataclass, e.g. `train.method`).
- `train/loss` per epoch and per-component `train/<method>/...` stats.
- `val/<rule>/<metric>` per eval epoch (e.g. `val/CPL/clu_f1`, `val/HUN-t0.90/clu_rec`).
- All `viz/*` PNG artefacts and the per-epoch CluPrec/CluRec scatter.

The defaults in [`config.py`](config.py) use the placeholder workspace
`your-comet-workspace` and project `cpl-cifar-clustering`; override
both via the CLI:

```bash
export COMET_API_KEY=...
python main.py --method hungarian --dataset cifar100 --epochs 300 \
    --comet-workspace your_workspace --comet-project your_project
```

## CLI reference

The most useful flags exposed by [`config.py`](config.py):

```
--method {cpl, bce, hungarian, ar}
--dataset {cifar10, cifar100}
--epochs N --batch-size B --learning-rate LR --seed S
--num-train-sets N --num-val-sets N --max-set-size N
--min-k N --max-k N --min-samples-per-cluster N --max-samples-per-cluster N

# Selection (shared CPL/AR)
--iterations N

# CPL
--pre-eos-weight W --post-eos-weight W

# BCE
--bce-thresholds 0.7,0.8,0.9 --bce-primary-threshold 0.9 --pos-weight W

# Hungarian
--hungarian-cls-weight W --hungarian-dist-weight W
--hungarian-consistency-weight W --hungarian-entropy-weight W
--hungarian-distance {l2, cosine}
--hungarian-dist-feature-space {transformer, resnet}
--hungarian-pos-weight W
--hungarian-thresholds 0.7,0.8,0.9 --hungarian-primary-threshold 0.9

# AR
--ar-decoder-layers N --ar-decoder-heads N --ar-decoder-ffn-dim N
--ar-decoder-dropout F --ar-decoder-norm-first / --no-ar-decoder-norm-first
--ar-max-selection-steps N --ar-pre-eos-weight W --ar-post-eos-weight W

# Visualisation / Comet
--viz-embedding {tsne, umap, pca} --num-viz-samples N --no-viz
--no-comet --comet-workspace W --comet-project P

# Resume / eval-only / runtime
--load-model --eval-only --checkpoint-path PATH
--log-inference-timing --inference-timing-warmup-batches N
--config path/to/overlay.yaml
```
