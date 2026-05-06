"""Dataclass-based configuration plus argparse and YAML overlay helpers.

The single :class:`Config` object is passed around the training pipeline.
``parse_args`` builds a fully populated :class:`Config` from CLI flags
(optionally overlaying a YAML file via ``--config``) and is the entry
point used by ``main.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

PADDING_CLUSTER = -1
BIG_NEG = -1e9

CIFAR_DISPLAY_NAME = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
CIFAR_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Dataset construction knobs (CIFAR variant + bag sampler ranges)."""

    dataset: str = "cifar100"
    feature_batch_size: int = 256
    max_set_size: int = 160
    min_k: int = 3
    max_k: int = 10
    min_samples_per_cluster: int = 12
    max_samples_per_cluster: int = 40
    num_train_sets: int = 5000
    num_val_sets: int = 1000
    data_root: str = "./data"
    num_workers: int = 2


@dataclass
class ModelConfig:
    """Transformer encoder hyperparameters shared by CPL and BCE.

    ``pair_heads`` is CPL-specific: it controls the number of bilinear
    heads used to compute the pairwise score matrix ``W`` (must divide
    ``hidden``).
    """

    transformer_dim: int = 512
    transformer_heads: int = 4
    transformer_ffn_dim: int = 1024
    transformer_layers: int = 3
    transformer_dropout: float = 0.1
    transformer_activation: str = "gelu"  # "gelu" | "relu"
    transformer_norm_first: bool = True
    pair_heads: int = 4
    hidden: int = 256


@dataclass
class SelectionConfig:
    """Knobs shared by permutation-based selection losses (CPL, AR).

    ``iterations`` controls how many random GT permutations each loss
    averages over per batch. Both losses iterate over all ``K_b + 1``
    prefixes per sampled permutation, so this is the *outer* loop count
    (the per-prefix passes are vectorised inside the loss).
    """

    iterations: int = 10


@dataclass
class CplConfig:
    """CPL-specific knobs: EOS weighting and greedy decode budget."""

    pre_eos_weight: float = 5.0
    post_eos_weight: float = 1.0
    max_selection_steps: int = 20


@dataclass
class BceConfig:
    """BCE baseline knobs.

    ``pos_weight`` is None to auto-set to ``n_neg / n_pos`` per batch.
    ``primary_threshold`` is what the visualisation overlays.
    """

    pos_weight: float | None = None
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    percentiles: tuple[int, ...] = (50, 60, 70, 80, 90)
    primary_threshold: float = 0.5


@dataclass
class HungarianConfig:
    """Hungarian-matching knobs.

    The matching cost for each (token i, gt j) pair is
    ``cls_weight * (-log sigmoid(logit_i)) + dist_weight * d(f_i, f_gt_j)``
    where ``d`` is controlled by ``distance`` (``l2`` or ``cosine``).
    Vectors ``f`` come from ``dist_feature_space``: ``transformer`` uses
    post-encoder ``h``; ``resnet`` uses per-token bag features ``X``
    (frozen ResNet-18 in this repo). The classification term always uses
    ``logit_i`` from the transformer head.

    ``consistency_weight`` scales the SmoothL1 (or cosine-surrogate)
    consistency loss between ``h[match_i]`` and ``h[gt_j]`` on matched
    pairs (autograd ON), always in transformer ``h`` space.

    ``entropy_weight`` scales an optional mean Shannon entropy penalty on
    normalized ``sigmoid(logits)`` over valid bag tokens (sharpening); ``0``
    disables it.

    Inference uses the same sigmoid-threshold / percentile sweep as
    :class:`BceConfig`; the extra knobs here mirror that config so the
    two modes can diverge independently.

    When ``exclude_identity_match`` is True, the cost cell for pairing a
    GT column with the same token index (identity row) is set very high
    before Hungarian assignment so the solver picks the best *other*
    token for that column. Skipped when there is only one valid token
    (no alternative).
    """

    cls_weight: float = 10.0
    dist_weight: float = 0.01
    consistency_weight: float = 0.0
    entropy_weight: float = 0.0
    distance: str = "l2"  # "l2" | "cosine"
    dist_feature_space: str = "resnet"  # "transformer" | "resnet"
    exclude_identity_match: bool = True
    pos_weight: float | None = None
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    percentiles: tuple[int, ...] = (50, 60, 70, 80, 90)
    primary_threshold: float = 0.5


@dataclass
class ArConfig:
    """Auto-regressive set-selection knobs.

    The bag is encoded once by the shared transformer encoder; a small
    ``nn.TransformerDecoder`` consumes a teacher-forced length-``K+1``
    sequence (BOS + GT history) and emits per-step pointer-network logits
    over the ``N+1`` bag-plus-EOS slots. The loss is the same
    cross-entropy with asymmetric EOS weighting used by CPL: at each
    prefix step we ``log_softmax`` the slot logits (with ``BIG_NEG``
    masking on already-selected and padded bag slots), then add
    ``-mean(log p[remaining])`` plus ``pre_eos_weight * p[EOS]``; at the
    final prefix step (all GTs covered) we add ``post_eos_weight *
    (-log p[EOS])``. The number of random GT permutations averaged per
    batch is shared with CPL via :class:`SelectionConfig`.
    """

    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_ffn_dim: int = 1024
    decoder_dropout: float = 0.1
    decoder_norm_first: bool = True
    max_selection_steps: int = 20
    pre_eos_weight: float = 5.0
    post_eos_weight: float = 1.0


@dataclass
class TrainConfig:
    """Training-loop control: method, optimizer, epochs, checkpointing."""

    method: str = "cpl"  # "cpl" | "bce" | "hungarian" | "ar"
    seed: int = 42
    batch_size: int = 256
    epochs: int = 300
    learning_rate: float = 5e-4
    eval_every: int = 1
    checkpoint_dir: str = "runs"
    run_name: str | None = None
    load_model: bool = False
    eval_only: bool = False
    checkpoint_path: str | None = None
    log_inference_timing: bool = False
    inference_timing_warmup_batches: int = 100


@dataclass
class VizConfig:
    """Visualisation knobs (2D embedding, image grid, frequency)."""

    enabled: bool = True
    embedding: str = "tsne"  # "tsne", "umap", or "pca"
    num_viz_samples: int = 6
    viz_every_n_epochs: int = 1
    image_grid: bool = True
    score_histogram: bool = True
    w_heatmap: bool = True


@dataclass
class CometConfig:
    """Comet logging knobs (opt-in via ``COMET_API_KEY``).

    Logging only activates when ``COMET_API_KEY`` is set in the environment
    *and* ``enabled`` is True. Override ``workspace`` / ``project`` via the
    ``--comet-workspace`` / ``--comet-project`` CLI flags or a YAML overlay.
    """

    enabled: bool = True
    workspace: str = "your-comet-workspace"
    project: str = "cpl-cifar-clustering"


@dataclass
class Config:
    """Top-level configuration tree consumed by the training pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    cpl: CplConfig = field(default_factory=CplConfig)
    bce: BceConfig = field(default_factory=BceConfig)
    hungarian: HungarianConfig = field(default_factory=HungarianConfig)
    ar: ArConfig = field(default_factory=ArConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    comet: CometConfig = field(default_factory=CometConfig)

    @property
    def num_classes(self) -> int:
        return CIFAR_NUM_CLASSES[self.data.dataset]

    @property
    def auto_run_name(self) -> str:
        """Default folder/Comet-experiment name when ``run_name`` is unset."""
        return f"{self.train.method}_{self.data.dataset}"

    @property
    def run_name(self) -> str:
        return self.train.run_name or self.auto_run_name

    @property
    def run_dir(self) -> str:
        return os.path.join(self.train.checkpoint_dir, self.run_name)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Recursively convert a :class:`Config` to a JSON-friendly dict."""
    return _asdict_normalised(cfg)


def _asdict_normalised(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict_normalised(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_asdict_normalised(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict_normalised(v) for k, v in obj.items()}
    return obj


def save_config_json(path: str, cfg: Config) -> None:
    """Dump the resolved configuration as JSON next to the checkpoints."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2)


def flatten_for_comet(cfg: Config, prefix: str = "") -> dict[str, Any]:
    """Flatten the nested config to ``a.b.c -> value`` pairs for Comet."""
    flat: dict[str, Any] = {}
    raw = config_to_dict(cfg)

    def _walk(obj: Any, key: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{key}.{k}" if key else k)
        elif isinstance(obj, list):
            flat[key] = ", ".join(str(v) for v in obj)
        else:
            flat[key] = obj

    _walk(raw, prefix)
    return flat


# ---------------------------------------------------------------------------
# YAML overlay (optional)
# ---------------------------------------------------------------------------


def _merge_yaml_into(cfg: Config, yaml_path: str) -> None:
    """Apply a YAML file's keys onto ``cfg`` in place (best-effort)."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - optional dep
        raise SystemExit(f"--config requires pyyaml ({e})") from e
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML at {yaml_path} must be a mapping at top level")
    _apply_dict(cfg, data)


def _apply_dict(target: Any, src: dict[str, Any]) -> None:
    for k, v in src.items():
        if not hasattr(target, k):
            continue
        cur = getattr(target, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _apply_dict(cur, v)
        else:
            if isinstance(v, list) and isinstance(cur, tuple):
                v = tuple(v)
            setattr(target, k, v)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", choices=("cifar10", "cifar100"), default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--feature-batch-size", type=int, default=None)
    p.add_argument("--max-set-size", type=int, default=None)
    p.add_argument("--min-k", type=int, default=None)
    p.add_argument("--max-k", type=int, default=None)
    p.add_argument("--min-samples-per-cluster", type=int, default=None)
    p.add_argument("--max-samples-per-cluster", type=int, default=None)
    p.add_argument("--num-train-sets", type=int, default=None)
    p.add_argument("--num-val-sets", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--transformer-dim", type=int, default=None)
    p.add_argument("--transformer-heads", type=int, default=None)
    p.add_argument("--transformer-ffn-dim", type=int, default=None)
    p.add_argument("--transformer-layers", type=int, default=None)
    p.add_argument("--transformer-dropout", type=float, default=None)
    p.add_argument(
        "--transformer-activation",
        choices=("gelu", "relu"),
        default=None,
    )
    p.add_argument(
        "--transformer-norm-first",
        dest="transformer_norm_first",
        action="store_true",
        default=None,
    )
    p.add_argument(
        "--no-transformer-norm-first",
        dest="transformer_norm_first",
        action="store_false",
        default=None,
    )
    p.add_argument(
        "--pair-heads",
        type=int,
        default=None,
        help="CPL only: number of bilinear heads for the pairwise W matrix",
    )
    p.add_argument("--hidden", type=int, default=None)


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=("Shared CPL/AR: average training loss over this many random GT permutations per batch (each permutation is unrolled over all K_b + 1 prefixes inside the loss)"),
    )


def _add_cpl_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pre-eos-weight", type=float, default=None)
    p.add_argument("--post-eos-weight", type=float, default=None)
    p.add_argument("--max-selection-steps", type=int, default=None)


def _add_bce_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="BCE positive-class weight; default auto = n_neg / n_pos",
    )
    p.add_argument(
        "--bce-thresholds",
        type=str,
        default=None,
        help="Comma-separated decision thresholds for the BCE sweep",
    )
    p.add_argument("--bce-primary-threshold", type=float, default=None)


def _add_hungarian_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--hungarian-cls-weight",
        type=float,
        default=None,
        help="Hungarian classification-cost weight (alpha in the cost matrix)",
    )
    p.add_argument(
        "--hungarian-dist-weight",
        type=float,
        default=None,
        help="Hungarian embedding-cost weight (beta in the cost matrix)",
    )
    p.add_argument(
        "--hungarian-consistency-weight",
        type=float,
        default=None,
        help="Weight of the SmoothL1 consistency loss on matched pairs",
    )
    p.add_argument(
        "--hungarian-entropy-weight",
        type=float,
        default=None,
        help=("Weight of mean Shannon entropy on normalized sigmoid(logits) over valid tokens (0 disables)"),
    )
    p.add_argument(
        "--hungarian-distance",
        choices=("l2", "cosine"),
        default=None,
        help="Embedding distance used in the cost matrix and consistency loss",
    )
    p.add_argument(
        "--hungarian-dist-feature-space",
        choices=("transformer", "resnet"),
        default=None,
        help=("Feature vectors for the Hungarian *distance* term only: transformer h vs frozen ResNet bag features X (consistency stays on h)"),
    )
    p.add_argument(
        "--hungarian-exclude-identity-match",
        action="store_true",
        default=False,
        help=("For each GT column, forbid assigning the token with the same index (forces second-best by cost when another valid token exists)"),
    )
    p.add_argument(
        "--hungarian-pos-weight",
        type=float,
        default=None,
        help="BCE positive-class weight for the matched-targets loss",
    )
    p.add_argument(
        "--hungarian-thresholds",
        type=str,
        default=None,
        help="Comma-separated decision thresholds for the Hungarian eval sweep",
    )
    p.add_argument("--hungarian-primary-threshold", type=float, default=None)


def _add_ar_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ar-decoder-layers", type=int, default=None)
    p.add_argument("--ar-decoder-heads", type=int, default=None)
    p.add_argument("--ar-decoder-ffn-dim", type=int, default=None)
    p.add_argument("--ar-decoder-dropout", type=float, default=None)
    p.add_argument(
        "--ar-decoder-norm-first",
        dest="ar_decoder_norm_first",
        action="store_true",
        default=None,
    )
    p.add_argument(
        "--no-ar-decoder-norm-first",
        dest="ar_decoder_norm_first",
        action="store_false",
        default=None,
    )
    p.add_argument("--ar-max-selection-steps", type=int, default=None)
    p.add_argument(
        "--ar-pre-eos-weight",
        type=float,
        default=None,
        help=("AR: weight on the EOS probability term added at every prefix step before the all-covered step (penalises early EOS)"),
    )
    p.add_argument(
        "--ar-post-eos-weight",
        type=float,
        default=None,
        help=("AR: weight on -log p[EOS] at the all-covered prefix step (rewards picking EOS once everything is selected)"),
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", choices=("cpl", "bce", "hungarian", "ar"), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--load-model", action="store_true", default=False)
    p.add_argument("--eval-only", action="store_true", default=False)
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument(
        "--log-inference-timing",
        action="store_true",
        default=False,
        help=("Print encoder / heads / decode breakdown during validation snapshot collection"),
    )
    p.add_argument(
        "--inference-timing-warmup-batches",
        type=int,
        default=None,
        help=("With --log-inference-timing: run this many val batches without hooks before measurement (CUDA/cuDNN warm-up). Default from config (2). Use 0 to skip."),
    )


def _add_viz_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-viz", dest="viz_enabled", action="store_false", default=None)
    p.add_argument("--viz-embedding", choices=("tsne", "umap", "pca"), default=None)
    p.add_argument("--num-viz-samples", type=int, default=None)
    p.add_argument("--viz-every-n-epochs", type=int, default=None)


def _add_comet_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--no-comet",
        dest="comet_enabled",
        action="store_false",
        default=None,
        help="Disable Comet logging entirely (otherwise auto-enabled when COMET_API_KEY is set)",
    )
    p.add_argument("--comet-workspace", type=str, default=None)
    p.add_argument("--comet-project", type=str, default=None)


def build_argparser() -> argparse.ArgumentParser:
    """Build the unified CLI parser used by :func:`parse_args`."""
    p = argparse.ArgumentParser(description="CPL / BCE / Hungarian cluster selection on CIFAR")
    p.add_argument("--config", type=str, default=None, help="YAML overlay")
    _add_data_args(p)
    _add_model_args(p)
    _add_selection_args(p)
    _add_cpl_args(p)
    _add_bce_args(p)
    _add_hungarian_args(p)
    _add_ar_args(p)
    _add_train_args(p)
    _add_viz_args(p)
    _add_comet_args(p)
    return p


def _apply_args(cfg: Config, args: argparse.Namespace) -> None:
    """Override fields from CLI args; only non-None values are applied."""
    a = args
    # Data
    if a.dataset is not None:
        cfg.data.dataset = a.dataset
    if a.data_root is not None:
        cfg.data.data_root = a.data_root
    for src, dst in (
        ("feature_batch_size", "feature_batch_size"),
        ("max_set_size", "max_set_size"),
        ("min_k", "min_k"),
        ("max_k", "max_k"),
        ("min_samples_per_cluster", "min_samples_per_cluster"),
        ("max_samples_per_cluster", "max_samples_per_cluster"),
        ("num_train_sets", "num_train_sets"),
        ("num_val_sets", "num_val_sets"),
        ("num_workers", "num_workers"),
    ):
        v = getattr(a, src, None)
        if v is not None:
            setattr(cfg.data, dst, v)

    # Model
    for src, dst in (
        ("transformer_dim", "transformer_dim"),
        ("transformer_heads", "transformer_heads"),
        ("transformer_ffn_dim", "transformer_ffn_dim"),
        ("transformer_layers", "transformer_layers"),
        ("transformer_dropout", "transformer_dropout"),
        ("transformer_activation", "transformer_activation"),
        ("transformer_norm_first", "transformer_norm_first"),
        ("pair_heads", "pair_heads"),
        ("hidden", "hidden"),
    ):
        v = getattr(a, src, None)
        if v is not None:
            setattr(cfg.model, dst, v)

    # Selection (shared CPL/AR)
    if getattr(a, "iterations", None) is not None:
        cfg.selection.iterations = a.iterations

    # CPL
    for src, dst in (
        ("pre_eos_weight", "pre_eos_weight"),
        ("post_eos_weight", "post_eos_weight"),
        ("max_selection_steps", "max_selection_steps"),
    ):
        v = getattr(a, src, None)
        if v is not None:
            setattr(cfg.cpl, dst, v)

    # BCE
    if a.pos_weight is not None:
        cfg.bce.pos_weight = a.pos_weight
    if a.bce_thresholds:
        cfg.bce.thresholds = tuple(float(s) for s in a.bce_thresholds.split(","))
    if a.bce_primary_threshold is not None:
        cfg.bce.primary_threshold = a.bce_primary_threshold

    # Hungarian
    for src, dst in (
        ("hungarian_cls_weight", "cls_weight"),
        ("hungarian_dist_weight", "dist_weight"),
        ("hungarian_consistency_weight", "consistency_weight"),
        ("hungarian_entropy_weight", "entropy_weight"),
        ("hungarian_distance", "distance"),
        ("hungarian_dist_feature_space", "dist_feature_space"),
        ("hungarian_pos_weight", "pos_weight"),
        ("hungarian_primary_threshold", "primary_threshold"),
    ):
        v = getattr(a, src, None)
        if v is not None:
            setattr(cfg.hungarian, dst, v)
    if getattr(a, "hungarian_thresholds", None):
        cfg.hungarian.thresholds = tuple(float(s) for s in a.hungarian_thresholds.split(","))
    if getattr(a, "hungarian_exclude_identity_match", False):
        cfg.hungarian.exclude_identity_match = True

    # AR
    for src, dst in (
        ("ar_decoder_layers", "decoder_layers"),
        ("ar_decoder_heads", "decoder_heads"),
        ("ar_decoder_ffn_dim", "decoder_ffn_dim"),
        ("ar_decoder_dropout", "decoder_dropout"),
        ("ar_decoder_norm_first", "decoder_norm_first"),
        ("ar_max_selection_steps", "max_selection_steps"),
        ("ar_pre_eos_weight", "pre_eos_weight"),
        ("ar_post_eos_weight", "post_eos_weight"),
    ):
        v = getattr(a, src, None)
        if v is not None:
            setattr(cfg.ar, dst, v)

    # Train
    for src, dst in (
        ("method", "method"),
        ("seed", "seed"),
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("learning_rate", "learning_rate"),
        ("eval_every", "eval_every"),
        ("checkpoint_dir", "checkpoint_dir"),
        ("run_name", "run_name"),
        ("checkpoint_path", "checkpoint_path"),
    ):
        v = getattr(a, src, None)
        if v is not None:
            setattr(cfg.train, dst, v)
    if a.load_model:
        cfg.train.load_model = True
    if a.eval_only:
        cfg.train.eval_only = True
    if getattr(a, "log_inference_timing", False):
        cfg.train.log_inference_timing = True
    if getattr(a, "inference_timing_warmup_batches", None) is not None:
        cfg.train.inference_timing_warmup_batches = int(a.inference_timing_warmup_batches)

    # Viz
    if a.viz_enabled is False:
        cfg.viz.enabled = False
    if a.viz_embedding is not None:
        cfg.viz.embedding = a.viz_embedding
    if a.num_viz_samples is not None:
        cfg.viz.num_viz_samples = a.num_viz_samples
    if a.viz_every_n_epochs is not None:
        cfg.viz.viz_every_n_epochs = a.viz_every_n_epochs

    # Comet
    if a.comet_enabled is False:
        cfg.comet.enabled = False
    if a.comet_workspace is not None:
        cfg.comet.workspace = a.comet_workspace
    if a.comet_project is not None:
        cfg.comet.project = a.comet_project


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse the CLI into a fully resolved :class:`Config` object."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = Config()
    if args.config:
        _merge_yaml_into(cfg, args.config)
    _apply_args(cfg, args)
    if cfg.data.min_k > cfg.num_classes or cfg.data.max_k > cfg.num_classes:
        raise SystemExit(f"{cfg.data.dataset}: require min_k and max_k <= {cfg.num_classes} (got {cfg.data.min_k}, {cfg.data.max_k})")
    return cfg


__all__ = [
    "BIG_NEG",
    "CIFAR_DISPLAY_NAME",
    "CIFAR_NUM_CLASSES",
    "PADDING_CLUSTER",
    "ArConfig",
    "BceConfig",
    "CometConfig",
    "Config",
    "CplConfig",
    "DataConfig",
    "HungarianConfig",
    "ModelConfig",
    "SelectionConfig",
    "TrainConfig",
    "VizConfig",
    "build_argparser",
    "config_to_dict",
    "flatten_for_comet",
    "parse_args",
    "save_config_json",
]
