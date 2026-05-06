"""Single source of truth for road-cpl: data generation, model, training, and logging.

The procedural generator, training hyperparameters, and optional YAML overrides
use this module. :class:`GeneratorConfig` and ``DEFAULT_CONFIG`` are re-exported
from the legacy :mod:`config` module for backward compatibility.
"""

from __future__ import annotations

import json
import math
import os
import types
import warnings
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

# ---------------------------------------------------------------------------
# Procedural data generation
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Single source of truth for the procedural generator."""

    H: int = 256
    W: int = 256
    D: int = 8
    lane_width_px: int = 12
    canvas_margin_px: int = 8

    K_MAX: int = 6
    # Per-path point cap after arc-length resampling and per-cell deduplication.
    # Max plausible path length on a 256x256 canvas (with leaf extension) is well
    # under M_MAX * D*sqrt(2) ~ 64 * 11.3 = 720 px, so 64 is comfortable.
    M_MAX: int = 64

    branch_prob: float = 0.5
    triple_branch_prob: float = 0.08
    min_fork_dist_from_canvas_lw: float = 4.0

    max_edges: int = 16
    extension_extra_edges: int = 12
    max_growth_steps: int = 32
    retries_per_segment: int = 12
    max_sample_retries: int = 20
    seed_min_forward_lw: float = 6.0
    seed_max_nearest_edge_dist_frac: float = 0.15
    seed_closest_edge_tie_eps_px: float = 1.0
    seed_away_from_tangent_min_rad: float = 0.08726646259971647 * 2
    seed_max_root_attempts: int = 256

    long_mean_lw: float = 8.0
    long_std_lw: float = 4.0
    long_min_lw: float = 4.0
    long_max_lw: float = 16.0

    lat_std_lw: float = 2.5
    lat_max_lw: float = 5.0
    lat_branch_offset_lw: float = 3.0
    sharp_junction_prob: float = 0.5
    sharp_2way_t_weight: float = 0.5

    min_arc_radius_lw: float = 2.5

    polyline_samples_per_lw: float = 1.0
    collision_seam_eps_px: float = 7.0

    ego_arrow_length_lw: float = 2.0
    ego_arrow_tip_ratio: float = 0.4
    ego_arrow_thickness_px: int = 2

    # Ego pose channel: asymmetric directional Gaussian heatmap (see generator.rasterize_ego_heatmap).
    # Sigma values are defined at lane_width_px=12; at runtime they are scaled by lane_width_px/12.
    ego_heatmap_sigma_lateral: float = 2.0
    ego_heatmap_sigma_long_forward: float = 6.0
    ego_heatmap_sigma_long_backward: float = 1.5

    n_samples: int = 10000
    train_frac: float = 0.8
    num_workers: int = 8
    output_dir: str = "data"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.H % self.D != 0 or self.W % self.D != 0:
            raise ValueError(f"H ({self.H}) and W ({self.W}) must be divisible by D ({self.D}).")
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1).")
        if self.K_MAX < 1:
            raise ValueError("K_MAX must be >= 1.")
        if self.M_MAX < 2:
            raise ValueError("M_MAX must be >= 2 (at least two points per path).")
        if not 0.0 <= self.sharp_junction_prob <= 1.0:
            raise ValueError("sharp_junction_prob must be in [0, 1].")
        if not 0.0 <= self.sharp_2way_t_weight <= 1.0:
            raise ValueError("sharp_2way_t_weight must be in [0, 1].")
        if not 0.0 < self.seed_max_nearest_edge_dist_frac <= 1.0:
            raise ValueError("seed_max_nearest_edge_dist_frac must be in (0, 1].")
        if not 0.0 < self.seed_away_from_tangent_min_rad < 0.5 * math.pi:
            raise ValueError("seed_away_from_tangent_min_rad must be in (0, pi/2) radians.")
        if self.seed_closest_edge_tie_eps_px < 0.0:
            raise ValueError("seed_closest_edge_tie_eps_px must be non-negative.")
        if self.seed_max_root_attempts < 1:
            raise ValueError("seed_max_root_attempts must be >= 1.")
        if self.min_fork_dist_from_canvas_lw < 0.0:
            raise ValueError("min_fork_dist_from_canvas_lw must be non-negative.")
        for name in (
            "ego_heatmap_sigma_lateral",
            "ego_heatmap_sigma_long_forward",
            "ego_heatmap_sigma_long_backward",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")

    @property
    def label_H(self) -> int:
        return self.H // self.D

    @property
    def label_W(self) -> int:
        return self.W // self.D


DEFAULT_CONFIG = GeneratorConfig()

# ---------------------------------------------------------------------------
# Model, losses, and training
# ---------------------------------------------------------------------------


@dataclass
class CplHeadConfig:
    """Configuration for the CPL ordering head."""

    max_steps: int = 100
    eos_in_graph: bool = True
    mlp_num_layers: int = 2
    # CPL token-loss mode:
    # - "one_hot_prob": softmax -> probability vector, one-hot target, NLL.
    # - "index_ce": standard PyTorch cross-entropy with class indices.
    loss_type: str = "one_hot_prob"
    eos_init_std: float = 0.02

    def __post_init__(self) -> None:
        lt = (self.loss_type or "").lower()
        if lt not in ("one_hot_prob", "index_ce"):
            raise ValueError("cpl_head.loss_type must be 'one_hot_prob' or 'index_ce'.")
        self.loss_type = lt


@dataclass
class ArHeadConfig:
    """Configuration for the autoregressive decoder head.

    The head is a TransformerDecoder + pointer-network readout over the encoder
    cell embeddings. ``bos_init_std`` initialises the learnable BOS query;
    ``eos_init_std`` initialises the learnable EOS key vector;
    ``step_pos_init_std`` initialises the per-step query positional embedding.
    """

    max_steps: int = 100
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    ffn_dim_multiplier: int = 4
    activation: str = "gelu"
    # Pre-norm transformer decoder (matches modern best practice).
    norm_first: bool = True
    # AR token-loss mode:
    # - "one_hot_prob": softmax -> probability vector, one-hot target, NLL.
    # - "index_ce": standard PyTorch cross-entropy with class indices.
    loss_type: str = "one_hot_prob"
    bos_init_std: float = 0.02
    eos_init_std: float = 0.02
    step_pos_init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("ar_head.max_steps must be >= 1.")
        if self.num_layers < 1:
            raise ValueError("ar_head.num_layers must be >= 1.")
        if self.num_heads < 1:
            raise ValueError("ar_head.num_heads must be >= 1.")
        if self.ffn_dim_multiplier < 1:
            raise ValueError("ar_head.ffn_dim_multiplier must be >= 1.")
        if self.dropout < 0.0:
            raise ValueError("ar_head.dropout must be non-negative.")
        a = (self.activation or "gelu").lower()
        if a not in ("gelu", "relu"):
            raise ValueError("ar_head.activation must be 'gelu' or 'relu'.")
        lt = (self.loss_type or "").lower()
        if lt not in ("one_hot_prob", "index_ce"):
            raise ValueError("ar_head.loss_type must be 'one_hot_prob' or 'index_ce'.")
        self.loss_type = lt
        for name in ("bos_init_std", "eos_init_std", "step_pos_init_std"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"ar_head.{name} must be positive.")


@dataclass
class HeatmapHeadConfig:
    """Per-token MLP widths for the heatmap (logits) and offset (dx, dy) heads."""

    heatmap_hidden: int = 64
    offset_hidden: int = 64


@dataclass
class MultiHypothesisConfig:
    """Configuration for the baseline K-way multi-hypothesis (MH) head.

    When ``enabled`` is true, the model predicts ``num_hypotheses`` parallel
    ``(heatmap, offset)`` maps and a per-hypothesis selector logit. Training
    uses a winner-take-all path loss plus a cross-entropy selector loss
    (weighted by ``ce_weight``). Mutually exclusive with ``run.use_cpl`` and
    ``run.use_ar``.
    """

    enabled: bool = False
    num_hypotheses: int = 6
    ce_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.num_hypotheses < 1:
            raise ValueError("multi_hypothesis.num_hypotheses must be >= 1")
        if self.ce_weight < 0.0:
            raise ValueError("multi_hypothesis.ce_weight must be non-negative")


@dataclass
class ModelConfig:
    in_channels: int = 2
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    grid_size: int = 32
    transformer_dropout: float = 0.1
    ffn_dim_multiplier: int = 4
    transformer_activation: str = "gelu"
    pos_init_std: float = 0.02
    backbone_c1: int = 64
    backbone_c2: int = 128
    cpl_head: CplHeadConfig = field(default_factory=CplHeadConfig)
    ar_head: ArHeadConfig = field(default_factory=ArHeadConfig)
    heatmap_head: HeatmapHeadConfig = field(default_factory=HeatmapHeadConfig)
    multi_hypothesis: MultiHypothesisConfig = field(default_factory=MultiHypothesisConfig)

    def __post_init__(self) -> None:
        if not (self.ffn_dim_multiplier >= 1):
            raise ValueError("ffn_dim_multiplier must be >= 1")
        a = (self.transformer_activation or "gelu").lower()
        if a not in ("gelu", "relu"):
            raise ValueError("transformer_activation must be 'gelu' or 'relu'")


@dataclass
class HeatmapBceConfig:
    """Per-pixel class weights for heatmap ``BCEWithLogits`` (:class:`path_heads_loss.PathHeadsLoss`).

    Used whenever the heatmap classification term is trained (grid or Hungarian matching).
    """

    matched_bce_weight: float = 1.0
    unmatched_bce_weight: float = 1.0


@dataclass
class HungarianCostConfig:
    """Cost terms for predicted-point Hungarian matching against GT points.

    The per-cell cost is::

        cost = dist_weight * ||pred_point - gt_point||
             + prob_weight * (-log_sigmoid(heatmap_logit))

    where ``pred_point = (j*D + dx, i*D + dy)`` in full-resolution pixels.
    """

    dist_weight: float = 1.0
    prob_weight: float = 1.0


@dataclass
class OptimizerConfig:
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class EvalMetricsConfig:
    """Thresholds for converting heatmap+offset outputs to point sets at eval time.

    Validation metrics operate in full-resolution pixel coordinates: the
    drivable-area map is the original road channel binarized at
    ``road_binarize_threshold``; predicted points are sub-pixel ``(y, x)``
    derived from active cells (``sigmoid(heatmap_logit) > path_score_min_prob``,
    or the top-K cells by probability when none clear the threshold).

    ``path_score_min_prob`` is the primary threshold used for ambiguity stratification
    (per ``n_paths`` buckets) and legacy ``val_*`` metric keys.

    When ``path_score_min_prob_grid`` is non-empty and CPL is off, the trainer runs
    additional eval passes at each grid probability and logs ``val_p{tag}_*`` headline
    metrics (global means only) so you can compare scores across heatmap cutoffs.
    """

    road_binarize_threshold: float = 0.1
    path_score_min_prob: float = 0.5
    # path_score_min_prob_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    path_score_min_prob_grid: tuple[float, ...] | None = None
    # Inference-runtime profiling (only honoured in --test-only). When True
    # the test-only path runs a dedicated warmup pass, then attaches
    # forward-timing hooks, runs the eval pass, and writes
    # ``inference_timing.json`` next to the checkpoint.
    log_inference_timing: bool = False
    inference_timing_warmup_batches: int = 100

    def __post_init__(self) -> None:
        if self.inference_timing_warmup_batches < 0:
            raise ValueError("inference_timing_warmup_batches must be non-negative.")


@dataclass
class DataConfig:
    # Default to multiple workers so the GPU is not starved by sync I/O on the
    # main thread (lazy NPZ loads + per-sample tensor work). Tune per machine.
    train_num_workers: int = 8
    val_num_workers: int = 4
    # Keep workers alive across epochs (avoids fork+import cost every epoch).
    persistent_workers: bool = True
    # Number of batches each worker prefetches ahead. Higher hides loader stalls
    # at the cost of RAM. PyTorch default is 2.
    prefetch_factor: int = 4
    # When True (default) datasets open each ``.npz`` on demand inside
    # ``__getitem__`` (lazy_dataset.RoadCplLazyDataset). When False the entire
    # split is preloaded into RAM (in_memory_dataset.RoadCplInMemoryDataset).
    lazy: bool = False


@dataclass
class CometConfig:
    """Comet logging knobs (opt-in via ``COMET_API_KEY``).

    Logging only activates when ``COMET_API_KEY`` is set in the environment
    *and* ``enabled`` is True. Override ``workspace`` / ``project_name`` via a
    YAML overlay or by editing this dataclass.
    """

    enabled: bool = True
    workspace: str = "your-comet-workspace"
    project_name: str = "cpl-road"


@dataclass
class VizConfig:
    num_viz_val_samples: int = 5
    viz_seed: int | None = 42
    val_path_metrics_max_batches: int | None = None


_VALID_MATCHING = ("grid", "hungarian")
_VALID_OFFSET_LOSS = ("l1", "smooth_l1", "mse", "none")


@dataclass
class TrainRunConfig:
    """Top-level training/eval knobs.

    The loss is composed of two orthogonal axes:

    * ``matching``: how heatmap (and offset) targets are assigned to GT points.
      ``"grid"`` places each GT point in the cell it falls into; ``"hungarian"``
      runs bipartite matching between predicted sub-pixel points and GT points.
    * ``use_cpl`` / ``use_ar``: optional sequence heads (mutually exclusive).
      When either sequence head is enabled, it replaces the heatmap term
      **unless** ``matching == \"hungarian\"``: sequence + Hungarian keeps the
      heatmap BCE term so predicted-point matching stays meaningful.
      The offset L1 term always uses ``matching`` to assign offset targets.

    Loss term weights:

    * ``heatmap_weight``: scales BCEWithLogits on the heatmap (grid/hungarian).
    * ``cpl_weight``: scales the CPL ordering loss (only used when ``use_cpl``).
    * ``ar_weight``: scales the AR next-token loss (only used when ``use_ar``).
    * ``offset_weight``: scales the offset regression loss (``offset_loss``).
    * ``offset_loss``: ``"l1"``, ``"smooth_l1"``, ``"mse"``, or ``"none"`` to disable it.
    """

    matching: str = "grid"
    use_cpl: bool = False
    use_ar: bool = False
    use_flip_augmentation: bool = True
    heatmap_weight: float = 10.0
    offset_weight: float = 5.0
    cpl_weight: float = 1.0
    ar_weight: float = 1.0
    offset_loss: str = "l1"  # "l1", "smooth_l1", "mse", or "none"

    batch_size: int = 256
    learning_rate: float = 0.0003
    num_epochs: int = 300
    data_root: str | None = None
    # data_root: str = "data/10k_test_center/"
    checkpoint_path: str | None = None
    test_only: bool = False
    # If set, checkpoints and viz go here; otherwise `finalize_train_run` sets a default under train/<run_label>/.
    save_dir: str | None = None
    comet_name: str | None = None
    # If set (e.g. CLI --name), used as the directory name under train/<run_label>/ and as Comet experiment name.
    run_stem: str | None = None
    # Filled by `finalize_train_run` (not set from YAML; kept for checkpoint JSON):
    resolved_save_dir: str = field(default="")
    resolved_experiment_stem: str = field(default="")

    def __post_init__(self) -> None:
        if self.matching not in _VALID_MATCHING:
            raise ValueError(f"matching must be one of {_VALID_MATCHING}, got {self.matching!r}.")
        if self.offset_loss not in _VALID_OFFSET_LOSS:
            raise ValueError(f"offset_loss must be one of {_VALID_OFFSET_LOSS}, got {self.offset_loss!r}.")
        if self.use_cpl and self.use_ar:
            raise ValueError("use_cpl and use_ar are mutually exclusive.")
        if self.heatmap_weight < 0.0 or self.cpl_weight < 0.0 or self.ar_weight < 0.0 or self.offset_weight < 0.0:
            raise ValueError("loss weights must be non-negative.")

    @property
    def run_label(self) -> str:
        """Short, filesystem-safe descriptor of the loss configuration.

        Used as a folder under ``train/`` and as the prefix for default Comet
        experiment names.
        """
        if self.use_cpl:
            head = "cpl"
        elif self.use_ar:
            head = "ar"
        else:
            head = "sup"
        return f"{head}_{self.matching}"

    @property
    def name(self) -> str:
        """Comet / human-readable run name (set by :func:`finalize_train_run`)."""
        if self.comet_name:
            return self.comet_name
        if self.resolved_experiment_stem:
            return f"{self.run_label}_{self.resolved_experiment_stem}"
        if self.run_stem:
            return str(self.run_stem)
        return f"{self.run_label}_pending"


@dataclass
class RoadCplConfig:
    """Root configuration: generator, training, model, and auxiliary sections."""

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    run: TrainRunConfig = field(default_factory=TrainRunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hungarian: HungarianCostConfig = field(default_factory=HungarianCostConfig)
    heatmap_bce: HeatmapBceConfig = field(default_factory=HeatmapBceConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    eval: EvalMetricsConfig = field(default_factory=EvalMetricsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    comet: CometConfig = field(default_factory=CometConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    def __post_init__(self) -> None:
        # Mutual exclusion across the three head modes (CPL / AR / multi-hypothesis).
        # The MH flag lives under model.multi_hypothesis.enabled but is logically
        # a head-mode toggle, so we enforce the invariant at the top level here.
        active = [
            ("use_cpl", bool(self.run.use_cpl)),
            ("use_ar", bool(self.run.use_ar)),
            ("model.multi_hypothesis.enabled", bool(self.model.multi_hypothesis.enabled)),
        ]
        enabled = [name for name, on in active if on]
        if len(enabled) > 1:
            raise ValueError(f"At most one of use_cpl, use_ar, model.multi_hypothesis.enabled may be True, got: {enabled}")

    @property
    def use_cpl(self) -> bool:
        return bool(self.run.use_cpl)

    @property
    def use_ar(self) -> bool:
        return bool(self.run.use_ar)

    @property
    def use_multi_hypothesis(self) -> bool:
        return bool(self.model.multi_hypothesis.enabled)

    @property
    def learns_heatmap_score(self) -> bool:
        """Whether the heatmap output is supervised with BCE-with-logits.

        Sequence-head + grid (CPL or AR) only trains offsets + sequence ordering
        (heatmap logits are not penalised). Sequence-head + Hungarian adds
        heatmap supervision so Hungarian costs align with trained probabilities.
        Multi-hypothesis baseline always supervises heatmaps (winner-take-all
        BCE on the selected hypothesis).
        """
        r = self.run
        if self.use_multi_hypothesis:
            return True
        use_sequence_head = r.use_cpl or r.use_ar
        return (not use_sequence_head) or (r.matching == "hungarian")


# ---------------------------------------------------------------------------
# Load / merge / export
# ---------------------------------------------------------------------------


def get_default_road_cpl_config() -> RoadCplConfig:
    return RoadCplConfig()


def _is_union_tp(t: Any) -> bool:
    o = get_origin(t)
    if o is Union:
        return True
    u = getattr(types, "UnionType", None)
    if u is not None and o is u:
        return True
    return False


def _is_optional(t: Any) -> bool:
    if not _is_union_tp(t):
        return False
    return type(None) in get_args(t)


def _type_without_none(t: Any) -> Any:
    args = [a for a in get_args(t) if a is not type(None)]
    if len(args) == 1:
        return args[0]
    return t


def _reconstruct_value(th: Any, v: Any) -> Any:
    if v is None and _is_optional(th):
        return None
    inner = th if not _is_optional(th) else _type_without_none(th)
    if is_dataclass(inner) and isinstance(v, dict):
        return _reconstruct_dataclass_from_dict(inner, v)
    o = get_origin(th)
    if o is tuple and isinstance(v, (list, tuple)):
        return tuple(v)
    return v


def _reconstruct_dataclass_from_dict(cls: type, d: dict[str, Any]) -> Any:
    hints = get_type_hints(cls, globalns=globals(), localns=vars())
    names = {f.name for f in fields(cls)}
    for key in d:
        if key not in names:
            raise KeyError(f"Unknown key {key!r} for {cls.__name__}")
    out: dict[str, Any] = {}
    for f in fields(cls):
        th = hints.get(f.name, f.type)
        if f.name in d:
            v = d[f.name]
            out[f.name] = _reconstruct_value(th, v)
        elif f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            out[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            raise KeyError(f"Missing required key {f.name!r} for {cls.__name__}")
    return cls(**out)


def _strict_deep_merge(base: dict[str, Any], patch: dict[str, Any], path: str) -> None:
    for k, v in patch.items():
        p = f"{path}.{k}" if path else k
        if k not in base:
            raise KeyError(f"Unknown config key: {p}")
        b = base[k]
        if isinstance(b, dict) and isinstance(v, dict):
            _strict_deep_merge(b, v, p)
        else:
            base[k] = v


def as_nested_dict_for_merge(cfg: RoadCplConfig) -> dict[str, Any]:
    return asdict(cfg)


def _migrate_legacy_hungarian_bce_yaml(raw: dict[str, Any]) -> None:
    """Move deprecated ``hungarian.{matched,unmatched}_bce_weight`` into ``heatmap_bce``."""
    h = raw.get("hungarian")
    if not isinstance(h, dict):
        return
    moved: dict[str, Any] = {}
    for k in ("matched_bce_weight", "unmatched_bce_weight"):
        if k in h:
            moved[k] = h.pop(k)
    if not moved:
        return
    warnings.warn(
        "hungarian.matched_bce_weight and hungarian.unmatched_bce_weight are deprecated; use heatmap_bce.matched_bce_weight and heatmap_bce.unmatched_bce_weight instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    hb = raw.setdefault("heatmap_bce", {})
    if not isinstance(hb, dict):
        raise ValueError("heatmap_bce must be a mapping when present in YAML")
    for k, v in moved.items():
        hb.setdefault(k, v)


def merge_road_cpl_config(base: RoadCplConfig, patch: dict[str, Any]) -> RoadCplConfig:
    """Return a new config: ``base`` updated with ``patch``. Unknown keys raise KeyError."""
    b = as_nested_dict_for_merge(base)
    _strict_deep_merge(b, patch, "")
    return _reconstruct_dataclass_from_dict(RoadCplConfig, b)  # type: ignore[return-value]


def load_road_cpl_config_from_path(path: Path) -> RoadCplConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    p = path.expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return get_default_road_cpl_config()
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping or empty (null) for defaults")
    _migrate_legacy_hungarian_bce_yaml(raw)
    return merge_road_cpl_config(get_default_road_cpl_config(), raw)


def road_cpl_config_to_json_dict(cfg: RoadCplConfig) -> dict[str, Any]:
    return asdict(cfg)


def save_road_cpl_config_json(path: str | Path, cfg: RoadCplConfig) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(road_cpl_config_to_json_dict(cfg), f, indent=2, sort_keys=True)
        f.write("\n")


def _flatten_d(prefix: str, d: dict[str, Any], out: dict[str, Any]) -> None:
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_d(p, v, out)
        elif isinstance(v, (list, tuple)):
            out[p] = json.dumps(v) if v and isinstance(v[0], (int, float, str, bool, type(None))) else str(v)
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[p] = v
        else:
            out[p] = str(v)


def road_cpl_config_to_comet_params(cfg: RoadCplConfig) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    _flatten_d("cfg", asdict(cfg), flat)
    return flat


def _safe_run_stem(name: str) -> str:
    """Single path segment for train/<run_label>/<stem>/; no directory separators."""
    s = name.strip()
    if not s:
        raise ValueError("run name must be non-empty")
    for sep in (os.sep, os.altsep or ""):
        if sep and sep in s:
            raise ValueError(f"run name must not contain path separators: {name!r}")
    return s


def finalize_train_run(cfg: RoadCplConfig, *, at: datetime | None = None) -> None:
    """Set ``run.resolved_experiment_stem``, ``resolved_save_dir``, and ``comet_name``."""
    t = at or datetime.now()
    r = cfg.run
    label = r.run_label
    if r.run_stem is not None:
        stem = _safe_run_stem(r.run_stem)
        r.resolved_experiment_stem = stem
        r.comet_name = stem
    else:
        stem = f"road_detection_{t.strftime('%Y%m%d_%H%M%S')}"
        r.resolved_experiment_stem = stem
        if r.comet_name is None:
            r.comet_name = f"{label}_{stem}"
    if r.save_dir is not None:
        r.resolved_save_dir = os.path.abspath(r.save_dir)
    else:
        r.resolved_save_dir = f"train/{label}/{r.resolved_experiment_stem}"


def resolve_run_work_dirs(cfg: RoadCplConfig) -> None:
    """After CLI/YAML merge: set output directory and comet name for training or test-only runs."""
    r = cfg.run
    label = r.run_label
    if r.test_only:
        if r.save_dir is not None:
            r.resolved_save_dir = os.path.abspath(r.save_dir)
        elif r.checkpoint_path is not None:
            r.resolved_save_dir = str(Path(r.checkpoint_path).resolve().parent)
        else:
            raise ValueError("test-only requires --checkpoint and/or --out-dir")
        if r.run_stem is not None:
            stem = _safe_run_stem(r.run_stem)
            r.resolved_experiment_stem = stem
            r.comet_name = stem
        else:
            if not r.resolved_experiment_stem:
                r.resolved_experiment_stem = "test"
            if r.comet_name is None:
                r.comet_name = f"{label}_test"
    else:
        finalize_train_run(cfg)


__all__ = [
    "DEFAULT_CONFIG",
    "ArHeadConfig",
    "CometConfig",
    "CplHeadConfig",
    "DataConfig",
    "EvalMetricsConfig",
    "GeneratorConfig",
    "HeatmapBceConfig",
    "HeatmapHeadConfig",
    "HungarianCostConfig",
    "ModelConfig",
    "MultiHypothesisConfig",
    "OptimizerConfig",
    "RoadCplConfig",
    "TrainRunConfig",
    "VizConfig",
    "finalize_train_run",
    "get_default_road_cpl_config",
    "load_road_cpl_config_from_path",
    "merge_road_cpl_config",
    "resolve_run_work_dirs",
    "road_cpl_config_to_comet_params",
    "road_cpl_config_to_json_dict",
    "save_road_cpl_config_json",
    "yaml",
]
