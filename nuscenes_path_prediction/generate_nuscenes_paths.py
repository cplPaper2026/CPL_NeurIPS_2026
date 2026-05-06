"""Map-only nuScenes generator for the road-cpl ``heatmap_offset_rgb_v1`` schema.

This script does **not** use the ``NuScenes`` object, scenes, samples, or
sensor ego poses. It needs only the four ``NuScenesMap`` instances. Each
sample is built from a randomly-sampled point on the lane network:

1. Pick a lane across the configured maps (optionally including
   lane connectors via CLI).
2. Pick a uniform arclength offset within that lane to define the
   "ego" pose ``(x, y, yaw)``.
3. Render a 3-channel RGB BEV centred on that pose. Layers and colours
   are taken from nuScenes' ``MapExplorer.color_map``:
     * ``drivable_area``  ``#a6cee3`` filled polygons with a thin black
       outline (mirrors ``nusc_map.render_map_patch``).
     * ``road_divider``   ``#cab2d6`` (stroked polylines)
     * ``lane_divider``   ``#6a3d9a`` (stroked polylines)
   Painted on a white background, then a red ego triangle on top.
4. Enumerate all forward-only driving paths (DFS over outgoing lane
   connectivity) starting at the sampled offset. Each path runs until it
   reaches a leaf lane or leaves the BEV image.

Output schema per ``.npz`` (consumed by :class:`in_memory_dataset.RoadCplInMemoryDataset`):

* ``image``        ``uint8 (3, H, W)``           RGB BEV with the ego marker baked in.
* ``points``       ``float32 (K_MAX, M_MAX, 2)`` candidate paths, one per slot.
* ``num_points``   ``int32 (K_MAX,)``
* ``valid_paths``  ``uint8 (K_MAX,)``

There is no canonical GT path; every valid slot is an equally legal
forward continuation, which is exactly what the CPL ordering loss expects
when training on map-derived multi-modal labels.

Usage:
    python generate_nuscenes_paths.py \\
        --dataroot /path/to/nuscenes/v1.0/ \\
        --n-samples 10000 \\
        --out data/nuscenes/ \\
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import traceback
import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import descartes
import matplotlib as mpl
import numpy as np
from generator import prepare_paths_targets, rasterize_ego_heatmap
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon, Polygon as ShapelyPolygon
from shapely.ops import unary_union
from tqdm import tqdm
from train_config import DEFAULT_CONFIG, GeneratorConfig

mpl.use("Agg", force=True)

# ``descartes`` triggers a Shapely deprecation warning per call when it
# casts polygon coordinates via the ``.coords`` array interface. The
# warning is harmless for our pinned shapely 1.x but, at thousands of
# samples per generation run, would otherwise drown the tqdm output.
warnings.filterwarnings(
    "ignore",
    message="The array interface is deprecated.*",
)

_DEFAULT_MAP_LOCATIONS: tuple[str, ...] = (
    "boston-seaport",
    "singapore-onenorth",
    "singapore-hollandvillage",
    "singapore-queenstown",
)

# Colours match nuScenes' ``MapExplorer.color_map`` exactly. The
# drivable-area fill is rendered at devkit alpha (``0.5``); dividers
# and the polygon outline use higher alpha + thinner strokes for a
# crisper, less washed-out look at our 0.5 m/px BEV resolution.
_COLOR_BG: tuple[int, int, int] = (255, 255, 255)
_COLOR_DRIVABLE_HEX: str = "#a6cee3"
_COLOR_ROAD_DIVIDER_HEX: str = "#cab2d6"
_COLOR_LANE_DIVIDER_HEX: str = "#6a3d9a"
_COLOR_DRIVABLE_BOUNDARY: str = "#222222"
_COLOR_EGO: tuple[int, int, int] = (220, 30, 30)
# Lane-direction chevron colour: a dark teal that contrasts with the
# pale-blue drivable fill, the purples used by both divider layers, and
# the saturated red ego marker.
_COLOR_LANE_DIRECTION_HEX: str = "#0e7c66"

# Drivable-area fill alpha (matches ``MapExplorer.render_map_patch``'s
# ``alpha=0.5``).
_DRIVABLE_FILL_ALPHA: float = 0.5
# A thin nearly-opaque outline gives the polygon strict boundaries, like
# the devkit's reference image (``nusc_map.render_map_patch``).
_DRIVABLE_BOUNDARY_ALPHA: float = 0.85
_DRIVABLE_BOUNDARY_LW: float = 0.6
# Dividers are bumped to higher alpha + finer linewidth so they remain
# legible without washing the road colour underneath.
_DIVIDER_ALPHA: float = 0.9
_DIVIDER_LW: float = 0.6
# Chevrons are stroked with the same crisp matplotlib pipeline as
# dividers (no Gaussian blur). Slightly thicker linewidth so the small
# arms (~2 m long) read clearly at the default 0.5 m/px BEV resolution.
_LANE_DIRECTION_ALPHA: float = 0.9
_LANE_DIRECTION_LW: float = 0.6
# Half-opening angle of each chevron (so the full ``>`` opens at 70 deg).
_LANE_DIRECTION_HALF_ANGLE_DEG: float = 35.0

# Map layers we paint, ordered so that overlays end up on top of the
# drivable polygons.
_LINE_LAYERS: tuple[tuple[str, str], ...] = (
    ("road_divider", _COLOR_ROAD_DIVIDER_HEX),
    ("lane_divider", _COLOR_LANE_DIVIDER_HEX),
)
# Polygon layers feeding ``_render_bev_map``'s drivable-surface fill.
# ``drivable_area`` records carry ``polygon_tokens`` (plural list); see
# :func:`_iter_polygon_tokens`.
_DRIVABLE_POLYGON_LAYERS: tuple[str, ...] = ("drivable_area",)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NuScenesGenConfig:
    """All settings specific to the (map-only) nuScenes generator.

    The grid / per-cell sparsification (``D``, ``K_MAX``, ``M_MAX``) is
    governed by the shared :class:`train_config.GeneratorConfig`.

    Canonical ego framing is expressed in meters:
      * ``ego_behind_m`` behind the ego and ``ego_ahead_m`` ahead of the
        ego along the heading axis.
      * The lateral axis is centered, with optional bounded jitter.
    With the defaults (``bev_size=256``, ``meters_per_pixel=0.5``) this gives
    a 128 m span with the ego at 40 m from the rear and 88 m from the front.
    """

    dataroot: str = "data/nuscenes/v1.0/"
    out_dir: str = "data/nuscenes/"
    map_locations: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_MAP_LOCATIONS)

    bev_size: int = 256
    meters_per_pixel: float = 0.5
    ego_behind_m: float = 40.0
    ego_ahead_m: float = 88.0
    ego_lateral_jitter_m: float = 5.0
    # ``None`` -> auto-derive from the BEV diagonal so candidate paths always
    # reach the image edge. Set explicitly to cap the outgoing-lane DFS.
    candidate_lookahead_m: float | None = None
    # Extend each candidate path forward along its last tangent until it
    # hits the BEV edge. Off by default because, with
    # ``require_fully_connected_bev`` on, paths already follow real lanes
    # all the way to the canvas boundary; turning it on extrapolates
    # through whatever happens to be outside the lane network (often
    # non-drivable pixels), which is undesirable for this dataset.
    extend_paths_to_bev_edge: bool = False
    # Reject any sampled BEV that contains a "dangling" lane endpoint
    # (a ``lane`` or ``lane_connector`` whose head / tail lies inside the
    # canvas yet has no incoming / outgoing topology link). With this on,
    # every road element rendered in the canvas is guaranteed to either
    # connect to another mapped element or exit the BEV cleanly, so
    # candidate paths never trail off into non-drivable pixels.
    require_fully_connected_bev: bool = True
    fully_connected_bev_margin_px: float = 2.0
    # Forward-DFS entry sources around ego. Kept at 0 by default so paths start
    # from the sampled primary lane only, which keeps an implicit shared prefix
    # from the ego before route divergence. Set >0 to re-enable nearby-lane
    # multi-source entry search.
    lane_search_radius_m: float = 0.0
    lane_search_max_angle_deg: float = 60.0
    lane_search_max_sources: int = 6
    # Spatial dedup tolerance in meters. Two candidate paths whose per-point
    # distance never exceeds this value over their common length are
    # considered the same topological route — collapsing parallel-lane
    # duplicates into a single path, while paths that diverge at a junction
    # (lateral separation grows fast) stay distinct. Default 4.0 m, slightly
    # wider than a typical 3.5 m lane.
    lane_dedup_tol_m: float = 8.0

    # Per-lane offset search ("best of N"): for the assigned
    # ``(map, lane_token)`` we sample this many candidate ego offsets
    # along the lane and keep the one whose forward-path DFS yields the
    # most distinct paths. Biases the dataset toward multi-path BEVs
    # (junctions, exits) without changing the per-job lane assignment.
    # Set to 1 to disable (use the offset coming straight from the job).
    # The job's original offset is always tried first, so this only
    # *adds* candidates on top of the existing uniform sample.
    paths_search_offsets: int = 6
    # Reject any base sample whose best (post-search) forward-path DFS
    # returns fewer than this many paths. With ``paths_search_offsets``
    # the search returns the *best* offset on the assigned lane, so a
    # rejection here means the lane itself has no multi-path location
    # within ``candidate_lookahead_m`` — there is nothing the offset
    # search can do; we skip and let the worker pool move on. Set to 1
    # to disable the threshold (any nonzero-path sample is accepted).
    # Compensate the lower acceptance rate by scaling ``n_samples`` up.
    min_paths_per_sample: int = 1

    draw_ego_marker: bool = True

    # Lane-direction chevrons: small ``>`` markers placed at fixed
    # arclength intervals along every ``lane`` record's centerline,
    # pointing in the lane's legal driving direction. Junction
    # connectors (``lane_connector``) are intentionally excluded — they
    # cross multiple lanes and would clutter the BEV. Drawn with the
    # same sharp matplotlib pipeline as dividers (no Gaussian blur).
    draw_lane_direction: bool = True
    lane_direction_spacing_m: float = 6.0
    lane_direction_arm_m: float = 2.0
    lane_direction_color: str = _COLOR_LANE_DIRECTION_HEX
    # Job sampling domain: by default sample ego starts from ``lane`` only
    # (not ``lane_connector``) so starts are kept out of junction connectors.
    include_lane_connectors_in_jobs: bool = False

    # Total accepted samples; deterministic 80/20 split by index (mirrors
    # ``generate_data.py``). Set ``n_samples`` higher than the desired output
    # to compensate for failures from empty lanes / fully-clipped paths.
    n_samples: int = 10000
    train_frac: float = 0.8
    num_workers: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.bev_size <= 0:
            raise ValueError("bev_size must be positive.")
        if self.meters_per_pixel <= 0.0:
            raise ValueError("meters_per_pixel must be positive.")
        if self.ego_behind_m <= 0.0 or self.ego_ahead_m <= 0.0:
            raise ValueError("ego_behind_m and ego_ahead_m must be positive.")
        bev_span_m = self.bev_size * self.meters_per_pixel
        span_tol_m = max(2.0, 0.08 * bev_span_m)
        if not math.isclose(self.ego_behind_m + self.ego_ahead_m, bev_span_m, abs_tol=span_tol_m):
            raise ValueError(
                f"ego_behind_m + ego_ahead_m must approximately match total BEV span ({bev_span_m:.3f} m at current resolution; tolerance ±{span_tol_m:.3f} m).",
            )
        if self.ego_lateral_jitter_m < 0.0:
            raise ValueError("ego_lateral_jitter_m must be non-negative.")
        if self.candidate_lookahead_m is None:
            diag_m = self.bev_size * self.meters_per_pixel * math.sqrt(2.0)
            self.candidate_lookahead_m = max(30.0, diag_m * 1.2)
        if self.candidate_lookahead_m <= 0.0:
            raise ValueError("candidate_lookahead_m must be positive.")
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1).")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        if self.paths_search_offsets < 1:
            raise ValueError("paths_search_offsets must be >= 1.")
        if self.min_paths_per_sample < 1:
            raise ValueError("min_paths_per_sample must be >= 1.")
        if self.lane_direction_spacing_m <= 0.0:
            raise ValueError("lane_direction_spacing_m must be positive.")
        if self.lane_direction_arm_m <= 0.0:
            raise ValueError("lane_direction_arm_m must be positive.")
        self.ego_behind_m = float(self.ego_behind_m)
        self.ego_ahead_m = float(self.ego_ahead_m)
        self.ego_lateral_jitter_m = float(self.ego_lateral_jitter_m)
        self.map_locations = tuple(self.map_locations)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    gen_cfg = _gen_cfg_from_args(args)
    cfg = _generator_cfg_for_bev(gen_cfg)

    out_dir = Path(gen_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(gen_cfg)
    if not jobs:
        print("No lanes available in the requested maps.", file=sys.stderr)
        return 1

    successes, failures, failure_reasons = _run_pool(jobs, cfg, gen_cfg)

    summary = {
        "requested": len(jobs),
        "successes": successes,
        "failures": failures,
        "train_count": sum(1 for j in jobs if j.split == "train"),
        "val_count": sum(1 for j in jobs if j.split == "val"),
        "failure_reasons": failure_reasons,
        "output_dir": str(out_dir.resolve()),
        "source": "nuscenes_map_only",
        "schema": "heatmap_offset_rgb_v1",
        "map_locations": list(gen_cfg.map_locations),
    }
    with open(out_dir / "generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    d = NuScenesGenConfig()
    p = argparse.ArgumentParser(description="Generate the road-cpl nuScenes dataset (map-only, drivable-area BEV).")
    p.add_argument("--dataroot", type=str, default=d.dataroot, help="nuScenes dataroot (must contain maps/).")
    p.add_argument("--out", type=str, default=d.out_dir, help="Output directory (default: %(default)s).")
    p.add_argument(
        "--maps",
        type=str,
        default=",".join(d.map_locations),
        help="Comma-separated map locations to sample from (default: all four).",
    )
    p.add_argument("-n", "--n-samples", type=int, default=d.n_samples, help="Total samples to attempt (default: %(default)s).")
    p.add_argument("--train-frac", type=float, default=d.train_frac, help="Train fraction in [0, 1] (default: %(default)s).")
    p.add_argument("--workers", type=int, default=d.num_workers, help="Parallel worker processes (default: %(default)s).")
    p.add_argument("--seed", type=int, default=d.seed, help="Base seed (default: %(default)s).")
    p.add_argument("--bev-size", type=int, default=d.bev_size, help="BEV side length in pixels (default: %(default)s).")
    p.add_argument("--meters-per-pixel", type=float, default=d.meters_per_pixel, help="BEV resolution m/px (default: %(default)s).")
    p.add_argument(
        "--ego-behind-m",
        type=float,
        default=d.ego_behind_m,
        help="Meters of context behind ego along BEV heading (default: %(default)s).",
    )
    p.add_argument(
        "--ego-ahead-m",
        type=float,
        default=d.ego_ahead_m,
        help="Meters of context ahead of ego along BEV heading (default: %(default)s).",
    )
    p.add_argument(
        "--ego-lateral-jitter-m",
        type=float,
        default=d.ego_lateral_jitter_m,
        help="Max deterministic lateral ego jitter (uniform in [-m, +m]) (default: %(default)s).",
    )
    p.add_argument(
        "--extend-paths",
        dest="extend_paths_to_bev_edge",
        action="store_true",
        default=d.extend_paths_to_bev_edge,
        help="Extrapolate each candidate path's last tangent to the BEV edge (off by default; only safe when require_fully_connected_bev is also off).",
    )
    p.add_argument(
        "--no-fully-connected-bev",
        dest="require_fully_connected_bev",
        action="store_false",
        help="Allow sampling BEVs that contain dangling lane endpoints inside the canvas.",
    )
    p.set_defaults(require_fully_connected_bev=d.require_fully_connected_bev)
    p.add_argument(
        "--candidate-lookahead-m",
        type=float,
        default=None,
        help="Outgoing-lane DFS arclength budget in meters (default: auto, ~1.2x BEV diagonal so paths reach the image edge).",
    )
    p.add_argument("--no-ego-marker", dest="draw_ego_marker", action="store_false", help="Skip the directional ego Gaussian overlay.")
    p.add_argument(
        "--no-lane-direction",
        dest="draw_lane_direction",
        action="store_false",
        help="Skip the lane-direction chevron overlay.",
    )
    p.set_defaults(draw_lane_direction=d.draw_lane_direction)
    p.add_argument(
        "--lane-direction-spacing-m",
        type=float,
        default=d.lane_direction_spacing_m,
        help="Arclength interval between consecutive chevrons on a lane, in meters (default: %(default)s).",
    )
    p.add_argument(
        "--lane-direction-arm-m",
        type=float,
        default=d.lane_direction_arm_m,
        help="Half-length of each chevron arm, in meters (default: %(default)s).",
    )
    p.add_argument(
        "--lane-direction-color",
        type=str,
        default=d.lane_direction_color,
        help="Matplotlib colour spec for chevrons (default: %(default)s).",
    )
    p.add_argument(
        "--include-lane-connectors-in-jobs",
        dest="include_lane_connectors_in_jobs",
        action="store_true",
        default=d.include_lane_connectors_in_jobs,
        help=("Also sample base ego starts from lane_connector tokens (off by default; when off, starts are sampled from lane tokens only)."),
    )
    p.add_argument(
        "--paths-search-offsets",
        type=int,
        default=d.paths_search_offsets,
        help=(
            "Per-lane best-of-N offset search: evaluate this many ego offsets along the assigned lane "
            "and keep the one yielding the most forward paths (default: %(default)s; 1 disables)."
        ),
    )
    p.add_argument(
        "--min-paths-per-sample",
        type=int,
        default=d.min_paths_per_sample,
        help=("Reject any base whose best (post-search) DFS returns fewer than this many forward paths (default: %(default)s; 1 disables, accepting every nonzero-path sample)."),
    )
    return p.parse_args()


def _gen_cfg_from_args(args: argparse.Namespace) -> NuScenesGenConfig:
    maps = tuple(m.strip() for m in args.maps.split(",") if m.strip())
    return NuScenesGenConfig(
        dataroot=args.dataroot,
        out_dir=args.out,
        map_locations=maps or _DEFAULT_MAP_LOCATIONS,
        bev_size=args.bev_size,
        meters_per_pixel=args.meters_per_pixel,
        ego_behind_m=args.ego_behind_m,
        ego_ahead_m=args.ego_ahead_m,
        ego_lateral_jitter_m=args.ego_lateral_jitter_m,
        candidate_lookahead_m=args.candidate_lookahead_m,
        extend_paths_to_bev_edge=bool(args.extend_paths_to_bev_edge),
        require_fully_connected_bev=bool(args.require_fully_connected_bev),
        draw_ego_marker=bool(args.draw_ego_marker),
        draw_lane_direction=bool(args.draw_lane_direction),
        lane_direction_spacing_m=float(args.lane_direction_spacing_m),
        lane_direction_arm_m=float(args.lane_direction_arm_m),
        lane_direction_color=str(args.lane_direction_color),
        include_lane_connectors_in_jobs=bool(args.include_lane_connectors_in_jobs),
        paths_search_offsets=args.paths_search_offsets,
        min_paths_per_sample=args.min_paths_per_sample,
        n_samples=args.n_samples,
        train_frac=args.train_frac,
        num_workers=args.workers,
        seed=args.seed,
    )


def _generator_cfg_for_bev(gen_cfg: NuScenesGenConfig) -> GeneratorConfig:
    """Synthesize a :class:`GeneratorConfig` whose ``H``/``W`` match the BEV.

    ``D`` must divide ``bev_size``; we keep ``DEFAULT_CONFIG.D`` when
    possible and fall back to the largest valid divisor in (8, 4, 2, 1).
    """
    d = int(DEFAULT_CONFIG.D)
    if gen_cfg.bev_size % d != 0:
        for cand in (8, 4, 2, 1):
            if gen_cfg.bev_size % cand == 0:
                d = cand
                break
    return replace(DEFAULT_CONFIG, H=gen_cfg.bev_size, W=gen_cfg.bev_size, D=d)


# ---------------------------------------------------------------------------
# Job enumeration: sample (map_name, lane_token, offset_frac) start points.
# ---------------------------------------------------------------------------


@dataclass
class SampleJob:
    idx: int
    map_name: str
    lane_token: str
    offset_frac: float
    split: str


def _build_jobs(gen_cfg: NuScenesGenConfig) -> list[SampleJob]:
    """Pick ``n_samples`` lane-and-offset start points uniformly across maps.

    Sampling is done in the parent process so the worker pool only has to
    do per-sample rendering. We sample (a) a ``(map_name, lane_token)``
    pair uniformly from all available ``lane`` records across the configured
    maps (optionally including ``lane_connector``) and (b) an arclength
    offset uniform in [0, 1].
    The 80/20 split is then applied deterministically by index.
    """
    flat: list[tuple[str, str]] = []
    for loc in gen_cfg.map_locations:
        m = NuScenesMap(dataroot=gen_cfg.dataroot, map_name=loc)
        flat.extend((loc, lane["token"]) for lane in m.lane)
        if gen_cfg.include_lane_connectors_in_jobs:
            flat.extend((loc, lc["token"]) for lc in m.lane_connector)
    if not flat:
        return []

    rng = np.random.default_rng(gen_cfg.seed)
    pick_idx = rng.integers(0, len(flat), size=gen_cfg.n_samples)
    offsets = rng.uniform(0.0, 1.0, size=gen_cfg.n_samples).astype(np.float64)

    n_train = int(gen_cfg.n_samples * gen_cfg.train_frac)
    jobs: list[SampleJob] = []
    for i in range(gen_cfg.n_samples):
        m_name, lane_tok = flat[int(pick_idx[i])]
        split = "train" if i < n_train else "val"
        jobs.append(SampleJob(idx=i, map_name=m_name, lane_token=lane_tok, offset_frac=float(offsets[i]), split=split))
    return jobs


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------


_WORKER_GEN: NuScenesDatasetGenerator | None = None


def _init_worker(cfg_dict: dict, gen_cfg_dict: dict) -> None:
    global _WORKER_GEN  # noqa: PLW0603 - per-process worker singleton
    cfg = GeneratorConfig(**cfg_dict)
    gen_cfg = NuScenesGenConfig(**gen_cfg_dict)
    _WORKER_GEN = NuScenesDatasetGenerator(cfg, gen_cfg)


def _worker(args: tuple[int, str, str, float, str, str]) -> tuple[int, int, str]:
    """Generate one sample for one base lane-offset job.

    Returns ``(base_idx, n_written, failure_msg)``. ``n_written == 0``
    indicates the base was filtered out. ``failure_msg`` carries the
    traceback if an exception escaped, else empty (``filtered`` for the
    no-output case).
    """
    idx, map_name, lane_tok, offset_frac, split, out_dir = args
    try:
        assert _WORKER_GEN is not None, "worker not initialised"
        sample = _WORKER_GEN.generate(map_name, lane_tok, offset_frac, sample_idx=idx)
        if sample is None:
            return idx, 0, "filtered"
        sample_id = f"{idx:08d}"
        _write_sample(Path(out_dir), sample_id, sample, split)
        return idx, 1, ""
    except Exception:
        return idx, 0, traceback.format_exc()


def _run_pool(
    jobs: list[SampleJob],
    cfg: GeneratorConfig,
    gen_cfg: NuScenesGenConfig,
) -> tuple[int, int, dict[str, int]]:
    out_dir = str(Path(gen_cfg.out_dir).resolve())
    cfg_dict = asdict(cfg)
    gen_cfg_dict = asdict(gen_cfg)
    tasks = [(j.idx, j.map_name, j.lane_token, j.offset_frac, j.split, out_dir) for j in jobs]

    if gen_cfg.num_workers <= 1:
        _init_worker(cfg_dict, gen_cfg_dict)
        iterator: Iterator[tuple[int, int, str]] = (_worker(t) for t in tasks)
        pool = None
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=gen_cfg.num_workers,
            initializer=_init_worker,
            initargs=(cfg_dict, gen_cfg_dict),
        )
        iterator = pool.imap_unordered(_worker, tasks, chunksize=4)

    successes = 0
    failures = 0
    failure_reasons: dict[str, int] = {}
    pbar = tqdm(total=len(tasks), desc="Generating", unit="sample")
    for _idx, n_written, msg in iterator:
        pbar.update(1)
        if n_written > 0:
            successes += n_written
        if n_written < 1:
            failures += 1
            short = (msg.splitlines()[-1] if msg else "filtered")[:80]
            failure_reasons[short] = failure_reasons.get(short, 0) + 1
    pbar.close()

    if pool is not None:
        pool.close()
        pool.join()
    return successes, failures, failure_reasons


def _write_sample(out_dir: Path, sample_id: str, sample: dict, split: str) -> None:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_dir / f"{sample_id}.npz",
        image=sample["image"].astype(np.uint8),
        points=sample["points"].astype(np.float32),
        num_points=sample["num_points"].astype(np.int32),
        valid_paths=sample["valid_paths"].astype(np.uint8),
    )
    meta = dict(sample["meta"])
    meta["sample_id"] = sample_id
    meta["split"] = split
    with open(split_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class NuScenesDatasetGenerator:
    """Per-worker holder for the four ``NuScenesMap`` instances.

    Constructing a :class:`nuscenes.map_expansion.map_api.NuScenesMap`
    parses ~100 MB of map JSON; the parent process avoids paying that
    cost by deferring the import to ``_init_worker``.
    """

    def __init__(self, cfg: GeneratorConfig, gen_cfg: NuScenesGenConfig) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.maps: dict[str, NuScenesMap] = {loc: NuScenesMap(dataroot=gen_cfg.dataroot, map_name=loc) for loc in gen_cfg.map_locations}

    def generate(self, map_name: str, lane_token: str, offset_frac: float, sample_idx: int) -> dict | None:
        """Build one map-only sample dict, or ``None`` to skip the base.

        Pipeline (shared cost is paid once per accepted base):

        1. Best-of-N offset search on the assigned lane: try
           ``paths_search_offsets`` candidate offsets and keep the one
           whose connectivity filter + forward-path DFS yields the most
           distinct paths (rendering is *not* run during the search).
           This biases the dataset toward multi-path BEVs without
           changing the per-job lane assignment.
        2. Render one canonical heading-right BEV at the selected offset,
           using deterministic lateral-only ego jitter and no augmentation.
        """
        nusc_map = self.maps.get(map_name)
        if nusc_map is None:
            return None

        # Connectivity filter is checked per-offset, but with the *base*
        # ego_pixel (yaw-invariant in this filter), so we draw it once
        # outside the search and reuse it.
        base_ego_pixel = _sample_ego_pixel(self.gen_cfg, sample_idx)

        best = _pick_best_offset_on_lane(
            nusc_map,
            lane_token,
            seed_offset=offset_frac,
            sample_idx=sample_idx,
            base_ego_pixel=base_ego_pixel,
            cfg=self.cfg,
            gen_cfg=self.gen_cfg,
        )
        if best is None:
            return None
        yaw = float(best.base_yaw)
        return self._build_one_sample(
            nusc_map=nusc_map,
            ego_xy=best.ego_xy,
            yaw=yaw,
            ego_pixel=base_ego_pixel,
            candidates_global=best.candidates_global,
            map_name=map_name,
            lane_token=lane_token,
            offset_frac=best.offset_frac,
            sample_idx=sample_idx,
            base_yaw=float(best.base_yaw),
        )

    def _build_one_sample(
        self,
        *,
        nusc_map,
        ego_xy: np.ndarray,
        yaw: float,
        ego_pixel: np.ndarray,
        candidates_global: list[np.ndarray],
        map_name: str,
        lane_token: str,
        offset_frac: float,
        sample_idx: int,
        base_yaw: float,
    ) -> dict | None:
        """Render one sample (canonical yaw + ego_pixel) into the schema dict."""
        candidates_px = _candidates_global_to_pixels(
            candidates_global,
            ego_xy=ego_xy,
            yaw=yaw,
            ego_pixel=ego_pixel,
            gen_cfg=self.gen_cfg,
        )
        candidates_anchored = [_prepend_ego_anchor(c, ego_pixel) for c in candidates_px]
        # Pixel-space dedup runs here, after clipping + anchoring, so we
        # collapse paths that look identical in the BEV (e.g. parallel
        # sibling lanes, or paths whose only divergence is outside the
        # canvas). Tolerance is converted from the user-facing meters
        # config so the same lane-width threshold works at any
        # meters_per_pixel.
        tol_px = self.gen_cfg.lane_dedup_tol_m / max(self.gen_cfg.meters_per_pixel, 1e-6)
        candidates_anchored = _drop_redundant_paths_pixel(candidates_anchored, tol_px)
        candidates_anchored = candidates_anchored[: self.cfg.K_MAX]
        if not candidates_anchored:
            return None

        ego_theta_in_bev = _ego_heading_theta_in_bev(
            ego_xy=ego_xy,
            base_yaw=base_yaw,
            yaw=yaw,
            ego_pixel=ego_pixel,
            m_per_px=self.gen_cfg.meters_per_pixel,
        )
        image = _render_bev_map(
            nusc_map,
            ego_xy,
            yaw,
            ego_pixel,
            self.gen_cfg,
            self.cfg,
            ego_theta_in_bev,
        )

        points, num_points, valid_paths = prepare_paths_targets(self.cfg, candidates_anchored)
        # Final ``min_paths_per_sample`` gate: enforced post-dedup so the
        # threshold matches the on-disk ``n_paths`` field. Pixel-space
        # dedup can collapse sibling lanes that the search-time
        # (world-space) count saw as distinct, hence the second check.
        if int(valid_paths.sum()) < self.gen_cfg.min_paths_per_sample:
            return None

        meta = {
            "format_version": 6,
            "schema": "heatmap_offset_rgb_v1",
            "source": "nuscenes_map_only",
            "map_name": map_name,
            "lane_token": lane_token,
            "lane_offset_frac": float(offset_frac),
            "ego_global": {"x": float(ego_xy[0]), "y": float(ego_xy[1]), "yaw_rad": float(yaw)},
            "ego_pixel": [float(ego_pixel[0]), float(ego_pixel[1])],
            "augmentation": {
                "base_idx": int(sample_idx),
                "aug_idx": 0,
                "yaw_delta_rad": 0.0,
                "base_yaw_rad": float(base_yaw),
            },
            "n_paths": int(valid_paths.sum()),
            "config": {
                "H": self.cfg.H,
                "W": self.cfg.W,
                "D": self.cfg.D,
                "K_MAX": self.cfg.K_MAX,
                "M_MAX": self.cfg.M_MAX,
                "meters_per_pixel": self.gen_cfg.meters_per_pixel,
                "candidate_lookahead_m": self.gen_cfg.candidate_lookahead_m,
                "ego_behind_m": self.gen_cfg.ego_behind_m,
                "ego_ahead_m": self.gen_cfg.ego_ahead_m,
                "ego_lateral_jitter_m": self.gen_cfg.ego_lateral_jitter_m,
                "extend_paths_to_bev_edge": self.gen_cfg.extend_paths_to_bev_edge,
                "require_fully_connected_bev": self.gen_cfg.require_fully_connected_bev,
                "lane_dedup_tol_m": self.gen_cfg.lane_dedup_tol_m,
                "augmentations_per_sample": 1,
                "aug_yaw_max_deg": 0.0,
                "aug_translate_frac": [0.0, 0.0],
                "paths_search_offsets": self.gen_cfg.paths_search_offsets,
                "min_paths_per_sample": self.gen_cfg.min_paths_per_sample,
                "include_lane_connectors_in_jobs": self.gen_cfg.include_lane_connectors_in_jobs,
                "draw_lane_direction": self.gen_cfg.draw_lane_direction,
                "lane_direction_spacing_m": self.gen_cfg.lane_direction_spacing_m,
                "lane_direction_arm_m": self.gen_cfg.lane_direction_arm_m,
                "lane_direction_color": self.gen_cfg.lane_direction_color,
            },
        }
        return {
            "image": image,
            "points": points,
            "num_points": num_points,
            "valid_paths": valid_paths,
            "meta": meta,
        }


# ---------------------------------------------------------------------------
# Lane sampling: start pose from a ``(lane_token, offset_frac)`` pair.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OffsetEvaluation:
    """Result of evaluating one candidate ego offset on a lane.

    Holds everything ``generate`` needs to render the BEV without
    redoing the connectivity filter or the forward-path DFS.
    """

    ego_xy: np.ndarray
    base_yaw: float
    start_idx: int
    offset_frac: float
    candidates_global: list[np.ndarray]
    lane_poses_xy: np.ndarray


def _pick_best_offset_on_lane(
    nusc_map,
    lane_token: str,
    *,
    seed_offset: float,
    sample_idx: int,
    base_ego_pixel: np.ndarray,
    cfg: GeneratorConfig,
    gen_cfg: NuScenesGenConfig,
) -> _OffsetEvaluation | None:
    """Pick the offset on ``lane_token`` that yields the most forward paths.

    Discretises the lane once, then evaluates up to
    ``gen_cfg.paths_search_offsets`` candidate offsets — including the
    job's original ``seed_offset`` — running only the cheap stages
    (connectivity filter + the forward-path DFS, no rendering). Keeps
    the offset whose DFS returned the most distinct candidate paths and
    short-circuits as soon as we reach ``cfg.K_MAX`` paths (any extras
    would be truncated downstream anyway). The result is then also
    required to clear ``gen_cfg.min_paths_per_sample`` — if even the
    best offset on the lane is below that threshold, we return ``None``
    and let the worker pool move on to the next job assignment.

    Returns ``None`` when every candidate is rejected by the
    connectivity filter, yields zero forward paths, or the best
    candidate has fewer than ``min_paths_per_sample`` paths.
    """
    poses, lane_poses_xy = _discretise_lane(nusc_map, lane_token)
    if poses is None:
        return None
    target_paths = min(int(cfg.K_MAX), max(1, int(gen_cfg.min_paths_per_sample)))

    best: _OffsetEvaluation | None = None
    for offset in _candidate_offsets(seed_offset, gen_cfg, sample_idx):
        evaluation = _evaluate_offset_for_paths(
            nusc_map,
            lane_token=lane_token,
            poses=poses,
            lane_poses_xy=lane_poses_xy,
            offset_frac=offset,
            base_ego_pixel=base_ego_pixel,
            cfg=cfg,
            gen_cfg=gen_cfg,
        )
        if evaluation is None:
            continue
        if best is None or len(evaluation.candidates_global) > len(best.candidates_global):
            best = evaluation
        # Stop once we hit the K_MAX slot ceiling: extras would be
        # truncated. We *do not* early-stop at ``target_paths`` because
        # going further might find an even richer offset and we want
        # the dataset biased toward the maximum reachable on the lane.
        if len(best.candidates_global) >= cfg.K_MAX:
            break
    if best is None or len(best.candidates_global) < target_paths:
        return None
    return best


def _discretise_lane(
    nusc_map,
    lane_token: str,
) -> tuple[list | None, np.ndarray | None]:
    """Resolve the lane's arcline once and return both pose forms used downstream.

    ``poses`` (list of ``(x, y, yaw)``) is what we use to fetch the pose
    at any offset; ``lane_poses_xy`` (``(N, 2)``) is what the DFS needs.
    """
    try:
        arc = nusc_map.get_arcline_path(lane_token)
    except ValueError:
        return None, None
    poses = arcline_path_utils.discretize_lane(arc, 0.5)
    if not poses:
        return None, None
    lane_poses_xy = np.asarray([(p[0], p[1]) for p in poses], dtype=np.float64)
    return poses, lane_poses_xy


def _candidate_offsets(
    seed_offset: float,
    gen_cfg: NuScenesGenConfig,
    sample_idx: int,
) -> list[float]:
    """Build the list of offsets to try, deterministic per ``sample_idx``.

    The job's original ``seed_offset`` is always first so the search
    only ever *expands* the candidate set — disabling the search
    (``paths_search_offsets == 1``) reproduces the previous behaviour
    exactly.
    """
    n_search = max(1, int(gen_cfg.paths_search_offsets))
    if n_search <= 1:
        return [float(seed_offset)]
    rng = np.random.default_rng(
        (
            int(gen_cfg.seed) & 0xFFFFFFFF,
            int(sample_idx) & 0xFFFFFFFF,
            0xCAFEBABE,
        )
    )
    extras = rng.uniform(0.0, 1.0, size=n_search - 1)
    return [float(seed_offset)] + [float(o) for o in extras]


def _evaluate_offset_for_paths(
    nusc_map,
    *,
    lane_token: str,
    poses: list,
    lane_poses_xy: np.ndarray,
    offset_frac: float,
    base_ego_pixel: np.ndarray,
    cfg: GeneratorConfig,
    gen_cfg: NuScenesGenConfig,
) -> _OffsetEvaluation | None:
    """Cheap-only evaluation of one offset: filter + DFS, no rendering."""
    n = len(poses)
    start_idx = int(np.clip(round(offset_frac * (n - 1)), 0, n - 1))
    x, y, yaw = poses[start_idx]
    ego_xy = np.array([float(x), float(y)], dtype=np.float64)
    base_yaw = float(yaw)

    if gen_cfg.require_fully_connected_bev and not _bev_is_fully_connected(nusc_map, ego_xy, base_yaw, base_ego_pixel, gen_cfg):
        return None

    candidates_global = _enumerate_forward_paths_multi_source(
        nusc_map,
        ego_xy=ego_xy,
        yaw=base_yaw,
        primary_lane=lane_token,
        primary_poses=lane_poses_xy,
        primary_start_idx=start_idx,
        max_arclen_m=gen_cfg.candidate_lookahead_m or 0.0,
        max_paths=max(8, cfg.K_MAX * 2),
        search_radius_m=gen_cfg.lane_search_radius_m,
        max_angle_deg=gen_cfg.lane_search_max_angle_deg,
        max_sources=gen_cfg.lane_search_max_sources,
    )
    if not candidates_global:
        return None
    return _OffsetEvaluation(
        ego_xy=ego_xy,
        base_yaw=base_yaw,
        start_idx=start_idx,
        offset_frac=float(offset_frac),
        candidates_global=candidates_global,
        lane_poses_xy=lane_poses_xy,
    )


def _canonical_ego_pixel(gen_cfg: NuScenesGenConfig) -> np.ndarray:
    """Canonical ego pixel from meter-based framing constraints.

    Longitudinal placement is determined by the requested
    ``(ego_behind_m, ego_ahead_m)`` ratio and then scaled to the current
    BEV span. Lateral placement is centered before deterministic jitter.
    """
    span = float(gen_cfg.ego_behind_m + gen_cfg.ego_ahead_m)
    if span <= 0.0:
        raise ValueError("ego_behind_m + ego_ahead_m must be positive.")
    bev = float(gen_cfg.bev_size)
    px = bev * (float(gen_cfg.ego_behind_m) / span)
    py = 0.5 * bev
    px = float(np.clip(px, 0.0, bev - 1.0))
    py = float(np.clip(py, 0.0, bev - 1.0))
    return np.array([px, py], dtype=np.float64)


def _sample_ego_pixel(gen_cfg: NuScenesGenConfig, sample_idx: int) -> np.ndarray:
    """Sample canonical ego pixel with deterministic lateral-only jitter."""
    base = _canonical_ego_pixel(gen_cfg)
    jitter_m = float(gen_cfg.ego_lateral_jitter_m)
    if jitter_m <= 0.0:
        return base
    rng = np.random.default_rng(
        (
            int(gen_cfg.seed) & 0xFFFFFFFF,
            int(sample_idx) & 0xFFFFFFFF,
            0x5EEDC0DE,
        )
    )
    # Positive lateral shift means ego-left, which maps to smaller image y.
    lateral_shift_m = float(rng.uniform(-jitter_m, jitter_m))
    py = float(base[1] - lateral_shift_m / gen_cfg.meters_per_pixel)
    py = float(np.clip(py, 0.0, float(gen_cfg.bev_size - 1)))
    return np.array([float(base[0]), py], dtype=np.float64)


# ---------------------------------------------------------------------------
# BEV transforms / rendering
# ---------------------------------------------------------------------------


def _global_to_bev_pixels(
    pts_xy: np.ndarray,
    ego_xy: np.ndarray,
    yaw: float,
    m_per_px: float,
    ego_pixel: np.ndarray,
) -> np.ndarray:
    """Heading-right BEV transform: ego forward = right, ego left = up."""
    if pts_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    d = pts_xy.astype(np.float64) - ego_xy.reshape(1, 2)
    c, s = math.cos(-yaw), math.sin(-yaw)
    u = d[:, 0] * c - d[:, 1] * s  # forward (m)
    v = d[:, 0] * s + d[:, 1] * c  # left (m)
    px = float(ego_pixel[0]) + u / m_per_px
    py = float(ego_pixel[1]) - v / m_per_px
    return np.stack([px, py], axis=1)


def _ego_heading_theta_in_bev(
    ego_xy: np.ndarray,
    base_yaw: float,
    yaw: float,
    ego_pixel: np.ndarray,
    m_per_px: float,
) -> float:
    """Direction of the ego's forward vector in BEV pixel coordinates.

    The Gaussian ego marker is supposed to point along the actual lane
    direction, so the network gets a directional hint of where to start
    predicting. With canonical heading-right framing and no yaw
    augmentation, ``yaw == base_yaw`` and the forward direction is
    image-right (theta = 0). We still compute the on-screen heading by
    projecting the world forward step through the same transform used
    for the candidate paths, then taking ``atan2(dy, dx)`` of the
    resulting pixel vector.
    """
    step_world = np.stack(
        [ego_xy, ego_xy + np.array([math.cos(base_yaw), math.sin(base_yaw)], dtype=np.float64)],
        axis=0,
    )
    step_pixel = _global_to_bev_pixels(step_world, ego_xy, yaw, m_per_px, ego_pixel)
    direction = step_pixel[1] - step_pixel[0]
    return float(math.atan2(direction[1], direction[0]))


def _render_bev_map(
    nusc_map,
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
    cfg: GeneratorConfig,
    ego_theta_in_bev: float,
) -> np.ndarray:
    """Render the BEV by mirroring ``NuScenesMap.render_map_patch`` exactly.

    The devkit reference uses matplotlib + ``descartes.PolygonPatch`` with
    ``alpha=0.5`` for polygons and ``ax.plot(..., alpha=0.5)`` for line
    layers. We do the same here: polygons / lines are projected from
    world coords into BEV pixel coords (so we get our heading-right
    rotation for free), and matplotlib renders them as vector graphics,
    so polygon edges are anti-aliased at sub-pixel precision and the
    saturated colours fade onto the white axes background just like the
    devkit reference image. The asymmetric ego Gaussian is then painted
    on top of the matplotlib output at the final resolution so its
    analytic shape is preserved exactly.
    """
    canvas = _render_map_layers_devkit(nusc_map, ego_xy, yaw, ego_pixel, gen_cfg)
    if gen_cfg.draw_ego_marker:
        _draw_ego_gaussian(canvas, ego_pixel, cfg, ego_theta_in_bev)
    return np.transpose(canvas, (2, 0, 1)).copy()  # -> (3, H, W)


def _render_map_layers_devkit(
    nusc_map,
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> np.ndarray:
    """Render polygon + line layers via matplotlib at devkit fidelity.

    Polygons are added as ``descartes.PolygonPatch`` instances with the
    devkit colour map and ``alpha=0.5`` (mirroring
    :meth:`NuScenesMapExplorer._render_polygon_layer`). Line layers
    (``road_divider``, ``lane_divider``) are drawn with ``ax.plot`` at
    the same alpha. The figure is sized so that ``figsize * dpi``
    matches ``bev_size`` exactly, giving an output of ``(H, W, 3)``
    ``uint8`` pixels.
    """
    fig, ax = _create_bev_figure(gen_cfg.bev_size)
    try:
        records = _query_map_records(nusc_map, ego_xy, ego_pixel, gen_cfg)
        _add_polygon_patches(ax, nusc_map, records, ego_xy, yaw, ego_pixel, gen_cfg)
        _add_divider_lines(ax, nusc_map, records, ego_xy, yaw, ego_pixel, gen_cfg)
        # Chevrons are stroked AFTER drivable fill + dividers so they
        # sit on top of both, but BEFORE ``_figure_to_rgb_array`` so
        # they share the same crisp matplotlib pipeline. The ego
        # Gaussian is then alpha-blended onto the resulting numpy
        # canvas in :func:`_render_bev_map`, which keeps the chevrons
        # blur-free.
        _add_lane_direction_chevrons(ax, nusc_map, ego_xy, yaw, ego_pixel, gen_cfg)
        return _figure_to_rgb_array(fig, gen_cfg.bev_size)
    finally:
        fig.clear()


def _create_bev_figure(bev_size: int) -> tuple[Figure, Axes]:
    """Build a matplotlib figure whose pixel size is exactly ``bev_size``."""
    dpi = 100
    inches = bev_size / dpi
    fig = Figure(figsize=(inches, inches), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, float(bev_size))
    # Image-y axis: pixel 0 is at the top, ``bev_size`` at the bottom.
    ax.set_ylim(float(bev_size), 0.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    return fig, ax


def _figure_to_rgb_array(fig: Figure, bev_size: int) -> np.ndarray:
    """Render ``fig`` to an ``(H, W, 3)`` ``uint8`` array sized to ``bev_size``."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[..., :3].copy()
    if img.shape[0] != bev_size or img.shape[1] != bev_size:
        img = cv2.resize(img, (bev_size, bev_size), interpolation=cv2.INTER_AREA)
    return img


def _add_polygon_patches(
    ax,
    nusc_map,
    records: dict[str, list[str]],
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> None:
    """Add ``descartes.PolygonPatch`` for each polygon layer, devkit-style.

    The devkit's ``_render_polygon_layer`` iterates each record's
    polygons individually and calls ``ax.add_patch(PolygonPatch(...,
    alpha=0.5))``. nuScenes' ``drivable_area`` records OVERLAP at
    junctions (the junction polygon plus the long road-strip polygons
    that extend into it), so the per-polygon alpha-stacking turns those
    overlaps into visibly darker rectangles inside the road. We
    therefore union all polygons of a layer once with
    ``shapely.ops.unary_union`` BEFORE rendering, and emit one
    ``PolygonPatch`` per resulting piece — this preserves the devkit's
    soft ``alpha=0.5`` colour while eliminating the alpha-stacking
    artefacts that produced the "many overlapping rectangles" effect.
    Real holes (roundabout islands, medians) survive the union and are
    rendered correctly by ``PolygonPatch``.
    """
    for layer_name in _DRIVABLE_POLYGON_LAYERS:
        polygons = list(_collect_layer_polygons(nusc_map, records.get(layer_name, []), layer_name))
        if not polygons:
            continue
        for piece in _iter_simple_polygons(_safe_unary_union(polygons)):
            bev_poly = _polygon_world_to_bev(piece, ego_xy, yaw, ego_pixel, gen_cfg)
            if bev_poly is None or bev_poly.is_empty:
                continue
            # Soft alpha-blended fill (no edge) — gives the devkit-style
            # pale-blue road surface.
            ax.add_patch(
                descartes.PolygonPatch(
                    bev_poly,
                    fc=_COLOR_DRIVABLE_HEX,
                    ec="none",
                    alpha=_DRIVABLE_FILL_ALPHA,
                )
            )
            # Crisp boundary (no fill) drawn on top — matches the strict
            # black outline visible in ``nusc_map.render_map_patch``.
            ax.add_patch(
                descartes.PolygonPatch(
                    bev_poly,
                    fc="none",
                    ec=_COLOR_DRIVABLE_BOUNDARY,
                    linewidth=_DRIVABLE_BOUNDARY_LW,
                    alpha=_DRIVABLE_BOUNDARY_ALPHA,
                    joinstyle="round",
                )
            )


def _collect_layer_polygons(nusc_map, tokens: list[str], layer_name: str):
    """Yield every Shapely ``Polygon`` belonging to the layer's records.

    ``extract_polygon`` may return a ``MultiPolygon`` when a record
    represents multiple disjoint pieces; we flatten via
    :func:`_iter_simple_polygons`.
    """
    for tok in tokens:
        rec = nusc_map.get(layer_name, tok)
        for poly_tok in _iter_polygon_tokens(rec, layer_name):
            geom = nusc_map.extract_polygon(poly_tok)
            yield from _iter_simple_polygons(geom)


def _safe_unary_union(polygons):
    """``unary_union`` with a graceful fall-back for bad geometry.

    A single self-intersecting polygon would otherwise abort the union;
    in that case we return a ``MultiPolygon`` of the original inputs so
    the renderer can still paint them (the alpha-stacking artefact
    re-appears on that one record only, which is much rarer than the
    typical road-strip-over-junction overlap we wanted to fix).
    """
    try:
        return unary_union(polygons)
    except Exception:
        return ShapelyMultiPolygon([p for p in polygons if not p.is_empty])


def _add_divider_lines(
    ax,
    nusc_map,
    records: dict[str, list[str]],
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> None:
    """Stroke each ``road_divider`` / ``lane_divider`` polyline via ``ax.plot``."""
    for layer_name, colour in _LINE_LAYERS:
        for tok in records.get(layer_name, []):
            rec = nusc_map.get(layer_name, tok)
            line_tok = rec.get("line_token")
            if not line_tok:
                continue
            line = nusc_map.extract_line(line_tok)
            if line.is_empty:
                continue
            xy = np.asarray(line.coords, dtype=np.float64)
            if xy.shape[0] < 2:
                continue
            pix = _global_to_bev_pixels(xy, ego_xy, yaw, gen_cfg.meters_per_pixel, ego_pixel)
            ax.plot(
                pix[:, 0],
                pix[:, 1],
                color=colour,
                alpha=_DIVIDER_ALPHA,
                linewidth=_DIVIDER_LW,
                solid_capstyle="round",
            )


def _add_lane_direction_chevrons(
    ax,
    nusc_map,
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> None:
    """Stroke ``>``-shaped chevrons along every visible ``lane`` centerline.

    Junction connectors (``lane_connector``) are intentionally excluded:
    those records cross through intersections and would clutter the BEV
    with chevrons that overlap multiple legal flows. Each chevron is
    placed at a fixed arclength on the lane's directed centerline (via
    :func:`arcline_path_utils.discretize_lane`, same call the path
    enumeration uses) so it always points along the legal driving
    direction. Vertices are computed in world coords and projected with
    the same heading-right :func:`_global_to_bev_pixels` transform used
    by every other map layer.
    """
    if not gen_cfg.draw_lane_direction:
        return
    spacing = float(gen_cfg.lane_direction_spacing_m)
    arm = float(gen_cfg.lane_direction_arm_m)
    half_angle_rad = math.radians(_LANE_DIRECTION_HALF_ANGLE_DEG)
    cos_a = math.cos(half_angle_rad)
    sin_a = math.sin(half_angle_rad)

    lane_tokens = _query_lane_tokens_in_bev(nusc_map, ego_xy, ego_pixel, gen_cfg)
    for lane_tok in lane_tokens:
        chevrons_world = _chevron_world_triples(
            nusc_map,
            lane_tok,
            spacing=spacing,
            arm=arm,
            cos_a=cos_a,
            sin_a=sin_a,
        )
        if not chevrons_world:
            continue
        _stroke_chevrons(ax, chevrons_world, ego_xy, yaw, ego_pixel, gen_cfg)


def _query_lane_tokens_in_bev(
    nusc_map,
    ego_xy: np.ndarray,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> list[str]:
    """Lane records (no junction connectors) intersecting the BEV's AABB patch.

    Reuses :func:`_bev_aabb_patch` with the current ``ego_pixel`` so
    asymmetric longitudinal framing remains covered. ``lane_connector``
    is deliberately omitted — see
    :func:`_add_lane_direction_chevrons` for the rationale.
    """
    patch = _bev_aabb_patch(ego_xy, ego_pixel, gen_cfg)
    res = nusc_map.get_records_in_patch(patch, ["lane"], mode="intersect")
    return list(res.get("lane", []))


def _chevron_world_triples(
    nusc_map,
    lane_token: str,
    *,
    spacing: float,
    arm: float,
    cos_a: float,
    sin_a: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(tip, back_left, back_right)`` world-coord triples for the lane.

    Chevron tips are placed at arclengths ``spacing/2, 3*spacing/2, ...``
    along the directed centerline so:
      * a long lane gets multiple chevrons (one per ``spacing`` meters);
      * a short lane gets at least one chevron whenever
        ``length >= spacing/2`` (and zero below that threshold).
    """
    try:
        arc = nusc_map.get_arcline_path(lane_token)
    except ValueError:
        return []
    poses = arcline_path_utils.discretize_lane(arc, 0.5)
    if len(poses) < 2:
        return []
    xy = np.asarray([(p[0], p[1]) for p in poses], dtype=np.float64)
    seg = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < spacing * 0.5:
        return []
    targets = np.arange(spacing * 0.5, total + 1e-6, spacing, dtype=np.float64)
    return [_chevron_at_arclength(xy, seg, seg_len, cum, t, arm=arm, cos_a=cos_a, sin_a=sin_a) for t in targets if _segment_index_for_arclength(cum, t) is not None]


def _segment_index_for_arclength(cum: np.ndarray, target: float) -> int | None:
    """Index ``i`` such that ``cum[i-1] <= target <= cum[i]``, or ``None``."""
    if target < 0.0 or target > float(cum[-1]):
        return None
    idx = int(np.searchsorted(cum, target, side="right"))
    return max(1, min(idx, len(cum) - 1))


def _chevron_at_arclength(
    xy: np.ndarray,
    seg: np.ndarray,
    seg_len: np.ndarray,
    cum: np.ndarray,
    target: float,
    *,
    arm: float,
    cos_a: float,
    sin_a: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one ``(tip, back_left, back_right)`` triple at arclength ``target``."""
    idx = _segment_index_for_arclength(cum, target)
    assert idx is not None  # guarded by caller
    s = float(seg_len[idx - 1])
    if s < 1e-9:
        tip = xy[idx]
        forward = np.array([1.0, 0.0], dtype=np.float64)
    else:
        forward = seg[idx - 1] / s
        alpha = (target - float(cum[idx - 1])) / s
        tip = xy[idx - 1] + alpha * seg[idx - 1]
    left = np.array([-forward[1], forward[0]], dtype=np.float64)
    back = -arm * cos_a * forward
    side = arm * sin_a * left
    return tip, tip + back + side, tip + back - side


def _stroke_chevrons(
    ax,
    chevrons_world: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> None:
    """Project each chevron triple to BEV pixels and draw it as a polyline."""
    for tip, back_left, back_right in chevrons_world:
        verts_world = np.stack([back_left, tip, back_right], axis=0)
        pix = _global_to_bev_pixels(
            verts_world,
            ego_xy,
            yaw,
            gen_cfg.meters_per_pixel,
            ego_pixel,
        )
        ax.plot(
            pix[:, 0],
            pix[:, 1],
            color=gen_cfg.lane_direction_color,
            alpha=_LANE_DIRECTION_ALPHA,
            linewidth=_LANE_DIRECTION_LW,
            solid_capstyle="round",
            solid_joinstyle="miter",
        )


def _polygon_world_to_bev(
    poly,
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
):
    """Project a Shapely polygon (with optional holes) from world XY to BEV pixels.

    Returns ``None`` if the projected exterior has fewer than three
    distinct vertices — those would otherwise crash
    ``descartes.PolygonPatch``.
    """
    ext_xy = np.asarray(list(poly.exterior.coords), dtype=np.float64)
    if ext_xy.shape[0] < 3:
        return None
    ext_pix = _global_to_bev_pixels(ext_xy, ego_xy, yaw, gen_cfg.meters_per_pixel, ego_pixel)
    holes_pix: list[np.ndarray] = []
    for ring in poly.interiors:
        ring_xy = np.asarray(list(ring.coords), dtype=np.float64)
        if ring_xy.shape[0] < 3:
            continue
        holes_pix.append(_global_to_bev_pixels(ring_xy, ego_xy, yaw, gen_cfg.meters_per_pixel, ego_pixel))
    return ShapelyPolygon(ext_pix, holes=holes_pix)


def _query_map_records(
    nusc_map,
    ego_xy: np.ndarray,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> dict[str, list[str]]:
    """Records of all painted layers intersecting the BEV's worst-case footprint.

    Uses an axis-aligned bounding box big enough to enclose the rotated
    BEV for the current ego pixel (including asymmetric longitudinal
    framing).
    """
    patch = _bev_aabb_patch(ego_xy, ego_pixel, gen_cfg)
    layer_names = list(_DRIVABLE_POLYGON_LAYERS) + [name for name, _ in _LINE_LAYERS]
    return nusc_map.get_records_in_patch(patch, layer_names, mode="intersect")


def _bev_aabb_patch(
    ego_xy: np.ndarray,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> tuple[float, float, float, float]:
    """Yaw-invariant axis-aligned box that contains the BEV footprint.

    Radius is the farthest corner distance from ``ego_pixel`` in meters,
    so asymmetric longitudinal ego placement remains fully covered.
    """
    bev = float(gen_cfg.bev_size)
    corners = np.array(
        [[0.0, 0.0], [bev - 1.0, 0.0], [bev - 1.0, bev - 1.0], [0.0, bev - 1.0]],
        dtype=np.float64,
    )
    delta = corners - np.asarray(ego_pixel, dtype=np.float64).reshape(1, 2)
    radius_m = float(np.linalg.norm(delta, axis=1).max()) * float(gen_cfg.meters_per_pixel)
    return (
        float(ego_xy[0] - radius_m),
        float(ego_xy[1] - radius_m),
        float(ego_xy[0] + radius_m),
        float(ego_xy[1] + radius_m),
    )


# ---------------------------------------------------------------------------
# Connectivity filter: every visible lane endpoint must connect somewhere.
# ---------------------------------------------------------------------------


def _bev_is_fully_connected(
    nusc_map,
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> bool:
    """Return ``True`` iff every ``lane`` / ``lane_connector`` whose centerline
    extends into the BEV either continues into another mapped element or
    exits the canvas cleanly.

    Concretely, for each lane and lane_connector intersecting the BEV's
    AABB patch, we check the centerline's *start_pose* and *end_pose*:

    * If a ``start_pose`` lies inside the canvas, the lane must have at
      least one ``incoming_lane_id`` (otherwise the road appears out of
      thin air mid-canvas).
    * If an ``end_pose`` lies inside the canvas, the lane must have at
      least one ``outgoing_lane_id`` (otherwise the road dead-ends with
      no annotated continuation, which is what causes path extensions
      to extrapolate into non-drivable pixels).

    Endpoints are read straight off the arcline segments (``arc[0][
    'start_pose']``, ``arc[-1]['end_pose']``) which is ~180x faster
    than a full ``discretize_lane`` call.
    """
    patch = _bev_aabb_patch(ego_xy, ego_pixel, gen_cfg)
    res = nusc_map.get_records_in_patch(patch, ["lane", "lane_connector"], mode="intersect")
    margin = float(gen_cfg.fully_connected_bev_margin_px)
    bev = gen_cfg.bev_size

    for layer in ("lane", "lane_connector"):
        for tok in res.get(layer, []):
            ends = _lane_endpoints_global(nusc_map, tok)
            if ends is None:
                continue
            head_xy, tail_xy = ends
            head_pix, tail_pix = _global_to_bev_pixels(
                np.stack([head_xy, tail_xy], axis=0),
                ego_xy,
                yaw,
                gen_cfg.meters_per_pixel,
                ego_pixel,
            )
            if _is_inside_bev(head_pix, bev, margin) and not nusc_map.get_incoming_lane_ids(tok):
                return False
            if _is_inside_bev(tail_pix, bev, margin) and not nusc_map.get_outgoing_lane_ids(tok):
                return False
    return True


def _lane_endpoints_global(nusc_map, lane_token: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(head_xy, tail_xy)`` global coords for a lane / lane_connector,
    pulled directly from the arcline segments (no full discretization).
    """
    try:
        arc = nusc_map.get_arcline_path(lane_token)
    except ValueError:
        return None
    if not arc:
        return None
    head = arc[0].get("start_pose")
    tail = arc[-1].get("end_pose")
    if not head or not tail:
        return None
    return (
        np.array([float(head[0]), float(head[1])], dtype=np.float64),
        np.array([float(tail[0]), float(tail[1])], dtype=np.float64),
    )


def _is_inside_bev(pixel: np.ndarray, bev_size: int, margin: float) -> bool:
    return bool(margin <= float(pixel[0]) <= bev_size - 1.0 - margin and margin <= float(pixel[1]) <= bev_size - 1.0 - margin)


def _iter_polygon_tokens(record: dict, layer_name: str):
    """Yield the polygon tokens for a polygon-layer record (handles plural vs singular).

    Two record shapes show up in nuScenes:
      * ``drivable_area`` carries ``polygon_tokens`` (plural list).
      * Every other polygon layer (``lane``, ``lane_connector``,
        ``road_segment``, ``ped_crossing``, ...) carries ``polygon_token``
        (singular).
    """
    if layer_name == "drivable_area":
        yield from record.get("polygon_tokens", [])
        return
    poly_tok = record.get("polygon_token")
    if poly_tok:
        yield poly_tok


def _iter_simple_polygons(geom):
    """Yield the constituent ``Polygon``s of a (possibly empty/MultiPolygon) geometry.

    ``extract_polygon`` may return a ``MultiPolygon`` when a record
    represents multiple disjoint pieces; downstream callers want a flat
    sequence of plain ``Polygon`` objects.
    """
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "MultiPolygon":
        for sub in geom.geoms:
            if not sub.is_empty:
                yield sub
    elif geom.geom_type == "Polygon":
        yield geom


def _draw_ego_gaussian(
    canvas: np.ndarray,
    ego_pixel: np.ndarray,
    cfg: GeneratorConfig,
    theta: float,
) -> None:
    """Paint a directional asymmetric Gaussian at ``ego_pixel`` over all 3 RGB channels.

    Uses :func:`generator.rasterize_ego_heatmap` so the shape matches the
    synthetic generator: a long forward tongue (``sigma_long_forward``)
    and a short backward stub (``sigma_long_backward``) make the heading
    visible at a glance. The grayscale heatmap is alpha-blended into the
    RGB canvas with ``_COLOR_EGO`` so the heading is encoded jointly
    across R, G, B channels.

    ``theta`` is the on-screen heading direction in pixel coords (math
    convention, i.e. ``atan2(dy, dx)`` of the heading vector). In the
    canonical no-augmentation setup it is ``0`` (forward = image-right),
    but the value stays transform-derived so the marker remains correctly
    tied to the lane heading if conventions change.
    """
    heat = rasterize_ego_heatmap(cfg, pos=np.asarray(ego_pixel, dtype=np.float64), theta=theta)
    alpha = heat.astype(np.float32) / 255.0  # (H, W) in [0, 1]
    if not float(alpha.max()) > 0.0:
        return
    color = np.asarray(_COLOR_EGO, dtype=np.float32)  # (3,)
    canvas_f = canvas.astype(np.float32)
    blended = canvas_f * (1.0 - alpha[..., None]) + color * alpha[..., None]
    canvas[:] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Forward-only candidate paths
# ---------------------------------------------------------------------------


def _enumerate_forward_paths_multi_source(
    nusc_map,
    ego_xy: np.ndarray,
    yaw: float,
    primary_lane: str,
    primary_poses: np.ndarray,
    primary_start_idx: int,
    *,
    max_arclen_m: float,
    max_paths: int,
    search_radius_m: float,
    max_angle_deg: float,
    max_sources: int,
) -> list[np.ndarray]:
    """Enumerate forward paths from the primary lane PLUS nearby
    same-direction parallel lanes near the sampled start point.

    Why: at most 4-way intersections, the through lane only has outgoing
    connectors for "straight" + "right". The "left turn" option is encoded
    on a *separate* parallel lane (the left-turn pocket). Sampling a single
    lane therefore misses left-turn candidates entirely. We instead enter
    the DFS from every nearby lane whose centerline passes near ``ego_xy``
    and whose tangent is roughly aligned with ``yaw``; connectors are still
    reached through outgoing-lane traversal from those lane entries.
    """
    entries = _collect_path_entries(
        nusc_map,
        ego_xy=ego_xy,
        yaw=yaw,
        primary_lane=primary_lane,
        primary_poses=primary_poses,
        primary_start_idx=primary_start_idx,
        search_radius_m=search_radius_m,
        max_angle_deg=max_angle_deg,
        max_sources=max_sources,
    )
    paths: list[np.ndarray] = []
    per_entry = max(2, max_paths // max(1, len(entries)))
    for lane_tok, sidx, sposes in entries:
        paths.extend(
            _enumerate_forward_paths(
                nusc_map,
                first_lane=lane_tok,
                first_lane_poses=sposes,
                start_idx=sidx,
                max_arclen_m=max_arclen_m,
                max_paths=per_entry,
            )
        )
    return paths


def _collect_path_entries(
    nusc_map,
    *,
    ego_xy: np.ndarray,
    yaw: float,
    primary_lane: str,
    primary_poses: np.ndarray,
    primary_start_idx: int,
    search_radius_m: float,
    max_angle_deg: float,
    max_sources: int,
) -> list[tuple[str, int, np.ndarray]]:
    """Pick (lane_token, start_idx, poses) entry tuples for the multi-source DFS."""
    entries: list[tuple[str, int, np.ndarray]] = [(primary_lane, primary_start_idx, primary_poses)]
    if search_radius_m <= 0.0 or max_sources <= 1:
        return entries

    seen = {primary_lane}
    nearby_tokens = _query_nearby_lanes(nusc_map, ego_xy, search_radius_m)
    ego_dir = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    cos_thresh = math.cos(math.radians(max_angle_deg))

    for tok in nearby_tokens:
        if tok in seen or len(entries) >= max_sources:
            continue
        seen.add(tok)
        entry = _entry_for_lane(nusc_map, tok, ego_xy, ego_dir, search_radius_m, cos_thresh)
        if entry is not None:
            entries.append(entry)
    return entries


def _query_nearby_lanes(nusc_map, ego_xy: np.ndarray, radius_m: float) -> list[str]:
    """Nearby lane records whose polygon intersects a small AABB around
    ``ego_xy`` (radius is half the AABB side, in meters).

    Intentionally excludes ``lane_connector`` as an entry seed so candidate
    path starts stay anchored on lane centerlines before entering junction
    connectors via normal outgoing-lane traversal.
    """
    patch = (
        float(ego_xy[0] - radius_m),
        float(ego_xy[1] - radius_m),
        float(ego_xy[0] + radius_m),
        float(ego_xy[1] + radius_m),
    )
    res = nusc_map.get_records_in_patch(patch, ["lane"], mode="intersect")
    return list(res.get("lane", []))


def _entry_for_lane(
    nusc_map,
    lane_token: str,
    ego_xy: np.ndarray,
    ego_dir: np.ndarray,
    search_radius_m: float,
    cos_thresh: float,
) -> tuple[str, int, np.ndarray] | None:
    """Return ``(lane_token, start_idx, poses_xy)`` if the lane qualifies as a
    same-direction entry near ``ego_xy``, else ``None``.
    """
    try:
        arc = nusc_map.get_arcline_path(lane_token)
    except ValueError:
        return None
    poses = arcline_path_utils.discretize_lane(arc, 0.5)
    if len(poses) < 2:
        return None
    poses_xy = np.asarray([(p[0], p[1]) for p in poses], dtype=np.float64)
    dists = np.linalg.norm(poses_xy - ego_xy.reshape(1, 2), axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] > search_radius_m:
        return None
    nxt = idx + 1 if idx + 1 < poses_xy.shape[0] else idx
    prv = idx - 1 if idx > 0 else idx
    tangent = poses_xy[nxt] - poses_xy[prv]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        return None
    if float((tangent / norm) @ ego_dir) < cos_thresh:
        return None
    return lane_token, idx, poses_xy


def _enumerate_forward_paths(
    nusc_map,
    first_lane: str,
    first_lane_poses: np.ndarray,
    start_idx: int,
    max_arclen_m: float,
    max_paths: int,
) -> list[np.ndarray]:
    """Enumerate every forward driving path starting on ``first_lane`` at ``start_idx``.

    The first lane is trimmed at ``start_idx`` so the first point is the
    sampled "ego". Outgoing lanes are walked via DFS; each branch yields
    one candidate. Cumulative arclength is bounded by ``max_arclen_m`` so
    we don't run unbounded loops on connected DAGs.
    """
    seqs = _enumerate_lane_sequences(nusc_map, first_lane, max_arclen_m, max_paths)
    out: list[np.ndarray] = []
    for seq in seqs:
        global_xy = _concat_lane_centerlines(
            nusc_map,
            seq,
            first_lane_poses=first_lane_poses,
            start_idx=start_idx,
        )
        if global_xy.shape[0] >= 2:
            out.append(global_xy)
    return out


def _enumerate_lane_sequences(
    nusc_map,
    start: str,
    max_arclen_m: float,
    max_paths: int,
) -> list[list[str]]:
    """Iterative DFS over outgoing lanes / lane connectors (forward only)."""
    paths: list[list[str]] = []
    stack: list[tuple[list[str], float]] = [([start], 0.0)]
    while stack and len(paths) < max_paths:
        seq, used = stack.pop()
        last = seq[-1]
        try:
            outgoing = list(nusc_map.get_outgoing_lane_ids(last))
        except ValueError:
            outgoing = []
        outgoing = [o for o in outgoing if o not in seq]
        if not outgoing or used >= max_arclen_m:
            paths.append(seq)
            continue
        stack.extend(([*seq, o], used + _lane_length_m(nusc_map, o)) for o in outgoing)
    return paths


def _concat_lane_centerlines(
    nusc_map,
    seq: list[str],
    first_lane_poses: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    """Stitch ``seq``'s discretised centerlines into a single ``(N, 2)`` polyline.

    The first lane's prefix is sliced at ``start_idx``; subsequent lanes
    are appended whole, with a tiny dedup at the seam.
    """
    pieces: list[np.ndarray] = []
    for i, lane_tok in enumerate(seq):
        if i == 0:
            xy = first_lane_poses[start_idx:]
        else:
            try:
                arc = nusc_map.get_arcline_path(lane_tok)
            except ValueError:
                continue
            poses = arcline_path_utils.discretize_lane(arc, 0.5)
            if not poses:
                continue
            xy = np.asarray([(p[0], p[1]) for p in poses], dtype=np.float64)
        if pieces and xy.shape[0] > 0 and np.allclose(pieces[-1][-1], xy[0], atol=0.05):
            xy = xy[1:]
        if xy.shape[0] > 0:
            pieces.append(xy)
    if not pieces:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(pieces, axis=0)


def _lane_length_m(nusc_map, lane_tok: str) -> float:
    try:
        arc = nusc_map.get_arcline_path(lane_tok)
    except ValueError:
        return 0.0
    poses = arcline_path_utils.discretize_lane(arc, 1.0)
    if len(poses) < 2:
        return 0.0
    pts = np.asarray([(p[0], p[1]) for p in poses], dtype=np.float64)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


# ---------------------------------------------------------------------------
# Pixel-space helpers (BEV clipping, dedup, anchor)
# ---------------------------------------------------------------------------


def _candidates_global_to_pixels(
    candidates_global: list[np.ndarray],
    ego_xy: np.ndarray,
    yaw: float,
    ego_pixel: np.ndarray,
    gen_cfg: NuScenesGenConfig,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for poly in candidates_global:
        pix = _global_to_bev_pixels(poly, ego_xy, yaw, gen_cfg.meters_per_pixel, ego_pixel)
        if gen_cfg.extend_paths_to_bev_edge:
            pix = _extend_polyline_to_bev_edge(pix, gen_cfg.bev_size)
        pix = _clip_polyline_to_bev(pix, gen_cfg.bev_size)
        if pix.shape[0] >= 2:
            out.append(pix)
    out.sort(key=lambda p: -p.shape[0])
    return out


def _extend_polyline_to_bev_edge(pix: np.ndarray, bev_size: int) -> np.ndarray:
    """If the polyline ends inside the BEV, extrapolate the last tangent forward
    until it hits the canvas boundary.

    nuScenes annotates ~17 % of lanes with no outgoing connectors at all
    (verified on boston-seaport). Without this, paths that walk into one
    of those "dangling" lanes visibly stop mid-image — what you see in
    image 1. Extending in pixel space is exact (the world->pixel
    transform is affine), so this is equivalent to extrapolating the
    underlying centerline and re-projecting.
    """
    if pix.shape[0] < 2:
        return pix
    last = pix[-1]
    margin = 0.5
    if last[0] <= margin or last[0] >= bev_size - 1.0 - margin or last[1] <= margin or last[1] >= bev_size - 1.0 - margin:
        return pix
    direction = last - pix[-2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return pix
    direction = direction / norm
    t_max = _t_to_canvas_edge(last, direction, bev_size)
    if t_max is None:
        return pix
    new_end = last + t_max * direction
    return np.concatenate([pix, new_end.reshape(1, 2)], axis=0)


def _t_to_canvas_edge(start: np.ndarray, direction: np.ndarray, bev_size: int) -> float | None:
    """Smallest positive ``t`` for which ``start + t*direction`` exits the
    ``[0, bev_size-1]^2`` canvas. ``None`` if the ray points along the edge.
    """
    candidates: list[float] = []
    for axis in (0, 1):
        d = float(direction[axis])
        if d > 1e-6:
            candidates.append((bev_size - 1.0 - float(start[axis])) / d)
        elif d < -1e-6:
            candidates.append((0.0 - float(start[axis])) / d)
    candidates = [t for t in candidates if t > 0.0]
    if not candidates:
        return None
    return min(candidates)


def _clip_polyline_to_bev(pix: np.ndarray, bev_size: int) -> np.ndarray:
    """Keep the leading prefix that lies inside the BEV image; truncate at the
    first point that leaves the canvas (retaining that boundary point as a
    final anchor so per-cell sparsification still has a terminal cell).
    """
    if pix.shape[0] == 0:
        return pix
    inside = (pix[:, 0] >= 0) & (pix[:, 0] < bev_size) & (pix[:, 1] >= 0) & (pix[:, 1] < bev_size)
    if not bool(inside[0]):
        return np.zeros((0, 2), dtype=np.float64)
    last = int(inside.shape[0])
    for i in range(1, inside.shape[0]):
        if not inside[i]:
            last = i + 1
            break
    return pix[:last]


def _prepend_ego_anchor(pix: np.ndarray, ego_pixel: np.ndarray) -> np.ndarray:
    """Ensure the polyline starts exactly at ``ego_pixel = (px, py)``."""
    anchor = np.asarray(ego_pixel, dtype=np.float64).reshape(1, 2)
    if pix.shape[0] == 0:
        return anchor.copy()
    if np.linalg.norm(pix[0] - anchor[0]) < 1e-6:
        return pix.astype(np.float64, copy=False)
    return np.concatenate([anchor, pix.astype(np.float64, copy=False)], axis=0)


def _drop_redundant_paths_pixel(paths: list[np.ndarray], tol_px: float) -> list[np.ndarray]:
    """Spatial dedup of candidate paths in BEV pixel coordinates.

    Pixel-space (post-clip, post-anchor) dedup is what we actually care
    about for the rendered training sample: any divergence between two
    paths *outside* the BEV is invisible to the network, so two paths
    whose visible portions match within ``tol_px`` should collapse to
    one. Pre-clip world dedup misses this case — paths that diverge far
    downstream survive as exact pixel duplicates inside the BEV.

    Two paths are considered the same on-screen route if their per-point
    Euclidean distance never exceeds ``tol_px`` over their common length.
    With ``tol_px`` slightly wider than a lane (~8 px at 0.5 m/px),
    parallel-lane duplicates collapse but junction-divergent paths
    (which separate by tens of pixels after the fork) stay distinct.

    Paths are ordered longest-first; when a duplicate is detected we
    keep the longer one, since it encodes the same on-screen path PLUS
    more reach toward the BEV edge.
    """
    paths = [p for p in paths if p.shape[0] >= 2]
    paths.sort(key=lambda p: -p.shape[0])
    kept: list[np.ndarray] = []
    for cand in paths:
        if any(_paths_overlap_px(cand, k, tol_px) for k in kept):
            continue
        kept.append(cand)
    return kept


def _paths_overlap_px(short: np.ndarray, long: np.ndarray, tol_px: float) -> bool:
    """True iff ``short`` is within ``tol_px`` of ``long`` over their common length."""
    n = int(min(short.shape[0], long.shape[0]))
    if n < 2:
        return False
    return bool(np.linalg.norm(short[:n] - long[:n], axis=1).max() < tol_px)


if __name__ == "__main__":
    sys.exit(main())
