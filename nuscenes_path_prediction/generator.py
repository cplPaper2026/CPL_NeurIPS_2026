"""Map-based procedural engine for the road-cpl synthetic dataset.

The generator builds a tree of road segments. Each edge is a single
:class:`path_segments.LineOrArcPath` (line or circular arc, fitted by pure
pursuit). Each node is a 2D position with an ``arrival_heading``.

Growth proceeds forward from a randomly placed ego/root, popping nodes from
a frontier and attempting one (or, with low probability, two or three)
forward-fork children per pop. Children differ by a forced lateral target
(usually a mild offset); by default that keeps splits shallow, but
``GeneratorConfig.sharp_junction_prob > 0`` can produce ~90 deg fork arms (T,
dual-90, or 3-arm plus) via :class:`SharpBranchStraight` /
:class:`SharpBranchQuarter` targets. Every successful candidate creates a
fresh node, so paths never rejoin: two paths whose buffers would overlap are
rejected by the collision check.

After growth, every root-to-leaf path whose terminal node is flagged as
``is_exit`` is enumerated and resampled to a sparse list of full-resolution
(x, y) points starting from the ego, with the property that no two points
fall into the same ``D x D`` grid cell. Samples whose graph yields more than
``cfg.K_MAX`` exit paths are rejected so that every legal path is always
labeled.

Public API:
    generate_sample(cfg, seed) -> dict with keys:
        image          uint8   [2, H, W]                 road map + ego heatmap
        points         float32 [K_MAX, M_MAX, 2]         padded full-res (x, y) points
        num_points     int32   [K_MAX]                   number of valid points per path
        valid_paths    uint8   [K_MAX]                   1 = path k is valid
        meta           dict                              JSON-serialisable
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
from path_segments import LineOrArcPath, normalize_angle
from shapely.geometry import Point, Polygon
from train_config import GeneratorConfig

# ---------------------------------------------------------------------------
# Root (ego) seed placement
# ---------------------------------------------------------------------------


def _root_border_distances(x: float, y: float, cfg: GeneratorConfig) -> tuple[float, float, float, float]:
    """Per-side distances to the full canvas (0, W) x (0, H)."""
    d_left = x
    d_right = float(cfg.W) - x
    d_top = y
    d_bottom = float(cfg.H) - y
    return d_left, d_right, d_top, d_bottom


def _root_d_nearest(x: float, y: float, cfg: GeneratorConfig) -> float:
    dL, dR, dT, dB = _root_border_distances(x, y, cfg)
    return min(dL, dR, dT, dB)


def _root_in_outer_seed_band(x: float, y: float, cfg: GeneratorConfig) -> bool:
    cap = cfg.seed_max_nearest_edge_dist_frac * min(cfg.H, cfg.W)
    return _root_d_nearest(x, y, cfg) <= cap


def _root_inward_normals_closest_borders(x: float, y: float, cfg: GeneratorConfig) -> list[tuple[float, float]]:
    """Inward unit normals (into the image) for all sides tied for min border dist."""
    dL, dR, dT, dB = _root_border_distances(x, y, cfg)
    d_n = min(dL, dR, dT, dB)
    eps = float(cfg.seed_closest_edge_tie_eps_px)
    out: list[tuple[float, float]] = []
    if dL - d_n <= eps:
        out.append((1.0, 0.0))
    if dR - d_n <= eps:
        out.append((-1.0, 0.0))
    if dT - d_n <= eps:
        out.append((0.0, 1.0))
    if dB - d_n <= eps:
        out.append((0.0, -1.0))
    return out


def _root_heading_away_from_closest_borders(x: float, y: float, theta: float, cfg: GeneratorConfig) -> bool:
    # Logic: Cannot point toward ANY border that is within a quarter image from the root.
    quarter_H = 0.25 * float(cfg.H)
    quarter_W = 0.25 * float(cfg.W)
    dL, dR, dT, dB = _root_border_distances(x, y, cfg)
    c, s = math.cos(theta), math.sin(theta)
    if dL <= quarter_W and c < 0:
        return False
    if dR <= quarter_W and c > 0:
        return False
    if dT <= quarter_H and s < 0:
        return False
    if dB <= quarter_H and s > 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Graph data model
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: int
    pos: np.ndarray  # shape (2,) in pixel coords
    arrival_heading: float  # heading along which incoming edge ends
    is_root: bool = False
    is_exit: bool = False  # the segment that produced this node crossed the canvas border
    # ``is_decoration_seed`` marks the phantom co-located node we use to grow
    # the "behind-ego" sub-tree. Its sub-tree contributes to the road-occupancy
    # channel only; ``enumerate_exit_paths`` never starts here.
    is_decoration_seed: bool = False


@dataclass
class Edge:
    id: int
    path: LineOrArcPath
    src_node: int
    dst_node: int
    polyline: np.ndarray  # cached (M, 2) dense samples in pixel coords
    footprint: Polygon  # cached buffer polygon at lane_width/2


@dataclass(frozen=True)
class SharpBranchStraight:
    """Sharp fork arm: l = 0 after the drawn segment length, i.e. straight."""


@dataclass(frozen=True)
class SharpBranchQuarter:
    """Sharp fork arm: l = sign * m so pure pursuit gives ~90 deg; sign in {-1, 1}."""

    sign: int

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError("SharpBranchQuarter sign must be -1 or 1.")


# ``None`` = sample lateral (Gaussian), ``float`` = fixed offset in pixels,
# sharp types bind l to 0 or ±m for ~90 deg geometry.
LateralTarget = Union[None, float, SharpBranchStraight, SharpBranchQuarter]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Two nodes at distance below this are considered to share the same seam (e.g.
# the ego root and the co-located decoration seed, or sibling fork children).
# Edges anchored at the same seam are exempt from cross-collision because
# their flat-cap buffers are designed to meet at that seam.
_SEAM_COLOCATION_PX = 1.0
# Minimum overlap area (in pixels^2) above which two edge buffers count as a
# real collision; smaller overlaps are tolerated as floating-point slop.
_COLLISION_AREA_TOL = 1e-3


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class Graph:
    """DAG of :class:`Node` connected by :class:`Edge` (LineOrArcPath)."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self.out_edges: dict[int, list[int]] = {}
        self.in_edges: dict[int, list[int]] = {}
        self._next_node_id = 0
        self._next_edge_id = 0
        self.root_id: int | None = None
        self._decoration_seeds: list[int] = []
        # Bumped to ``max_edges + extension_extra_edges`` only during
        # :meth:`_extend_leaves_to_border` so the mandatory border reach is not
        # throttled by the same cap as the sparse main growth.
        self._edge_limit: int = cfg.max_edges

    # -- ID helpers --------------------------------------------------------
    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _new_edge_id(self) -> int:
        eid = self._next_edge_id
        self._next_edge_id += 1
        return eid

    def _commit_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.out_edges.setdefault(node.id, [])
        self.in_edges.setdefault(node.id, [])

    def _commit_edge(self, edge: Edge) -> None:
        self.edges[edge.id] = edge
        self.out_edges.setdefault(edge.src_node, []).append(edge.id)
        self.in_edges.setdefault(edge.dst_node, []).append(edge.id)

    # -- Canvas / collision checks ----------------------------------------
    def _check_inside_canvas(self, footprint: Polygon, allow_boundary: bool = False) -> bool:
        cfg = self.cfg
        bbox = footprint.bounds
        if allow_boundary:
            slack = 0.5 * cfg.lane_width_px + 1.0
        else:
            slack = 1.0
        return bbox[0] >= -slack and bbox[1] >= -slack and bbox[2] <= cfg.W + slack and bbox[3] <= cfg.H + slack

    def _shrink_at_seam(self, poly: Polygon, seam: np.ndarray) -> Polygon:
        eps = self.cfg.collision_seam_eps_px
        shrink = Point(float(seam[0]), float(seam[1])).buffer(eps)
        try:
            return poly.difference(shrink)
        except Exception:
            return poly

    def _nodes_sharing_seam(self, anchor_pos: np.ndarray) -> set[int]:
        """All nodes whose position coincides with ``anchor_pos`` (within slop)."""
        return {nid for nid, n in self.nodes.items() if float(np.linalg.norm(n.pos - anchor_pos)) < _SEAM_COLOCATION_PX}

    def _check_collision(self, footprint: Polygon, src_node_id: int) -> bool:
        """Return True iff ``footprint`` does not overlap any non-sibling edge.

        Edges anchored at the same seam (same source position as the candidate)
        are skipped: their flat-cap buffers are designed to meet at that seam.
        """
        src_pos = self.nodes[src_node_id].pos
        check_poly = self._shrink_at_seam(footprint, src_pos)
        src_seam_nodes = self._nodes_sharing_seam(src_pos)

        for edge in self.edges.values():
            if edge.src_node in src_seam_nodes:
                continue
            other = edge.footprint
            if not check_poly.intersects(other):
                continue
            if check_poly.touches(other):
                continue
            if check_poly.intersection(other).area > _COLLISION_AREA_TOL:
                return False
        return True

    # -- Seed --------------------------------------------------------------
    def _distance_to_border(self, pos: np.ndarray, theta: float) -> float:
        cfg = self.cfg
        dx, dy = math.cos(theta), math.sin(theta)
        ts: list[float] = []
        if abs(dx) > 1e-9:
            ts.append((0.0 - pos[0]) / dx)
            ts.append((cfg.W - pos[0]) / dx)
        if abs(dy) > 1e-9:
            ts.append((0.0 - pos[1]) / dy)
            ts.append((cfg.H - pos[1]) / dy)
        ts = [t for t in ts if t > 0]
        return min(ts) if ts else 0.0

    def _seed_root(self, rng: random.Random) -> bool:
        cfg = self.cfg
        m = cfg.canvas_margin_px + cfg.lane_width_px
        if cfg.W - 2 * m <= 0 or cfg.H - 2 * m <= 0:
            return False
        for _ in range(cfg.seed_max_root_attempts):
            x = rng.uniform(m, cfg.W - m)
            y = rng.uniform(m, cfg.H - m)
            if not _root_in_outer_seed_band(x, y, cfg):
                continue
            theta = rng.uniform(-math.pi, math.pi)
            if not _root_heading_away_from_closest_borders(x, y, theta, cfg):
                continue
            forward = self._distance_to_border(np.array([x, y], dtype=np.float64), theta)
            if forward >= cfg.seed_min_forward_lw * cfg.lane_width_px:
                pos = np.array([x, y], dtype=np.float64)
                node = Node(
                    id=self._new_node_id(),
                    pos=pos,
                    arrival_heading=theta,
                    is_root=True,
                )
                self._commit_node(node)
                self.root_id = node.id
                # Backward ("behind-ego") decoration seed: a phantom node at
                # the same position with the opposite heading. Its descendants
                # add visual road behind the ego but never participate in the
                # ground-truth path enumeration.
                deco = Node(
                    id=self._new_node_id(),
                    pos=pos.copy(),
                    arrival_heading=normalize_angle(theta + math.pi),
                    is_root=False,
                    is_decoration_seed=True,
                )
                self._commit_node(deco)
                self._decoration_seeds.append(deco.id)
                return True
        return False

    # -- Growth at a single port ------------------------------------------
    def _try_grow_child(
        self,
        node_id: int,
        rng: random.Random,
        lateral: LateralTarget,
    ) -> tuple[int, bool] | None:
        """Try to grow one child segment from ``node_id``.

        ``lateral`` is ``None`` (Gaussian), a fixed offset in pixels, or a
        :class:`SharpBranchStraight` / :class:`SharpBranchQuarter` for ~90 deg
        arms (l=0 or l=sign*m after the drawn length ``m``).

        Returns ``(dst_id, is_exit)`` on success, ``None`` on retry exhaustion.
        Every successful growth always allocates a fresh destination node, so
        the resulting graph is a tree (no rejoining of paths).
        """
        cfg = self.cfg
        node = self.nodes[node_id]
        h_in = node.arrival_heading

        for _ in range(cfg.retries_per_segment):
            if len(self.edges) >= self._edge_limit:
                return None
            m_lw = rng.gauss(cfg.long_mean_lw, cfg.long_std_lw)
            m_lw = max(cfg.long_min_lw, min(cfg.long_max_lw, m_lw))
            m_px = m_lw * cfg.lane_width_px
            if lateral is None:
                l_lw = rng.gauss(0.0, cfg.lat_std_lw)
                l_lw = max(-cfg.lat_max_lw, min(cfg.lat_max_lw, l_lw))
                l_px = l_lw * cfg.lane_width_px
            elif isinstance(lateral, SharpBranchStraight):
                l_px = 0.0
            elif isinstance(lateral, SharpBranchQuarter):
                l_px = float(lateral.sign) * m_px
            else:
                l_px = float(lateral)

            ch, sh = math.cos(h_in), math.sin(h_in)
            x_t = float(node.pos[0]) + m_px * ch - l_px * sh
            y_t = float(node.pos[1]) + m_px * sh + l_px * ch

            path = LineOrArcPath(float(node.pos[0]), float(node.pos[1]), float(h_in), x_t, y_t)
            if path.is_arc() and path.radius < cfg.min_arc_radius_lw * cfg.lane_width_px:
                continue

            cross_m = path.crosses_canvas_bounds(cfg.W, cfg.H)
            if cross_m is not None:
                if cross_m < 0.5 * cfg.long_min_lw * cfg.lane_width_px:
                    continue
                path = path.truncate_to_bounds(cfg.W, cfg.H)
                is_exit = True
            else:
                is_exit = False

            try:
                polyline = path.polyline_samples(
                    samples_per_lw=cfg.polyline_samples_per_lw,
                    lane_width_px=cfg.lane_width_px,
                )
                footprint = path.buffer_polygon(
                    half_width=cfg.lane_width_px / 2.0,
                    samples_per_lw=cfg.polyline_samples_per_lw,
                    lane_width_px=cfg.lane_width_px,
                )
            except Exception:
                continue
            if footprint.is_empty or not footprint.is_valid:
                continue

            if not self._check_inside_canvas(footprint, allow_boundary=is_exit):
                continue

            end_pt = path.last_point()
            end_h = path.last_heading()

            if not self._check_collision(footprint, node_id):
                continue

            dst_node = Node(
                id=self._new_node_id(),
                pos=end_pt.copy(),
                arrival_heading=end_h,
                is_root=False,
                is_exit=is_exit,
            )
            self._commit_node(dst_node)
            dst_id = dst_node.id

            edge = Edge(
                id=self._new_edge_id(),
                path=path,
                src_node=node_id,
                dst_node=dst_id,
                polyline=polyline,
                footprint=footprint,
            )
            self._commit_edge(edge)
            return dst_id, is_exit

        return None

    # -- Top-level build ---------------------------------------------------
    def _sample_branch_count(self, rng: random.Random) -> int:
        cfg = self.cfg
        r = rng.random()
        if r < cfg.triple_branch_prob:
            return 3
        if r < cfg.triple_branch_prob + cfg.branch_prob:
            return 2
        return 1

    def build(self, rng: random.Random) -> bool:
        cfg = self.cfg
        if not self._seed_root(rng):
            return False
        assert self.root_id is not None
        frontier: list[int] = [self.root_id, *self._decoration_seeds]
        open_set: set[int] = set(frontier)
        steps = 0

        no_fork_dist_px = cfg.min_fork_dist_from_canvas_lw * cfg.lane_width_px

        while frontier and steps < cfg.max_growth_steps and len(self.edges) < cfg.max_edges:
            idx = rng.randrange(len(frontier))
            node_id = frontier.pop(idx)
            open_set.discard(node_id)
            steps += 1

            # Suppress forks while the node is still close to any canvas
            # border: visually two paths are indistinguishable from one wide
            # road until they have travelled at least one segment inwards.
            pos = self.nodes[node_id].pos
            d_border = min(
                float(pos[0]),
                float(cfg.W) - float(pos[0]),
                float(pos[1]),
                float(cfg.H) - float(pos[1]),
            )
            if d_border < no_fork_dist_px:
                n_children = 1
            else:
                n_children = self._sample_branch_count(rng)
            offset = cfg.lat_branch_offset_lw * cfg.lane_width_px
            use_sharp = n_children in (2, 3) and cfg.sharp_junction_prob > 0.0 and rng.random() < cfg.sharp_junction_prob
            if n_children == 1:
                targets: list[LateralTarget] = [None]
            elif n_children == 2:
                if use_sharp:
                    if rng.random() < cfg.sharp_2way_t_weight:
                        q = SharpBranchQuarter(1 if rng.random() < 0.5 else -1)
                        t_list: list[LateralTarget] = [
                            SharpBranchStraight(),
                            q,
                        ]
                    else:
                        t_list = [
                            SharpBranchQuarter(1),
                            SharpBranchQuarter(-1),
                        ]
                    rng.shuffle(t_list)
                    targets = t_list
                else:
                    sign = rng.choice([1.0, -1.0])
                    targets = [+offset * sign, -offset * sign]
            elif use_sharp:
                t_list2: list[LateralTarget] = [
                    SharpBranchQuarter(1),
                    SharpBranchStraight(),
                    SharpBranchQuarter(-1),
                ]
                rng.shuffle(t_list2)
                targets = t_list2
            else:
                targets = [+offset, 0.0, -offset]

            for lateral in targets:
                if len(self.edges) >= self._edge_limit:
                    break
                result = self._try_grow_child(node_id, rng, lateral)
                if result is None:
                    continue
                dst_id, is_exit = result
                if not is_exit and dst_id not in open_set:
                    frontier.append(dst_id)
                    open_set.add(dst_id)

        # No interior dead-ends: every open chain must reach a frame exit. This
        # applies to both the ego (forward) tree and the co-located decoration
        # (behind-ego) sub-tree, so a road always terminates on the image border.
        if not self._extend_leaves_to_border(rng):
            return False
        return True

    def _extend_leaves_to_border(self, rng: random.Random) -> bool:
        """Grows from any leaf that is not yet a border ``is_exit`` until the
        chain reaches the frame edge. Runs for both the forward (ego) tree and
        the co-located decoration (behind-ego) sub-tree. Returns False if
        any dead-end remains (usually collision / edge budget) so the sample
        can be discarded.
        """
        cfg = self.cfg
        old_limit = self._edge_limit
        self._edge_limit = cfg.max_edges + cfg.extension_extra_edges
        try:
            max_rounds = max(cfg.max_growth_steps, 1) * 8
            for _ in range(max_rounds):
                if len(self.edges) >= self._edge_limit:
                    return self._all_leaves_exit()
                non_exit: list[int] = [nid for nid, n in self.nodes.items() if not self.out_edges.get(nid) and not n.is_exit]
                if not non_exit:
                    return True
                rng.shuffle(non_exit)
                progressed = False
                for nid in non_exit:
                    if len(self.edges) >= self._edge_limit:
                        return self._all_leaves_exit()
                    if self._try_grow_child(nid, rng, None) is not None:
                        progressed = True
                if not progressed:
                    return False
            return self._all_leaves_exit()
        finally:
            self._edge_limit = old_limit

    def _all_leaves_exit(self) -> bool:
        for nid, n in self.nodes.items():
            if not self.out_edges.get(nid) and not n.is_exit:
                return False
        return True

    # -- Path enumeration --------------------------------------------------
    def enumerate_exit_paths(self) -> list[list[int]]:
        """DFS from root, return every edge_id list that terminates at an exit node."""
        if self.root_id is None:
            return []
        results: list[list[int]] = []
        # Small headroom above K_MAX is enough: any return value strictly
        # greater than ``K_MAX`` is treated as "too many paths" and rejected
        # by ``generate_sample``.
        max_paths = self.cfg.K_MAX + 4
        max_depth = self.cfg.max_edges * 2

        def dfs(node_id: int, edge_path: list[int], depth: int) -> None:
            if len(results) >= max_paths or depth > max_depth:
                return
            outgoing = self.out_edges.get(node_id, [])
            if not outgoing:
                if self.nodes[node_id].is_exit:
                    results.append(list(edge_path))
                return
            for eid in outgoing:
                edge = self.edges[eid]
                edge_path.append(eid)
                dfs(edge.dst_node, edge_path, depth + 1)
                edge_path.pop()

        dfs(self.root_id, [], 0)
        return results


# ---------------------------------------------------------------------------
# Polyline / rasterisation helpers
# ---------------------------------------------------------------------------


def _polyline_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sqrt((diffs**2).sum(axis=1)).sum())


def _path_polyline(graph: Graph, edge_path: list[int]) -> np.ndarray:
    """Concatenate dense polylines along an ordered edge path from root."""
    if not edge_path:
        return np.zeros((0, 2), dtype=np.float64)
    pieces: list[np.ndarray] = []
    cur = graph.nodes[graph.root_id].pos if graph.root_id is not None else None
    if cur is None:
        return np.zeros((0, 2), dtype=np.float64)
    for eid in edge_path:
        edge = graph.edges[eid]
        poly = edge.polyline
        if poly.shape[0] < 2:
            continue
        if np.linalg.norm(poly[0] - cur) > np.linalg.norm(poly[-1] - cur):
            poly = poly[::-1]
        if pieces:
            poly = poly[1:]
        pieces.append(poly)
        if poly.shape[0] > 0:
            cur = poly[-1]
    if not pieces:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(pieces, axis=0)


def _polygon_to_int_array(poly: Polygon) -> np.ndarray:
    if poly.is_empty:
        return np.zeros((0, 2), dtype=np.int32)
    if poly.geom_type == "Polygon":
        coords = np.array(poly.exterior.coords, dtype=np.float32)
        return np.round(coords).astype(np.int32)
    return np.zeros((0, 2), dtype=np.int32)


def rasterize_road_map(graph: Graph) -> np.ndarray:
    cfg = graph.cfg
    canvas = np.zeros((cfg.H, cfg.W), dtype=np.uint8)
    polys: list[np.ndarray] = []
    for edge in graph.edges.values():
        fp = edge.footprint
        if fp.is_empty:
            continue
        if fp.geom_type == "Polygon":
            arr = _polygon_to_int_array(fp)
            if arr.size > 0:
                polys.append(arr)
        elif fp.geom_type == "MultiPolygon":
            for sub in fp.geoms:
                arr = _polygon_to_int_array(sub)
                if arr.size > 0:
                    polys.append(arr)
    # Fill each edge footprint in its own call. A single `fillPoly` with
    # multiple contours uses the even-odd fill rule, so overlapping buffers
    # (e.g. at a fork) would render as holes; separate fills OR them to 1.
    for arr in polys:
        if arr.size > 0:
            cv2.fillPoly(canvas, [arr], color=1)
    return np.clip(np.rint(canvas * 255.0), 0, 255).astype(np.uint8)


def rasterize_ego_heatmap(cfg: GeneratorConfig, pos: np.ndarray, theta: float) -> np.ndarray:
    """Ego pose as an asymmetric directional Gaussian in image space.

    The heatmap is evaluated on the full (H, W) grid in closed form. Local
    coordinates (u along heading, v lateral) make forward vs. backward
    unambiguous: longitudinal spread is wider for u > 0 (ahead) than for u < 0.

    Returns:
        uint8 array of shape (H, W) with values in [0, 255] (continuous values
        quantized; peak is 255 at the ego position when the maximum is 1.0).
    """
    h, w = int(cfg.H), int(cfg.W)
    cos_theta = math.cos(float(theta))
    sin_theta = math.sin(float(theta))

    # Scales for lane_width; keeps heatmap size consistent when lane width changes.
    scale = float(cfg.lane_width_px) / 12.0
    eps = 1.0e-6
    sigma_lateral = max(float(cfg.ego_heatmap_sigma_lateral) * scale, eps)
    sigma_long_forward = max(float(cfg.ego_heatmap_sigma_long_forward) * scale, eps)
    sigma_long_backward = max(float(cfg.ego_heatmap_sigma_long_backward) * scale, eps)

    y_coords, x_coords = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    px, py = float(pos[0]), float(pos[1])
    du_x = x_coords - px
    du_y = y_coords - py

    # Ego frame: u forward along heading, v to the left of the heading vector.
    u = du_x * cos_theta + du_y * sin_theta
    v = -du_x * sin_theta + du_y * cos_theta

    sigma_u = np.where(u > 0.0, sigma_long_forward, sigma_long_backward)
    # Avoid division by zero if misconfigured; sigmas are already clamped to eps.
    inv_sigma_u2 = 1.0 / (sigma_u * sigma_u)
    inv_sigma_v2 = 1.0 / (sigma_lateral * sigma_lateral)
    heat = np.exp(-0.5 * (u * u * inv_sigma_u2 + v * v * inv_sigma_v2))
    # Peak at (u,v)=(0,0) is 1.0; guard numeric noise.
    heat = np.clip(heat, 0.0, 1.0)
    return np.clip(np.rint(heat * 255.0), 0, 255).astype(np.uint8)


def _arc_length_resample(poly: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at a uniform arc-length spacing.

    Args:
        poly: ``(M, 2)`` array of ``(x, y)`` coordinates in pixels (float).
        step: target spacing along arc length, in pixels. Must be > 0.

    Returns:
        ``(M', 2)`` array of resampled points. The first point of ``poly`` is
        always preserved; the last point of ``poly`` is preserved when its
        cumulative length is at least ``step`` away from the previous sampled
        point.
    """
    if poly.shape[0] < 2 or step <= 0.0:
        return poly.astype(np.float64, copy=True)
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total <= step:
        return poly[[0, -1]].astype(np.float64, copy=True)
    n = int(np.floor(total / step))
    targets = np.arange(0, n + 1, dtype=np.float64) * step
    if total - float(targets[-1]) > 1e-6:
        targets = np.concatenate([targets, [total]])
    out = np.empty((targets.shape[0], 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < cum.shape[0] - 1 and cum[j + 1] < t:
            j += 1
        seg_total = cum[j + 1] - cum[j]
        if seg_total <= 0.0:
            out[i] = poly[j]
        else:
            alpha = (t - cum[j]) / seg_total
            out[i] = poly[j] + alpha * (poly[j + 1] - poly[j])
    return out


def _dedupe_by_cell(poly: np.ndarray, D: int) -> np.ndarray:
    """Drop consecutive points that share the same ``floor(x/D), floor(y/D)`` cell.

    The first point is always kept; subsequent points are kept only if their
    grid cell index differs from the previous kept point's cell index.
    """
    if poly.shape[0] == 0:
        return poly.astype(np.float64, copy=True)
    cells = np.floor(poly / float(D)).astype(np.int64)
    keep = np.ones(poly.shape[0], dtype=bool)
    last = cells[0]
    for i in range(1, poly.shape[0]):
        c = cells[i]
        if int(c[0]) == int(last[0]) and int(c[1]) == int(last[1]):
            keep[i] = False
        else:
            last = c
    return poly[keep].astype(np.float64, copy=True)


def sparsify_polyline(poly: np.ndarray, D: int) -> np.ndarray:
    """Arc-length resample a polyline and drop per-cell duplicates.

    The chosen step is ``D * sqrt(2)`` so that no two consecutive samples can
    fall into the same ``D x D`` grid cell on a straight segment; the final
    dedupe pass handles tight curves and the resampler's last-point handling.
    """
    if poly.shape[0] < 2:
        return poly.astype(np.float64, copy=True)
    step = float(D) * math.sqrt(2.0)
    resampled = _arc_length_resample(poly, step)
    return _dedupe_by_cell(resampled, D)


def prepare_paths_targets(cfg: GeneratorConfig, polylines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparsify each polyline and pack the result into fixed-size arrays.

    Args:
        cfg: Generator configuration; uses ``D``, ``K_MAX``, and ``M_MAX``.
        polylines: List of full-resolution ``(M_k, 2)`` polylines (``(x, y)``)
            beginning at the ego.

    Returns:
        ``(points, num_points, valid_paths)`` where:

        * ``points``: ``(K_MAX, M_MAX, 2) float32`` zero-padded points (pixels).
        * ``num_points``: ``(K_MAX,) int32`` number of valid points per path.
        * ``valid_paths``: ``(K_MAX,) uint8`` 1 if the slot has >= 2 points.

    Paths whose sparsified length exceeds ``M_MAX`` are truncated to
    ``M_MAX`` points (keeping the prefix from the ego).
    """
    K = cfg.K_MAX
    M = cfg.M_MAX
    points = np.zeros((K, M, 2), dtype=np.float32)
    num_points = np.zeros(K, dtype=np.int32)
    valid = np.zeros(K, dtype=np.uint8)
    for k, poly in enumerate(polylines[:K]):
        if poly.shape[0] < 2:
            continue
        sparse = sparsify_polyline(poly, cfg.D)
        if sparse.shape[0] < 2:
            continue
        if sparse.shape[0] > M:
            sparse = sparse[:M]
        n = int(sparse.shape[0])
        points[k, :n, :] = sparse.astype(np.float32, copy=False)
        num_points[k] = n
        valid[k] = 1
    return points, num_points, valid


# ---------------------------------------------------------------------------
# Top-level sample generation
# ---------------------------------------------------------------------------


def _rng_for_sample_attempt(seed: int, attempt: int) -> random.Random:
    """Deterministic stream per (seed, attempt) so ``max_sample_retries`` is not
    all identical to the first draw."""
    s = (seed * 0x1_0000_00_3D + attempt) & 0x7FFFFFFF
    if s == 0:
        s = 0x1_0000_00_3D
    return random.Random(s)


def generate_sample(cfg: GeneratorConfig, seed: int) -> dict | None:
    """Generate one sample. Returns ``None`` if all retries fail."""
    np.random.seed(seed & 0xFFFFFFFF)

    for attempt in range(cfg.max_sample_retries):
        rng = _rng_for_sample_attempt(seed, attempt)
        graph = Graph(cfg)
        if not graph.build(rng):
            continue
        edge_paths = graph.enumerate_exit_paths()
        if not edge_paths:
            continue

        polylines: list[np.ndarray] = []
        path_lengths: list[float] = []
        path_exits: list[tuple[float, float]] = []
        path_segments_dicts: list[list[dict]] = []
        for edge_path in edge_paths:
            poly = _path_polyline(graph, edge_path)
            if poly.shape[0] < 2:
                continue
            polylines.append(poly)
            path_lengths.append(_polyline_length(poly))
            path_exits.append((float(poly[-1, 0]), float(poly[-1, 1])))
            path_segments_dicts.append([graph.edges[eid].path.to_dict() for eid in edge_path])

        if not polylines:
            continue

        # Reject samples whose graph yields more legal exit paths than the
        # label tensor can hold: silently dropping paths leaves road on the
        # canvas with no matching label channel, which hurts learning.
        if len(polylines) > cfg.K_MAX:
            continue

        road_map = rasterize_road_map(graph)
        if int(road_map.sum()) == 0:
            continue

        assert graph.root_id is not None
        root = graph.nodes[graph.root_id]
        ego_heatmap = rasterize_ego_heatmap(cfg, root.pos, root.arrival_heading)
        points_arr, num_points_arr, valid = prepare_paths_targets(cfg, polylines)
        if int(valid.sum()) == 0:
            continue

        image = np.stack([road_map, ego_heatmap], axis=0)

        types_seen: set[str] = set()
        for edge in graph.edges.values():
            types_seen.add("arc" if edge.path.is_arc() else "line")
        primitives_present = sorted(types_seen)

        meta = {
            "format_version": 3,
            "schema": "heatmap_offset_v1",
            "seed": int(seed),
            "attempt": int(attempt),
            "config": {
                "H": cfg.H,
                "W": cfg.W,
                "D": cfg.D,
                "lane_width_px": cfg.lane_width_px,
                "K_MAX": cfg.K_MAX,
                "M_MAX": cfg.M_MAX,
            },
            "ego": {
                "x": float(root.pos[0]),
                "y": float(root.pos[1]),
                "theta_rad": float(root.arrival_heading),
            },
            "primitives_present": primitives_present,
            "n_paths": int(valid.sum()),
            "paths": [
                {
                    "segments": path_segments_dicts[k],
                    "length_px": int(round(path_lengths[k])),
                    "exits_frame_at": [
                        int(round(path_exits[k][0])),
                        int(round(path_exits[k][1])),
                    ],
                    "n_points": int(num_points_arr[k]),
                }
                for k in range(len(polylines))
            ],
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
        }

        return {
            "image": image,
            "points": points_arr,
            "num_points": num_points_arr,
            "valid_paths": valid,
            "meta": meta,
        }

    return None
