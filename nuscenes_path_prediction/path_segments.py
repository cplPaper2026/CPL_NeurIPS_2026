"""Line/arc path primitives for road-cpl segment-based generation.

Ported and lightly extended from
``gen3-pocs/road_poc/model/path_utils.py``. The math is invariant under a
y-axis flip, so the same formulas work directly in canvas pixel coordinates
(y grows downward). The "clockwise" flag inside :class:`ArcPath` therefore
reflects the standard mathematical convention; visually on a y-down screen
the rotation reads as the opposite handedness, which is purely cosmetic.

Three classes are exposed:

* :class:`LinePath` - straight segment between two endpoints.
* :class:`ArcPath` - circular arc with explicit center / radius / sweep.
* :class:`LineOrArcPath` - smart wrapper that picks line or arc from a
  ``(x0, y0, h0, x1, y1)`` triple via the pure-pursuit closed form.

Compared to the upstream module we add helpers required by the road-cpl
generator: ``last_point``, ``last_heading``, ``polyline_samples``,
``buffer_polygon``, ``crosses_canvas_bounds``, ``truncate_to_bounds`` and a
``radius`` property.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from shapely.geometry import LineString, Polygon

Point = np.ndarray  # shape (2,) representing (x, y)

_DEFAULT_ARC_SAMPLES = 64
_MIN_POLYLINE_SAMPLES = 4


def normalize_angle(angle: float) -> float:
    """Wrap ``angle`` (radians) to the half-open interval (-pi, pi]."""
    a = (angle + np.pi) % (2 * np.pi) - np.pi
    if a <= -np.pi:
        a += 2 * np.pi
    return float(a)


def direction_from_angle(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)


class BasePath(ABC):
    """Common interface shared by line and arc representations."""

    @abstractmethod
    def path2cartesian(self, p: Point) -> Point: ...

    @abstractmethod
    def cartesian2path(self, p: Point) -> Point: ...

    @abstractmethod
    def heading(self, m: float = 0.0) -> float: ...

    @abstractmethod
    def inv_r(self, m: float) -> float: ...

    @abstractmethod
    def first_m(self) -> float: ...

    @abstractmethod
    def last_m(self) -> float: ...


class LinePath(BasePath):
    """Straight line between two endpoints in canvas pixel coordinates."""

    def __init__(
        self,
        x0: Point | None = None,
        x1: Point | None = None,
        heading: float | None = None,
        length: float | None = None,
    ) -> None:
        self.x0 = np.zeros(2) if x0 is None else np.asarray(x0, dtype=np.float64)
        self._l = 0.0
        self.v = np.array([1.0, 0.0], dtype=np.float64)
        self.vt = np.array([0.0, 1.0], dtype=np.float64)
        self._h: float | None = None
        self._h_valid = False

        if x1 is not None and heading is None:
            self.x1 = np.asarray(x1, dtype=np.float64)
            v = self.x1 - self.x0
            self._l = float(np.linalg.norm(v))
            if self._l < 1e-7:
                self.x1 = self.x1.copy()
                self.x1[0] += 1e-7
                v = self.x1 - self.x0
                self._l = float(np.linalg.norm(v))
            self.v = v / self._l
            self.vt = np.array([-self.v[1], self.v[0]], dtype=np.float64)
        elif heading is not None and length is not None:
            self._l = float(length)
            self._h = float(heading)
            self._h_valid = True
            self.v = direction_from_angle(self._h)
            self.vt = np.array([-self.v[1], self.v[0]], dtype=np.float64)
            self.x1 = self.x0 + self._l * self.v
        else:
            self.x1 = self.x0.copy()

    @classmethod
    def from_endpoints(cls, x0: Point, x1: Point) -> LinePath:
        return cls(x0=x0, x1=x1)

    @classmethod
    def from_start_heading_length(cls, start: Point, heading: float, length: float) -> LinePath:
        return cls(x0=start, heading=heading, length=length)

    def cartesian2path(self, p: Point) -> Point:
        d = np.asarray(p, dtype=np.float64) - self.x0
        return np.array([float(np.dot(self.v, d)), float(np.dot(self.vt, d))])

    def path2cartesian(self, p: Point) -> Point:
        p = np.asarray(p, dtype=np.float64)
        return self.x0 + p[0] * self.v + p[1] * self.vt

    def heading(self, m: float = 0.0) -> float:
        if not self._h_valid:
            self._h = float(np.arctan2(self.v[1], self.v[0]))
            self._h_valid = True
        return float(self._h)  # type: ignore[arg-type]

    def inv_r(self, m: float) -> float:
        return 0.0

    def first_m(self) -> float:
        return 0.0

    def last_m(self) -> float:
        return self._l


class ArcPath(BasePath):
    """Circular arc parameterised by ``(center, radius, theta_start, length, cw)``."""

    def __init__(
        self,
        center: Point,
        radius: float,
        theta_start: float,
        arc_length: float,
        clockwise: bool,
    ) -> None:
        self._c = np.asarray(center, dtype=np.float64)
        self._radius = float(abs(radius))
        self._theta_start = float(theta_start)
        self._l = float(abs(arc_length))
        self._s = -1.0 if clockwise else 1.0
        self._clockwise = bool(clockwise)

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def center(self) -> np.ndarray:
        return self._c.copy()

    @property
    def clockwise(self) -> bool:
        return self._clockwise

    @property
    def theta_start(self) -> float:
        return self._theta_start

    def cartesian2path(self, p: Point) -> Point:
        d = np.asarray(p, dtype=np.float64) - self._c
        r = float(np.linalg.norm(d))
        theta = float(np.arctan2(d[1], d[0]))
        dtheta = theta - self._theta_start
        if self._s > 0:
            while dtheta < 0:
                dtheta += 2 * np.pi
            while dtheta > 2 * np.pi:
                dtheta -= 2 * np.pi
        else:
            while dtheta > 0:
                dtheta -= 2 * np.pi
            while dtheta < -2 * np.pi:
                dtheta += 2 * np.pi
            dtheta = -dtheta
        m = dtheta * self._radius
        lateral = self._s * (self._radius - r)
        return np.array([m, lateral], dtype=np.float64)

    def path2cartesian(self, p: Point) -> Point:
        p = np.asarray(p, dtype=np.float64)
        theta = self._theta_start + self._s * p[0] / self._radius
        r = self._radius - self._s * p[1]
        return self._c + r * direction_from_angle(theta)

    def heading(self, m: float = 0.0) -> float:
        theta = self._theta_start + self._s * float(m) / self._radius
        return normalize_angle(theta + self._s * np.pi / 2)

    def inv_r(self, m: float) -> float:
        return self._s / self._radius

    def first_m(self) -> float:
        return 0.0

    def last_m(self) -> float:
        return self._l


class LineOrArcPath(BasePath):
    """Either a :class:`LinePath` or :class:`ArcPath` chosen by pure pursuit.

    Given a start pose ``(x0, y0, h0)`` and an endpoint ``(x1, y1)``, the
    constructor solves for the unique circle of curvature that passes through
    the start and the endpoint with the given starting tangent. If the
    resulting radius is finite (and the endpoint lies forward of the start in
    the heading-aligned frame), an :class:`ArcPath` is created. Otherwise the
    path collapses to a straight :class:`LinePath`.
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        h0: float,
        x1: float,
        y1: float,
    ) -> None:
        self._lp: LinePath | None = None
        self._ap: ArcPath | None = None
        self._isArc = False
        self._l = 0.0
        self._x0 = float(x0)
        self._y0 = float(y0)
        self._h0 = float(h0)
        self._x1 = float(x1)
        self._y1 = float(y1)

        t_x1 = x1 - x0
        t_y1 = y1 - y0
        cos_h = float(np.cos(-h0))
        sin_h = float(np.sin(-h0))
        r_x1 = cos_h * t_x1 - sin_h * t_y1
        r_y1 = sin_h * t_x1 + cos_h * t_y1

        denom = 2 * abs(r_y1)
        denom = max(denom, 1e-7)
        radius = (r_x1 * r_x1 + r_y1 * r_y1) / denom
        theta_delta = 2 * float(np.arctan2(abs(r_y1), r_x1))

        if radius < 1.0e5 and r_x1 > 0 and theta_delta > 1e-4:
            self._isArc = True
            if r_y1 > 0:
                theta_start = normalize_angle(-np.pi / 2 + h0)
                clockwise = False
            else:
                theta_start = normalize_angle(np.pi / 2 + h0)
                clockwise = True
            cos_ts = float(np.cos(theta_start))
            sin_ts = float(np.sin(theta_start))
            cx = x0 - radius * cos_ts
            cy = y0 - radius * sin_ts
            arc_length = radius * theta_delta
            self._ap = ArcPath(
                center=np.array([cx, cy], dtype=np.float64),
                radius=radius,
                theta_start=theta_start,
                arc_length=arc_length,
                clockwise=clockwise,
            )
            self._l = self._ap.last_m()
        else:
            self._isArc = False
            self._lp = LinePath.from_endpoints(
                x0=np.array([x0, y0], dtype=np.float64),
                x1=np.array([x1, y1], dtype=np.float64),
            )
            self._l = self._lp.last_m()

    # ---- type predicates ------------------------------------------------
    def is_arc(self) -> bool:
        return self._isArc

    def is_line(self) -> bool:
        return not self._isArc

    @property
    def radius(self) -> float:
        if self._isArc:
            assert self._ap is not None
            return self._ap.radius
        return float("inf")

    # ---- BasePath interface --------------------------------------------
    def path2cartesian(self, p: Point) -> Point:
        if self._isArc:
            assert self._ap is not None
            return self._ap.path2cartesian(p)
        assert self._lp is not None
        return self._lp.path2cartesian(p)

    def cartesian2path(self, p: Point) -> Point:
        if self._isArc:
            assert self._ap is not None
            return self._ap.cartesian2path(p)
        assert self._lp is not None
        return self._lp.cartesian2path(p)

    def heading(self, m: float = 0.0) -> float:
        if self._isArc:
            assert self._ap is not None
            return self._ap.heading(m)
        assert self._lp is not None
        return self._lp.heading(m)

    def inv_r(self, m: float) -> float:
        if self._isArc:
            assert self._ap is not None
            return self._ap.inv_r(m)
        return 0.0

    def first_m(self) -> float:
        return 0.0

    def last_m(self) -> float:
        return self._l

    # ---- start / end pose ----------------------------------------------
    def start_point(self) -> np.ndarray:
        return np.array([self._x0, self._y0], dtype=np.float64)

    def start_heading(self) -> float:
        return self._h0

    def last_point(self) -> np.ndarray:
        return self.path2cartesian(np.array([self._l, 0.0], dtype=np.float64))

    def last_heading(self) -> float:
        return self.heading(self._l)

    # ---- sampling / geometry helpers -----------------------------------
    def polyline_samples(self, samples_per_lw: float = 1.0, lane_width_px: float = 1.0) -> np.ndarray:
        """Return a dense (M, 2) polyline along the path.

        For lines, exactly two endpoints are returned. For arcs we sample
        adaptively at roughly ``samples_per_lw`` points per lane-width of arc
        length, with a soft minimum of :data:`_MIN_POLYLINE_SAMPLES`.
        """
        if not self._isArc:
            assert self._lp is not None
            return np.stack([self._lp.x0, self._lp.x1], axis=0)
        density = max(1e-6, samples_per_lw / max(1e-6, lane_width_px))
        n = int(np.ceil(self._l * density)) + 1
        n = max(_MIN_POLYLINE_SAMPLES, min(n, _DEFAULT_ARC_SAMPLES))
        ms = np.linspace(0.0, self._l, n)
        pts = np.stack(
            [self.path2cartesian(np.array([m, 0.0], dtype=np.float64)) for m in ms],
            axis=0,
        )
        return pts

    def buffer_polygon(
        self,
        half_width: float,
        samples_per_lw: float = 1.0,
        lane_width_px: float = 1.0,
    ) -> Polygon:
        """Return a flat-cap buffer polygon around the dense polyline."""
        pts = self.polyline_samples(samples_per_lw, lane_width_px)
        if pts.shape[0] < 2:
            return Polygon()
        line = LineString(pts)
        return line.buffer(half_width, cap_style=2, join_style=2, resolution=8)

    # ---- canvas-bounds clipping ----------------------------------------
    def _outside_canvas(self, p: np.ndarray, W: float, H: float) -> bool:
        return bool(p[0] < 0 or p[0] > W or p[1] < 0 or p[1] > H)

    def crosses_canvas_bounds(self, W: float, H: float) -> float | None:
        """Return the smallest ``m > 0`` at which the path leaves ``[0,W]x[0,H]``.

        ``None`` if the entire segment is fully inside the canvas.
        """
        n_probe = _DEFAULT_ARC_SAMPLES if self._isArc else 8
        ms = np.linspace(0.0, self._l, n_probe)
        first_outside = -1
        for i, m in enumerate(ms):
            if i == 0:
                continue
            p = self.path2cartesian(np.array([m, 0.0], dtype=np.float64))
            if self._outside_canvas(p, W, H):
                first_outside = i
                break
        if first_outside < 0:
            return None
        lo = float(ms[first_outside - 1])
        hi = float(ms[first_outside])
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            pmid = self.path2cartesian(np.array([mid, 0.0], dtype=np.float64))
            if self._outside_canvas(pmid, W, H):
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-4:
                break
        return 0.5 * (lo + hi)

    def truncate_to_bounds(self, W: float, H: float) -> LineOrArcPath:
        """Return a copy clipped at the first canvas-border crossing.

        If the segment lies entirely inside the canvas, returns ``self``.
        """
        m_cross = self.crosses_canvas_bounds(W, H)
        if m_cross is None:
            return self
        end = self.path2cartesian(np.array([m_cross, 0.0], dtype=np.float64))
        return LineOrArcPath(self._x0, self._y0, self._h0, float(end[0]), float(end[1]))

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        if self._isArc:
            assert self._ap is not None
            arc = self._ap
            start_point = arc.path2cartesian(np.array([0.0, 0.0]))
            return {
                "type": "arc",
                "center_x": float(arc._c[0]),
                "center_y": float(arc._c[1]),
                "radius": float(arc.radius),
                "theta_start": float(arc.theta_start),
                "arc_length": float(arc.last_m()),
                "clockwise": bool(arc.clockwise),
                "start_x": float(start_point[0]),
                "start_y": float(start_point[1]),
                "start_heading": float(arc.heading(0.0)),
            }
        assert self._lp is not None
        line = self._lp
        return {
            "type": "line",
            "x0": float(line.x0[0]),
            "y0": float(line.x0[1]),
            "x1": float(line.x1[0]),
            "y1": float(line.x1[1]),
            "length": float(line.last_m()),
            "heading": float(line.heading()),
        }

    @classmethod
    def from_dict(cls, params: dict) -> LineOrArcPath:
        if params["type"] == "arc":
            sign = -1.0 if params["clockwise"] else 1.0
            theta_end = params["theta_start"] + sign * params["arc_length"] / params["radius"]
            end_x = params["center_x"] + params["radius"] * np.cos(theta_end)
            end_y = params["center_y"] + params["radius"] * np.sin(theta_end)
            return cls(
                params["start_x"],
                params["start_y"],
                params["start_heading"],
                float(end_x),
                float(end_y),
            )
        return cls(
            params["x0"],
            params["y0"],
            params["heading"],
            params["x1"],
            params["y1"],
        )

    def __repr__(self) -> str:
        if self._isArc:
            return f"LineOrArcPath(arc len={self._l:.2f}, r={self.radius:.2f})"
        return f"LineOrArcPath(line len={self._l:.2f})"
