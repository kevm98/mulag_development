"""Standalone planar reference trajectories and nearest-point geometry.

The old trajectory code stored only a monotone ``y(x)`` function.  That is
adequate for a road-following line, but it cannot represent a circle or a
rounded square because those paths are not single-valued in ``x``.  This
module is deliberately independent of Isaac Sim and of the hydraulic models:
it is the geometry contract used by visualization, preflight, observations,
and reward code.

Coordinates are metres in the caller's planar frame.  ``s`` is arc length;
closed paths do not repeat their first vertex in the stored samples, and wrap
around when queried.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _as_xy(value: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    xy = np.asarray(value, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise ValueError(f"xy must have shape (N, 2), got {xy.shape}")
    if not np.isfinite(xy).all():
        raise ValueError("trajectory coordinates must be finite")
    return xy


def _resample_polyline(vertices: np.ndarray, count: int, *, closed: bool) -> np.ndarray:
    vertices = _as_xy(vertices)
    count = int(count)
    if count < 16:
        raise ValueError("count must be at least 16")
    if closed:
        edges = np.vstack((vertices[1:] - vertices[:-1], vertices[0] - vertices[-1]))
    else:
        edges = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1.0e-10):
        raise ValueError("trajectory contains a zero-length edge")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    targets = np.linspace(0.0, total, count, endpoint=not closed)
    result = np.empty((count, 2), dtype=np.float64)
    for i, distance in enumerate(targets):
        edge = int(np.searchsorted(cumulative, distance, side="right") - 1)
        edge = min(max(edge, 0), len(edges) - 1)
        alpha = (distance - cumulative[edge]) / lengths[edge]
        result[i] = vertices[edge] + alpha * edges[edge]
    return result


@dataclass(frozen=True)
class NearestPointResult:
    """Nearest-point query result in the trajectory's planar frame."""

    point_xy: np.ndarray
    tangent_xy: np.ndarray
    normal_xy: np.ndarray
    signed_normal_m: np.ndarray
    distance_m: np.ndarray
    s_m: np.ndarray


@dataclass(frozen=True)
class Trajectory2D:
    """Arc-length sampled planar trajectory."""

    name: str
    xy: np.ndarray
    s_m: np.ndarray
    tangent_xy: np.ndarray
    normal_xy: np.ndarray
    closed: bool
    metadata: dict[str, float | int | str]

    def __post_init__(self) -> None:
        xy = _as_xy(self.xy)
        s = np.asarray(self.s_m, dtype=np.float64)
        tangent = _as_xy(self.tangent_xy)
        normal = _as_xy(self.normal_xy)
        if len(xy) != len(s) or tangent.shape != xy.shape or normal.shape != xy.shape:
            raise ValueError("trajectory arrays must have the same length")
        if s[0] != 0.0 or np.any(np.diff(s) <= 0.0):
            raise ValueError("s_m must start at zero and increase strictly")
        if float(s[-1]) <= 0.0:
            raise ValueError("trajectory length must be positive")
        if not all(np.isfinite(value).all() for value in (xy, s, tangent, normal)):
            raise ValueError("trajectory arrays must be finite")

    @property
    def length_m(self) -> float:
        return float(self.s_m[-1])

    @property
    def sample_count(self) -> int:
        return int(self.xy.shape[0])

    def sample(self, s_m: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate point, tangent and normal at arc length ``s_m``."""
        query = np.asarray(s_m, dtype=np.float64)
        if self.closed:
            query = np.mod(query, self.length_m)
        else:
            query = np.clip(query, 0.0, self.length_m)
        if self.closed:
            s_ext = np.r_[self.s_m, self.length_m]
            xy_ext = np.vstack((self.xy, self.xy[0]))
            tan_ext = np.vstack((self.tangent_xy, self.tangent_xy[0]))
            norm_ext = np.vstack((self.normal_xy, self.normal_xy[0]))
        else:
            s_ext, xy_ext, tan_ext, norm_ext = (
                self.s_m,
                self.xy,
                self.tangent_xy,
                self.normal_xy,
            )
        point = np.stack([np.interp(query, s_ext, xy_ext[:, i]) for i in range(2)], axis=-1)
        tangent = np.stack([np.interp(query, s_ext, tan_ext[:, i]) for i in range(2)], axis=-1)
        normal = np.stack([np.interp(query, s_ext, norm_ext[:, i]) for i in range(2)], axis=-1)
        tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True).clip(min=1.0e-9)
        normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
        return point, tangent, normal

    def nearest_point(self, points_xy: np.ndarray | Iterable[Iterable[float]]) -> NearestPointResult:
        """Find the nearest point on the polyline, including segment interiors."""
        points = np.asarray(points_xy, dtype=np.float64)
        scalar = points.ndim == 1
        if scalar:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
            raise ValueError(f"points_xy must have shape (N, 2), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("query points must be finite")
        start = self.xy
        if self.closed:
            end = np.roll(self.xy, -1, axis=0)
            edge_s = self.s_m
            edge_lengths = np.diff(np.r_[self.s_m, self.length_m])
        else:
            start = self.xy[:-1]
            end = self.xy[1:]
            edge_s = self.s_m[:-1]
            edge_lengths = np.diff(self.s_m)
        edge = end - start
        denom = np.sum(edge * edge, axis=1).clip(min=1.0e-12)
        delta = points[:, None, :] - start[None, :, :]
        alpha = np.sum(delta * edge[None, :, :], axis=-1) / denom[None, :]
        alpha = np.clip(alpha, 0.0, 1.0)
        candidates = start[None, :, :] + alpha[..., None] * edge[None, :, :]
        distance_sq = np.sum((points[:, None, :] - candidates) ** 2, axis=-1)
        indices = np.argmin(distance_sq, axis=1)
        rows = np.arange(points.shape[0])
        nearest = candidates[rows, indices]
        s = edge_s[indices] + alpha[rows, indices] * edge_lengths[indices]
        if self.closed:
            s = np.mod(s, self.length_m)
        _, tangent, normal = self.sample(s)
        signed = np.sum((points - nearest) * normal, axis=-1)
        result = NearestPointResult(
            point_xy=nearest,
            tangent_xy=tangent,
            normal_xy=normal,
            signed_normal_m=signed,
            distance_m=np.sqrt(distance_sq[rows, indices]),
            s_m=s,
        )
        if scalar:
            return NearestPointResult(*(value[0] for value in result.__dict__.values()))
        return result


def _from_vertices(name: str, vertices: np.ndarray, *, closed: bool, count: int, metadata: dict) -> Trajectory2D:
    xy = _resample_polyline(vertices, count, closed=closed)
    if closed:
        edge = np.roll(xy, -1, axis=0) - xy
        lengths = np.linalg.norm(edge, axis=1)
        s = np.r_[0.0, np.cumsum(lengths[:-1])]
        tangent = edge / lengths[:, None]
    else:
        edge = np.diff(xy, axis=0)
        lengths = np.linalg.norm(edge, axis=1)
        s = np.r_[0.0, np.cumsum(lengths)]
        tangent = np.empty_like(xy)
        tangent[0] = edge[0] / lengths[0]
        tangent[-1] = edge[-1] / lengths[-1]
        tangent[1:-1] = xy[2:] - xy[:-2]
        tangent[1:-1] /= np.linalg.norm(tangent[1:-1], axis=1, keepdims=True).clip(min=1.0e-9)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    return Trajectory2D(name, xy, s, tangent, normal, closed, dict(metadata))


def from_xy_samples(
    xy: np.ndarray | Iterable[Iterable[float]],
    *,
    name: str = "trajectory",
    closed: bool = False,
) -> Trajectory2D:
    """Build the same contract from an already sampled polyline.

    This adapter is useful when a plant-backed JSON library provides the
    reference points but not tangent/normal columns.  It resamples by arc
    length so nearest-point queries have the same numerical behaviour as the
    built-in factories.
    """
    value = _as_xy(xy)
    return _from_vertices(str(name), value, closed=bool(closed), count=len(value), metadata={})


def line(*, length_m: float = 2.5, count: int = 241) -> Trajectory2D:
    length_m = float(length_m)
    if length_m <= 0.0:
        raise ValueError("line length must be positive")
    return _from_vertices("line", np.array([[-0.5 * length_m, 0.0], [0.5 * length_m, 0.0]]), closed=False, count=count, metadata={"length_m": length_m})


def sine_chain(*, length_m: float = 3.0, amplitude_m: float = 0.18, cycles: float = 3.0, count: int = 361) -> Trajectory2D:
    length_m, amplitude_m, cycles = float(length_m), float(amplitude_m), float(cycles)
    if length_m <= 0.0 or amplitude_m < 0.0 or cycles <= 0.0:
        raise ValueError("sine_chain length/amplitude/cycles are invalid")
    x = np.linspace(-0.5 * length_m, 0.5 * length_m, max(count, 32))
    y = amplitude_m * np.sin(2.0 * math.pi * cycles * (x + 0.5 * length_m) / length_m)
    return _from_vertices("sine_chain", np.column_stack((x, y)), closed=False, count=count, metadata={"length_m": length_m, "amplitude_m": amplitude_m, "cycles": cycles})


def circle(*, radius_m: float = 0.35, count: int = 361) -> Trajectory2D:
    radius_m = float(radius_m)
    if radius_m <= 0.0:
        raise ValueError("circle radius must be positive")
    theta = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    vertices = radius_m * np.column_stack((np.cos(theta), np.sin(theta)))
    return _from_vertices("circle", vertices, closed=True, count=count, metadata={"radius_m": radius_m})


def rounded_square(*, side_m: float = 0.80, corner_radius_m: float = 0.12, count: int = 401) -> Trajectory2D:
    side_m, radius = float(side_m), float(corner_radius_m)
    if side_m <= 0.0 or radius <= 0.0 or radius > 0.5 * side_m:
        raise ValueError("rounded-square requires 0 < corner_radius <= side/2")
    h = 0.5 * side_m
    vertices: list[np.ndarray] = [np.array([-h + radius, -h]), np.array([h - radius, -h])]
    corners = (
        (np.array([h - radius, -h + radius]), -math.pi / 2.0, 0.0),
        (np.array([h - radius, h - radius]), 0.0, math.pi / 2.0),
        (np.array([-h + radius, h - radius]), math.pi / 2.0, math.pi),
        (np.array([-h + radius, -h + radius]), math.pi, 1.5 * math.pi),
    )
    straight_endpoints = (
        np.array([h, h - radius]),
        np.array([h - radius, h]),
        np.array([-h, h - radius]),
        np.array([-h + radius, -h]),
    )
    # Add each quarter circle, then the next straight segment.  The final point
    # is the initial point and is removed for the closed representation.
    for (center, start_angle, end_angle), straight_end in zip(corners, straight_endpoints):
        angles = np.linspace(start_angle, end_angle, 16)[1:]
        vertices.extend(center + radius * np.column_stack((np.cos(angles), np.sin(angles))))
        vertices.append(straight_end)
    raw = np.asarray(vertices[:-1], dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1.0e-8]
    raw = raw[keep]
    if np.linalg.norm(raw[-1] - raw[0]) <= 1.0e-8:
        raw = raw[:-1]
    return _from_vertices("rounded_square", raw, closed=True, count=count, metadata={"side_m": side_m, "corner_radius_m": radius})


TRAJECTORY_FACTORIES = {
    "line": line,
    "sine_chain": sine_chain,
    "circle": circle,
    "rounded_square": rounded_square,
}


def make_trajectory(name: str, **kwargs) -> Trajectory2D:
    try:
        factory = TRAJECTORY_FACTORIES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown 2-D trajectory {name!r}; expected {tuple(TRAJECTORY_FACTORIES)}") from exc
    return factory(**kwargs)


__all__ = [
    "NearestPointResult",
    "Trajectory2D",
    "TRAJECTORY_FACTORIES",
    "circle",
    "from_xy_samples",
    "line",
    "make_trajectory",
    "rounded_square",
    "sine_chain",
]
