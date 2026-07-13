"""Map-backed ground-level cutter paths with cylinder IK seeds.

Every reference comes from, or is interpolated between, physics-settled poses in
``ground_band_map.json``.  No frozen-linkage FK or analytic Jacobian is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist

DEFAULT_MAP = Path(__file__).resolve().parents[1] / "models/ground_band_map.json"


@dataclass(frozen=True)
class GroundPath:
    name: str
    goals: np.ndarray
    cylinder_seeds: np.ndarray
    direct_targets: np.ndarray
    cutter_world_z: np.ndarray
    cumulative_m: np.ndarray

    @property
    def length_m(self) -> float:
        return float(self.cumulative_m[-1])

    def sample(self, distance_m: float, *, pingpong: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
        """Interpolate cutter goal, cylinder seed, and mapped world-Z by arc length."""
        if pingpong:
            period = 2.0 * self.length_m
            value = float(distance_m) % period
            distance_m = value if value <= self.length_m else period - value
        else:
            distance_m = float(np.clip(distance_m, 0.0, self.length_m))
        upper = int(np.searchsorted(self.cumulative_m, distance_m, side="right"))
        upper = min(max(upper, 1), len(self.cumulative_m) - 1)
        lower = upper - 1
        span = max(float(self.cumulative_m[upper] - self.cumulative_m[lower]), 1.0e-9)
        alpha = (distance_m - float(self.cumulative_m[lower])) / span
        goal = (1.0 - alpha) * self.goals[lower] + alpha * self.goals[upper]
        seed = (1.0 - alpha) * self.cylinder_seeds[lower] + alpha * self.cylinder_seeds[upper]
        world_z = (1.0 - alpha) * self.cutter_world_z[lower] + alpha * self.cutter_world_z[upper]
        return goal.astype(np.float32), seed.astype(np.float32), float(world_z)

    def reversed(self) -> "GroundPath":
        goals = self.goals[::-1].copy()
        seeds = self.cylinder_seeds[::-1].copy()
        targets = self.direct_targets[::-1].copy()
        world_z = self.cutter_world_z[::-1].copy()
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]
        return GroundPath(self.name, goals, seeds, targets, world_z, cumulative)


class GroundPathPlanner:
    def __init__(self, map_path: str | Path = DEFAULT_MAP, *, use_standalone: bool = True):
        self.map_path = Path(map_path).expanduser().resolve()
        payload = json.loads(self.map_path.read_text())
        # Runtime paths use the complete physics-settled band.  The optional
        # standalone verification samples remain diagnostic metadata; filtering
        # them created a discontinuous path rather than a usable mowing branch.
        standalone_arc = []
        band = payload.get("ground_band", [])
        self._standalone_ordered = False
        if len(band) < (4 if standalone_arc else 8):
            raise ValueError(f"Ground map has only {len(band)} feasible poses: {self.map_path}")
        self.points = np.asarray([sample["cutter_truck"] for sample in band], dtype=np.float64)
        self.seeds = np.asarray([sample["cylinder_x"] for sample in band], dtype=np.float64)
        self.direct_targets = np.asarray([sample["target_x"] for sample in band], dtype=np.float64)
        self.world_z = np.asarray([sample["cutter_world_z"] for sample in band], dtype=np.float64)
        self._spatial = cdist(self.points, self.points)
        self._cylinder = cdist(self.seeds, self.seeds)

    def _graph(self, max_spatial_edge_m: float, max_cylinder_edge_m: float = 0.08) -> csr_matrix:
        mask = (
            (self._spatial > 1.0e-7)
            & (self._spatial <= float(max_spatial_edge_m))
            & (self._cylinder <= float(max_cylinder_edge_m))
        )
        return csr_matrix(np.where(mask, self._spatial, 0.0))

    @staticmethod
    def _reconstruct(predecessors: np.ndarray, start: int, end: int) -> list[int]:
        path = [int(end)]
        current = int(end)
        while current != int(start):
            current = int(predecessors[current])
            if current < 0:
                raise RuntimeError("Map graph path reconstruction failed")
            path.append(current)
        path.reverse()
        return path

    def _diameter_path(self, graph: csr_matrix, candidates: np.ndarray | None = None) -> list[int]:
        if candidates is None:
            candidates = np.arange(len(self.points), dtype=np.int64)
        sub = graph[candidates][:, candidates]
        count, labels = connected_components(sub, directed=False)
        sizes = np.bincount(labels, minlength=count)
        component = int(np.argmax(sizes))
        local_ids = np.where(labels == component)[0]
        if len(local_ids) < 3:
            raise RuntimeError("No connected ground-band branch with at least three poses")
        connected = sub[local_ids][:, local_ids]
        first_dist = dijkstra(connected, directed=False, indices=0)
        first = int(np.argmax(np.where(np.isfinite(first_dist), first_dist, -1.0)))
        second_dist, predecessors = dijkstra(
            connected, directed=False, indices=first, return_predecessors=True
        )
        second = int(np.argmax(np.where(np.isfinite(second_dist), second_dist, -1.0)))
        local_path = self._reconstruct(predecessors, first, second)
        return candidates[local_ids[np.asarray(local_path, dtype=np.int64)]].tolist()

    def _build(self, name: str, indices: list[int], spacing_m: float = 0.02) -> GroundPath:
        # Drop consecutive duplicate/near-duplicate poses before interpolation.
        raw_p = self.points[np.asarray(indices, dtype=np.int64)]
        raw_q = self.seeds[np.asarray(indices, dtype=np.int64)]
        raw_t = self.direct_targets[np.asarray(indices, dtype=np.int64)]
        raw_z = self.world_z[np.asarray(indices, dtype=np.int64)]
        keep = np.r_[True, np.linalg.norm(np.diff(raw_p, axis=0), axis=1) > 1.0e-5]
        raw_p, raw_q, raw_t, raw_z = raw_p[keep], raw_q[keep], raw_t[keep], raw_z[keep]
        raw_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw_p, axis=0), axis=1))]
        if raw_s[-1] < 1.5:
            raise RuntimeError(f"{name} map branch is only {raw_s[-1]:.3f} m; need >=1.5 m")
        query = np.linspace(0.0, raw_s[-1], max(int(np.ceil(raw_s[-1] / spacing_m)) + 1, 2))
        goals = np.column_stack([np.interp(query, raw_s, raw_p[:, axis]) for axis in range(3)])
        seeds = np.column_stack([np.interp(query, raw_s, raw_q[:, axis]) for axis in range(4)])
        targets = np.column_stack([np.interp(query, raw_s, raw_t[:, axis]) for axis in range(4)])
        world_z = np.interp(query, raw_s, raw_z)
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]
        return GroundPath(name, goals, seeds, targets, world_z, cumulative)

    def arc(self, *, max_abs_z: float | None = None, max_cylinder_edge_m: float = 0.12) -> GroundPath:
        """Longest smooth branch of the settled ground band (natural slew arc).

        ``max_abs_z`` restricts waypoints to a tight height band (the full band
        spans z in [-0.100, +0.096]; its deep stretch made long sweeps dig to
        -0.126 m live). The map holds 106 samples with |z|<=0.05 over 5.9x8.6 m,
        so a flat branch exists -- select it instead of lifting goals post hoc.
        """
        if self._standalone_ordered:
            return self._build("arc", list(range(len(self.points))))
        # The 6-point-per-cylinder grid leaves roughly 0.6 m spacing along the
        # reproducible single-environment branch. Interpolation supplies dense
        # references between these physics-settled IK seeds.
        graph = self._graph(max_spatial_edge_m=0.75, max_cylinder_edge_m=float(max_cylinder_edge_m))
        candidates = None
        if max_abs_z is not None:
            candidates = np.where(np.abs(self.world_z) <= float(max_abs_z))[0]
        return self._build("arc", self._diameter_path(graph, candidates))

    def arc_slew(self, *, max_abs_z: float = 0.05) -> GroundPath:
        """Flat-band mowing sweep: one flat sample per SLEW grid level, chosen to
        keep the boom cylinders (1..3) maximally continuous level-to-level. The
        physical sweep is monotone in slew; a graph-diameter path through the
        dense cloud zig-zags instead."""
        idx = np.where(np.abs(self.world_z) <= float(max_abs_z))[0]
        slew = self.direct_targets[idx, 0]  # grid TARGET levels (6), not measured
        levels = np.unique(np.round(slew, 4))
        kept: list[int] = []
        for lv in levels:
            cand = idx[np.isclose(slew, lv, atol=5e-4)]
            if len(cand) == 0:
                continue
            if not kept:
                boom = self.seeds[cand, 1:]
                pick = cand[int(np.argmin(np.linalg.norm(boom - np.median(boom, axis=0), axis=1)))]
            else:
                prev = self.seeds[kept[-1], 1:]
                pick = cand[int(np.argmin(np.linalg.norm(self.seeds[cand, 1:] - prev, axis=1)))]
            kept.append(int(pick))
        return self._build("arc_slew", kept)

    def make(self, shape: str) -> GroundPath:
        if shape == "arc":
            return self.arc()
        raise ValueError(
            f"{shape!r} is not an arm-only ground-manifold path. "
            "Use long_trajectory_gate.py: the truck supplies a world line and "
            "truck advance plus reciprocal slew supplies the serpentine."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=str(DEFAULT_MAP))
    parser.add_argument("--shapes", default="arc")
    args = parser.parse_args()
    planner = GroundPathPlanner(args.map)
    for name in (part.strip() for part in args.shapes.split(",") if part.strip()):
        path = planner.make(name)
        print(
            f"[GROUND-PATH] {name}: length={path.length_m:.3f}m waypoints={len(path.goals)} "
            f"mapped_world_z=[{path.cutter_world_z.min():+.3f},{path.cutter_world_z.max():+.3f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
