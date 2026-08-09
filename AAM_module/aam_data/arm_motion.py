"""Stage 10 compact arm-motion trajectory set (AAM data collection).

Reuses the existing trajectory generators; no new shape complexity is added.
Sources:

- ``mulag_plant/models/plant_reachable_trajectory_library_v1.json``: exact
  shapes line, arc, sine, circle, square (plus procedural variants), each a
  dense 241-sample path of tool goals, cylinder seeds, and arc length.
- ``mulag_plant/models/mowing_arc_head_down.json``: the legacy head-down
  ground arc, the only existing generator with a large boom (cylinder 0)
  excursion and varying tool height. It is the "vertical sweep" of the set.

The module samples each shape along arc length at the control dt with the same
smooth accel-cruise-decel profile the deployed env uses (mirror of
``mulag_plant.trajectories.time_scaling.cruise_profile``), then derives the
reference-implied action matrix ``U`` in normalized action space. Coverage
tags (extension band, velocity sign, reversals, holds, speed band,
single-cylinder vs coupled) are computed from the seed path so the Stage 10
report can show that the set covers the required state space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIBRARY_PATH = ROOT / "mulag_plant/models/plant_reachable_trajectory_library_v1.json"
LEGACY_ARC_PATH = ROOT / "mulag_plant/models/mowing_arc_head_down.json"

CONTROL_DT = 1.0 / 60.0          # native control step
ROM_DT = 0.05                    # command ROM tick
ACTION_SCALE = 0.02              # metres per normalized action unit (env cfg)
CRUISE_SPEED_MPS = 0.13          # env base path speed (policy env_cfg default)

# The compact set requested by the Stage 10 spec. "horizontal sweep" and
# "line" share the library line shape; "vertical sweep" is the legacy
# head-down arc (boom-dominated, tool height varies); "reach-and-return" is
# the line traversed A->B->A. circle/square stay available as coupled-motion
# extras but are not required for the first collection.
COMPACT_SET = (
    ("horizontal_sweep", "line"),
    ("vertical_sweep", "legacy_arc"),
    ("line", "line"),
    ("arc", "arc"),
    ("sine", "sine"),
    ("reach_and_return", "line"),
)
EXTRAS = ("circle", "square")


def _cruise_profile(length_m: float, cruise_speed_mps: float, dt: float,
                    accel_time_s: float = 1.0) -> np.ndarray:
    """Sample arc positions ``s(t)`` at dt, mirroring the deployed cruise profile.

    Same smoothstep accel-cruise-decel math as
    ``mulag_plant.trajectories.time_scaling.cruise_profile``: blend distance
    ``0.5 * v_c * t_a`` per ramp, symmetric ramps, cruise speed reduced when
    the stroke is too short to reach the requested speed.
    """
    v_c = float(cruise_speed_mps)
    t_a = max(float(accel_time_s), 1e-6)
    length = max(float(length_m), 1e-9)
    if v_c * t_a > length:  # 0.5*v_c*t_a + 0.5*v_c*t_a > length
        v_c = length / t_a
    d_ramp = 0.5 * v_c * t_a
    d_cruise = max(length - 2.0 * d_ramp, 0.0)
    t_cruise = d_cruise / max(v_c, 1e-9)
    total = 2.0 * t_a + t_cruise
    n = int(np.ceil(total / dt)) + 1
    t = np.arange(n, dtype=np.float64) * dt
    s = np.zeros(n)
    # accelerate
    m = t < t_a
    tau = np.clip(t[m] / t_a, 0.0, 1.0)
    s[m] = v_c * t_a * (tau**3 - 0.5 * tau**4)
    # cruise
    m = (t >= t_a) & (t < t_a + t_cruise)
    s[m] = d_ramp + v_c * (t[m] - t_a)
    # decelerate (mirror)
    m = t >= t_a + t_cruise
    tau = np.clip((total - t[m]) / t_a, 0.0, 1.0)
    s[m] = length - v_c * t_a * (tau**3 - 0.5 * tau**4)
    return np.clip(s, 0.0, length)


def load_library_shapes() -> dict[str, dict]:
    return json.loads(LIBRARY_PATH.read_text())["shapes"]


def load_legacy_arc() -> dict:
    payload = json.loads(LEGACY_ARC_PATH.read_text())
    teacher = payload["cylinder_teacher"]
    theta = np.asarray([row["theta_deg"] for row in teacher], dtype=np.float64)
    order = np.argsort(theta)
    goals = np.asarray([row["cutter_truck"] for row in teacher], dtype=np.float64)[order]
    seeds = np.asarray([row["cylinder_cmd"] for row in teacher], dtype=np.float64)[order]
    world_z = np.asarray([row["cutter_world_z"] for row in teacher], dtype=np.float64)[order]
    path_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]
    return {
        "base_shape": "legacy_arc",
        "speed_factor": 1.0,
        "cylinder_span_m": (seeds.max(axis=0) - seeds.min(axis=0)).tolist(),
        "path_s_m": path_s,
        "goals_m": goals,
        "seeds": seeds,
        "world_z": world_z,
    }


def shape_arrays(name: str) -> dict:
    """Return ``{path_s_m, goals_m, seeds, world_z, speed_factor, span}``."""
    if name == "legacy_arc":
        return load_legacy_arc()
    shapes = load_library_shapes()
    if name not in shapes:
        raise ValueError(f"unknown shape {name!r}; have {sorted(shapes)}")
    sh = shapes[name]
    rows = sh["samples"]
    return {
        "base_shape": sh.get("base_shape", name),
        "speed_factor": float(sh.get("speed_factor", 1.0)),
        "cylinder_span_m": list(sh.get("cylinder_span_m", [0.0] * 4)),
        "path_s_m": np.asarray([row["path_s_m"] for row in rows], dtype=np.float64),
        "goals_m": np.asarray([row["cutter_truck"] for row in rows], dtype=np.float64),
        "seeds": np.asarray([row["cylinder_cmd"] for row in rows], dtype=np.float64),
        "world_z": np.asarray([row["cutter_world_z"] for row in rows], dtype=np.float64),
    }


def _interp(path_s: np.ndarray, values: np.ndarray, s: float) -> np.ndarray:
    return np.asarray(
        [np.interp(s, path_s, values[:, i]) for i in range(values.shape[1])],
        dtype=np.float64,
    )


@dataclass
class ArmTrajectory:
    name: str
    source: str
    dt: float
    cruise_speed_mps: float
    goals: np.ndarray            # (T, 3)
    seeds: np.ndarray            # (T, 4) learned cylinder extension
    world_z: np.ndarray          # (T,)
    path_s: np.ndarray           # (T,) arc position per control step
    action: np.ndarray           # (T, 4) reference-implied normalized action
    coverage: dict = field(default_factory=dict)

    @property
    def arc_length_m(self) -> float:
        return float(self.path_s[-1])

    @property
    def num_steps(self) -> int:
        return int(self.action.shape[0])


def build_trajectory(
    name: str,
    source: str,
    *,
    control_dt: float = CONTROL_DT,
    cruise_speed_mps: float = CRUISE_SPEED_MPS,
    speed_factor: float = 1.0,
    reach_and_return: bool = False,
) -> ArmTrajectory:
    """Sample one shape along arc length and derive its reference action matrix.

    ``action[t] = (seeds[t+1] - seeds[t]) / action_scale`` is the normalized
    action that reproduces the library seed path under the deployed adapter
    (``target_delta_x = action * action_scale``, learned metres per step; the
    per-channel mapping sign is a constant and is omitted). The last row is
    zeroed so ``action`` has exactly one row per control step.
    """
    data = shape_arrays(source)
    path_s = data["path_s_m"]
    goals = data["goals_m"]
    seeds = data["seeds"]
    world_z = data["world_z"]
    length = float(path_s[-1])
    speed = float(cruise_speed_mps) * float(speed_factor)

    s_fwd = _cruise_profile(length, speed, control_dt)
    goal_fwd = np.stack([_interp(path_s, goals, s) for s in s_fwd], axis=0)
    seed_fwd = np.stack([_interp(path_s, seeds, s) for s in s_fwd], axis=0)
    wz_fwd = np.asarray([np.interp(s, path_s, world_z) for s in s_fwd], dtype=np.float64)

    if reach_and_return:
        # A->B then B->A: direction reversal + both velocity signs in one run.
        goal_bwd = goal_fwd[::-1]
        seed_bwd = seed_fwd[::-1]
        wz_bwd = wz_fwd[::-1]
        s_bwd = s_fwd[::-1]
        goals_all = np.concatenate([goal_fwd, goal_bwd[1:]], axis=0)
        seeds_all = np.concatenate([seed_fwd, seed_bwd[1:]], axis=0)
        wz_all = np.concatenate([wz_fwd, wz_bwd[1:]])
        # Arc length must stay monotone within each leg for interpolation-free
        # bookkeeping; store the A->B arc positions, then B->A positions mapped
        # back to forward arc coordinates (length - s).
        s_all = np.concatenate([s_fwd, length - s_bwd[1:]])
    else:
        goals_all, seeds_all, wz_all, s_all = goal_fwd, seed_fwd, wz_fwd, s_fwd

    delta = np.diff(seeds_all, axis=0)
    action = np.zeros_like(seeds_all)
    action[:-1] = delta / ACTION_SCALE
    action = np.clip(action, -1.0, 1.0)

    return ArmTrajectory(
        name=name,
        source=source,
        dt=control_dt,
        cruise_speed_mps=speed,
        goals=goals_all.astype(np.float32),
        seeds=seeds_all.astype(np.float32),
        world_z=wz_all.astype(np.float32),
        path_s=s_all.astype(np.float32),
        action=action.astype(np.float32),
    )


def compact_set(
    *,
    slow_factor: float = 0.5,
) -> list[ArmTrajectory]:
    """Build the Stage 10 compact set, slow + medium speed for each shape."""
    out: list[ArmTrajectory] = []
    for name, source in COMPACT_SET:
        reach_and_return = name == "reach_and_return"
        out.append(
            build_trajectory(
                name, source, speed_factor=1.0, reach_and_return=reach_and_return
            )
        )
        out.append(
            build_trajectory(
                f"{name}_slow",
                source,
                speed_factor=slow_factor,
                reach_and_return=reach_and_return,
            )
        )
    for name in EXTRAS:
        out.append(build_trajectory(name, name, speed_factor=1.0))
    return out


def coverage_stats(traj: ArmTrajectory) -> dict:
    """Coverage tags required by the Stage 10 spec (item 26)."""
    dt = traj.dt
    delta = np.diff(traj.seeds, axis=0)                    # per-step learned delta
    speed = np.abs(delta) / dt                             # m/s per cylinder
    n_cyl = traj.seeds.shape[1]
    eps = 1.0e-6

    signs = np.sign(delta)
    reversals = [
        int(np.count_nonzero(np.diff(signs[:, c] != 0)))
        for c in range(n_cyl)
    ]
    holds = [int(np.count_nonzero(np.abs(delta[:, c]) <= eps)) for c in range(n_cyl)]
    both_signs = [
        bool(np.any(delta[:, c] > eps) and np.any(delta[:, c] < -eps))
        for c in range(n_cyl)
    ]
    slow_steps = [int(np.count_nonzero((speed[:, c] > 1.0e-4) & (speed[:, c] < 0.02))) for c in range(n_cyl)]
    med_steps = [int(np.count_nonzero((speed[:, c] >= 0.02) & (speed[:, c] < 0.06))) for c in range(n_cyl)]

    ext_min = traj.seeds.min(axis=0)
    ext_max = traj.seeds.max(axis=0)
    span = ext_max - ext_min
    dominant = int(np.argmax(span)) if span.max() > 0 else -1
    total_var = np.sum(delta**2, axis=0)
    dominant_frac = (
        float(total_var[dominant] / total_var.sum()) if total_var.sum() > 0 else 0.0
    )
    coupled = bool((span > 1.0e-4).sum() >= 2)

    return {
        "extension_min_m": [float(v) for v in ext_min],
        "extension_max_m": [float(v) for v in ext_max],
        "extension_span_m": [float(v) for v in span],
        "dominant_cylinder": dominant,
        "dominant_span_frac": round(dominant_frac, 4),
        "single_cylinder_dominant": bool(dominant_frac >= 0.8),
        "coupled_multi_cylinder": coupled,
        "velocity_both_signs": both_signs,
        "direction_reversals": reversals,
        "hold_steps": holds,
        "slow_speed_steps": slow_steps,
        "medium_speed_steps": med_steps,
        "world_z_min_m": float(traj.world_z.min()),
        "world_z_max_m": float(traj.world_z.max()),
    }


def annotate_coverage(traj: ArmTrajectory) -> ArmTrajectory:
    traj.coverage = coverage_stats(traj)
    return traj
