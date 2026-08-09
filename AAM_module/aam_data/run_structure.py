"""Stage 13 per-run structure: task realism + coverage + identifiability.

Each run concatenates settle, DLS posture holds with identification
excitation, DLS workspace trajectories, a coupled trajectory, a
reach-and-return, and a safe stow. The plan is pure Python (no Isaac): DLS
segments carry the reference seed path they would track (the closed-loop DLS
produces the actual action at collection time), and identification segments
carry the bounded probe action schedule from Stage 12.

Postures A/B/C are taken from the DLS pilot's own teacher
(``mowing_arc_head_down.json``): a safe, reachable, verified configuration
triplet. DLS transitions are cosine-blended seed ramps (the same shape the
pilot uses to deploy without an impulse); DLS workspace segments reuse the
Stage 10 trajectory builder.

Segment durations: settle / posture-ramp / ID / stow are controlled by their
``*_s`` parameters; the DLS workspace, coupled, and reach-and-return segments
derive their length from the underlying trajectory shapes (arc length and
cruise speed), so with the defaults the run is 11304 steps / 188.4 s (workspace
1215 steps, coupled 1480 steps, reach-and-return 2429 steps).

The probe used for identification segments defaults to PRBS at the Stage 12
measured amplitude (0.015 per channel, stroke-headroom-limited by cyl0) with
the pilot-grounded frequency band; multisine is available as an alternative.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .arm_motion import CONTROL_DT, build_trajectory
from .excitation import PRBSSchedule


ROOT = Path(__file__).resolve().parents[2]
LEGACY_ARC_PATH = ROOT / "mulag_plant/models/mowing_arc_head_down.json"


@dataclass
class RunSegment:
    name: str
    kind: str
    start: int
    end: int  # exclusive
    reference_seeds: np.ndarray | None = None   # (T, 4) learned extension
    reference_goals: np.ndarray | None = None   # (T, 3) tool goals (DLS)
    probe_action: np.ndarray | None = None      # (T, 4) normalized probe (ID)
    params: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(self.end - self.start)

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, **self.params}


def teacher_postures() -> dict[str, np.ndarray]:
    """Three verified, reachable postures from the DLS pilot's teacher."""
    payload = json.loads(LEGACY_ARC_PATH.read_text())
    teacher = payload["cylinder_teacher"]
    theta = np.asarray([row["theta_deg"] for row in teacher], dtype=np.float64)
    order = np.argsort(theta)
    seeds = np.asarray([row["cylinder_cmd"] for row in teacher], dtype=np.float64)[order]
    n = len(seeds)
    return {
        "A": seeds[0].astype(np.float64),
        "B": seeds[n // 2].astype(np.float64),
        "C": seeds[-1].astype(np.float64),
    }


def _cosine_ramp(start: np.ndarray, target: np.ndarray, steps: int) -> np.ndarray:
    """Cosine-blended seed ramp (deployment shape used by the DLS pilot)."""
    u = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    blend = 0.5 * (1.0 - np.cos(math.pi * u))
    return start[None, :] + blend[:, None] * (target - start)[None, :]


def _posture_hold(seed: np.ndarray, steps: int) -> np.ndarray:
    return np.repeat(seed[None, :], steps, axis=0)


@dataclass
class RunPlan:
    segments: list[RunSegment]
    control_dt: float
    postures: dict[str, np.ndarray]
    probe_name: str
    probe_amplitude: float
    probe_seed: int
    probe_hold_steps: int
    trajectory_stride: int

    @property
    def total_steps(self) -> int:
        return self.segments[-1].end if self.segments else 0

    def segment_table(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "kind": s.kind,
                "start": s.start,
                "end": s.end,
                "steps": s.steps,
                "duration_s": round(s.steps * self.control_dt, 2),
                "params": s.as_dict(),
            }
            for s in self.segments
        ]

    def probe_schedule(self) -> np.ndarray:
        """Composite probe action: probe in ID segments, zeros elsewhere."""
        U = np.zeros((self.total_steps, 4), dtype=np.float64)
        for s in self.segments:
            if s.probe_action is not None:
                U[s.start : s.end] = s.probe_action
        return U


def build_run_plan(
    *,
    control_dt: float = CONTROL_DT,
    settle_s: float = 2.0,
    posture_ramp_s: float = 3.0,
    id_s: float = 30.0,
    stow_s: float = 4.0,
    probe_name: str = "prbs",
    probe_amplitude: float = 0.015,
    probe_seed: int = 20260807,
    probe_hold_steps: int = 10,
    trajectory_stride: int = 1,
    workspace_shape: str = "line",
    coupled_shape: str = "sine",
) -> RunPlan:
    """Build the 11-step Stage 13 plan."""
    if trajectory_stride < 1:
        raise ValueError("trajectory_stride must be >= 1")
    postures = teacher_postures()
    dt = float(control_dt)

    def steps(s: float) -> int:
        return max(int(round(s / dt)), 4)

    segments: list[RunSegment] = []
    cursor = 0

    def add(name: str, kind: str, reference, probe=None, params=None) -> None:
        nonlocal cursor
        n = reference.shape[0] if reference is not None else probe.shape[0]
        segments.append(
            RunSegment(
                name=name,
                kind=kind,
                start=cursor,
                end=cursor + n,
                reference_seeds=reference,
                probe_action=probe,
                params=params or {},
            )
        )
        cursor += n

    # 1. SETTLE at the start posture (A).
    add("settle", "settle", _posture_hold(postures["A"], steps(settle_s)))

    # 2. DLS -> posture A (already at A after settle; short confirm ramp).
    add("dls_to_A", "dls_posture", _posture_hold(postures["A"], steps(1.0)))

    # 3. Identification excitation at posture A.
    add(
        "id_A",
        "id_excitation",
        _posture_hold(postures["A"], steps(id_s)),
        probe=_probe(
            steps(id_s), probe_name, probe_amplitude,
            seed=probe_seed, hold_steps=probe_hold_steps,
        ),
        params={"posture": "A", "probe": probe_name, "amplitude": probe_amplitude,
                "seed": probe_seed, "hold_steps": probe_hold_steps},
    )

    # 4. DLS workspace trajectory (single-cylinder-dominant line A->B).
    traj = build_trajectory("workspace", workspace_shape, control_dt=dt)
    add(
        "dls_workspace",
        "dls_trajectory",
        _stride_path(traj.seeds.astype(np.float64), trajectory_stride),
        params={"shape": workspace_shape, "source": traj.source},
    )

    # 5. DLS -> posture B.
    add(
        "dls_to_B",
        "dls_posture",
        _cosine_ramp(traj.seeds[-1].astype(np.float64), postures["B"], steps(posture_ramp_s)),
        params={"posture": "B"},
    )

    # 6. Identification excitation at posture B.
    add(
        "id_B",
        "id_excitation",
        _posture_hold(postures["B"], steps(id_s)),
        probe=_probe(
            steps(id_s), probe_name, probe_amplitude,
            seed=probe_seed + 1, hold_steps=probe_hold_steps,
        ),
        params={"posture": "B", "probe": probe_name, "amplitude": probe_amplitude,
                "seed": probe_seed + 1, "hold_steps": probe_hold_steps},
    )

    # 7. Coordinated/coupled DLS trajectory.
    coupled = build_trajectory("coupled", coupled_shape, control_dt=dt)
    add(
        "dls_coupled",
        "dls_trajectory",
        _stride_path(coupled.seeds.astype(np.float64), trajectory_stride),
        params={"shape": coupled_shape, "source": coupled.source},
    )

    # 8. DLS -> posture C.
    add(
        "dls_to_C",
        "dls_posture",
        _cosine_ramp(coupled.seeds[-1].astype(np.float64), postures["C"], steps(posture_ramp_s)),
        params={"posture": "C"},
    )

    # 9. Identification excitation at posture C.
    add(
        "id_C",
        "id_excitation",
        _posture_hold(postures["C"], steps(id_s)),
        probe=_probe(
            steps(id_s), probe_name, probe_amplitude,
            seed=probe_seed + 2, hold_steps=probe_hold_steps,
        ),
        params={"posture": "C", "probe": probe_name, "amplitude": probe_amplitude,
                "seed": probe_seed + 2, "hold_steps": probe_hold_steps},
    )

    # 10. Reach-and-return / direction reversal (line A->B->A).
    rr = build_trajectory(
        "reach_and_return", "line", control_dt=dt, reach_and_return=True
    )
    add(
        "reach_and_return",
        "dls_trajectory",
        _stride_path(rr.seeds.astype(np.float64), trajectory_stride),
        params={"shape": "line", "reach_and_return": True},
    )

    # 11. Safe stow back to posture A.
    add(
        "safe_stow",
        "stow",
        _cosine_ramp(rr.seeds[-1].astype(np.float64), postures["A"], steps(stow_s)),
        params={"target": "A"},
    )

    return RunPlan(
        segments=segments,
        control_dt=dt,
        postures=postures,
        probe_name=probe_name,
        probe_amplitude=probe_amplitude,
        probe_seed=int(probe_seed),
        probe_hold_steps=int(probe_hold_steps),
        trajectory_stride=int(trajectory_stride),
    )


def _stride_path(path: np.ndarray, stride: int) -> np.ndarray:
    """Subsample a reference path for bounded test flights, retaining its end."""
    if stride <= 1:
        return path
    indices = list(range(0, len(path), stride))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)
    return path[np.asarray(indices, dtype=np.int64)]


def _probe(
    steps: int, name: str, amplitude: float, *, seed: int, hold_steps: int
) -> np.ndarray:
    if name == "prbs":
        schedule = PRBSSchedule(
            amplitude=amplitude,
            per_cylinder_steps=steps,
            seed=int(seed),
            hold_steps=int(hold_steps),
        )
    else:
        from .excitation import MultisineSchedule

        schedule = MultisineSchedule(
            per_cylinder_steps=steps,
            amplitudes=tuple([amplitude] * 4),
            frequencies_hz=(0.2, 0.8, 2.0),
        )
    action, _ = schedule.build(total_steps=steps)
    return action.astype(np.float64)
