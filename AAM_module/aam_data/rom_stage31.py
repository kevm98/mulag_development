"""Stage 31 data contract for the reduced-order ROM deployment plant.

The ROM is a command-interface plant: the classical collection controller
supplies a desired cylinder motion and the command ROM emits the valve command
that is applied for the next ROM interval.  The learning data therefore keeps
both sides of that transition explicitly:

* SD input action: ``[previous_valve, current_valve]`` (8 values).
* AAM observable row: ``[joint_q, joint_qdot, dP, previous_valve]`` (18
  values).  The previous valve is the last command already available before
  the current ID command is produced, so the row is causal at deployment.

Desired cylinder references are retained in the raw trajectory for the ID
transfer loss, but are deliberately excluded from the AAM input.
"""

from __future__ import annotations

import numpy as np


CONTROL_DT = 1.0 / 60.0
ROM_DT = 0.05
STEPS_PER_ROM_TICK = 3
HYD_STATE_DIM = 12
SD_ACTION_DIM = 8
AAM_FEATURE_DIM = 18
FACTOR_KEYS = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
AAM_HISTORY_LAGS = (5, 10, 20, 40)

AAM_FEATURE_GROUPS = {
    "joint_position": (0, 5),
    "joint_velocity": (5, 10),
    "cylinder_pressure": (10, 14),
    "previous_valve": (14, 18),
}


def rom_ticks(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return exact raw control-step indices at which the ROM updates."""
    marker = np.asarray(arrays["rom_tick"], dtype=np.float64)
    if marker.ndim != 2 or marker.shape[1] != 1:
        raise ValueError(f"rom_tick must have shape (N, 1), got {marker.shape}")
    ticks = np.flatnonzero(marker[:, 0] > 0.5).astype(np.int64)
    if ticks.size == 0 or ticks[0] != 0:
        raise ValueError("ROM tick grid must start at control step zero")
    if ticks.size > 1 and not np.all(np.diff(ticks) == STEPS_PER_ROM_TICK):
        raise ValueError("ROM tick grid is not uniformly spaced at 3 control steps")
    if int(ticks[-1]) + STEPS_PER_ROM_TICK >= arrays["cylinder_extension"].shape[0]:
        raise ValueError("last ROM tick has no complete next-state transition")
    return ticks


def previous_valve(arrays: dict[str, np.ndarray], tick: int) -> np.ndarray:
    """Valve command already available when the command at ``tick`` is made."""
    if tick == 0:
        return np.zeros(4, dtype=np.float64)
    return np.asarray(
        arrays["rom_predicted_valve_cmd"][tick - STEPS_PER_ROM_TICK],
        dtype=np.float64,
    )


def state_row(arrays: dict[str, np.ndarray], row: int) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(arrays["cylinder_extension"][row], dtype=np.float64),
            np.asarray(arrays["cylinder_velocity"][row], dtype=np.float64),
            np.asarray(arrays["dP"][row], dtype=np.float64),
        )
    )


def transition(arrays: dict[str, np.ndarray], tick: int) -> dict[str, np.ndarray | int]:
    """Build one exact 50 ms ROM transition from raw rows."""
    t = int(tick)
    end = t + STEPS_PER_ROM_TICK
    prev = previous_valve(arrays, t)
    current = np.asarray(arrays["rom_predicted_valve_cmd"][t], dtype=np.float64)
    start = state_row(arrays, t)
    target = state_row(arrays, end) - start
    q = np.asarray(arrays["joint_position"][t], dtype=np.float64)
    qdot = np.asarray(arrays["joint_velocity"][t], dtype=np.float64)
    u = np.concatenate((prev, current))
    return {
        "tick_control_step": t,
        "state_start": start,
        "target": target,
        "q": np.concatenate((q, qdot)),
        "u": u,
        "aam_feature": np.concatenate((q, qdot, start[8:12], prev)),
        "previous_valve": prev,
        "current_valve": current,
    }


def build_transitions(arrays: dict[str, np.ndarray]) -> list[dict]:
    return [transition(arrays, int(tick)) for tick in rom_ticks(arrays)]


def causal_stack(rows: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Most-recent-first causal windows with no padding or cross-run rows."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"rows must be rank 2, got {rows.shape}")
    if lag < 1:
        raise ValueError("lag must be positive")
    valid = np.arange(rows.shape[0]) >= lag - 1
    ends = np.flatnonzero(valid)
    windows = np.stack(
        [np.concatenate([rows[end - k] for k in range(lag)]) for end in ends],
        axis=0,
    ) if ends.size else np.zeros((0, lag * rows.shape[1]), dtype=np.float64)
    return windows, valid


def factor_vector(metadata: dict) -> np.ndarray:
    return np.asarray([float(metadata[key]) for key in FACTOR_KEYS], dtype=np.float64)

