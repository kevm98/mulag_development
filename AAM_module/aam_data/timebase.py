"""Learning timebase for the frozen master dataset (AAM Stage 17, item 46).

Raw data is logged at the native Isaac Lab control frequency. The learning
timebase is the ROM's native update interval, and learning samples are built
as exact aligned transitions on that grid -- no interpolation anywhere:

    physics_dt = 1/120 s      PhysX physics substep
    control_dt = 1/60 s       native control step (recording frequency)
    rom_dt     = 0.05 s       command-ROM update interval = 3 control steps

The command ROM recomputes every 3rd control step (``counter == 0`` before the
step; recorded as ``rom_tick == 1`` after it). Its 50 ms output is delivered
to PhysX as three equal increments over control steps t, t+1, t+2, so the
plant transition of one ROM interval is exactly

    state row t  --command row t-->  state row t+3

with ``t`` on the ROM tick grid (0, 3, 6, ...). Every sample is built from
exact raw rows; nothing is interpolated or re-sampled.

The 24-D ROM input row for tick ``t`` is reconstructed from recorded signals
(same layout as ``rom_schema``: Extension, ExtensionVelocity, dP,
PreviousValveCmd, DesiredDeltaExtension, DesiredDeltaExtensionVelocity):

    extension  = state[t].cylinder_extension
    velocity   = state[t].cylinder_velocity
    dP         = state[t].dP
    prev_valve = rom_predicted_valve_cmd[t-3]   (0 at the first tick; the ROM
                 starts with a zero previous valve on reset and holds the
                 valve between ticks)
    command    = cat(desired_delta_extension[t], desired_delta_extension_velocity[t])
"""

from __future__ import annotations

import numpy as np


PHYSICS_DT = 1.0 / 120.0
CONTROL_DT = 1.0 / 60.0
ROM_DT = 0.05
STEPS_PER_ROM_TICK = int(round(ROM_DT / CONTROL_DT))  # 3

STATE_16_NAMES = (
    "extension_4",
    "velocity_4",
    "dP_4",
    "prev_valve_4",
)
COMMAND_8_NAMES = (
    "desired_delta_extension_4",
    "desired_delta_extension_velocity_4",
)


def rom_tick_indices(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Control-step rows where the command ROM recomputed (``rom_tick``)."""
    rt = np.asarray(arrays["rom_tick"], dtype=np.float64)
    return np.flatnonzero(rt[:, 0] > 0.5).astype(np.int64)


def check_rom_grid(
    arrays: dict[str, np.ndarray],
    *,
    control_dt: float = CONTROL_DT,
    rom_dt: float = ROM_DT,
) -> dict:
    """Verify the recorded ROM tick grid is exactly one tick every 3 steps.

    Returns a report dict; ``issues`` is empty when the grid is clean.
    """
    ticks = rom_tick_indices(arrays)
    n_steps = int(arrays["rom_tick"].shape[0])
    expected = STEPS_PER_ROM_TICK
    issues: list[str] = []
    if abs(float(control_dt) - CONTROL_DT) > 1e-12:
        issues.append(f"control_dt {control_dt} != {CONTROL_DT}")
    if abs(float(rom_dt) - ROM_DT) > 1e-12:
        issues.append(f"rom_dt {rom_dt} != {ROM_DT}")
    if float(rom_dt) / float(control_dt) != expected:
        issues.append(f"rom_dt/control_dt = {rom_dt / control_dt} != {expected}")
    if ticks.size:
        if int(ticks[0]) != 0:
            issues.append(f"first tick at {int(ticks[0])}, expected 0")
        gaps = np.diff(ticks)
        if not bool((gaps == expected).all()):
            bad = int(np.flatnonzero(gaps != expected)[0])
            issues.append(f"non-uniform tick gap at tick {bad}: {int(gaps[bad])} != {expected}")
        last_end = int(ticks[-1]) + expected
        if last_end > n_steps:
            issues.append(f"last tick {int(ticks[-1])} + {expected} > {n_steps} rows")
    else:
        issues.append("no ROM ticks recorded")
    return {
        "n_ticks": int(ticks.size),
        "first_tick": int(ticks[0]) if ticks.size else None,
        "last_tick": int(ticks[-1]) if ticks.size else None,
        "gap": expected,
        "uniform": bool(ticks.size) and bool((np.diff(ticks) == expected).all())
        and int(ticks[0]) == 0
        and int(ticks[-1]) + expected <= n_steps,
        "n_control_steps": n_steps,
        "issues": issues,
    }


def _prev_valve_at(arrays: dict[str, np.ndarray], tick_idx: int) -> np.ndarray:
    """Exact previous-valve row entering the ROM at this tick (no interpolation)."""
    if tick_idx == 0:
        return np.zeros(4, dtype=np.float64)
    prev = tick_idx - STEPS_PER_ROM_TICK
    return np.asarray(arrays["rom_predicted_valve_cmd"][prev], dtype=np.float64)


def transition_sample(arrays: dict[str, np.ndarray], tick_idx: int) -> dict:
    """One aligned 50 ms transition sample from exact raw rows.

    ``state_start``/``state_end`` are the 16-D states entering the ROM at this
    tick and at the next tick (t+3); ``target`` is the measured 12-D delta
    over the interval (extension, velocity, dP). ``rom_delta_extension_sum``
    is the sum of the ROM's three per-step deltas across the interval.
    """
    t = int(tick_idx)
    end = t + STEPS_PER_ROM_TICK
    s_ext = np.asarray(arrays["cylinder_extension"], dtype=np.float64)
    s_vel = np.asarray(arrays["cylinder_velocity"], dtype=np.float64)
    s_dp = np.asarray(arrays["dP"], dtype=np.float64)
    prev = _prev_valve_at(arrays, t)
    valve_now = np.asarray(arrays["rom_predicted_valve_cmd"][t], dtype=np.float64)

    state_start = np.concatenate((s_ext[t], s_vel[t], s_dp[t], prev))
    state_end = np.concatenate((s_ext[end], s_vel[end], s_dp[end], valve_now))
    target = np.concatenate(
        (
            s_ext[end] - s_ext[t],
            s_vel[end] - s_vel[t],
            s_dp[end] - s_dp[t],
        )
    )
    command = np.concatenate(
        (
            np.asarray(arrays["desired_delta_extension"][t], dtype=np.float64),
            np.asarray(arrays["desired_delta_extension_velocity"][t], dtype=np.float64),
        )
    )
    rom_delta = np.stack(
        [
            np.asarray(arrays["delta_extension_per_step"][t + k], dtype=np.float64)
            for k in range(STEPS_PER_ROM_TICK)
        ],
        axis=0,
    )
    flags = {
        name: np.asarray(arrays[name][t], dtype=np.float64)
        for name in (
            "cylinder_limit_flag",
            "command_saturation_flag",
            "valve_saturation_flag",
            "invalid_physics_flag",
        )
    }
    return {
        "tick_control_step": t,
        "state_start_idx": t,
        "state_end_idx": end,
        "command_idx": t,
        "input": np.concatenate((state_start, command)),
        "state_start": state_start,
        "state_end": state_end,
        "target": target,
        "command": command,
        "rom_delta_extension_sum": rom_delta.sum(axis=0),
        "flags": flags,
    }


def build_transitions(
    arrays: dict[str, np.ndarray],
    *,
    control_dt: float = CONTROL_DT,
    rom_dt: float = ROM_DT,
) -> dict:
    """All aligned 50 ms transitions for one run, grid-checked."""
    grid = check_rom_grid(arrays, control_dt=control_dt, rom_dt=rom_dt)
    if grid["issues"]:
        raise ValueError(f"ROM grid check failed: {grid['issues']}")
    samples = [transition_sample(arrays, int(t)) for t in rom_tick_indices(arrays)]
    return {"grid": grid, "samples": samples}
