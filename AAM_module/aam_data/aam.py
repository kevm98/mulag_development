"""Pressure-inclusive AAM dataset contract (Stage 30; supersedes Stage 24).

The online AAM estimates the privileged latent ``z_hat`` from a causal window
of OBSERVABLE state-action history only. The dataset stores, per ROM tick
(exact 50 ms grid, Stage 17), the deployment-available feature row and per-run
the stopped-gradient target ``z = mu(e)`` from the frozen Stage 23 encoder.

Per-step feature row (30-D), all exact raw rows, no interpolation:

    applied action      8   [desired_delta_extension(4);
                             desired_delta_extension_velocity(4)]
    cylinder extension  4   cylinder_extension[t]
    cylinder velocity   4   cylinder_velocity[t]
    cylinder pressure   4   dP[t] (deployment-observable hydraulic state)
    joint position      5   joint_position[t]
    joint velocity      5   joint_velocity[t]

Excluded by construction: physical factors ``e``, true latent ``z`` (target
only, never an input), ROM hidden state (prev valve, predicted valve,
effective actuation), future observations (windows are causal), and the
desired task reference (desired_cylinder_position/velocity, x_ref/xdot_ref).
Measured cylinder pressure ``dP`` is included because it is available in the
reduced-order hydraulic deployment state.

History windows are causal: for lag ``L`` the window ending at tick ``t`` is
``[feature(t), feature(t-1), ..., feature(t-L+1)]`` (most-recent-first, same
convention as the SD ``history_L*`` files). Windows whose history reaches
before tick 0 are dropped for that variant; nothing is padded.
"""

from __future__ import annotations

import numpy as np


APPLIED_ACTION_DIM = 8      # desired_delta_extension 4 + ..._velocity 4
CYLINDER_DIM = 8            # extension 4 + velocity 4
PRESSURE_DIM = 4            # cylinder pressure difference
JOINT_CONTEXT_DIM = 10      # joint_position 5 + joint_velocity 5
AAM_FEATURE_DIM = APPLIED_ACTION_DIM + CYLINDER_DIM + PRESSURE_DIM + JOINT_CONTEXT_DIM  # 30

HISTORY_LAG_CANDIDATES = (5, 10, 20, 40)  # Stage 25: 0.25/0.5/1.0/2.0 s

FEATURE_GROUPS = {
    "applied_action": (0, 8),
    "cylinder_extension": (8, 12),
    "cylinder_velocity": (12, 16),
    "cylinder_pressure": (16, 20),
    "joint_position": (20, 25),
    "joint_velocity": (25, 30),
}


def applied_action(arrays: dict[str, np.ndarray], tick: int) -> np.ndarray:
    """8-D action applied to the plant at this ROM tick (Stage 1 interface)."""
    return np.concatenate(
        (
            np.asarray(arrays["desired_delta_extension"][tick], dtype=np.float64),
            np.asarray(
                arrays["desired_delta_extension_velocity"][tick], dtype=np.float64
            ),
        )
    )


def aam_feature(arrays: dict[str, np.ndarray], tick: int) -> np.ndarray:
    """30-D observable feature row at control step ``tick`` (exact raw rows)."""
    return np.concatenate(
        (
            applied_action(arrays, tick),
            np.asarray(arrays["cylinder_extension"][tick], dtype=np.float64),
            np.asarray(arrays["cylinder_velocity"][tick], dtype=np.float64),
            np.asarray(arrays["dP"][tick], dtype=np.float64),
            np.asarray(arrays["joint_position"][tick], dtype=np.float64),
            np.asarray(arrays["joint_velocity"][tick], dtype=np.float64),
        )
    )


def build_aam_features(arrays: dict[str, np.ndarray]) -> dict:
    """All AAM feature rows on the ROM tick grid (grid-checked, no padding)."""
    from .timebase import check_rom_grid, rom_tick_indices

    grid = check_rom_grid(arrays)
    if grid["issues"]:
        raise ValueError(f"ROM grid check failed: {grid['issues']}")
    ticks = rom_tick_indices(arrays)
    features = np.stack([aam_feature(arrays, int(t)) for t in ticks], axis=0)
    flags = {
        name: np.stack(
            [np.asarray(arrays[name][t], dtype=np.float64) for t in ticks], axis=0
        )
        for name in (
            "cylinder_limit_flag",
            "command_saturation_flag",
            "valve_saturation_flag",
            "invalid_physics_flag",
        )
    }
    return {
        "grid": grid,
        "tick_control_step": ticks.astype(np.int64),
        "features": features,
        "flags": flags,
        "n_ticks": int(ticks.size),
    }


def aam_window_stack(
    features: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Causal ``lag``-tick windows of feature rows, most-recent-first.

    Returns ``(stack, valid_mask)`` with ``stack`` shape
    ``(n - lag + 1, lag * AAM_FEATURE_DIM)``; the first ``lag - 1`` ticks of
    each run cannot be represented without padding and are dropped
    (``valid_mask`` False), matching the SD history convention. The valid
    rows are indexed by their window end tick, not their start tick.
    """
    rows = np.asarray(features, dtype=np.float64)
    if lag < 1:
        raise ValueError(f"lag must be positive, got {lag}")
    if rows.ndim != 2 or rows.shape[1] != AAM_FEATURE_DIM:
        raise ValueError(
            f"features must have shape (n, {AAM_FEATURE_DIM}), got {rows.shape}"
        )
    n = rows.shape[0]
    valid = np.arange(n) >= lag - 1
    n_valid = int(valid.sum())
    stack = np.zeros((n_valid, lag * AAM_FEATURE_DIM), dtype=np.float64)
    idx = np.flatnonzero(valid)
    for k in range(lag):
        stack[:, k * AAM_FEATURE_DIM : (k + 1) * AAM_FEATURE_DIM] = rows[idx - k]
    return stack, valid
