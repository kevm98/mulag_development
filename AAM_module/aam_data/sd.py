"""SD dataset contract (AAM Stage 18, items 47-49).

The new SD is trained entirely from the new Isaac Lab ROM transitions (the
aligned 50 ms samples of Stage 17). Primary hydraulic state and target:

    s_t^H = [x_t, xdot_t, dP_t]   (12-D: Extension 4, Velocity 4, dP 4)
    target = Delta s_{t+1}^H = s_{t+1}^H - s_t^H   (12-D)

Mechanical context (item 48): joint position/velocity enter as CONTEXT ONLY;
joint states are never predicted:

    candidate input = [s_t^H; q_t, qdot_t; u_t; z_i]  (12 + 10 + 8 + 6)

with q_t/qdot_t the 5 arm joints at tick t, u_t the 8-D ROM command, and z_i
the fixed per-run factor vector.

History (item 49): the hydraulic plant has memory, so the dataset is built
with the instantaneous sample plus short causal state/action history
variants (default lags 3 and 6 ROM ticks), each variant stacking
20-D blocks ordered most-recent-first: ``[s_t^H; u_t]``, ``[s_{t-1}^H;
u_{t-1}]``, ..., ``[s_{t-L+1}^H; u_{t-L+1}]``. Samples whose history reaches
before tick 0 are dropped for that variant (counts reported). The final
minimum useful history is decided by measured delays, residual
autocorrelation, and validation rollout performance (Stage 19+).
"""

from __future__ import annotations

import numpy as np


HYD_STATE_DIM = 12          # Extension 4 + Velocity 4 + dP 4
MECH_CONTEXT_DIM = 10       # joint_position 5 + joint_velocity 5
CMD_DIM = 8                 # DesiredDeltaExtension 4 + DesiredDeltaExtensionVelocity 4
FACTOR_DIM = 6              # Stage 5 vector (flow, compliance, seal, pressure, mass_scale, com_z)
TARGET_DIM = 12             # Delta s_{t+1}^H
HISTORY_CHANNEL_DIM = HYD_STATE_DIM + CMD_DIM  # 20 per lag
DEFAULT_HISTORY_LAGS = (3, 6)

FACTOR_KEYS = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")


def hydraulic_state(sample: dict) -> np.ndarray:
    """s_t^H = first 12 channels of the tick's 16-D state_start row."""
    return np.asarray(sample["state_start"], dtype=np.float64)[:HYD_STATE_DIM]


def mechanical_context(arrays: dict[str, np.ndarray], tick: int) -> np.ndarray:
    """q_t, qdot_t at the ROM tick (exact raw rows; context only)."""
    return np.concatenate(
        (
            np.asarray(arrays["joint_position"][tick], dtype=np.float64),
            np.asarray(arrays["joint_velocity"][tick], dtype=np.float64),
        )
    )


def factor_vector(metadata: dict) -> np.ndarray:
    """Fixed per-run privileged factor vector z_i (Stage 5 order)."""
    return np.asarray([float(metadata[k]) for k in FACTOR_KEYS], dtype=np.float64)


def sd_sample(
    arrays: dict[str, np.ndarray],
    tick: int,
    z: np.ndarray,
    *,
    transition: dict | None = None,
) -> dict:
    """One SD sample: input context, hydraulic target, and flags.

    ``transition`` is the Stage 17 aligned transition for this tick (rebuilt
    here if not given). ``target`` is the measured Delta s_{t+1}^H; q/qdot
    appear only in the input context.
    """
    if transition is None:
        from .timebase import transition_sample

        transition = transition_sample(arrays, tick)
    s_h = hydraulic_state(transition)
    q = mechanical_context(arrays, tick)
    u = np.asarray(transition["command"], dtype=np.float64)
    return {
        "tick_control_step": int(transition["tick_control_step"]),
        "s_h": s_h,
        "target": np.asarray(transition["target"], dtype=np.float64).copy(),
        "q": q,
        "u": u,
        "z": np.asarray(z, dtype=np.float64).copy(),
        "input_instantaneous": np.concatenate((s_h, q, u, z)),
        "history_row": np.concatenate((s_h, u)),  # 20-D per lag
        "flags": transition["flags"],
    }


def history_stack(
    samples: list[dict], lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack causal 20-D blocks, most-recent-first, ``lag`` blocks per row.

    Returns ``(stack, valid_mask)``; samples whose history reaches before
    tick 0 are dropped (mask False) so no padding is invented.
    """
    rows = np.stack([s["history_row"] for s in samples])
    # Causal history needs lag-1 previous SAMPLES; the first lag-1 positions
    # of each run cannot be represented without padding and are dropped.
    valid = np.arange(len(samples)) >= lag - 1
    n = int(valid.sum())
    stack = np.zeros((n, lag * HISTORY_CHANNEL_DIM), dtype=np.float64)
    idx = np.flatnonzero(valid)
    for k in range(lag):
        stack[:, k * HISTORY_CHANNEL_DIM : (k + 1) * HISTORY_CHANNEL_DIM] = rows[
            idx - k
        ]
    return stack, valid


def build_sd_samples(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    *,
    history_lags: tuple[int, ...] = DEFAULT_HISTORY_LAGS,
) -> dict:
    """All SD samples for one run (instantaneous + history variants)."""
    from .timebase import build_transitions

    built = build_transitions(arrays)
    z = factor_vector(metadata)
    samples = [
        sd_sample(arrays, int(t["tick_control_step"]), z) for t in built["samples"]
    ]
    histories: dict[int, dict] = {}
    for lag in history_lags:
        stack, valid = history_stack(samples, lag)
        histories[lag] = {
            "stack": stack,
            "valid_mask": valid,
            "n_valid": int(valid.sum()),
            "n_dropped": int((~valid).sum()),
        }
    return {
        "grid": built["grid"],
        "samples": samples,
        "histories": histories,
        "n_samples": len(samples),
    }
