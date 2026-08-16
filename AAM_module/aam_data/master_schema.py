"""Master raw Isaac Lab dataset contract (AAM Stage 8).

The fundamental unit is one deployment trajectory: for domain run i the
dataset stores (tau_i, e_i) where e_i is the fixed physical factor vector and
tau_i = {o_0, u_0, o_1, u_1, ..., o_T} is the per-control-step recording of the
full command chain and plant response. Everything else (SD transitions, AAM
history windows) is derived from this master set.

Layout per run:

    <out>/runs/<run_id:05d>/
        metadata.json      run-level metadata (see REQUIRED_METADATA)
        trajectory.npz     state (T+1), command (T), flag (T) arrays

Availability: signals listed in UNAVAILABLE_SIGNALS are stored as NaN in the
deployed command-ROM path because the plant does not expose them (no physical
valve, no contact sensor, DLS not used by env-driven collection). The metadata
records the availability map so consumers can distinguish "NaN = absent" from
a genuine non-finite reading.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "master_v1"

# --- Factor layout (Stage 5) -------------------------------------------------
DIM_ORDER = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
RANGES = {
    # Superset of the original Stage-5 box and the Stage-22-wide collection
    # profile. Hydraulic bounds are the clamps audited in HydraulicROM.forward;
    # validation must reject values that would be silently clipped by the ROM.
    "flow": (-0.25, 0.25),
    "compliance": (-0.25, 0.10),
    "seal": (0.0, 1.0),
    "pressure": (-0.15, 0.15),
    "mass_scale": (0.8, 1.5),
    "com_z": (-0.10, 0.05),
}
HYDRAULIC_DIMS = ("flow", "compliance", "seal", "pressure")
MECHANICAL_DIMS = ("mass_scale", "com_z")

# --- Signal layout -----------------------------------------------------------
# Observed state per control step, shape (T+1, dim). "CylinderExtension" and
# "CylinderVelocity" are the learned/Simscape-space quantities the ROM consumes
# (adapter output); the raw USD joint values are recorded as joint_position.
STATE_SIGNALS = {
    "cylinder_extension": 4,
    "cylinder_velocity": 4,
    "dP": 4,
    "joint_position": 5,
    "joint_velocity": 5,
    "ee_pos_local": 3,
    "ee_pos_w": 3,
    "ee_quat_w": 4,
    "ee_lin_vel_w": 3,
    "ee_ang_vel_w": 3,
}

# Command chain per control step, shape (T, dim). The verified plant input is
# delta_extension_per_step (learned m) and delta_isaac (USD m after the
# Simscape -> Isaac mapping); cylinder_pos_target is the integrated absolute
# target written to PhysX.
COMMAND_SIGNALS = {
    "probe_excitation": 4,                  # injected action port command
    "controller_output": 4,                 # clamped PPO action (_current_action)
    "desired_cylinder_position": 4,         # _cylinder_ref (x_ref)
    "desired_cylinder_velocity": 4,         # _cylinder_vel_ref (xdot_ref)
    "desired_delta_extension": 4,           # _action_target_delta_x_learned
    "desired_delta_extension_velocity": 4,  # _action_target_delta_xdot_learned
    "command_before_limits": 4,             # _action_target_delta_x_pre_gain_learned
    "command_after_limits": 4,              # post safety clamp
    # Compatibility fields are retained for the shared schema. In the
    # command-ROM plant there is no external valve request or applied valve:
    # requested_valve_cmd is only an alias of the ROM's internal prediction.
    "requested_valve_cmd": 4,               # compatibility alias, not an input
    "applied_valve_cmd": 4,                 # NaN: no physical valve in command-ROM path
    "rom_predicted_valve_cmd": 4,           # internal ROM output / ID target
    "rom_effective_actuation": 4,           # internal ROM actuation
    "delta_extension_per_step": 4,          # ROM delta_extension per control step
    "delta_isaac": 4,                       # _delta_isaac (mapped USD increment)
    "cylinder_pos_target": 4,               # absolute target written to PhysX
    "rom_tick": 1,                          # True on the step the ROM recomputed
}

# Quality / safety per control step, shape (T, dim).
FLAG_SIGNALS = {
    "cylinder_limit_flag": 4,        # cylinder joint at soft limit
    "command_saturation_flag": 4,    # any delta-x / delta-xdot clamp active
    "valve_saturation_flag": 4,      # ROM predicted valve at +-1 bound
    "invalid_physics_flag": 1,       # env-level non-finite ROM/obs reading
    "dls_condition_number": 1,       # NaN: DLS not used in env-driven collection
    "dls_damping": 1,                # NaN: same
    "contact_state": 1,              # NaN: no contact sensor in this env
    "contact_force": 3,              # NaN: same
}

UNAVAILABLE_SIGNALS = {
    "applied_valve_cmd",
    "dls_condition_number",
    "dls_damping",
    "contact_state",
    "contact_force",
}

# --- Metadata contract (#20) -------------------------------------------------
REQUIRED_METADATA = [
    "schema_version",
    "run_id",
    "domain",
    "split",
    "random_seed",
    "trajectory_schedule",       # {name, params}
    "segment_ids",               # [{name, start_step, end_step}]
    "physics_dt",
    "control_dt",
    "rom_dt",
    "controller_gains",          # dict of cfg values that shape the chain
    "dls_parameters",            # null when DLS is not used
    "excitation_config",         # full deterministic schedule definition
    "hydraulic_draw",            # e_H (4)
    "mechanical_draw",           # e_M (2)
    "applied_system_factors",    # 7-D ROM factor channel actually applied
    "applied_mass_scale",
    "applied_com_z",
    "start_time",
    "end_time",
    "task_name",
    "rom_checkpoint",
    "num_control_steps",
    "availability",              # per-signal bool map
]


def e_to_system_factors(e: dict[str, float]) -> list[float]:
    """Map the Stage 5 hydraulic draw onto the ROM 7-D factor channel.

    Channel order is [compliance, flow, pressure, seal, seal, seal, seal]
    (see Stage 5 application section and hydraulic_rom.py).
    """
    return [
        float(e["compliance"]),
        float(e["flow"]),
        float(e["pressure"]),
        float(e["seal"]),
        float(e["seal"]),
        float(e["seal"]),
        float(e["seal"]),
    ]


def e_to_mechanical(e: dict[str, float]) -> tuple[float, float]:
    """Return (mass_scale, com_z_shift_m) for the cutter body."""
    return float(e["mass_scale"]), float(e["com_z"])


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_run(run_dir: Path, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    """Persist one run as metadata.json + trajectory.npz."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "metadata.json", metadata)
    np.savez_compressed(run_dir / "trajectory.npz", **arrays)


def read_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    with np.load(run_dir / "trajectory.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return {"metadata": metadata, "arrays": arrays}


def validate_run(run_dir: Path, manifest_row: dict | None = None) -> list[str]:
    """Return contract violations; empty list means the run is valid."""
    run_dir = Path(run_dir)
    issues: list[str] = []
    try:
        data = read_run(run_dir)
    except FileNotFoundError as exc:
        return [f"missing file: {exc}"]
    metadata, arrays = data["metadata"], data["arrays"]

    for key in REQUIRED_METADATA:
        if key not in metadata:
            issues.append(f"metadata missing key: {key}")

    if metadata.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            f"schema_version {metadata.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )

    n_state = int(metadata.get("num_control_steps", -1)) + 1
    n_cmd = int(metadata.get("num_control_steps", -1))
    for name, dim in STATE_SIGNALS.items():
        arr = arrays.get(name)
        if arr is None:
            issues.append(f"missing state signal: {name}")
        elif arr.ndim != 2 or arr.shape != (n_state, dim):
            issues.append(f"state signal {name}: shape {arr.shape}, expected ({n_state}, {dim})")
        elif name not in UNAVAILABLE_SIGNALS and not np.isfinite(arr).all():
            issues.append(f"state signal {name} contains non-finite values")
    for name, dim in COMMAND_SIGNALS.items():
        arr = arrays.get(name)
        if arr is None:
            issues.append(f"missing command signal: {name}")
        elif arr.ndim != 2 or arr.shape != (n_cmd, dim):
            issues.append(f"command signal {name}: shape {arr.shape}, expected ({n_cmd}, {dim})")
        elif name not in UNAVAILABLE_SIGNALS and not np.isfinite(arr).all():
            issues.append(f"command signal {name} contains non-finite values")
    for name, dim in FLAG_SIGNALS.items():
        arr = arrays.get(name)
        if arr is None:
            issues.append(f"missing flag signal: {name}")
        elif arr.ndim != 2 or arr.shape != (n_cmd, dim):
            issues.append(f"flag signal {name}: shape {arr.shape}, expected ({n_cmd}, {dim})")

    availability = metadata.get("availability", {})
    for name in COMMAND_SIGNALS:
        if name in UNAVAILABLE_SIGNALS:
            arr = arrays.get(name)
            if arr is not None and not np.isnan(arr).all():
                issues.append(f"signal {name} marked unavailable but is not all-NaN")

    e = {k: metadata.get(k) for k in DIM_ORDER}
    if all(v is not None for v in e.values()):
        for name in DIM_ORDER:
            lo, hi = RANGES[name]
            if not (lo - 1e-9 <= float(e[name]) <= hi + 1e-9):
                issues.append(f"factor {name} outside audited collection range: {e[name]}")
    if manifest_row is not None:
        for name in DIM_ORDER:
            expected = float(manifest_row[name])
            got = float(e[name])
            if abs(expected - got) > 1e-6:
                issues.append(f"factor {name}: manifest {expected} != run {got}")

    sf = metadata.get("applied_system_factors")
    if sf is not None and e.get("compliance") is not None:
        expected_sf = e_to_system_factors(e)
        if not all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(sf, expected_sf)):
            issues.append("applied_system_factors does not match the hydraulic draw")

    t = arrays.get("time_s")
    if t is not None and t.size and not bool(np.all(np.diff(t) > 0)):
        issues.append("time_s is not strictly increasing")
    return issues
