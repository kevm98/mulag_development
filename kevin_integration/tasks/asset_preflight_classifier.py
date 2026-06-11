from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


BLOCKED_NONFINITE = "BLOCKED_NONFINITE"
BLOCKED_RESET_DURING_PREFLIGHT = "BLOCKED_RESET_DURING_PREFLIGHT"
BLOCKED_BASE_STABILITY = "BLOCKED_BASE_STABILITY"
BLOCKED_TARGET_WRITER = "BLOCKED_TARGET_WRITER"
BLOCKED_DRIVE_AUTHORITY = "BLOCKED_DRIVE_AUTHORITY"
BLOCKED_LINKAGE_TRANSMISSION = "BLOCKED_LINKAGE_TRANSMISSION"
ASSET_TRANSMISSION_PASS = "ASSET_TRANSMISSION_PASS"


@dataclass(frozen=True)
class AssetPreflightGates:
    min_target_motion_m: float = 1.0e-4
    min_cylinder_motion_m: float = 0.002
    min_mechanical_motion: float = 1.0e-4
    max_pulse_root_z_motion_m: float = 0.02
    max_pulse_roll_pitch_motion_rad: float = 0.03490658503988659
    max_pre_pulse_root_vertical_velocity_mps: float = 0.05


def classify_asset_transmission_preflight(
    *,
    finite: bool,
    reset_during_preflight: bool,
    pulse_root_z_motion_m: float,
    pulse_root_roll_pitch_motion_rad: float,
    root_vertical_velocity_before_pulse_mps: float,
    target_motion_m: list[float] | tuple[float, ...],
    cylinder_motion_m: list[float] | tuple[float, ...],
    mechanical_family_motion: dict[str, float],
    gates: AssetPreflightGates = AssetPreflightGates(),
) -> str:
    """Classify asset preflight results in diagnostic priority order."""
    scalar_values = (
        pulse_root_z_motion_m,
        pulse_root_roll_pitch_motion_rad,
        root_vertical_velocity_before_pulse_mps,
        *target_motion_m,
        *cylinder_motion_m,
        *mechanical_family_motion.values(),
    )
    if not finite or not all(isfinite(float(value)) for value in scalar_values):
        return BLOCKED_NONFINITE
    if reset_during_preflight:
        return BLOCKED_RESET_DURING_PREFLIGHT
    if (
        abs(float(pulse_root_z_motion_m)) > float(gates.max_pulse_root_z_motion_m)
        or abs(float(pulse_root_roll_pitch_motion_rad)) > float(gates.max_pulse_roll_pitch_motion_rad)
        or abs(float(root_vertical_velocity_before_pulse_mps))
        > float(gates.max_pre_pulse_root_vertical_velocity_mps)
    ):
        return BLOCKED_BASE_STABILITY
    if min(abs(float(value)) for value in target_motion_m) < float(gates.min_target_motion_m):
        return BLOCKED_TARGET_WRITER
    if min(abs(float(value)) for value in cylinder_motion_m) < float(gates.min_cylinder_motion_m):
        return BLOCKED_DRIVE_AUTHORITY
    if any(abs(float(value)) < float(gates.min_mechanical_motion) for value in mechanical_family_motion.values()):
        return BLOCKED_LINKAGE_TRANSMISSION
    return ASSET_TRANSMISSION_PASS
