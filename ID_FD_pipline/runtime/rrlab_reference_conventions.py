from __future__ import annotations

from dataclasses import dataclass


OBSTACLE_ARM_JOINT_NAMES: tuple[str, ...] = (
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
    "Messerkopf_joint",
)

OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES: tuple[str, ...] = OBSTACLE_ARM_JOINT_NAMES[:4]

OBSTACLE_WHEEL_JOINT_NAMES: tuple[str, ...] = (
    "Wheel_Front_Left_joint",
    "Wheel_Front_Right_joint",
    "Wheel_Rear_Left_joint",
    "Wheel_Rear_Right_joint",
)

OBSTACLE_TASK_SPACE_BODY_NAME = "Messerkopf"
OBSTACLE_TASK_SPACE_COMMAND_NAME = "ee_pose"
OBSTACLE_TASK_SPACE_POS_X_RANGE_M = (6.5, 7.5)
OBSTACLE_TASK_SPACE_POS_Y_RANGE_M = (0.7, 2.2)
OBSTACLE_TASK_SPACE_POS_Z_RANGE_M = (0.5, 0.5)
OBSTACLE_TASK_SPACE_ROLL_RANGE_RAD = (0.0, 0.0)
OBSTACLE_TASK_SPACE_PITCH_RANGE_RAD = (3.141592653589793, 3.141592653589793)
OBSTACLE_TASK_SPACE_YAW_RANGE_RAD = (3.141592653589793, 3.141592653589793)
OBSTACLE_TASK_SPACE_RESAMPLING_TIME_RANGE_S = (10.0, 12.0)

OBSTACLE_WHEEL_RADIUS_M = 0.575
OBSTACLE_DRIVING_SPEED_RANGE_KMH = (3.0, 8.0)
OBSTACLE_DRIVING_SPEED_RESAMPLE_STEP_THRESHOLD = 10
OBSTACLE_DRIVING_SPEED_UPDATE_INTERVAL_S = 0.1


@dataclass(frozen=True)
class RRLABObstacleVehicleCommandCfg:
    wheel_joint_names: tuple[str, ...] = OBSTACLE_WHEEL_JOINT_NAMES
    speed_range_kmh: tuple[float, float] = OBSTACLE_DRIVING_SPEED_RANGE_KMH
    wheel_radius_m: float = OBSTACLE_WHEEL_RADIUS_M
    update_interval_s: float = OBSTACLE_DRIVING_SPEED_UPDATE_INTERVAL_S
    resample_step_threshold: int = OBSTACLE_DRIVING_SPEED_RESAMPLE_STEP_THRESHOLD


def kmh_to_wheel_rad_per_sec(speed_kmh: float, wheel_radius_m: float = OBSTACLE_WHEEL_RADIUS_M) -> float:
    if wheel_radius_m <= 0.0:
        raise ValueError(f"wheel_radius_m must be positive, got {wheel_radius_m!r}")
    return (float(speed_kmh) / 3.6) / float(wheel_radius_m)


def wheel_rad_per_sec_to_kmh(omega_rad_s: float, wheel_radius_m: float = OBSTACLE_WHEEL_RADIUS_M) -> float:
    if wheel_radius_m <= 0.0:
        raise ValueError(f"wheel_radius_m must be positive, got {wheel_radius_m!r}")
    return float(omega_rad_s) * float(wheel_radius_m) * 3.6


def obstacle_speed_range_rad_per_sec(
    speed_range_kmh: tuple[float, float] = OBSTACLE_DRIVING_SPEED_RANGE_KMH,
    wheel_radius_m: float = OBSTACLE_WHEEL_RADIUS_M,
) -> tuple[float, float]:
    lower, upper = speed_range_kmh
    return (
        kmh_to_wheel_rad_per_sec(lower, wheel_radius_m),
        kmh_to_wheel_rad_per_sec(upper, wheel_radius_m),
    )
