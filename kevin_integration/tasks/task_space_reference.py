from __future__ import annotations


TASK_SPACE_POSE_ORDER = ("x", "y", "z", "qw", "qx", "qy", "qz")
TASK_SPACE_QUATERNION_ORDER = ("qw", "qx", "qy", "qz")
TASK_SPACE_FRAME = "robot_base"
TASK_SPACE_EE_BODY = "Messerkopf"
TASK_SPACE_COMMAND_NAME = "ee_pose"
TASK_SPACE_COMMAND_DIM = 7
TASK_SPACE_OBSERVATION_DIM = 73

TASK_SPACE_COMMAND_POS_X_RANGE_M = (6.5, 7.5)
TASK_SPACE_COMMAND_POS_Y_RANGE_M = (0.7, 2.2)
TASK_SPACE_COMMAND_POS_Z_RANGE_M = (0.5, 0.5)
TASK_SPACE_COMMAND_RESAMPLING_TIME_RANGE_S = (10.0, 12.0)
TASK_SPACE_COMMAND_QUAT_WXYZ = (0.0, 0.0, 1.0, 0.0)

READY_FOR_PPO = "READY_FOR_PPO"
BLOCKED_EE_BODY_RESOLUTION = "BLOCKED_EE_BODY_RESOLUTION"
BLOCKED_TASKSPACE_COMMAND = "BLOCKED_TASKSPACE_COMMAND"
BLOCKED_TASKSPACE_OBSERVATION = "BLOCKED_TASKSPACE_OBSERVATION"
BLOCKED_TASKSPACE_REWARD = "BLOCKED_TASKSPACE_REWARD"
BLOCKED_VEHICLE_COMMAND = "BLOCKED_VEHICLE_COMMAND"
BLOCKED_FD_INPUT_CONTRACT = "BLOCKED_FD_INPUT_CONTRACT"
BLOCKED_FD_OUTPUT_NONFINITE = "BLOCKED_FD_OUTPUT_NONFINITE"
BLOCKED_FD_ACTUATOR_CONFLICT = "BLOCKED_FD_ACTUATOR_CONFLICT"

TASK_SPACE_OBSERVATION_ORDER = (
    ("ee_pose_command", 7),
    ("current_ee_pose", 7),
    ("ee_position_error", 3),
    ("ee_orientation_error", 1),
    ("mechanical_q", 5),
    ("mechanical_qdot", 5),
    ("cylinder_q", 4),
    ("cylinder_qdot", 4),
    ("previous_action", 4),
    ("previous_ValveCmd", 4),
    ("z_arm_hat", 24),
    ("fd_tau_hat", 4),
    ("vehicle_forward_speed_kmh", 1),
)


def task_space_observation_dim() -> int:
    return sum(dim for _, dim in TASK_SPACE_OBSERVATION_ORDER)


def validate_task_space_observation_contract() -> None:
    dim = task_space_observation_dim()
    if dim != TASK_SPACE_OBSERVATION_DIM:
        raise RuntimeError(f"Task-space observation contract mismatch: {dim} != {TASK_SPACE_OBSERVATION_DIM}")


def task_space_workspace_bounds() -> dict[str, tuple[float, float]]:
    return {
        "x": TASK_SPACE_COMMAND_POS_X_RANGE_M,
        "y": TASK_SPACE_COMMAND_POS_Y_RANGE_M,
        "z": TASK_SPACE_COMMAND_POS_Z_RANGE_M,
    }


def position_inside_workspace(position_m: tuple[float, float, float]) -> bool:
    bounds = task_space_workspace_bounds()
    return all(bounds[axis][0] <= value <= bounds[axis][1] for axis, value in zip(("x", "y", "z"), position_m))
