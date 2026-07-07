"""High-level PPO observation builder.

Task-space only. Cylinder internal states are excluded from PPO observation.
The low-level controller owns cylinder/ID/FD logic.
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse


HI_HORIZON_OBS_INDEX = {
    "ee_pos_b": (0, 3),
    "target_pos_b": (3, 6),
    "ee_pos_error_b": (6, 9),
    "ee_error_norm": (9, 10),
    "joint_pos": (10, 15),
    "joint_vel": (15, 20),
    "prev_hl_action": (20, 23),
    "unimog_pose": (23, 27),
    "unimog_pose_dot": (27, 30),
}
HI_HORIZON_OBS_DIM = 30

# Legacy cylinder-space obs: kept for reference, not used by high-level PPO
HORIZON_OBS_INDEX = {
    "ee_pos_b": (0, 3), "ee_quat_w": (3, 7), "target_pos_b": (7, 10),
    "ee_pos_error_b": (10, 13), "joint_pos": (13, 18), "joint_vel": (18, 23),
    "cylinder_ext": (23, 27), "cylinder_vel": (27, 31),
    "prev_ppo_action": (31, 35), "prev_delta_x_des_H": (35, 39),
    "prev_delta_x_fd_H": (39, 43), "prev_delta_x_cmd_H": (43, 47),
    "prev_delta_x_meas_H": (47, 51), "prev_guard_flags": (51, 55),
    "unimog_pose": (55, 59), "unimog_pose_dot": (59, 62),
}
HORIZON_OBS_DIM = 62


def _normalized_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _tcp_pos_base_frame(uw) -> torch.Tensor:
    ee_pos_w = uw.robot.data.body_link_state_w[:, uw._ee_body_id, :3]
    base_pos_w = uw.robot.data.root_pos_w
    base_quat_w = _normalized_quat(uw.robot.data.root_quat_w)
    return quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)


def _unimog_pose(uw) -> tuple[torch.Tensor, torch.Tensor]:
    env_origins = uw.scene.env_origins
    local_pos = uw.robot.data.root_pos_w - env_origins
    root_q = _normalized_quat(uw.robot.data.root_quat_w)
    _, _, yaw = euler_xyz_from_quat(root_q)
    x_norm = torch.clamp(local_pos[:, 0] / 25.0, -1.5, 1.5)
    y_norm = torch.clamp((local_pos[:, 1] - 4.3) / 4.0, -1.5, 1.5)
    pose = torch.stack([x_norm, y_norm, torch.sin(yaw), torch.cos(yaw)], dim=-1)
    lin_vel = uw.robot.data.root_lin_vel_w
    ang_vel = uw.robot.data.root_ang_vel_w
    pose_dot = torch.stack([lin_vel[:, 0], lin_vel[:, 1], ang_vel[:, 2]], dim=-1)
    return pose, pose_dot


def build_high_level_observation(uw, prev_hl_action: torch.Tensor) -> torch.Tensor:
    """Build the 30D high-level task-space observation.

    No cylinder internal states. No ID/FD variables.
    PPO sees only task-space robot/EE/goal state.
    """
    ee_state = uw.robot.data.body_link_state_w[:, uw._ee_body_id, :7]
    ee_pos_b = _tcp_pos_base_frame(uw)
    if hasattr(uw, "_command_pos_w") and uw._command_pos_w is not None:
        target_pos_b = uw._command_pos_w[:, :3]
    else:
        target_pos_b = torch.zeros((uw.num_envs, 3), device=ee_state.device)
    ee_pos_error_b = target_pos_b - ee_pos_b
    ee_error_norm = torch.norm(ee_pos_error_b, dim=-1, keepdim=True)
    joint_pos = uw.robot.data.joint_pos[:, uw._arm_joint_ids]
    joint_vel = uw.robot.data.joint_vel[:, uw._arm_joint_ids]
    unimog_pose, unimog_pose_dot = _unimog_pose(uw)
    obs = torch.cat([
        ee_pos_b, target_pos_b, ee_pos_error_b, ee_error_norm,
        joint_pos, joint_vel, prev_hl_action, unimog_pose, unimog_pose_dot,
    ], dim=-1)
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if obs.shape[-1] != HI_HORIZON_OBS_DIM:
        raise RuntimeError(f"High-level obs dim mismatch: got {obs.shape[-1]}, expected {HI_HORIZON_OBS_DIM}")
    return obs


# Legacy function preserved for backward compatibility
def build_horizon_observation(uw, prev_action, prev_delta_x_des_H,
                               prev_delta_x_fd_H, prev_delta_x_cmd_H,
                               prev_delta_x_meas_H, prev_guard_flags) -> torch.Tensor:
    """Build the 62D horizon observation (legacy cylinder-space mode).

    DEPRECATED: Use build_high_level_observation() for new high-level PPO.
    """
    ee_state = uw.robot.data.body_link_state_w[:, uw._ee_body_id, :7]
    ee_pos_b = _tcp_pos_base_frame(uw)
    ee_quat_w = _normalized_quat(ee_state[:, 3:7])
    if hasattr(uw, "_command_pos_w") and uw._command_pos_w is not None:
        target_pos_b = uw._command_pos_w[:, :3]
    else:
        target_pos_b = torch.zeros((uw.num_envs, 3), device=ee_state.device)
    ee_pos_error_b = target_pos_b - ee_pos_b
    joint_pos = uw.robot.data.joint_pos[:, uw._arm_joint_ids]
    joint_vel = uw.robot.data.joint_vel[:, uw._arm_joint_ids]
    cq = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
    cqd = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]
    if getattr(uw, "_cylinder_adapter", None) is not None:
        cylinder_ext, cylinder_vel = uw._cylinder_adapter.current_from_cylinder_joints(cq, cqd)
    else:
        cylinder_ext, cylinder_vel = cq[:, :4], cqd[:, :4]
    unimog_pose, unimog_pose_dot = _unimog_pose(uw)
    obs = torch.cat([
        ee_pos_b, ee_quat_w, target_pos_b, ee_pos_error_b,
        joint_pos, joint_vel, cylinder_ext, cylinder_vel,
        prev_action, prev_delta_x_des_H, prev_delta_x_fd_H,
        prev_delta_x_cmd_H, prev_delta_x_meas_H, prev_guard_flags,
        unimog_pose, unimog_pose_dot,
    ], dim=-1)
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if obs.shape[-1] != HORIZON_OBS_DIM:
        raise RuntimeError(f"Horizon observation dim mismatch: got {obs.shape[-1]}, expected {HORIZON_OBS_DIM}")
    return obs
