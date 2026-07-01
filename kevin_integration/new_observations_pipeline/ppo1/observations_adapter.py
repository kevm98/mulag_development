"""Horizon PPO observation builder.

Obs dim: 53

Index map:
  0:2   ee_pos[3]
  3:6   ee_quat[4]
  7:9   pos_error[3]
  10    orient_error[1]
  11:15 joint_pos[5]
  16:20 joint_vel[5]
  21:24 cyl_ext[4]
  25:28 cyl_vel[4]
  29:32 prev_valve_cmd[4]
  33:36 prev_fd_delta[4]
  37:40 prev_meas_delta[4]
  41:44 prev_cmd_delta[4]
  45:48 prev_action[4]
  49:52 prev_action_rate[4]
"""
import torch


def build_horizon_observation(uw, prev_valve_cmd, prev_fd_delta, prev_meas_delta,
                               prev_cmd_delta, prev_action) -> torch.Tensor:
    """Build 53D horizon observation from env state and horizon context."""
    ee_state = uw.robot.data.body_link_state_w[:, uw._ee_body_id, :7]
    ee_quat = ee_state[:, 3:7]
    ee_quat = ee_quat / (ee_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    ee_pose = torch.cat([ee_state[:, :3], ee_quat], dim=-1)

    if hasattr(uw, '_command_pos_w') and uw._command_pos_w is not None:
        pos_err = uw._command_pos_w[:, :3] - ee_pose[:, :3]
    else:
        pos_err = torch.zeros((uw.num_envs, 3), device=ee_pose.device)
    orient_err = torch.zeros((uw.num_envs, 1), device=ee_pose.device)

    jpos = uw.robot.data.joint_pos[:, uw._arm_joint_ids]
    jvel = uw.robot.data.joint_vel[:, uw._arm_joint_ids]
    cq = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
    cqd = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]

    horizon_ctx = torch.cat([prev_valve_cmd, prev_fd_delta,
                              prev_meas_delta, prev_cmd_delta], dim=-1)
    action_ctx = torch.cat([prev_action,
                             torch.zeros_like(prev_action)], dim=-1)

    return torch.cat([ee_pose, pos_err, orient_err, jpos, jvel, cq, cqd,
                      horizon_ctx, action_ctx], dim=-1)
