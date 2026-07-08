"""Horizon PPO reward adapter."""
from __future__ import annotations

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse


def _normalized_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _tcp_pos_base_frame(uw) -> torch.Tensor:
    ee_pos_w = uw.robot.data.body_link_state_w[:, uw._ee_body_id, :3]
    base_pos_w = uw.robot.data.root_pos_w
    base_quat_w = _normalized_quat(uw.robot.data.root_quat_w)
    return quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)


def _tracking_pos_base_frame(uw) -> torch.Tensor:
    if hasattr(uw, "_ppo1_tracking_point_pos_b"):
        return uw._ppo1_tracking_point_pos_b
    return _tcp_pos_base_frame(uw)


def _target_pos_base_frame(uw) -> torch.Tensor:
    if hasattr(uw, "_ppo1_target_point_pos_b"):
        return uw._ppo1_target_point_pos_b
    return uw._command_pos_w[:, :3]


def compute_tracking_error(uw) -> torch.Tensor:
    """Return active tracking-point-to-command distance in the base frame."""
    tracking_pos_b = _tracking_pos_base_frame(uw)
    target_pos_b = _target_pos_base_frame(uw)
    return torch.norm(tracking_pos_b - target_pos_b, dim=-1)


def _yaw_error(uw) -> torch.Tensor:
    ee_q = _normalized_quat(uw.robot.data.body_link_state_w[:, uw._ee_body_id, 3:7])
    cmd_q = _normalized_quat(uw._command_quat_w)
    _, _, ee_yaw = euler_xyz_from_quat(ee_q)
    _, _, cmd_yaw = euler_xyz_from_quat(cmd_q)
    return torch.abs(torch.atan2(torch.sin(ee_yaw - cmd_yaw), torch.cos(ee_yaw - cmd_yaw)))


def compute_horizon_reward(
    uw,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    error_before: torch.Tensor,
    error_after: torch.Tensor,
    delta_x_cmd_H: torch.Tensor,
    delta_x_meas_H: torch.Tensor,
    guard_flags: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute horizon-level tracking reward and diagnostic terms."""
    cfg = uw.cfg
    progress = error_before - error_after
    std = float(getattr(cfg, "position_tracking_std", 0.4))
    pos_tracking = torch.exp(-error_after / max(std, 1e-6))
    close_target = 1.0 - torch.tanh(error_after / max(std, 1e-6))

    yaw_offset = float(getattr(cfg, "yaw_tracking_offset_deg", 30.0)) * torch.pi / 180.0
    yaw_pitch = torch.clamp(_yaw_error(uw) - yaw_offset, min=0.0)

    tcp_ahead = torch.zeros_like(error_after)
    if getattr(uw, "_arm_base_body_id", None) is not None:
        ee_pos_w = uw.robot.data.body_link_state_w[:, uw._ee_body_id, :3]
        arm_base_pos_w = uw.robot.data.body_link_state_w[:, uw._arm_base_body_id, :3]
        tcp_ahead = ee_pos_w[:, 0] - arm_base_pos_w[:, 0]

    action_rate = torch.sum((action - prev_action) ** 2, dim=-1)
    joint_vel = torch.sum(uw.robot.data.joint_vel[:, uw._arm_joint_ids] ** 2, dim=-1)
    hydraulic_consistency = torch.norm(delta_x_meas_H - delta_x_cmd_H, dim=-1)
    guard_fraction = guard_flags.float().mean(dim=-1)

    reward = (
        2.0 * progress
        + 1.0 * pos_tracking
        + 0.5 * close_target
        - 0.25 * yaw_pitch
        + 0.02 * torch.clamp(tcp_ahead, min=-1.0, max=1.0)
        - 0.05 * action_rate
        - 0.001 * joint_vel
        - float(getattr(cfg, "horizon_hydraulic_consistency_weight", 5.0)) * hydraulic_consistency
        - float(getattr(cfg, "horizon_guard_penalty_weight", 0.5)) * guard_fraction
    )
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    terms = {
        "progress": progress.detach(),
        "position_tracking": pos_tracking.detach(),
        "close_target": close_target.detach(),
        "yaw_pitch": yaw_pitch.detach(),
        "tcp_ahead": tcp_ahead.detach(),
        "action_rate": action_rate.detach(),
        "joint_velocity": joint_vel.detach(),
        "hydraulic_consistency": hydraulic_consistency.detach(),
        "guard_fraction": guard_fraction.detach(),
    }
    return reward, terms


def compute_stage0_reward(
    uw,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    error_before: torch.Tensor,
    error_after: torch.Tensor,
    delta_x_cmd_H: torch.Tensor,
    delta_x_meas_H: torch.Tensor,
    guard_flags: torch.Tensor,
    clipping_fraction: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stage-0 local-reach reward: progress toward nearby target (rescaled).

    Reward components (target per-horizon ~[-5, +5]):
        +20.0 * clamp(progress, -0.02, 0.02)
        + 2.0 * exp(-8.0 * ee_error_after)
        - 1.0 * ee_error_after
        - 0.05 * ||action||
        - 0.10 * ||action - prev_action||
        - 1.0  * hydraulic_consistency_error
        - 5.0  * guard_fraction
        - 3.0  * clipping_fraction
    Plus do-no-harm: if progress < 0, add 30.0 * progress (penalizes moving away)
    """
    cfg = uw.cfg
    progress = error_before - error_after
    progress_clamped = torch.clamp(progress, -0.02, 0.02)
    proximity = torch.exp(-8.0 * error_after)
    action_norm = torch.norm(action, dim=-1)
    action_rate = torch.sum((action - prev_action) ** 2, dim=-1)
    hydraulic_consistency = torch.norm(delta_x_meas_H - delta_x_cmd_H, dim=-1)
    guard_fraction = guard_flags.float().mean(dim=-1)

    gpw = float(getattr(cfg, "horizon_guard_penalty_weight", 5.0))
    cpw = float(getattr(cfg, "horizon_clip_penalty_weight", 3.0))

    reward = (
        20.0 * progress_clamped
        + 2.0 * proximity
        - 1.0 * error_after
        - 0.05 * action_norm
        - 0.10 * action_rate
        - 1.0 * hydraulic_consistency
        - gpw * guard_fraction
        - cpw * clipping_fraction
    )

    # -- Do-no-harm / zero-action anchor: penalize moving further from target --
    do_no_harm = torch.where(progress < 0, 30.0 * progress, torch.zeros_like(progress))
    reward = reward + do_no_harm

    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    terms = {
        "progress": progress.detach(),
        "proximity": proximity.detach(),
        "ee_error_after": error_after.detach(),
        "action_norm": action_norm.detach(),
        "action_rate": action_rate.detach(),
        "hydraulic_consistency": hydraulic_consistency.detach(),
        "guard_fraction": guard_fraction.detach(),
        "clipping_fraction": clipping_fraction.detach(),
        "do_no_harm": do_no_harm.detach(),
    }
    return reward, terms


def compute_high_level_reward(
    uw,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    error_before: torch.Tensor,
    error_after: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """High-level task-space PPO reward.

    Teacher-student style: only task-space terms. No cylinder tracking.
    """
    cfg = uw.cfg
    progress = torch.clamp(error_before - error_after, -0.02, 0.02)
    proximity = torch.exp(-8.0 * error_after)
    action_rate = torch.sum((action - prev_action) ** 2, dim=-1)
    action_norm = torch.norm(action, dim=-1)
    joint_vel = torch.sum(uw.robot.data.joint_vel[:, uw._arm_joint_ids] ** 2, dim=-1)

    w_progress = 20.0
    w_tracking = 2.0
    w_error = 1.0
    w_action_rate = 0.05
    w_action_mag = 0.02
    w_joint_vel = 0.01

    reward = (
        w_progress * progress
        + w_tracking * proximity
        - w_error * error_after
        - w_action_rate * action_rate
        - w_action_mag * action_norm
        - w_joint_vel * joint_vel
    )
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    terms = {
        "progress": progress.detach(),
        "proximity": proximity.detach(),
        "ee_error_after": error_after.detach(),
        "action_rate": action_rate.detach(),
        "action_norm": action_norm.detach(),
        "joint_velocity": joint_vel.detach(),
        "guard_fraction": torch.zeros_like(error_after),
        "hydraulic_consistency": torch.zeros_like(error_after),
        "clipping_fraction": torch.zeros_like(error_after),
    }
    return reward, terms
