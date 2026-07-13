from __future__ import annotations

from typing import Iterable

import torch


def _check_last_dim(name: str, tensor: torch.Tensor, expected_dim: int) -> None:
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} must have last dimension {expected_dim}, got {tensor.shape[-1]}")


def _check_dims(tensors: Iterable[tuple[str, torch.Tensor, int]]) -> None:
    for name, tensor, expected_dim in tensors:
        _check_last_dim(name, tensor, expected_dim)


def build_aam_input(
    q: torch.Tensor,
    qdot: torch.Tensor,
    dP: torch.Tensor,
    prev_valve_cmd: torch.Tensor,
) -> torch.Tensor:
    _check_dims(
        (
            ("q", q, 5),
            ("qdot", qdot, 5),
            ("dP", dP, 4),
            ("prev_valve_cmd", prev_valve_cmd, 4),
        )
    )
    return torch.cat([q, qdot, dP, prev_valve_cmd], dim=-1)


def build_rl_obs(
    q: torch.Tensor,
    qdot: torch.Tensor,
    q_ref: torch.Tensor,
    qdot_ref: torch.Tensor,
    prev_action: torch.Tensor,
    prev_valve_cmd: torch.Tensor,
    z_arm_hat: torch.Tensor,
) -> torch.Tensor:
    _check_dims(
        (
            ("q", q, 5),
            ("qdot", qdot, 5),
            ("q_ref", q_ref, 5),
            ("qdot_ref", qdot_ref, 5),
            ("prev_action", prev_action, 4),
            ("prev_valve_cmd", prev_valve_cmd, 4),
            ("z_arm_hat", z_arm_hat, 24),
        )
    )
    q_error = q_ref - q
    qdot_error = qdot_ref - qdot
    return torch.cat(
        [
            q,
            qdot,
            q_error,
            qdot_error,
            prev_action,
            prev_valve_cmd,
            z_arm_hat,
        ],
        dim=-1,
    )


def build_task_space_rl_obs(
    ee_pose_command: torch.Tensor,
    current_ee_pose: torch.Tensor,
    ee_position_error: torch.Tensor,
    ee_orientation_error: torch.Tensor,
    mechanical_q: torch.Tensor,
    mechanical_qdot: torch.Tensor,
    cylinder_q: torch.Tensor,
    cylinder_qdot: torch.Tensor,
    prev_action: torch.Tensor,
    prev_valve_cmd: torch.Tensor,
    z_arm_hat: torch.Tensor,
    fd_tau_hat: torch.Tensor,
    vehicle_speed_kmh: torch.Tensor,
) -> torch.Tensor:
    _check_dims(
        (
            ("ee_pose_command", ee_pose_command, 7),
            ("current_ee_pose", current_ee_pose, 7),
            ("ee_position_error", ee_position_error, 3),
            ("ee_orientation_error", ee_orientation_error, 1),
            ("mechanical_q", mechanical_q, 5),
            ("mechanical_qdot", mechanical_qdot, 5),
            ("cylinder_q", cylinder_q, 4),
            ("cylinder_qdot", cylinder_qdot, 4),
            ("prev_action", prev_action, 4),
            ("prev_valve_cmd", prev_valve_cmd, 4),
            ("z_arm_hat", z_arm_hat, 24),
            ("fd_tau_hat", fd_tau_hat, 4),
            ("vehicle_speed_kmh", vehicle_speed_kmh, 1),
        )
    )
    ee_orientation_command = ee_pose_command[..., 3:7]
    return torch.cat(
        [
            ee_orientation_command,
            current_ee_pose,
            ee_position_error,
            ee_orientation_error,
            mechanical_q,
            mechanical_qdot,
            cylinder_q,
            cylinder_qdot,
            prev_action,
            prev_valve_cmd,
            z_arm_hat,
            fd_tau_hat,
            vehicle_speed_kmh,
        ],
        dim=-1,
    )


def build_task_space_id_fd_rl_obs(
    ee_pose_command: torch.Tensor,
    current_ee_pose: torch.Tensor,
    ee_position_error: torch.Tensor,
    ee_orientation_error: torch.Tensor,
    mechanical_q: torch.Tensor,
    mechanical_qdot: torch.Tensor,
    cylinder_q: torch.Tensor,
    cylinder_qdot: torch.Tensor,
    prev_action: torch.Tensor,
    prev_valve_cmd: torch.Tensor,
    fd_tau_hat: torch.Tensor,
    vehicle_speed_kmh: torch.Tensor,
) -> torch.Tensor:
    _check_dims(
        (
            ("ee_pose_command", ee_pose_command, 7),
            ("current_ee_pose", current_ee_pose, 7),
            ("ee_position_error", ee_position_error, 3),
            ("ee_orientation_error", ee_orientation_error, 1),
            ("mechanical_q", mechanical_q, 5),
            ("mechanical_qdot", mechanical_qdot, 5),
            ("cylinder_q", cylinder_q, 4),
            ("cylinder_qdot", cylinder_qdot, 4),
            ("prev_action", prev_action, 4),
            ("prev_valve_cmd", prev_valve_cmd, 4),
            ("fd_tau_hat", fd_tau_hat, 4),
            ("vehicle_speed_kmh", vehicle_speed_kmh, 1),
        )
    )
    ee_orientation_command = ee_pose_command[..., 3:7]
    return torch.cat(
        [
            ee_orientation_command,
            current_ee_pose,
            ee_position_error,
            ee_orientation_error,
            mechanical_q,
            mechanical_qdot,
            cylinder_q,
            cylinder_qdot,
            prev_action,
            prev_valve_cmd,
            fd_tau_hat,
            vehicle_speed_kmh,
        ],
        dim=-1,
    )


def build_id_input(q: torch.Tensor, qdot: torch.Tensor, qddot_cmd: torch.Tensor, dP: torch.Tensor) -> torch.Tensor:
    _check_dims((("q", q, 5), ("qdot", qdot, 5), ("dP", dP, 4)))
    if qddot_cmd.shape[-1] == 4:
        zero_tool_joint = torch.zeros_like(qddot_cmd[..., :1])
        qddot_cmd = torch.cat([qddot_cmd, zero_tool_joint], dim=-1)
    elif qddot_cmd.shape[-1] != 5:
        raise ValueError(f"qddot_cmd must have last dimension 4 or 5, got {qddot_cmd.shape[-1]}")
    return torch.cat([q, qdot, qddot_cmd, dP], dim=-1)


def build_fd_input(q: torch.Tensor, qdot: torch.Tensor, dP: torch.Tensor, valve_cmd: torch.Tensor) -> torch.Tensor:
    _check_dims((("q", q, 5), ("qdot", qdot, 5), ("dP", dP, 4), ("valve_cmd", valve_cmd, 4)))
    return torch.cat([q, qdot, dP, valve_cmd], dim=-1)


def build_sd_input(
    q: torch.Tensor,
    qdot: torch.Tensor,
    dP: torch.Tensor,
    fnet: torch.Tensor,
    valve_cmd: torch.Tensor,
) -> torch.Tensor:
    _check_dims(
        (
            ("q", q, 5),
            ("qdot", qdot, 5),
            ("dP", dP, 4),
            ("fnet", fnet, 4),
            ("valve_cmd", valve_cmd, 4),
        )
    )
    return torch.cat([q, qdot, dP, fnet, valve_cmd], dim=-1)
