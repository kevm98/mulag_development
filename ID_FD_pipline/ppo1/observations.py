"""Stage C: Enriched observation contract.
Adds cylinder goal, valve command, and FD delta so the policy can see
what the base controller is doing internally.
"""

from __future__ import annotations

import torch


# Stage C: +12 dims (cylinder_goal_x[4] + valve_cmd[4] + fd_delta[4])
EE_RESIDUAL_OBS_DIM = 33


def build_residual_observation(
    *,
    ee_pos_local: torch.Tensor,
    nominal_goal_local: torch.Tensor,
    ee_velocity_local: torch.Tensor,
    cylinder_x: torch.Tensor,
    cylinder_xdot: torch.Tensor,
    vehicle_speed_kmh: torch.Tensor,
    vehicle_yaw_rate: torch.Tensor,
    previous_residual_action: torch.Tensor,
    cylinder_goal_x: torch.Tensor | None = None,
    current_valve_cmd: torch.Tensor | None = None,
    fd_delta_corrected: torch.Tensor | None = None,
) -> torch.Tensor:
    position_yz = ee_pos_local[:, 1:3]
    goal_yz = nominal_goal_local[:, 1:3]
    error_yz = goal_yz - position_yz

    parts = [
        position_yz,                 # 2
        goal_yz,                     # 2
        error_yz,                    # 2
        ee_velocity_local[:, 1:3],   # 2
        cylinder_x,                  # 4
        cylinder_xdot,               # 4
        vehicle_speed_kmh.reshape(-1, 1) / 10.0,  # 1
        vehicle_yaw_rate.reshape(-1, 1),          # 1
        previous_residual_action,    # 3
    ]
    # Stage C: base-controller internal state for richer residual context
    if cylinder_goal_x is not None:
        parts.append(cylinder_goal_x)           # 4
    if current_valve_cmd is not None:
        parts.append(current_valve_cmd)          # 4
    if fd_delta_corrected is not None:
        parts.append(fd_delta_corrected)         # 4

    observation = torch.cat(parts, dim=-1)
    observation = torch.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
    if observation.shape[-1] != EE_RESIDUAL_OBS_DIM:
        raise RuntimeError(
            f"Residual observation width {observation.shape[-1]} != {EE_RESIDUAL_OBS_DIM}"
        )
    return observation
