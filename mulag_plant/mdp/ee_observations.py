"""EE-tracking observation builder for the ID-FD cylinder pipeline.

Observation (30D):
  [ee_pos_local3, ee_goal_local3, ee_error3, x4, xdot4, prev_action4, prev_valve4, fd_delta4, speed1]
"""

from __future__ import annotations

import torch


EE_TRACKING_OBS_DIM = 30


def build_ee_tracking_observation(
    *,
    ee_pos_local: torch.Tensor,
    ee_goal_local: torch.Tensor,
    cylinder_x: torch.Tensor,
    cylinder_xdot: torch.Tensor,
    prev_action: torch.Tensor,
    prev_valve_cmd: torch.Tensor,
    fd_delta_x: torch.Tensor,
    vehicle_speed_kmh: torch.Tensor,
) -> torch.Tensor:
    """Build the EE-tracking PPO observation.

    Layout:
    [ee_pos3, ee_goal3, ee_error3, x4, xdot4, prev_action4, prev_valve4, fd_delta4, speed1]  = 30D
    """
    ee_error = ee_goal_local - ee_pos_local
    speed = vehicle_speed_kmh.reshape(-1, 1) / 10.0
    obs = torch.cat(
        [
            ee_pos_local,
            ee_goal_local,
            ee_error,
            cylinder_x,
            cylinder_xdot,
            prev_action,
            prev_valve_cmd,
            fd_delta_x,
            speed,
        ],
        dim=-1,
    )
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if obs.shape[-1] != EE_TRACKING_OBS_DIM:
        raise RuntimeError(f"EE-tracking observation dim mismatch: got {obs.shape[-1]}, expected {EE_TRACKING_OBS_DIM}.")
    return obs

