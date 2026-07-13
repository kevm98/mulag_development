from __future__ import annotations

import torch


CYLINDER_OBS_DIM = 25


def build_cylinder_observation(
    *,
    cylinder_x: torch.Tensor,
    cylinder_xdot: torch.Tensor,
    cylinder_goal_x: torch.Tensor,
    prev_action: torch.Tensor,
    prev_valve_cmd: torch.Tensor,
    fd_delta_x: torch.Tensor,
    vehicle_speed_kmh: torch.Tensor,
) -> torch.Tensor:
    """Build the compact cylinder-space PPO observation.

    Layout:
    [x4, xdot4, goal_error4, prev_action4, prev_valve4, fd_delta4, vehicle_speed1]
    """
    goal_error = cylinder_goal_x - cylinder_x
    speed = vehicle_speed_kmh.reshape(-1, 1) / 10.0
    obs = torch.cat(
        [
            cylinder_x,
            cylinder_xdot,
            goal_error,
            prev_action,
            prev_valve_cmd,
            fd_delta_x,
            speed,
        ],
        dim=-1,
    )
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if obs.shape[-1] != CYLINDER_OBS_DIM:
        raise RuntimeError(f"Cylinder observation dim mismatch: got {obs.shape[-1]}, expected {CYLINDER_OBS_DIM}.")
    return obs

