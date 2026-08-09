"""Observation builder for the ID-FD Policy.

The observation is command-conditioned and contains no trajectory-shape label.
It supports the maintained three-action PPO1 interface and a future four-action
variant without changing the fixed plant/reference portion of the contract.
"""

from __future__ import annotations

import torch

RESIDUAL_OBS_FIXED_DIM = 38
RESIDUAL_OBS_DIM = 41
FOUR_CYLINDER_RESIDUAL_OBS_DIM = 42


def residual_observation_dim(action_dim: int) -> int:
    action_dim = int(action_dim)
    if action_dim not in (3, 4):
        raise ValueError(f"residual action dimension must be 3 or 4, got {action_dim}")
    return RESIDUAL_OBS_FIXED_DIM + action_dim


def build_residual_observation(
    *,
    tool_error_truck: torch.Tensor,
    measured_tool_velocity_truck: torch.Tensor,
    reference_tool_velocity_truck: torch.Tensor,
    reference_tool_acceleration_truck: torch.Tensor,
    measured_cylinder_extensions: torch.Tensor,
    cylinder_goal_error: torch.Tensor,
    measured_cylinder_velocities: torch.Tensor,
    previous_filtered_policy_action: torch.Tensor,
    current_valve_commands: torch.Tensor,
    corrected_fd_delta: torch.Tensor,
    reference_preview_xy_truck: torch.Tensor,
    trajectory_direction: torch.Tensor,
    cutter_bottom_world_clearance: torch.Tensor,
    head_down_alignment: torch.Tensor,
    head_down_alignment_rate: torch.Tensor,
) -> torch.Tensor:
    """Build 38 fixed plant/reference values plus one value per PPO action."""
    obs = torch.cat(
        [
            tool_error_truck,
            measured_tool_velocity_truck,
            reference_tool_velocity_truck,
            reference_tool_acceleration_truck,
            measured_cylinder_extensions,
            cylinder_goal_error,
            measured_cylinder_velocities,
            previous_filtered_policy_action,
            current_valve_commands,
            corrected_fd_delta,
            reference_preview_xy_truck,
            trajectory_direction.reshape(-1, 1),
            cutter_bottom_world_clearance.reshape(-1, 1),
            head_down_alignment.reshape(-1, 1),
            head_down_alignment_rate.reshape(-1, 1),
        ],
        dim=-1,
    )
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    expected = residual_observation_dim(previous_filtered_policy_action.shape[-1])
    if obs.shape[-1] != expected:
        raise RuntimeError(
            f"Residual observation dim mismatch: got {obs.shape[-1]}, expected {expected}."
        )
    return obs

