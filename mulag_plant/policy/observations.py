"""Observation builder for three- or four-cylinder residual PPO policies.

The policy corrects cylinders 1-3 while cylinder 4 remains at the deterministic
head-down teacher value. The observation is built in the truck frame against the
original unmodified tool reference.
"""

from __future__ import annotations

import torch

RESIDUAL_OBS_FIXED_DIM = 38
RESIDUAL_OBS_DIM = 41
FOUR_CYLINDER_RESIDUAL_OBS_DIM = 42
TRAJECTORY_GEOMETRY_CHANNELS = 3
TRAJECTORY_GEOMETRY_OBS_DIM = RESIDUAL_OBS_FIXED_DIM + TRAJECTORY_GEOMETRY_CHANNELS + 4


def residual_observation_dim(action_dim: int, *, trajectory_geometry: bool = False) -> int:
    action_dim = int(action_dim)
    if action_dim not in (3, 4):
        raise ValueError(f"residual action dimension must be 3 or 4, got {action_dim}")
    return RESIDUAL_OBS_FIXED_DIM + action_dim + (
        TRAJECTORY_GEOMETRY_CHANNELS if trajectory_geometry else 0
    )


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
    nearest_point_delta_xy: torch.Tensor | None = None,
    signed_normal_error_m: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the residual observation for a 3- or 4-action policy.

    Layout:
      0:3   tool_error_truck           = reference_tool_position - measured_tool_position
      3:6   measured_tool_velocity_truck
      6:9   reference_tool_velocity_truck
      9:12  reference_tool_acceleration_truck
     12:16  measured_cylinder_extensions
     16:20  cylinder_goal_error        = corrected_cylinder_goal - measured_cylinder_extensions
     20:24  measured_cylinder_velocities
     24:24+A previous_filtered_policy_action, where A is 3 or 4
     Remaining signals retain their order: valve command (4), ROM increment
     (4), XY preview (2), direction, clearance, head alignment, and head rate.

    If nearest-point geometry is supplied, three channels are appended after
    the legacy block: nearest-point delta XY and signed normal error.  This is
    opt-in so historical line checkpoints keep their exact observation width.
    """
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
    if (nearest_point_delta_xy is None) != (signed_normal_error_m is None):
        raise ValueError("nearest-point delta and signed normal must be supplied together")
    if nearest_point_delta_xy is not None:
        if nearest_point_delta_xy.shape[-1] != 2:
            raise ValueError("nearest_point_delta_xy must have two columns")
        obs = torch.cat((obs, nearest_point_delta_xy, signed_normal_error_m.reshape(-1, 1)), dim=-1)
    expected_dim = residual_observation_dim(
        previous_filtered_policy_action.shape[-1],
        trajectory_geometry=nearest_point_delta_xy is not None,
    )
    if obs.shape[-1] != expected_dim:
        raise RuntimeError(
            f"Residual observation dim mismatch: got {obs.shape[-1]}, expected {expected_dim}."
        )
    return obs
