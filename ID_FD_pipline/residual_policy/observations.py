"""41-dimensional observation builder for the residual PPO policy.

The policy corrects cylinders 1-3 while cylinder 4 remains at the deterministic
head-down teacher value. The observation is built in the truck frame against the
original unmodified tool reference.
"""

from __future__ import annotations

import torch

RESIDUAL_OBS_DIM = 41


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
    trajectory_phase_sin_cos: torch.Tensor,
    trajectory_direction: torch.Tensor,
    cutter_bottom_world_clearance: torch.Tensor,
    head_down_alignment: torch.Tensor,
    head_down_alignment_rate: torch.Tensor,
) -> torch.Tensor:
    """Build the 41-dimensional residual-policy observation.

    Layout:
      0:3   tool_error_truck           = reference_tool_position - measured_tool_position
      3:6   measured_tool_velocity_truck
      6:9   reference_tool_velocity_truck
      9:12  reference_tool_acceleration_truck
     12:16  measured_cylinder_extensions
     16:20  cylinder_goal_error        = corrected_cylinder_goal - measured_cylinder_extensions
     20:24  measured_cylinder_velocities
     24:27  previous_filtered_policy_action
     27:31  current_valve_commands
     31:35  corrected_fd_delta
     35:37  trajectory_phase_sin_cos
     37     trajectory_direction       (+1 for A→B, -1 for B→A)
     38     cutter_bottom_world_clearance
     39     head_down_alignment
     40     head_down_alignment_rate

    Total: 41.
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
            trajectory_phase_sin_cos,
            trajectory_direction.reshape(-1, 1),
            cutter_bottom_world_clearance.reshape(-1, 1),
            head_down_alignment.reshape(-1, 1),
            head_down_alignment_rate.reshape(-1, 1),
        ],
        dim=-1,
    )
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if obs.shape[-1] != RESIDUAL_OBS_DIM:
        raise RuntimeError(
            f"Residual observation dim mismatch: got {obs.shape[-1]}, expected {RESIDUAL_OBS_DIM}."
        )
    return obs
