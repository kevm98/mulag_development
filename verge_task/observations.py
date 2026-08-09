"""Observation builder for the sole-controller policy.

PPO2's observation carried ``corrected_cylinder_goal - measured_extension`` at
slice 16:20.  That signal only exists because a Jacobian cascade computed the
goal; handing it to PPO3 would mean the inverse kinematics is still solving the
task and the policy is still a corrector.  It is therefore removed from the
default layout and available only through the ``ik_hint`` ablation, which
appends it at the very end so the two layouts share a common prefix.

What replaces it is the state the cylinder DLS controller itself used to close
its loop -- measured extension, measured velocity, the standing lead between the
commanded PhysX target and the measurement, and the distance remaining to each
extension limit.  The policy is given the same information the deleted
controller had, and has to decide what to do with it.
"""

from __future__ import annotations

import torch


def finite_difference_kinematics(
    position: torch.Tensor,
    previous_position: torch.Tensor | None,
    previous_velocity: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Velocity/acceleration sample with a zero first-sample contract."""
    if previous_position is None:
        zeros = torch.zeros_like(position)
        return zeros, zeros
    safe_dt = max(float(dt), 1.0e-6)
    velocity = (position - previous_position) / safe_dt
    acceleration = (velocity - previous_velocity) / safe_dt
    return velocity, acceleration

SOLE_OBS_DIM = 50
IK_HINT_OBS_DIM = 4
SOLE_OBS_DIM_WITH_HINT = SOLE_OBS_DIM + IK_HINT_OBS_DIM


def sole_observation_dim(*, ik_hint: bool) -> int:
    return SOLE_OBS_DIM_WITH_HINT if ik_hint else SOLE_OBS_DIM


def build_sole_observation(
    *,
    tool_error_truck: torch.Tensor,
    measured_tool_velocity_truck: torch.Tensor,
    reference_tool_velocity_truck: torch.Tensor,
    reference_tool_acceleration_truck: torch.Tensor,
    measured_cylinder_extensions: torch.Tensor,
    measured_cylinder_velocities: torch.Tensor,
    cylinder_headroom_low: torch.Tensor,
    cylinder_headroom_high: torch.Tensor,
    previous_applied_action: torch.Tensor,
    current_valve_commands: torch.Tensor,
    rom_delta_extension: torch.Tensor,
    commanded_target_lead: torch.Tensor,
    reference_preview_xy_truck: torch.Tensor,
    trajectory_direction: torch.Tensor,
    cutter_bottom_world_clearance: torch.Tensor,
    head_down_alignment: torch.Tensor,
    head_down_alignment_rate: torch.Tensor,
    ik_hint_cylinder_error: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the sole-controller observation.

    Layout:
       0:3   tool_error_truck = reference_tool_position - measured_tool_position
       3:6   measured_tool_velocity_truck
       6:9   reference_tool_velocity_truck
       9:12  reference_tool_acceleration_truck
      12:16  measured_cylinder_extensions
      16:20  measured_cylinder_velocities
      20:24  cylinder_headroom_low  = measured - learned_min, normalized
      24:28  cylinder_headroom_high = learned_max - measured, normalized
      28:32  previous_applied_action
      32:36  current_valve_commands (predicted inside the ROM; plant state)
      36:40  rom_delta_extension (last applied ROM displacement)
      40:44  commanded_target_lead = commanded target - measured extension
      44:46  reference_preview_xy_truck
      46     trajectory_direction
      47     cutter_bottom_world_clearance
      48     head_down_alignment
      49     head_down_alignment_rate

    Total 50.  With ``ik_hint_cylinder_error`` the cascade's residual signal is
    appended at 50:54 for a total of 54.
    """
    columns = [
        tool_error_truck,
        measured_tool_velocity_truck,
        reference_tool_velocity_truck,
        reference_tool_acceleration_truck,
        measured_cylinder_extensions,
        measured_cylinder_velocities,
        cylinder_headroom_low,
        cylinder_headroom_high,
        previous_applied_action,
        current_valve_commands,
        rom_delta_extension,
        commanded_target_lead,
        reference_preview_xy_truck,
        trajectory_direction.reshape(-1, 1),
        cutter_bottom_world_clearance.reshape(-1, 1),
        head_down_alignment.reshape(-1, 1),
        head_down_alignment_rate.reshape(-1, 1),
    ]
    if ik_hint_cylinder_error is not None:
        columns.append(ik_hint_cylinder_error)

    obs = torch.nan_to_num(
        torch.cat(columns, dim=-1), nan=0.0, posinf=0.0, neginf=0.0
    )
    expected = sole_observation_dim(ik_hint=ik_hint_cylinder_error is not None)
    if obs.shape[-1] != expected:
        raise RuntimeError(
            f"Sole observation dim mismatch: got {obs.shape[-1]}, expected {expected}."
        )
    return obs
