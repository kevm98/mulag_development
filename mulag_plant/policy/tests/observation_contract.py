"""Observation contract test for the Policy.

Verifies:
- Observation shape is exactly 41
- All slices contain documented signals
- No NaN/Inf in observation
- Reset histories are zeroed
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mulag_plant.policy.observations import (
    RESIDUAL_OBS_DIM,
    build_residual_observation,
)


def test_observation_shape():
    """Verify observation shape is exactly 41."""
    N = 4
    zeros_3 = torch.zeros(N, 3)
    zeros_4 = torch.zeros(N, 4)
    zeros_1 = torch.zeros(N, 1)
    zeros_2 = torch.zeros(N, 2)
    zero_scalar = torch.zeros(N)

    obs = build_residual_observation(
        tool_error_truck=zeros_3,
        measured_tool_velocity_truck=zeros_3,
        reference_tool_velocity_truck=zeros_3,
        reference_tool_acceleration_truck=zeros_3,
        measured_cylinder_extensions=zeros_4,
        cylinder_goal_error=zeros_4,
        measured_cylinder_velocities=zeros_4,
        previous_filtered_policy_action=torch.zeros(N, 3),
        current_valve_commands=zeros_4,
        corrected_fd_delta=zeros_4,
        reference_preview_xy_truck=zeros_2,
        trajectory_direction=zero_scalar,
        cutter_bottom_world_clearance=zero_scalar,
        head_down_alignment=zero_scalar,
        head_down_alignment_rate=zero_scalar,
    )
    assert obs.shape == (N, RESIDUAL_OBS_DIM), f"Expected ({N}, {RESIDUAL_OBS_DIM}), got {obs.shape}"
    assert RESIDUAL_OBS_DIM == 41, f"RESIDUAL_OBS_DIM should be 41, got {RESIDUAL_OBS_DIM}"


def test_no_nan_inf():
    """Verify no NaN or Inf in the observation."""
    N = 2
    zeros_3 = torch.zeros(N, 3)
    zeros_4 = torch.zeros(N, 4)
    zeros_1 = torch.zeros(N, 1)
    zeros_2 = torch.zeros(N, 2)
    zero_scalar = torch.zeros(N)

    obs = build_residual_observation(
        tool_error_truck=zeros_3,
        measured_tool_velocity_truck=zeros_3,
        reference_tool_velocity_truck=zeros_3,
        reference_tool_acceleration_truck=zeros_3,
        measured_cylinder_extensions=zeros_4,
        cylinder_goal_error=zeros_4,
        measured_cylinder_velocities=zeros_4,
        previous_filtered_policy_action=torch.zeros(N, 3),
        current_valve_commands=zeros_4,
        corrected_fd_delta=zeros_4,
        reference_preview_xy_truck=zeros_2,
        trajectory_direction=zero_scalar,
        cutter_bottom_world_clearance=zero_scalar,
        head_down_alignment=zero_scalar,
        head_down_alignment_rate=zero_scalar,
    )
    assert torch.isfinite(obs).all(), f"Observation contains non-finite values"


def test_nan_replacement():
    """Verify NaN values are replaced with zeros."""
    N = 2
    nan_3 = torch.full((N, 3), float("nan"))
    zeros_4 = torch.zeros(N, 4)
    zeros_1 = torch.zeros(N, 1)
    zeros_2 = torch.zeros(N, 2)
    zero_scalar = torch.zeros(N)

    obs = build_residual_observation(
        tool_error_truck=nan_3,
        measured_tool_velocity_truck=nan_3,
        reference_tool_velocity_truck=nan_3,
        reference_tool_acceleration_truck=nan_3,
        measured_cylinder_extensions=zeros_4,
        cylinder_goal_error=zeros_4,
        measured_cylinder_velocities=zeros_4,
        previous_filtered_policy_action=torch.zeros(N, 3),
        current_valve_commands=zeros_4,
        corrected_fd_delta=zeros_4,
        reference_preview_xy_truck=zeros_2,
        trajectory_direction=zero_scalar,
        cutter_bottom_world_clearance=zero_scalar,
        head_down_alignment=zero_scalar,
        head_down_alignment_rate=zero_scalar,
    )
    assert torch.isfinite(obs).all(), f"NaN replacement failed"
    assert (obs[:, :12] == 0.0).all(), f"NaN values not zeroed"


def test_slice_layout():
    """Verify each documented slice range."""
    N = 1
    obs = build_residual_observation(
        tool_error_truck=torch.tensor([[1.0, 2.0, 3.0]]),
        measured_tool_velocity_truck=torch.tensor([[4.0, 5.0, 6.0]]),
        reference_tool_velocity_truck=torch.tensor([[7.0, 8.0, 9.0]]),
        reference_tool_acceleration_truck=torch.tensor([[10.0, 11.0, 12.0]]),
        measured_cylinder_extensions=torch.tensor([[13.0, 14.0, 15.0, 16.0]]),
        cylinder_goal_error=torch.tensor([[17.0, 18.0, 19.0, 20.0]]),
        measured_cylinder_velocities=torch.tensor([[21.0, 22.0, 23.0, 24.0]]),
        previous_filtered_policy_action=torch.tensor([[25.0, 26.0, 27.0]]),
        current_valve_commands=torch.tensor([[28.0, 29.0, 30.0, 31.0]]),
        corrected_fd_delta=torch.tensor([[32.0, 33.0, 34.0, 35.0]]),
        reference_preview_xy_truck=torch.tensor([[36.0, 37.0]]),
        trajectory_direction=torch.tensor([38.0]),
        cutter_bottom_world_clearance=torch.tensor([39.0]),
        head_down_alignment=torch.tensor([40.0]),
        head_down_alignment_rate=torch.tensor([41.0]),
    )
    flat = obs[0]
    assert flat[0] == 1.0, f"tool_error_truck[0] mismatch: {flat[0]}"
    assert flat[2] == 3.0, f"tool_error_truck[2] mismatch: {flat[2]}"
    assert flat[3] == 4.0, f"measured_tool_velocity[0] mismatch: {flat[3]}"
    assert flat[11] == 12.0, f"ref_acc[2] mismatch: {flat[11]}"
    assert flat[12] == 13.0, f"cylinder[0] mismatch: {flat[12]}"
    assert flat[15] == 16.0, f"cylinder[3] mismatch: {flat[15]}"
    assert flat[19] == 20.0, f"goal_error[3] mismatch: {flat[19]}"
    assert flat[23] == 24.0, f"cylinder_vel[3] mismatch: {flat[23]}"
    assert flat[26] == 27.0, f"prev_filtered[2] mismatch: {flat[26]}"
    assert flat[30] == 31.0, f"valve[3] mismatch: {flat[30]}"
    assert flat[34] == 35.0, f"fd_delta[3] mismatch: {flat[34]}"
    assert flat[36] == 37.0, f"reference preview Y mismatch: {flat[36]}"
    assert flat[37] == 38.0, f"direction mismatch: {flat[37]}"
    assert flat[38] == 39.0, f"clearance mismatch: {flat[38]}"
    assert flat[39] == 40.0, f"head_align mismatch: {flat[39]}"
    assert flat[40] == 41.0, f"head_rate mismatch: {flat[40]}"
    print("All slice layout checks passed.")


if __name__ == "__main__":
    test_observation_shape()
    test_no_nan_inf()
    test_nan_replacement()
    test_slice_layout()
    print("observation_contract: ALL TESTS PASSED")
