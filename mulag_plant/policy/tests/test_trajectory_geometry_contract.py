import torch

from mulag_plant.policy.observations import build_residual_observation, residual_observation_dim
from mulag_plant.policy.rewards import compute_residual_reward


def _base_kwargs(batch=2):
    return {
        "tool_error_truck": torch.zeros(batch, 3),
        "measured_tool_velocity_truck": torch.zeros(batch, 3),
        "reference_tool_velocity_truck": torch.zeros(batch, 3),
        "reference_tool_acceleration_truck": torch.zeros(batch, 3),
        "measured_cylinder_extensions": torch.zeros(batch, 4),
        "cylinder_goal_error": torch.zeros(batch, 4),
        "measured_cylinder_velocities": torch.zeros(batch, 4),
        "previous_filtered_policy_action": torch.zeros(batch, 4),
        "current_valve_commands": torch.zeros(batch, 4),
        "corrected_fd_delta": torch.zeros(batch, 4),
        "reference_preview_xy_truck": torch.zeros(batch, 2),
        "trajectory_direction": torch.ones(batch),
        "cutter_bottom_world_clearance": torch.zeros(batch),
        "head_down_alignment": torch.ones(batch),
        "head_down_alignment_rate": torch.zeros(batch),
    }


def test_geometry_channels_are_opt_in_and_widen_observation():
    kwargs = _base_kwargs()
    legacy = build_residual_observation(**kwargs)
    v2 = build_residual_observation(
        **kwargs,
        nearest_point_delta_xy=torch.tensor([[0.1, -0.2], [0.0, 0.03]]),
        signed_normal_error_m=torch.tensor([0.1, -0.03]),
    )
    assert legacy.shape[-1] == residual_observation_dim(4)
    assert v2.shape[-1] == residual_observation_dim(4, trajectory_geometry=True)
    assert torch.allclose(v2[:, -3:], torch.tensor([[0.1, -0.2, 0.1], [0.0, 0.03, -0.03]]))


def test_normal_reward_is_zero_for_on_path_and_positive_for_on_path():
    common = {
        "original_reference_tool_position": torch.zeros(2, 3),
        "measured_tool_position": torch.zeros(2, 3),
        "previous_error_norm": torch.zeros(2),
        "filtered_action": torch.zeros(2, 4),
        "previous_filtered_action": torch.zeros(2, 4),
        "head_down_alignment": torch.ones(2),
        "cutter_clearance": torch.zeros(2),
        "valve_command": torch.zeros(2, 4),
        "signed_normal_error_m": torch.tensor([0.0, 0.03]),
        "normal_tracking_std_m": 0.03,
        "normal_tracking_weight": 2.0,
    }
    terms = compute_residual_reward(**common)
    assert terms.normal_tracking[0] > terms.normal_tracking[1]
    assert terms.reward[0] > terms.reward[1]
