"""Reward function for the PPO Policy.

Reward is computed against the ORIGINAL UNMODIFIED tool reference, not a
policy-shifted reference. This ensures the policy cannot game the reward
by shifting the corrected cylinder target away from the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResidualRewardTerms:
    reward: torch.Tensor
    tracking: torch.Tensor
    progress: torch.Tensor
    action_magnitude: torch.Tensor
    action_rate: torch.Tensor
    head_penalty: torch.Tensor
    ground_penalty: torch.Tensor
    saturation_penalty: torch.Tensor
    error_norm: torch.Tensor
    valve_saturation_fraction: torch.Tensor
    normal_tracking: torch.Tensor


def compute_residual_reward(
    *,
    original_reference_tool_position: torch.Tensor,
    measured_tool_position: torch.Tensor,
    previous_error_norm: torch.Tensor,
    filtered_action: torch.Tensor,
    previous_filtered_action: torch.Tensor,
    head_down_alignment: torch.Tensor,
    cutter_clearance: torch.Tensor,
    valve_command: torch.Tensor,
    tracking_std_m: float = 0.02,
    tracking_weight: float = 4.0,
    progress_weight: float = 0.5,
    action_magnitude_weight: float = -0.02,
    action_rate_weight: float = -0.05,
    head_penalty_weight: float = -2.0,
    ground_penalty_weight: float = -2.0,
    saturation_penalty_weight: float = -1.0,
    head_threshold: float = 0.995,
    ground_band_m: float = 0.10,
    valve_limit: float = 1.0,
    valve_epsilon: float = 0.01,
    signed_normal_error_m: torch.Tensor | None = None,
    normal_tracking_std_m: float = 0.03,
    normal_tracking_weight: float = 0.0,
) -> ResidualRewardTerms:
    r"""Compute Policy reward against the original tool reference.

    Definitions:
        tool_error = original_reference_tool_position - measured_tool_position
        error_norm = norm(tool_error)

        tracking  = tracking_weight * exp(-(error_norm / tracking_std_m)^2)
        progress  = progress_weight * clamp((previous_error_norm - error_norm) / 0.005, -1, 1)
        action_magnitude = action_magnitude_weight * mean(filtered_action^2)
        action_rate = action_rate_weight * mean((filtered_action - previous_filtered_action)^2)
        head_penalty = head_penalty_weight * relu((head_threshold - head_down_alignment) / 0.02)^2
        ground_penalty = ground_penalty_weight * relu((abs(cutter_clearance) - ground_band_m) / 0.02)^2
        valve_saturation_fraction = mean(abs(valve_command) >= valve_limit - valve_epsilon)
        saturation_penalty = saturation_penalty_weight * valve_saturation_fraction

        total_reward = tracking + progress + action_magnitude + action_rate
                     + head_penalty + ground_penalty + saturation_penalty
    """
    tool_error = original_reference_tool_position - measured_tool_position
    error_norm = torch.linalg.norm(tool_error, dim=-1)
    tracking = float(tracking_weight) * torch.exp(
        -(error_norm / max(float(tracking_std_m), 1.0e-6)) ** 2
    )
    progress = float(progress_weight) * torch.clamp(
        (previous_error_norm - error_norm) / 0.005, -1.0, 1.0
    )
    action_magnitude = float(action_magnitude_weight) * torch.mean(
        torch.square(filtered_action), dim=-1
    )
    action_rate = float(action_rate_weight) * torch.mean(
        torch.square(filtered_action - previous_filtered_action), dim=-1
    )
    head_deviation = torch.relu(
        (float(head_threshold) - head_down_alignment) / 0.02
    )
    head_penalty = float(head_penalty_weight) * torch.square(head_deviation)
    ground_deviation = torch.relu(
        (torch.abs(cutter_clearance) - float(ground_band_m)) / 0.02
    )
    ground_penalty = float(ground_penalty_weight) * torch.square(ground_deviation)
    valve_saturation_fraction = torch.mean(
        (torch.abs(valve_command) >= (float(valve_limit) - float(valve_epsilon))).float(),
        dim=-1,
    )
    saturation_penalty = float(saturation_penalty_weight) * valve_saturation_fraction
    if signed_normal_error_m is None:
        normal_tracking = torch.zeros_like(tracking)
    else:
        normal_tracking = float(normal_tracking_weight) * torch.exp(
            -(signed_normal_error_m / max(float(normal_tracking_std_m), 1.0e-6)) ** 2
        )
    reward = (
        tracking
        + progress
        + action_magnitude
        + action_rate
        + head_penalty
        + ground_penalty
        + saturation_penalty
        + normal_tracking
    )
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    return ResidualRewardTerms(
        reward=reward,
        tracking=tracking.detach(),
        progress=progress.detach(),
        action_magnitude=action_magnitude.detach(),
        action_rate=action_rate.detach(),
        head_penalty=head_penalty.detach(),
        ground_penalty=ground_penalty.detach(),
        saturation_penalty=saturation_penalty.detach(),
        error_norm=error_norm.detach(),
        valve_saturation_fraction=valve_saturation_fraction.detach(),
        normal_tracking=normal_tracking.detach(),
    )
