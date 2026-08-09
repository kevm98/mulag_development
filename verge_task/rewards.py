"""Reward for the sole-controller policy.

The tracking core is PPO2's, unchanged and imported rather than copied, so a
PPO2 and a PPO3 tracking number mean the same thing and are directly
comparable.  Two terms are added, both of which exist because the inner loop is
gone:

``limit_penalty``  -- the DLS clamped its setpoint into ``[learned_min,
learned_max]`` every step, so PPO2 could not drive the arm into a hard stop.
PPO3 can, and a termination-only signal arrives too late to shape anything.

``rate_penalty``   -- the DLS had ``max_desired_delta_m`` and
``action_smoothing``; those bounds are what kept p95 tool acceleration inside
the promotion gate.  The action chain still enforces them as hard limits, but a
policy that rides the limit continuously is not a policy that has learned to be
smooth, so the cost is also made visible in the objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mulag_plant.policy.rewards import (
    ResidualRewardTerms,
    compute_residual_reward,
)


@dataclass(frozen=True)
class SoleRewardTerms:
    reward: torch.Tensor
    core: ResidualRewardTerms
    limit_penalty: torch.Tensor
    rate_penalty: torch.Tensor
    acceleration_penalty: torch.Tensor
    tracking_guard: torch.Tensor
    error_norm: torch.Tensor
    limit_proximity: torch.Tensor


def compute_sole_reward(
    *,
    original_reference_tool_position: torch.Tensor,
    measured_tool_position: torch.Tensor,
    previous_error_norm: torch.Tensor,
    applied_action: torch.Tensor,
    previous_applied_action: torch.Tensor,
    head_down_alignment: torch.Tensor,
    cutter_clearance: torch.Tensor,
    valve_command: torch.Tensor,
    cylinder_headroom_low_m: torch.Tensor,
    cylinder_headroom_high_m: torch.Tensor,
    tool_acceleration_norm: torch.Tensor,
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
    limit_penalty_weight: float = -1.0,
    limit_margin_m: float = 0.02,
    rate_penalty_weight: float = -0.10,
    acceleration_weight: float = -0.20,
    acceleration_scale_mps2: float = 30.0,
    tracking_guard_weight: float = 0.0,
    tracking_guard_scale_m: float = 0.04,
) -> SoleRewardTerms:
    """Sole-controller reward: PPO2 tracking core plus inner-loop replacements.

    ``limit_proximity`` is 0 while every cylinder has at least ``limit_margin_m``
    of travel left in both directions and rises to 1 at the limit itself, so the
    penalty is silent through the normal workspace and only bites in the region
    the deleted controller used to make unreachable.
    """
    core = compute_residual_reward(
        original_reference_tool_position=original_reference_tool_position,
        measured_tool_position=measured_tool_position,
        previous_error_norm=previous_error_norm,
        filtered_action=applied_action,
        previous_filtered_action=previous_applied_action,
        head_down_alignment=head_down_alignment,
        cutter_clearance=cutter_clearance,
        valve_command=valve_command,
        tracking_std_m=tracking_std_m,
        tracking_weight=tracking_weight,
        progress_weight=progress_weight,
        action_magnitude_weight=action_magnitude_weight,
        action_rate_weight=action_rate_weight,
        head_penalty_weight=head_penalty_weight,
        ground_penalty_weight=ground_penalty_weight,
        saturation_penalty_weight=saturation_penalty_weight,
        head_threshold=head_threshold,
        ground_band_m=ground_band_m,
    )

    margin = max(float(limit_margin_m), 1.0e-6)
    headroom = torch.minimum(cylinder_headroom_low_m, cylinder_headroom_high_m)
    limit_proximity = torch.clamp(1.0 - headroom / margin, 0.0, 1.0)
    limit_penalty = limit_penalty_weight * limit_proximity.square().amax(dim=-1)

    rate = (applied_action - previous_applied_action).square().mean(dim=-1)
    rate_penalty = rate_penalty_weight * rate

    scale = max(float(acceleration_scale_mps2), 1.0e-6)
    normalized_acceleration = torch.clamp(tool_acceleration_norm / scale, max=2.0)
    acceleration_penalty = acceleration_weight * normalized_acceleration.square()

    guard_scale = max(float(tracking_guard_scale_m), 1.0e-6)
    normalized_error = torch.clamp(core.error_norm / guard_scale, max=2.0)
    tracking_guard = tracking_guard_weight * normalized_error.square()

    reward = (
        core.reward
        + limit_penalty
        + rate_penalty
        + acceleration_penalty
        + tracking_guard
    )
    return SoleRewardTerms(
        reward=reward,
        core=core,
        limit_penalty=limit_penalty,
        rate_penalty=rate_penalty,
        acceleration_penalty=acceleration_penalty,
        tracking_guard=tracking_guard,
        error_norm=core.error_norm,
        limit_proximity=limit_proximity,
    )
