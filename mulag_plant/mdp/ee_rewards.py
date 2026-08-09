from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EETrackingRewardTerms:
    reward: torch.Tensor
    tracking: torch.Tensor
    progress: torch.Tensor
    action_rate: torch.Tensor
    valve_penalty: torch.Tensor
    fd_penalty: torch.Tensor
    goal_error_norm: torch.Tensor
    valve_abs_mean: torch.Tensor


def compute_ee_tracking_reward(
    *,
    ee_pos_local: torch.Tensor,
    ee_goal_local: torch.Tensor,
    previous_goal_error_norm: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    valve_cmd: torch.Tensor,
    fd_delta_x: torch.Tensor,
    tracking_std_m: float,
    progress_weight: float,
    action_rate_weight: float,
    valve_weight: float,
    fd_delta_weight: float,
) -> EETrackingRewardTerms:
    """Compute EE-tracking reward with standard penalties."""
    goal_error = ee_goal_local - ee_pos_local
    goal_error_norm = torch.linalg.norm(goal_error, dim=-1)
    tracking = torch.exp(-goal_error_norm / max(float(tracking_std_m), 1.0e-6))
    progress = float(progress_weight) * (previous_goal_error_norm - goal_error_norm)
    action_rate = -float(action_rate_weight) * torch.mean(torch.square(action - previous_action), dim=-1)
    valve_abs_mean = torch.mean(torch.abs(valve_cmd), dim=-1)
    valve_penalty = -float(valve_weight) * valve_abs_mean
    fd_penalty = -float(fd_delta_weight) * torch.mean(torch.square(fd_delta_x), dim=-1)
    reward = tracking + progress + action_rate + valve_penalty + fd_penalty
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    return EETrackingRewardTerms(
        reward=reward,
        tracking=tracking.detach(),
        progress=progress.detach(),
        action_rate=action_rate.detach(),
        valve_penalty=valve_penalty.detach(),
        fd_penalty=fd_penalty.detach(),
        goal_error_norm=goal_error_norm.detach(),
        valve_abs_mean=valve_abs_mean.detach(),
    )

