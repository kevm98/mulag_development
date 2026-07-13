"""Stage B: Restructured reward with exponential-weighted progress,
action smoothness, and ungated truck-speed credit.

No motion/action penalties (they caused the hold-still collapse).  The truck
progress term uses exp(-error/decay) so credit is always nonzero: policies that
track well get rewarded for driving; policies that track poorly still get some
gradient pulling them toward better tracking AND driving.
"""

from __future__ import annotations

import torch


def compute_residual_reward(
    ee_pos_local: torch.Tensor,
    nominal_goal_local: torch.Tensor,
    *,
    error_scale: float = 20.0,
    truck_kmh: torch.Tensor | None = None,
    progress_weight: float = 1.5,
    progress_error_gate_m: float = 0.05,
    truck_max_kmh: float = 6.0,
    progress_decay_m: float = 0.10,
    action_smooth_weight: float = 0.01,
    current_residual_action: torch.Tensor | None = None,
    previous_residual_action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute residual reward with exponential-weighted progress and action smoothing.

    Returns (reward, error, extras_dict).
    """
    error = torch.linalg.norm(nominal_goal_local - ee_pos_local, dim=-1)  # FULL 3-D
    tracking_reward = -float(error_scale) * error

    # Stage B: ungated progress via exponential weight.
    # truck_credit in [0,1]: error=0 -> credit=1.0; error>>decay -> credit->0.
    # This gives smooth gradient: better tracking -> more truck reward.
    truck_credit = torch.exp(-error / max(float(progress_decay_m), 1.0e-6))
    progress_term = torch.zeros_like(tracking_reward)
    truck_reward = torch.zeros_like(tracking_reward)
    if truck_kmh is not None:
        truck_reward = truck_credit * (
            truck_kmh / max(float(truck_max_kmh), 1.0e-6)
        )
        progress_term = float(progress_weight) * truck_reward

    # Stage B: action smoothness penalty (discourages jitter)
    action_smooth_term = torch.zeros_like(tracking_reward)
    action_smooth_penalty = torch.zeros_like(tracking_reward)
    if current_residual_action is not None and previous_residual_action is not None:
        action_delta = torch.linalg.norm(
            current_residual_action[:, :2] - previous_residual_action[:, :2], dim=-1
        )
        action_smooth_penalty = float(action_smooth_weight) * action_delta
        action_smooth_term = -action_smooth_penalty

    reward = tracking_reward + progress_term + action_smooth_term
    reward = torch.nan_to_num(reward, nan=-10.0, posinf=-10.0, neginf=-10.0)

    extras = {
        "residual/tracking_reward": tracking_reward.detach().mean(),
        "residual/progress_reward": progress_term.detach().mean(),
        "residual/truck_credit": truck_credit.detach().mean(),
        "residual/action_smooth": action_smooth_penalty.detach().mean(),
    }
    return reward, error.detach(), extras
