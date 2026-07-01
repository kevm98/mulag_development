"""Horizon PPO reward adapter.

Borrowed from task-space tracking branch, adapted to horizon level.
"""
import torch


def compute_horizon_reward(action: torch.Tensor) -> torch.Tensor:
    """Compute simple horizon reward: alive + action smoothness.

    Full reward shaping (EE tracking progress, success bonus, settling,
    hydraulic consistency, clipping/oscillation penalties) to be added
    after initial training converges.
    """
    B = action.shape[0]
    alive = torch.ones(B, device=action.device) * 0.01
    action_penalty = -0.01 * (action ** 2).sum(dim=-1)
    return alive + action_penalty
