"""Horizon PPO action adapter.

PPO action[0:4] in [-1, 1] -> delta_x_des_H = action * action_scale_m.
"""
import torch

DEFAULT_ACTION_SCALE_M = 0.01


def ppo_action_to_horizon_displacement(action: torch.Tensor,
                                        action_scale_m: float = DEFAULT_ACTION_SCALE_M) -> torch.Tensor:
    """Map PPO action to desired cylinder horizon displacement."""
    action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    action = torch.clamp(action, -1.0, 1.0)
    if action.shape[-1] != 4:
        raise ValueError(f"PPO action must be 4D, got {action.shape[-1]}")
    return action * action_scale_m
