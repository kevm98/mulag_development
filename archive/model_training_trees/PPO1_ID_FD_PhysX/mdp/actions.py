from __future__ import annotations

import torch


def sanitize_cylinder_action(actions: torch.Tensor, *, num_envs: int, action_dim: int = 4) -> torch.Tensor:
    """Return finite clipped PPO cylinder commands in [-1, 1]."""
    action = torch.nan_to_num(actions.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    expected = (int(num_envs), int(action_dim))
    if tuple(action.shape) != expected:
        raise RuntimeError(f"PPO action must have shape {expected}, got {tuple(action.shape)}.")
    return torch.clamp(action, -1.0, 1.0)

