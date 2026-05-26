from __future__ import annotations

import torch


def expand_action_to_id(action_4d: torch.Tensor) -> torch.Tensor:
    if action_4d.shape[-1] != 4:
        raise ValueError(f"action_4d must have last dimension 4, got {action_4d.shape[-1]}")
    zero_tool_joint = torch.zeros_like(action_4d[..., :1])
    return torch.cat([action_4d, zero_tool_joint], dim=-1)


def clamp_and_sanitize(value: torch.Tensor, max_abs: float | None = None) -> torch.Tensor:
    if max_abs is None:
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(value, nan=0.0, posinf=max_abs, neginf=-max_abs).clamp(-max_abs, max_abs)


def torque_to_joint_effort(torque_4d: torch.Tensor, state_joint_count: int = 5) -> torch.Tensor:
    if torque_4d.shape[-1] != 4:
        raise ValueError(f"torque_4d must have last dimension 4, got {torque_4d.shape[-1]}")
    if state_joint_count < 4:
        raise ValueError("state_joint_count must be at least 4")
    effort = torch.zeros(*torque_4d.shape[:-1], state_joint_count, dtype=torque_4d.dtype, device=torque_4d.device)
    effort[..., :4] = torque_4d
    return effort
