from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointReachRewardWeights:
    q_tracking: float = 8.0
    qdot: float = 0.25
    action_smoothness: float = 0.05
    valve_effort: float = 0.01
    torque_effort: float = 1.0e-6
    alive: float = 0.1
    termination: float = 5.0


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(value), dim=-1)


def joint_reach_reward(
    q: torch.Tensor,
    qdot: torch.Tensor,
    q_ref: torch.Tensor,
    action_delta: torch.Tensor,
    valve_cmd: torch.Tensor,
    torque: torch.Tensor,
    terminated: torch.Tensor,
    weights: JointReachRewardWeights,
) -> torch.Tensor:
    tracking = -weights.q_tracking * squared_norm(q_ref - q)
    damping = -weights.qdot * squared_norm(qdot)
    smoothness = -weights.action_smoothness * squared_norm(action_delta)
    valve_penalty = -weights.valve_effort * squared_norm(valve_cmd)
    torque_penalty = -weights.torque_effort * squared_norm(torque)
    alive = weights.alive * (1.0 - terminated.float())
    termination = -weights.termination * terminated.float()
    return tracking + damping + smoothness + valve_penalty + torque_penalty + alive + termination


def joint_safety_termination(
    q: torch.Tensor,
    qdot: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    *,
    joint_limit_margin: float,
    max_abs_joint_velocity: float,
) -> torch.Tensor:
    finite = torch.isfinite(q).all(dim=-1) & torch.isfinite(qdot).all(dim=-1)
    lower_violation = torch.any(q < lower_limits - joint_limit_margin, dim=-1)
    upper_violation = torch.any(q > upper_limits + joint_limit_margin, dim=-1)
    velocity_violation = torch.any(torch.abs(qdot) > max_abs_joint_velocity, dim=-1)
    return (~finite) | lower_violation | upper_violation | velocity_violation
