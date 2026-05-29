from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointReachRewardWeights:
    q_tracking: float = 8.0
    qdot: float = 0.25
    progress: float = 1.0
    action_smoothness: float = 0.05
    valve_effort: float = 0.01
    torque_effort: float = 1.0e-6
    alive: float = 0.1
    termination: float = 5.0


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(value), dim=-1)


def _as_joint_weights(
    q_error: torch.Tensor,
    q_tracking_joint_weights: torch.Tensor | list[float] | tuple[float, ...] | None,
) -> torch.Tensor:
    if q_tracking_joint_weights is None:
        return torch.ones_like(q_error)
    joint_weights = torch.as_tensor(q_tracking_joint_weights, device=q_error.device, dtype=q_error.dtype)
    if joint_weights.shape[-1] != q_error.shape[-1]:
        raise ValueError(
            f"q_tracking_joint_weights must have last dimension {q_error.shape[-1]}, got {joint_weights.shape[-1]}"
        )
    if joint_weights.dim() == 1:
        joint_weights = joint_weights.reshape(1, -1)
    return joint_weights


def _joint_reach_reward_components(
    q: torch.Tensor,
    qdot: torch.Tensor,
    q_ref: torch.Tensor,
    action_delta: torch.Tensor,
    valve_cmd: torch.Tensor,
    torque: torch.Tensor,
    terminated: torch.Tensor,
    weights: JointReachRewardWeights,
    *,
    prev_q_error_mean: torch.Tensor | None = None,
    torque_effort_weight: float | None = None,
    q_tracking_joint_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> dict[str, torch.Tensor]:
    q_error = q_ref - q
    joint_weights = _as_joint_weights(q_error, q_tracking_joint_weights)
    q_error_abs = torch.abs(q_error)
    q_error_mean_unweighted = torch.mean(q_error_abs, dim=-1)
    q_error_mean_weighted = torch.sum(joint_weights * q_error_abs, dim=-1) / torch.clamp(
        torch.sum(joint_weights, dim=-1), min=1.0e-8
    )
    q_tracking = -weights.q_tracking * torch.sum(joint_weights * torch.square(q_error), dim=-1)
    qdot_penalty = -weights.qdot * squared_norm(qdot)
    progress_reward = torch.zeros_like(q_tracking)
    if prev_q_error_mean is not None:
        progress_reward = weights.progress * (prev_q_error_mean - q_error_mean_weighted)
    action_smoothness_penalty = -weights.action_smoothness * squared_norm(action_delta)
    valve_effort_penalty = -weights.valve_effort * squared_norm(valve_cmd)
    effective_torque_effort = weights.torque_effort if torque_effort_weight is None else torque_effort_weight
    torque_effort_penalty = -effective_torque_effort * squared_norm(torque)
    alive = weights.alive * (1.0 - terminated.float())
    termination_penalty = -weights.termination * terminated.float()
    reward_total = (
        q_tracking
        + qdot_penalty
        + progress_reward
        + action_smoothness_penalty
        + valve_effort_penalty
        + torque_effort_penalty
        + alive
        + termination_penalty
    )
    return {
        "reward_total": reward_total,
        "q_tracking": q_tracking,
        "q_error_mean_unweighted": q_error_mean_unweighted,
        "q_error_mean_weighted": q_error_mean_weighted,
        "qdot_penalty": qdot_penalty,
        "progress_reward": progress_reward,
        "action_smoothness_penalty": action_smoothness_penalty,
        "valve_effort_penalty": valve_effort_penalty,
        "torque_effort_penalty": torque_effort_penalty,
        "alive": alive,
        "termination_penalty": termination_penalty,
    }


def joint_reach_reward(
    q: torch.Tensor,
    qdot: torch.Tensor,
    q_ref: torch.Tensor,
    action_delta: torch.Tensor,
    valve_cmd: torch.Tensor,
    torque: torch.Tensor,
    terminated: torch.Tensor,
    weights: JointReachRewardWeights,
    *,
    prev_q_error_mean: torch.Tensor | None = None,
    torque_effort_weight: float | None = None,
    q_tracking_joint_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    return _joint_reach_reward_components(
        q,
        qdot,
        q_ref,
        action_delta,
        valve_cmd,
        torque,
        terminated,
        weights,
        prev_q_error_mean=prev_q_error_mean,
        torque_effort_weight=torque_effort_weight,
        q_tracking_joint_weights=q_tracking_joint_weights,
    )["reward_total"]


def joint_reach_reward_debug(
    q: torch.Tensor,
    qdot: torch.Tensor,
    q_ref: torch.Tensor,
    action_delta: torch.Tensor,
    valve_cmd: torch.Tensor,
    torque: torch.Tensor,
    terminated: torch.Tensor,
    weights: JointReachRewardWeights,
    *,
    action: torch.Tensor | None = None,
    prev_q_error_mean: torch.Tensor | None = None,
    torque_effort_weight: float | None = None,
    q_tracking_joint_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms = _joint_reach_reward_components(
        q,
        qdot,
        q_ref,
        action_delta,
        valve_cmd,
        torque,
        terminated,
        weights,
        prev_q_error_mean=prev_q_error_mean,
        torque_effort_weight=torque_effort_weight,
        q_tracking_joint_weights=q_tracking_joint_weights,
    )
    q_error_abs = torch.abs(q - q_ref)
    diagnostics = {
        "q_error_mean": terms["q_error_mean_weighted"],
        "q_error_abs_per_joint": q_error_abs,
        "qdot_mean": torch.mean(torch.abs(qdot), dim=-1),
        "action_delta_mean": torch.mean(torch.abs(action_delta), dim=-1),
        "valve_mean": torch.mean(torch.abs(valve_cmd), dim=-1),
        "torque_mean": torch.mean(torch.abs(torque), dim=-1),
    }
    if action is not None:
        diagnostics["action_mean"] = torch.mean(torch.abs(action), dim=-1)
    terms.update(diagnostics)
    return terms["reward_total"], terms


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
