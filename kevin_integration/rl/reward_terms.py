from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointReachRewardWeights:
    q_tracking: float = 3.0
    qdot: float = 0.25
    progress: float = 2.0
    action_smoothness: float = 0.0
    action_rate: float = 0.05
    action_magnitude: float = 0.02
    action_saturation: float = 0.5
    action_saturation_threshold: float = 0.90
    settle_tracking: float = 2.0
    joint1_settle_tracking: float = 3.0
    joint1_settle_error_deadband: float = 0.005
    valve_effort: float = 0.01
    torque_effort: float = 1.0e-6
    alive: float = 0.1
    termination: float = 5.0


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(value), dim=-1)


def normalize_quaternion_wxyz(
    quaternion: torch.Tensor,
    *,
    name: str,
    validate: bool = True,
) -> torch.Tensor:
    """Normalize a quaternion tensor whose final dimension is ``[w, x, y, z]``."""
    if quaternion.shape[-1] != 4:
        raise RuntimeError(f"{name} must have last dimension 4, got {quaternion.shape[-1]}")
    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if validate:
        if not bool(torch.isfinite(quaternion).all()):
            raise RuntimeError(f"{name} contains a non-finite quaternion")
        if bool(torch.any(quaternion_norm <= 1.0e-8)):
            raise RuntimeError(f"{name} contains a quaternion with norm <= 1e-8")
    return quaternion / quaternion_norm.clamp_min(1.0e-8)


def quaternion_geodesic_error(
    target_quaternion: torch.Tensor,
    current_quaternion: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Return sign-invariant quaternion geodesic distance in radians."""
    target_normalized = normalize_quaternion_wxyz(
        target_quaternion,
        name="target_quaternion",
        validate=validate,
    )
    current_normalized = normalize_quaternion_wxyz(
        current_quaternion,
        name="current_quaternion",
        validate=validate,
    )
    quaternion_dot = torch.sum(target_normalized * current_normalized, dim=-1)
    quaternion_dot_abs = torch.abs(quaternion_dot).clamp(min=0.0, max=1.0)
    return 2.0 * torch.acos(quaternion_dot_abs)


def final_pose_orientation_reward(
    orientation_error: torch.Tensor,
    orientation_tolerance_rad: float,
) -> torch.Tensor:
    """Dense orientation reward before application of its configured weight."""
    if orientation_tolerance_rad <= 0.0:
        raise ValueError("orientation_tolerance_rad must be positive")
    return torch.exp(
        -torch.square(orientation_error / float(orientation_tolerance_rad))
    )


def final_pose_within_tolerance(
    position_error_norm: torch.Tensor,
    orientation_error: torch.Tensor,
    ee_linear_speed: torch.Tensor,
    ee_angular_speed: torch.Tensor,
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    linear_speed_tolerance_mps: float,
    angular_speed_tolerance_radps: float,
) -> torch.Tensor:
    """Return the per-environment final-pose accuracy and settling predicate."""
    return (
        (position_error_norm <= position_tolerance_m)
        & (orientation_error <= orientation_tolerance_rad)
        & (ee_linear_speed <= linear_speed_tolerance_mps)
        & (ee_angular_speed <= angular_speed_tolerance_radps)
    )


def update_consecutive_success_counter(
    success_counter: torch.Tensor,
    within_final_pose: torch.Tensor,
) -> torch.Tensor:
    """Increment consecutive valid steps and reset immediately on any failure."""
    return torch.where(
        within_final_pose,
        success_counter + 1,
        torch.zeros_like(success_counter),
    )


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
    action: torch.Tensor | None = None,
    prev_q_error_mean: torch.Tensor | None = None,
    torque_effort_weight: float | None = None,
    q_tracking_joint_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    is_settle_phase: torch.Tensor | None = None,
    use_settle_tracking_reward: bool = False,
    use_joint1_settle_tracking_reward: bool = False,
) -> dict[str, torch.Tensor]:
    q_error = q_ref - q
    joint_weights = _as_joint_weights(q_error, q_tracking_joint_weights).expand_as(q_error)
    q_error_abs = torch.abs(q_error)
    q_error_weighted_per_joint = joint_weights * q_error_abs
    q_error_squared_per_joint = torch.square(q_error)
    q_error_mean_unweighted = torch.mean(q_error_abs, dim=-1)
    q_error_mean_weighted = torch.sum(joint_weights * q_error_abs, dim=-1) / torch.clamp(
        torch.sum(joint_weights, dim=-1), min=1.0e-8
    )
    q_tracking_reward_per_joint = -weights.q_tracking * joint_weights * q_error_squared_per_joint
    q_tracking = torch.sum(q_tracking_reward_per_joint, dim=-1)
    qdot_penalty = -weights.qdot * squared_norm(qdot)
    progress_reward = torch.zeros_like(q_tracking)
    if prev_q_error_mean is not None:
        progress_reward = weights.progress * (prev_q_error_mean - q_error_mean_weighted)
    if is_settle_phase is None:
        is_settle_phase_float = torch.zeros_like(q_tracking)
    else:
        is_settle_phase_float = is_settle_phase.float().reshape(-1)
    weighted_squared_q_error_mean = torch.sum(joint_weights * q_error_squared_per_joint, dim=-1) / torch.clamp(
        torch.sum(joint_weights, dim=-1), min=1.0e-8
    )
    settle_tracking_reward = torch.zeros_like(q_tracking)
    if use_settle_tracking_reward:
        settle_tracking_reward = -weights.settle_tracking * weighted_squared_q_error_mean * is_settle_phase_float
    joint1_error_excess = torch.clamp(q_error_abs[:, 0] - weights.joint1_settle_error_deadband, min=0.0)
    joint1_settle_tracking_reward = torch.zeros_like(q_tracking)
    if use_joint1_settle_tracking_reward:
        joint1_settle_tracking_reward = (
            -weights.joint1_settle_tracking * torch.square(joint1_error_excess) * is_settle_phase_float
        )
    action_rate_penalty = -weights.action_rate * torch.mean(torch.square(action_delta), dim=-1)
    action_smoothness_penalty = action_rate_penalty
    if weights.action_smoothness != 0.0:
        action_smoothness_penalty = -weights.action_smoothness * torch.mean(torch.square(action_delta), dim=-1)
        action_rate_penalty = action_smoothness_penalty
    if action is None:
        action_abs = torch.zeros_like(action_delta)
    else:
        action_abs = torch.abs(action)
    action_saturation_excess = torch.clamp(action_abs - weights.action_saturation_threshold, min=0.0)
    action_saturation_penalty = -weights.action_saturation * torch.mean(
        torch.square(action_saturation_excess), dim=-1
    )
    action_magnitude_penalty = -weights.action_magnitude * torch.mean(torch.square(action_abs), dim=-1)
    action_saturation_fraction = torch.mean(
        (action_abs > weights.action_saturation_threshold).float(), dim=-1
    )
    valve_effort_penalty = -weights.valve_effort * squared_norm(valve_cmd)
    effective_torque_effort = weights.torque_effort if torque_effort_weight is None else torque_effort_weight
    torque_effort_penalty = -effective_torque_effort * squared_norm(torque)
    alive = weights.alive * (1.0 - terminated.float())
    termination_penalty = -weights.termination * terminated.float()
    reward_total = (
        q_tracking
        + qdot_penalty
        + progress_reward
        + settle_tracking_reward
        + joint1_settle_tracking_reward
        + action_rate_penalty
        + action_magnitude_penalty
        + action_saturation_penalty
        + valve_effort_penalty
        + torque_effort_penalty
        + alive
        + termination_penalty
    )
    return {
        "reward_total": reward_total,
        "q_tracking": q_tracking,
        "q_tracking_reward_total": q_tracking,
        "q_tracking_reward_per_joint": q_tracking_reward_per_joint,
        "q_tracking_reward_joint_weights": joint_weights,
        "q_error_mean_unweighted": q_error_mean_unweighted,
        "q_error_mean_weighted": q_error_mean_weighted,
        "q_error_weighted_per_joint": q_error_weighted_per_joint,
        "joint1_q_error_abs": q_error_abs[:, 0],
        "joint1_q_tracking_contribution": q_tracking_reward_per_joint[:, 0],
        "joint1_action_abs": action_abs[:, 0] if action_abs.shape[-1] > 0 else torch.zeros_like(q_tracking),
        "qdot_penalty": qdot_penalty,
        "progress_reward": progress_reward,
        "is_settle_phase": is_settle_phase_float,
        "settle_tracking_reward": settle_tracking_reward,
        "joint1_settle_tracking_reward": joint1_settle_tracking_reward,
        "joint1_error_excess": joint1_error_excess,
        "joint1_settle_error_deadband": torch.full_like(q_tracking, float(weights.joint1_settle_error_deadband)),
        "rew_settle_tracking": torch.full_like(q_tracking, float(weights.settle_tracking)),
        "rew_joint1_settle_tracking": torch.full_like(q_tracking, float(weights.joint1_settle_tracking)),
        "action_smoothness_penalty": action_smoothness_penalty,
        "action_rate_penalty": action_rate_penalty,
        "action_magnitude_penalty": action_magnitude_penalty,
        "action_saturation_penalty": action_saturation_penalty,
        "action_saturation_fraction": action_saturation_fraction,
        "action_abs_mean": torch.mean(action_abs, dim=-1),
        "action_abs_max": torch.max(action_abs, dim=-1).values,
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
    action: torch.Tensor | None = None,
    prev_q_error_mean: torch.Tensor | None = None,
    torque_effort_weight: float | None = None,
    q_tracking_joint_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    is_settle_phase: torch.Tensor | None = None,
    use_settle_tracking_reward: bool = False,
    use_joint1_settle_tracking_reward: bool = False,
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
        action=action,
        prev_q_error_mean=prev_q_error_mean,
        torque_effort_weight=torque_effort_weight,
        q_tracking_joint_weights=q_tracking_joint_weights,
        is_settle_phase=is_settle_phase,
        use_settle_tracking_reward=use_settle_tracking_reward,
        use_joint1_settle_tracking_reward=use_joint1_settle_tracking_reward,
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
    is_settle_phase: torch.Tensor | None = None,
    use_settle_tracking_reward: bool = False,
    use_joint1_settle_tracking_reward: bool = False,
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
        action=action,
        prev_q_error_mean=prev_q_error_mean,
        torque_effort_weight=torque_effort_weight,
        q_tracking_joint_weights=q_tracking_joint_weights,
        is_settle_phase=is_settle_phase,
        use_settle_tracking_reward=use_settle_tracking_reward,
        use_joint1_settle_tracking_reward=use_joint1_settle_tracking_reward,
    )
    q_error_abs = torch.abs(q - q_ref)
    diagnostics = {
        "q_error_mean": terms["q_error_mean_weighted"],
        "q_error_abs_per_joint": q_error_abs,
        "qdot_mean": torch.mean(torch.abs(qdot), dim=-1),
        "qdot_abs_max": torch.max(torch.abs(qdot), dim=-1).values,
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
