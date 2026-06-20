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
    torque_effort: float = 0.0
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


def near_target_linear_settling_reward(
    position_error_norm: torch.Tensor,
    orientation_error: torch.Tensor,
    ee_linear_speed: torch.Tensor,
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    linear_speed_tolerance_mps: float,
    position_gate_scale: float,
    orientation_gate_scale: float,
    speed_margin_mps: float,
    weight: float,
) -> torch.Tensor:
    """Smooth near-pose reward that prefers linear speed below the strict limit."""
    if position_tolerance_m <= 0.0:
        raise ValueError("position_tolerance_m must be positive")
    if orientation_tolerance_rad <= 0.0:
        raise ValueError("orientation_tolerance_rad must be positive")
    if linear_speed_tolerance_mps <= 0.0:
        raise ValueError("linear_speed_tolerance_mps must be positive")
    if position_gate_scale <= 0.0:
        raise ValueError("position_gate_scale must be positive")
    if orientation_gate_scale <= 0.0:
        raise ValueError("orientation_gate_scale must be positive")
    if speed_margin_mps <= 0.0:
        raise ValueError("speed_margin_mps must be positive")
    position_gate = torch.exp(
        -torch.square(
            position_error_norm / (float(position_tolerance_m) * float(position_gate_scale))
        )
    )
    orientation_gate = torch.exp(
        -torch.square(
            orientation_error / (float(orientation_tolerance_rad) * float(orientation_gate_scale))
        )
    )
    speed_margin = (
        float(linear_speed_tolerance_mps) - ee_linear_speed
    ) / float(speed_margin_mps)
    return float(weight) * position_gate * orientation_gate * torch.tanh(speed_margin)


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


def bounded_loss(value: torch.Tensor) -> torch.Tensor:
    """Smooth bounded loss in ``[0, 1)`` for normalized excess signals."""
    return torch.tanh(torch.square(torch.clamp(value, min=0.0)))


def normalized_task_space_potential(
    position_error_norm: torch.Tensor,
    orientation_error: torch.Tensor,
    *,
    position_scale_m: float,
    orientation_scale_rad: float,
    position_weight: float,
    orientation_weight: float,
) -> torch.Tensor:
    """Negative normalized pose error used for potential-based progress shaping."""
    if position_scale_m <= 0.0:
        raise ValueError("position_scale_m must be positive")
    if orientation_scale_rad <= 0.0:
        raise ValueError("orientation_scale_rad must be positive")
    normalized_position_error = position_error_norm / float(position_scale_m)
    normalized_orientation_error = orientation_error / float(orientation_scale_rad)
    return -(
        float(position_weight) * normalized_position_error
        + float(orientation_weight) * normalized_orientation_error
    )


def potential_progress_shaping(
    phi_current: torch.Tensor,
    phi_next: torch.Tensor,
    *,
    gamma: float,
    weight: float = 1.0,
    reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``weight * (gamma * phi_next - phi_current)`` with reset leakage masked out."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    shaping = float(weight) * (float(gamma) * phi_next - phi_current)
    if reset_mask is not None:
        shaping = torch.where(reset_mask, torch.zeros_like(shaping), shaping)
    return shaping


def avoidable_clamp_penalty(
    boundary_active: torch.Tensor,
    outward_request: torch.Tensor,
    blocked_outward_increment: torch.Tensor,
    *,
    increment_scale: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize only avoidable outward requests at a boundary, not static occupancy."""
    if increment_scale <= 0.0:
        raise ValueError("increment_scale must be positive")
    outward_violation = boundary_active & outward_request
    normalized_outward_request = torch.abs(blocked_outward_increment) / float(increment_scale)
    per_channel_loss = outward_violation.float() * bounded_loss(normalized_outward_request)
    penalty = -float(weight) * torch.mean(per_channel_loss, dim=-1)
    activation_fraction = torch.mean(outward_violation.float(), dim=-1)
    mean_blocked_request = torch.mean(
        outward_violation.float() * torch.abs(blocked_outward_increment),
        dim=-1,
    )
    return penalty, activation_fraction, mean_blocked_request


def valve_saturation_penalty(
    applied_valve_cmd: torch.Tensor,
    *,
    sat_start: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft bounded penalty based on applied ValveCmd excess above ``sat_start``."""
    if not 0.0 <= sat_start < 1.0:
        raise ValueError("sat_start must be in [0, 1)")
    sat_excess = torch.relu(torch.abs(applied_valve_cmd) - float(sat_start)) / (1.0 - float(sat_start))
    penalty = -float(weight) * torch.mean(bounded_loss(sat_excess), dim=-1)
    saturation_fraction = torch.mean((sat_excess > 0.0).float(), dim=-1)
    return penalty, saturation_fraction


def fd_load_excess_penalty(
    fd_tau_hat_normalized: torch.Tensor,
    *,
    soft_threshold: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded penalty for normalized FD load only above an evidence-backed soft threshold."""
    if soft_threshold <= 0.0:
        raise ValueError("soft_threshold must be positive")
    fd_excess = torch.relu(torch.abs(fd_tau_hat_normalized) - float(soft_threshold)) / float(soft_threshold)
    penalty = -float(weight) * torch.mean(bounded_loss(fd_excess), dim=-1)
    excess_fraction = torch.mean((fd_excess > 0.0).float(), dim=-1)
    return penalty, excess_fraction


def hold_progress_reward(
    success_hold_counter: torch.Tensor,
    *,
    required_hold_steps: int,
    weight: float,
    within_final_pose: torch.Tensor,
) -> torch.Tensor:
    """Bounded monotonic reward for consecutive settled steps before strict success."""
    if required_hold_steps <= 0:
        raise ValueError("required_hold_steps must be positive")
    hold_fraction = torch.clamp(
        success_hold_counter.to(dtype=torch.float32) / float(required_hold_steps),
        min=0.0,
        max=1.0,
    )
    return float(weight) * within_final_pose.float() * torch.square(hold_fraction)


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
