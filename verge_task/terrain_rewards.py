"""Reward for verge mowing.

Deliberately NOT PPO2's 3-D tool-tracking reward. That scores distance to a
stored trajectory point, which no longer exists here: the reference is the
sensed ground, and the objective is to hold the cutter a fixed height above
whatever the verge does.

Being explicit about comparability, because it matters for the write-up: PPO2's
promotion gate is a 3-D tracking error against a library path, and this is a cut
height error against sensed terrain. They are different quantities and a PPO3
verge number is NOT directly comparable to the 0.040 m PPO2-A gate. The safety
terms (head-down, tool acceleration, cylinder limits) are kept identical so
those, at least, remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def bounded_wrapped_yaw_reward(
    yaw_error_rad: torch.Tensor, *, weight: float = 0.5
) -> torch.Tensor:
    """Bounded yaw reward with useful gradient across the entire circle.

    A narrow Gaussian is effectively flat at the opposite heading: with the
    old 0.20 rad standard deviation and a measured 3.12 rad startup error its
    value and gradient both rounded to zero. This triangular wrapped-angle
    reward is ``weight`` at zero and zero at pi, with constant slope between.
    """
    wrapped = torch.atan2(torch.sin(yaw_error_rad), torch.cos(yaw_error_rad))
    normalized = torch.clamp(wrapped.abs() / math.pi, 0.0, 1.0)
    return float(weight) * (1.0 - normalized)


@dataclass(frozen=True)
class ContactSafetyRewardTerms:
    """The seven contact terms, kept separately so each can be logged."""

    total: torch.Tensor
    support: torch.Tensor
    contact_loss: torch.Tensor
    force_regulation: torch.Tensor
    excess_force: torch.Tensor
    hard_impact: torch.Tensor
    takeoff: torch.Tensor
    airborne_gap: torch.Tensor


def airborne_task_reward_bound(
    *,
    cut_tracking_weight: float,
    lateral_weight: float,
    cut_progress_weight: float,
    lateral_progress_weight: float,
) -> float:
    """Largest positive task reward an AIRBORNE environment can still collect.

    ``cut_tracking`` peaks at its weight, ``lateral`` at its weight, and both
    bounded progress terms at their magnitudes. The binary contact bonus is
    zero while airborne by construction, and both dense error terms are
    non-positive, so this is a strict upper bound.

    It exists to be asserted against: the contact-loss penalty must exceed it,
    or floating through a rise stays profitable and no amount of tuning the
    other terms will fix that.
    """
    return (
        abs(float(cut_tracking_weight))
        + abs(float(lateral_weight))
        + abs(float(cut_progress_weight))
        + abs(float(lateral_progress_weight))
    )


def compute_contact_safety_reward(
    *,
    contact: torch.Tensor,
    contact_acquired: torch.Tensor,
    contact_force_n: torch.Tensor,
    airborne_gap_m: torch.Tensor,
    cutter_vertical_velocity_mps: torch.Tensor,
    hard_impact: torch.Tensor,
    active: torch.Tensor,
    # Defaults mirror TerrainSolePolicyEnvCfg exactly, and
    # tests/test_contact_safety.py asserts they still do -- a signature default
    # that drifts from the config is a term that behaves one way under the
    # environment and another way under every unit test written against it.
    support_weight: float = 4.0,
    loss_weight: float = 20.0,
    force_weight: float = 2.0,
    force_target_n: float = 6000.0,
    force_std_n: float = 8000.0,
    excess_weight: float = 2.0,
    force_soft_limit_n: float = 20000.0,
    excess_max_ratio: float = 5.0,
    hard_impact_weight: float = 20.0,
    takeoff_weight: float = 4.0,
    takeoff_velocity_scale_mps: float = 0.10,
    takeoff_max_ratio: float = 5.0,
    weak_support_force_n: float = 500.0,
    gap_weight: float = 8.0,
    gap_scale_m: float = 0.005,
    gap_max_ratio: float = 4.0,
) -> ContactSafetyRewardTerms:
    """Price the CONTACT STATE, not just the contact bit.

    A binary contact bonus pays a violent impact exactly what it pays stable
    support, and the impact is what throws the tool off the verge one step
    later -- so raising that bonus buys more ejection, which is what
    ``terrain_contact_reward_weight`` 2.0 -> 4.0 measured. These seven terms
    separate the states that bonus conflates:

        r_support  = +w_support * contact
        r_loss     = -w_loss * acquired * (1 - contact)
        r_force    = +w_force * contact * exp(-((F - F_target)/F_std)^2)
        r_excess   = -w_excess * clamp(relu((F - F_soft)/F_soft), 0, m)^2
        r_hard     = -w_hard * hard_impact
        r_takeoff  = -w_takeoff * weak * clamp(relu(v_z)/v_scale, 0, m)^2
        r_gap      = -w_gap * clamp(gap/gap_scale, 0, m)^2

    ``r_force`` is a band, not a ramp: it peaks at ``force_target_n`` and falls
    off in BOTH directions, so pressing harder stops paying past the target
    instead of paying more. ``r_gap`` is continuous in the gap rather than a
    bonus that merely switches off, so a cutter 40 mm up is strictly worse than
    one 4 mm up -- with a binary term those two score identically and PPO has
    no gradient back down.

    Every term is bounded. ``active`` zeroes the whole group during
    deployment, preposition, reset correction and settling, where the arm is
    being driven by the reset ramp and none of this is attributable to the
    policy.
    """
    contact_f = contact.float()
    acquired_f = contact_acquired.float()
    force = torch.nan_to_num(contact_force_n, nan=0.0, posinf=0.0, neginf=0.0)

    support = float(support_weight) * contact_f

    contact_loss = -abs(float(loss_weight)) * acquired_f * (1.0 - contact_f)

    std = max(float(force_std_n), 1.0e-6)
    force_regulation = (
        float(force_weight)
        * contact_f
        * torch.exp(-((force - float(force_target_n)) / std).square())
    )

    soft = max(float(force_soft_limit_n), 1.0e-6)
    excess_ratio = torch.clamp(
        torch.relu((force - soft) / soft), max=max(float(excess_max_ratio), 0.0)
    )
    excess_force = -abs(float(excess_weight)) * excess_ratio.square()

    hard_impact_penalty = -abs(float(hard_impact_weight)) * hard_impact.float()

    # Weak support is the state an upward velocity is dangerous in: either no
    # contact at all, or contact so light it is about to be lost. Penalizing
    # upward motion while firmly supported would fight the tool following a
    # rising verge, which is the task.
    weak = ((~contact) | (force < float(weak_support_force_n))).float()
    v_scale = max(float(takeoff_velocity_scale_mps), 1.0e-6)
    takeoff_ratio = torch.clamp(
        torch.relu(cutter_vertical_velocity_mps) / v_scale,
        max=max(float(takeoff_max_ratio), 0.0),
    )
    takeoff = -abs(float(takeoff_weight)) * weak * takeoff_ratio.square()

    gap_ratio = torch.clamp(
        torch.relu(airborne_gap_m) / max(float(gap_scale_m), 1.0e-9),
        max=max(float(gap_max_ratio), 0.0),
    )
    airborne_gap = -abs(float(gap_weight)) * gap_ratio.square()

    active_f = active.float()
    terms = (
        support,
        contact_loss,
        force_regulation,
        excess_force,
        hard_impact_penalty,
        takeoff,
        airborne_gap,
    )
    scaled = tuple(term * active_f for term in terms)
    total = sum(scaled[1:], scaled[0])
    return ContactSafetyRewardTerms(
        total=total,
        support=scaled[0],
        contact_loss=scaled[1],
        force_regulation=scaled[2],
        excess_force=scaled[3],
        hard_impact=scaled[4],
        takeoff=scaled[5],
        airborne_gap=scaled[6],
    )


@dataclass(frozen=True)
class VergeRewardTerms:
    reward: torch.Tensor
    cut_tracking: torch.Tensor
    lateral: torch.Tensor
    smoothness: torch.Tensor
    head_penalty: torch.Tensor
    clearance_barrier: torch.Tensor
    contact_reward: torch.Tensor
    limit_penalty: torch.Tensor
    acceleration_penalty: torch.Tensor
    cut_height_error: torch.Tensor
    chassis_reaction_penalty: torch.Tensor | None = None


def sanitize_terminal_reward(
    reward: torch.Tensor,
    *,
    invalid_mask: torch.Tensor,
    terminal_reward: float = -100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace corrupt transition rewards with one finite terminal penalty.

    Clamping a reward term does not repair NaN because ``torch.clamp`` preserves
    it.  Return the combined bad-row mask as well as the scrubbed reward so the
    environment can carry an unexpected reward-only failure into its next done
    check.  Normal finite rewards are bit-preserved.
    """
    if reward.ndim != 1:
        raise ValueError(f"reward must be rank 1, got shape {tuple(reward.shape)}")
    invalid_mask = invalid_mask.to(device=reward.device, dtype=torch.bool)
    if invalid_mask.shape != reward.shape:
        raise ValueError(
            f"invalid mask shape {tuple(invalid_mask.shape)} != reward {tuple(reward.shape)}"
        )
    penalty = float(terminal_reward)
    if not math.isfinite(penalty):
        raise ValueError("terminal_reward must be finite")
    bad = invalid_mask | ~torch.isfinite(reward)
    finite_reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(bad, torch.full_like(reward, penalty), finite_reward), bad


def bounded_cost_progress(
    previous_cost: torch.Tensor,
    current_cost: torch.Tensor,
    *,
    scale: float,
    weight: float,
) -> torch.Tensor:
    """Reward one-step reduction of a non-negative task cost, with a hard cap.

    The terrain PPO collects only 32 steps per update while a production drive
    lasts 960 steps. Static occupancy rewards alone therefore give weak credit
    to the action that started a useful correction.  This transition term is
    positive when cost falls, negative when it rises, and bounded to
    ``[-abs(weight), +abs(weight)]`` so a reset or terrain edge cannot dominate
    the safety barriers.

    Deliberately do not scrub non-finite inputs here.  The environment's terminal
    sanitizer must see corruption rather than silently converting it into a
    plausible progress reward.
    """
    if previous_cost.shape != current_cost.shape:
        raise ValueError(
            "previous/current cost shapes differ: "
            f"{tuple(previous_cost.shape)} != {tuple(current_cost.shape)}"
        )
    safe_scale = max(float(scale), 1.0e-6)
    magnitude = abs(float(weight))
    direction = 1.0 if float(weight) >= 0.0 else -1.0
    normalized = torch.clamp(
        (previous_cost - current_cost) / safe_scale,
        min=-1.0,
        max=1.0,
    )
    return direction * magnitude * normalized


def lower_tail_weights(
    values: torch.Tensor,
    *,
    active_mask: torch.Tensor,
    fraction: float,
    multiplier: float,
) -> torch.Tensor:
    """Weight the lowest active fraction without changing non-tail rows."""
    if values.ndim != 1 or active_mask.shape != values.shape:
        raise ValueError("values and active_mask must be matching rank-1 tensors")
    weights = torch.ones_like(values)
    fraction = float(fraction)
    multiplier = float(multiplier)
    if fraction <= 0.0 or multiplier <= 1.0:
        return weights
    if fraction > 1.0 or not math.isfinite(multiplier):
        raise ValueError("tail fraction must be <= 1 and multiplier must be finite")
    eligible = torch.where(active_mask & torch.isfinite(values))[0]
    if eligible.numel() == 0:
        return weights
    count = max(1, int(math.ceil(fraction * int(eligible.numel()))))
    tail_local = torch.topk(values[eligible], count, largest=False).indices
    weights[eligible[tail_local]] = multiplier
    return weights


def update_failure_memory(
    memory: torch.Tensor,
    *,
    trigger: torch.Tensor,
    active_mask: torch.Tensor,
    horizon_steps: int,
) -> torch.Tensor:
    """Maintain a bounded per-environment countdown after a safety event."""
    if memory.ndim != 1 or trigger.shape != memory.shape or active_mask.shape != memory.shape:
        raise ValueError("memory, trigger, and active_mask must have matching rank-1 shapes")
    horizon = int(horizon_steps)
    if horizon <= 0:
        return torch.zeros_like(memory)
    updated = torch.where(active_mask, torch.clamp(memory - 1, min=0), memory)
    return torch.where(
        active_mask & trigger,
        torch.full_like(updated, horizon),
        updated,
    )


def compute_verge_reward(
    *,
    cut_height: torch.Tensor,
    target_cut_height_m: float,
    lateral_error: torch.Tensor,
    applied_action: torch.Tensor,
    previous_applied_action: torch.Tensor,
    head_down_alignment: torch.Tensor,
    cutter_contact: torch.Tensor,
    cylinder_headroom_low_m: torch.Tensor,
    cylinder_headroom_high_m: torch.Tensor,
    tool_acceleration_norm: torch.Tensor,
    scan_valid_fraction: torch.Tensor,
    cut_tracking_std_m: float = 0.03,
    cut_tracking_weight: float = 5.0,
    lateral_std_m: float = 0.30,
    lateral_weight: float = 0.5,
    action_magnitude_weight: float = -0.02,
    action_rate_weight: float = -0.10,
    head_penalty_weight: float = -2.0,
    head_threshold: float = 0.98,
    head_penalty_scale: float = 0.02,
    head_penalty_max_deviation: float = 2.0,
    head_penalty_far_weight: float = 0.0,
    clearance_barrier_weight: float = -2.0,
    clearance_margin_m: float = 0.02,
    clearance_barrier_scale_m: float = 0.02,
    clearance_barrier_max_deviation: float = 5.0,
    contact_reward_weight: float = 2.0,
    limit_penalty_weight: float = -1.0,
    limit_margin_m: float = 0.02,
    acceleration_weight: float = -0.20,
    acceleration_scale_mps2: float = 30.0,
    acceleration_max_ratio: float = 10.0,
    chassis_lateral_reaction_m: torch.Tensor | None = None,
    chassis_attitude_reaction_rad: torch.Tensor | None = None,
    chassis_reaction_weight: float = 0.0,
    chassis_lateral_scale_m: float = 0.005,
    chassis_attitude_scale_rad: float = 0.005,
    chassis_reaction_max_ratio: float = 5.0,
) -> VergeRewardTerms:
    """Hold the cutter at ``target_cut_height_m`` above the sensed verge.

    ``scan_valid_fraction`` gates the tracking term. Where the height scan saw
    nothing the measured cut height is a fallback constant, and rewarding the
    policy for matching a fallback would teach it to seek blind spots -- the
    exact shape of reward gaming this repository has been bitten by before.
    """
    error = cut_height - float(target_cut_height_m)

    # Only score tracking where the ground was actually measured.
    confidence = torch.clamp(scan_valid_fraction, 0.0, 1.0)
    cut_tracking = (
        cut_tracking_weight
        * confidence
        * torch.exp(-(error / max(cut_tracking_std_m, 1e-6)).square())
    )

    lateral = lateral_weight * torch.exp(
        -(lateral_error / max(lateral_std_m, 1e-6)).square()
    )

    magnitude = action_magnitude_weight * applied_action.square().mean(dim=-1)
    rate = action_rate_weight * (applied_action - previous_applied_action).square().mean(
        dim=-1
    )
    smoothness = magnitude + rate

    # Match PPO2's normalized quadratic safety barrier. The old unnormalized
    # linear term was only -0.165 at alignment=.9717, so gaining cut height by
    # tilting the head remained profitable against +7 of task reward.
    #
    # The quadratic is clamped so a flipped head cannot explode the value
    # targets, but the clamp alone makes the barrier a *plateau*: with the
    # production scale/max_deviation it saturates at alignment 0.955 and is then
    # exactly flat all the way to -1. Every inverted pose scores identically, so
    # PPO sees zero gradient in the region it must climb out of and a head that
    # flips once has no reason to come back. ``head_penalty_far_weight`` adds a
    # linear far-field term past the clamp: bounded slope, still finite at
    # alignment=-1, but strictly monotone so recovery is always rewarded.
    raw_head_deviation = torch.relu(
        (float(head_threshold) - head_down_alignment)
        / max(float(head_penalty_scale), 1.0e-6)
    )
    head_deviation = torch.clamp(
        raw_head_deviation, max=max(float(head_penalty_max_deviation), 0.0)
    )
    head_penalty = float(head_penalty_weight) * head_deviation.square() + float(
        head_penalty_far_weight
    ) * (raw_head_deviation - head_deviation)

    # Optional geometric guard.  The physical task uses a zero-height target and
    # real contact feedback, so production sets this weight to zero; retaining
    # the bounded term keeps the pure reward helper useful for safety screens.
    clearance_deviation = torch.relu(
        (float(clearance_margin_m) - cut_height)
        / max(float(clearance_barrier_scale_m), 1.0e-6)
    )
    clearance_deviation = torch.clamp(
        clearance_deviation,
        max=max(float(clearance_barrier_max_deviation), 0.0),
    )
    clearance_barrier = (
        float(clearance_barrier_weight) * clearance_deviation.square()
    )

    # Ground support is the requested operating state.  This is based on the
    # filtered PhysX cutter-to-verge interaction sensor, not a geometric proxy.
    contact_reward = contact_reward_weight * cutter_contact.float()

    margin = max(float(limit_margin_m), 1e-6)
    headroom = torch.minimum(cylinder_headroom_low_m, cylinder_headroom_high_m)
    limit_penalty = limit_penalty_weight * torch.clamp(
        1.0 - headroom / margin, 0.0, 1.0
    ).square().amax(dim=-1)

    scale = max(float(acceleration_scale_mps2), 1e-6)
    max_acceleration_ratio = max(float(acceleration_max_ratio), 0.0)
    acceleration_penalty = acceleration_weight * torch.clamp(
        tool_acceleration_norm / scale, min=0.0, max=max_acceleration_ratio
    ).square()

    # Arm-induced chassis reaction. The truck is a kinematic fixture, so its
    # pose error is corrected every substep and `truck_drift` therefore reads
    # 0.0000 -- which is why the visible sway was never in any metric. What the
    # fixture *cannot* hide is how hard it had to push: the pre-correction error
    # is exactly the reaction the arm fed into the chassis, and it is the only
    # part of the sway the policy can actually act on. Penalizing it trains the
    # arm to stop shoving the truck instead of relying on the fixture to mop up
    # afterwards, which is also what unloads a tyre (wheel_road_support).
    #
    # Lateral and attitude are kept as separate normalized channels because they
    # have different units and different scales; combining them into one metre
    # figure would silently weight 1 mm of slide against 1 mrad of roll.
    chassis_reaction_penalty = torch.zeros_like(cut_tracking)
    if float(chassis_reaction_weight) != 0.0:
        max_reaction_ratio = max(float(chassis_reaction_max_ratio), 0.0)
        channels = []
        if chassis_lateral_reaction_m is not None:
            channels.append(
                chassis_lateral_reaction_m / max(float(chassis_lateral_scale_m), 1e-6)
            )
        if chassis_attitude_reaction_rad is not None:
            channels.append(
                chassis_attitude_reaction_rad
                / max(float(chassis_attitude_scale_rad), 1e-6)
            )
        if channels:
            normalized = torch.stack(channels, dim=-1).abs().amax(dim=-1)
            chassis_reaction_penalty = float(chassis_reaction_weight) * torch.clamp(
                normalized, min=0.0, max=max_reaction_ratio
            ).square()

    reward = (
        cut_tracking
        + lateral
        + smoothness
        + head_penalty
        + clearance_barrier
        + contact_reward
        + limit_penalty
        + acceleration_penalty
        + chassis_reaction_penalty
    )
    return VergeRewardTerms(
        reward=reward,
        cut_tracking=cut_tracking,
        lateral=lateral,
        smoothness=smoothness,
        head_penalty=head_penalty,
        clearance_barrier=clearance_barrier,
        contact_reward=contact_reward,
        limit_penalty=limit_penalty,
        acceleration_penalty=acceleration_penalty,
        cut_height_error=error,
        chassis_reaction_penalty=chassis_reaction_penalty,
    )
