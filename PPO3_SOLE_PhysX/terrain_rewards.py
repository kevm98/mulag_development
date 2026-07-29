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
    head_deviation = torch.relu(
        (float(head_threshold) - head_down_alignment)
        / max(float(head_penalty_scale), 1.0e-6)
    )
    head_deviation = torch.clamp(
        head_deviation, max=max(float(head_penalty_max_deviation), 0.0)
    )
    head_penalty = float(head_penalty_weight) * head_deviation.square()

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

    reward = (
        cut_tracking
        + lateral
        + smoothness
        + head_penalty
        + clearance_barrier
        + contact_reward
        + limit_penalty
        + acceleration_penalty
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
    )
