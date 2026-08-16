"""Ten bounded reward terms for tracking, contact, and arm safety."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _bounded(value: torch.Tensor, scale: float, cap: float) -> torch.Tensor:
    """Linear in ``value / scale``, clipped at ``cap``. Gradient is constant.

    This is the workhorse shape. Unlike a Gaussian it has the SAME gradient at
    0.5 m of error as at 0.05 m, so a policy that is far from the reference is
    still told which way to move.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return torch.clamp(value.abs() / scale, max=float(cap))


def _huber(value: torch.Tensor, scale: float, cap: float) -> torch.Tensor:
    """Quadratic near zero, linear far away, clipped at ``cap``.

    Used where fine precision matters near the target (cut height) but the
    far-field gradient must not vanish.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    x = value.abs() / scale
    quadratic = 0.5 * x.square()
    linear = x - 0.5
    return torch.clamp(torch.where(x <= 1.0, quadratic, linear), max=float(cap))


@dataclass(frozen=True)
class RewardWeights:
    """The published budget. Every field is documented in the module docstring."""

    cut_height: float = 6.0
    lateral: float = 12.0
    ground_contact: float = 4.0
    head_alignment: float = -12.0
    contact_loss: float = -3.0
    action_rate: float = -1.0
    tool_acceleration: float = -1.0
    contact_force: float = -1.0
    cylinder_headroom: float = -2.0
    hard_impact: float = -4.0

    cut_height_scale_m: float = 0.04
    cut_height_cap: float = 2.5
    lateral_scale_m: float = 1.50
    lateral_fine_scale_m: float = 0.60
    lateral_fine_fraction: float = 0.40
    head_threshold: float = 0.98
    head_scale: float = 0.02          # fine: sharp right below the threshold
    head_coarse_scale: float = 1.00   # coarse: alive down to alignment ~= -0.02
    head_fine_fraction: float = 0.50
    action_rate_scale: float = 0.25
    acceleration_scale_mps2: float = 300.0
    contact_force_limit_n: float = 30_000.0
    contact_force_scale_n: float = 150_000.0
    headroom_margin_m: float = 0.02
    headroom_scale_m: float = 0.10

    def task_total(self) -> float:
        return self.cut_height + self.lateral + self.ground_contact

    def safety_total(self) -> float:
        return -(
            self.head_alignment
            + self.contact_loss
            + self.action_rate
            + self.tool_acceleration
            + self.contact_force
            + self.cylinder_headroom
            + self.hard_impact
        )


@dataclass
class RewardTerms:
    """Per-term breakdown. Logged individually so a regression is attributable."""

    cut_height: torch.Tensor
    lateral: torch.Tensor
    ground_contact: torch.Tensor
    head_alignment: torch.Tensor
    contact_loss: torch.Tensor
    action_rate: torch.Tensor
    tool_acceleration: torch.Tensor
    contact_force: torch.Tensor
    cylinder_headroom: torch.Tensor
    hard_impact: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return (
            self.cut_height
            + self.lateral
            + self.ground_contact
            + self.head_alignment
            + self.contact_loss
            + self.action_rate
            + self.tool_acceleration
            + self.contact_force
            + self.cylinder_headroom
            + self.hard_impact
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "cut_height": self.cut_height,
            "lateral": self.lateral,
            "ground_contact": self.ground_contact,
            "head_alignment": self.head_alignment,
            "contact_loss": self.contact_loss,
            "action_rate": self.action_rate,
            "tool_acceleration": self.tool_acceleration,
            "contact_force": self.contact_force,
            "cylinder_headroom": self.cylinder_headroom,
            "hard_impact": self.hard_impact,
        }


def compute_reward(
    *,
    cut_height_error: torch.Tensor,
    lateral_error: torch.Tensor,
    head_down_alignment: torch.Tensor,
    cutter_contact: torch.Tensor,
    applied_action: torch.Tensor,
    previous_applied_action: torch.Tensor,
    tool_acceleration_norm: torch.Tensor,
    contact_force_n: torch.Tensor,
    cylinder_headroom_low_m: torch.Tensor,
    cylinder_headroom_high_m: torch.Tensor,
    hard_impact: torch.Tensor | None = None,
    weights: RewardWeights | None = None,
) -> RewardTerms:
    """Ten bounded terms. No exponentials, no unbounded penalties.

    Sign convention, enforced by tests: the three task terms are >= 0 and the
    six safety terms are <= 0. That is what makes worst-case (CVaR) weighting
    safe to add later -- scaling a non-positive term by >= 1 can only lower the
    return, whereas scaling a positive reward for the WORST environments pays
    the policy to be bad. Measured, in the old stack: weighting the positive
    contact reward 4x for the worst 25 % made quality 0.49 pay 7.84 against
    quality 0.50 paying 2.00, and collapsed worst-family contact in one block.
    """
    w = weights or RewardWeights()
    contact = cutter_contact.float()

    cut_height = w.cut_height * (
        1.0
        - _huber(cut_height_error, w.cut_height_scale_m, w.cut_height_cap)
        / w.cut_height_cap
    )
    coarse = 1.0 - _bounded(lateral_error, w.lateral_scale_m, 1.0)
    fine = 1.0 - _bounded(lateral_error, w.lateral_fine_scale_m, 1.0)
    lateral = w.lateral * (
        (1.0 - w.lateral_fine_fraction) * coarse + w.lateral_fine_fraction * fine
    )
    ground_contact = w.ground_contact * contact

    head_deficit = torch.clamp(w.head_threshold - head_down_alignment, min=0.0)
    head_alignment = w.head_alignment * (
        w.head_fine_fraction * _bounded(head_deficit, w.head_scale, 1.0)
        + (1.0 - w.head_fine_fraction)
        * _bounded(head_deficit, w.head_coarse_scale, 1.0)
    )

    contact_loss = w.contact_loss * (1.0 - contact)

    rate = (applied_action - previous_applied_action).abs().mean(dim=-1)
    action_rate = w.action_rate * _bounded(rate, w.action_rate_scale, 1.0)

    tool_acceleration = w.tool_acceleration * _bounded(
        tool_acceleration_norm, w.acceleration_scale_mps2, 1.0
    )

    excess_force = torch.clamp(contact_force_n.abs() - w.contact_force_limit_n, min=0.0)
    contact_force = w.contact_force * _bounded(
        excess_force, w.contact_force_scale_n, 1.0
    )

    headroom = torch.minimum(
        cylinder_headroom_low_m.min(dim=-1).values,
        cylinder_headroom_high_m.min(dim=-1).values,
    )
    encroachment = torch.clamp(w.headroom_margin_m - headroom, min=0.0)
    cylinder_headroom = w.cylinder_headroom * _bounded(
        encroachment, w.headroom_scale_m, 1.0
    )

    if hard_impact is None:
        hard_impact_term = torch.zeros_like(contact_force)
    else:
        hard_impact_term = w.hard_impact * hard_impact.to(contact_force.dtype)

    return RewardTerms(
        cut_height=cut_height,
        lateral=lateral,
        ground_contact=ground_contact,
        head_alignment=head_alignment,
        contact_loss=contact_loss,
        action_rate=action_rate,
        tool_acceleration=tool_acceleration,
        contact_force=contact_force,
        cylinder_headroom=cylinder_headroom,
        hard_impact=hard_impact_term,
    )
