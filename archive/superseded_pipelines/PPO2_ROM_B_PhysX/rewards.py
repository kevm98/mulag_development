"""ROM-B safety shaping aligned with deterministic promotion gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ROMBSafetyRewardTerms:
    reward: torch.Tensor
    head_margin_penalty: torch.Tensor
    clearance_margin_penalty: torch.Tensor
    acceleration_margin_penalty: torch.Tensor
    truck_reaction_margin_penalty: torch.Tensor
    hard_safety_penalty: torch.Tensor
    acceleration_event_penalty: torch.Tensor
    hard_safety_violation: torch.Tensor
    acceleration_violation: torch.Tensor


def _bounded_margin(
    value: torch.Tensor, *, soft: float, hard: float, lower_is_bad: bool
) -> torch.Tensor:
    width = max(abs(float(hard) - float(soft)), 1.0e-6)
    distance = float(soft) - value if lower_is_bad else value - float(soft)
    return torch.clamp(torch.relu(distance) / width, max=1.0)


def compute_rom_b_safety_reward(
    *,
    head_alignment: torch.Tensor,
    clearance_error_m: torch.Tensor,
    acceleration_mps2: torch.Tensor,
    additional_safety_violation: torch.Tensor,
    truck_reaction_m: torch.Tensor | None = None,
    truck_reaction_soft_m: float = 0.035,
    truck_reaction_hard_m: float = 0.050,
    truck_reaction_margin_weight: float = 0.0,
    head_soft: float = 0.985,
    head_hard: float = 0.980,
    clearance_soft_m: float = 0.180,
    clearance_hard_m: float = 0.200,
    acceleration_soft_mps2: float = 35.0,
    acceleration_hard_mps2: float = 40.0,
    head_margin_weight: float = -4.0,
    clearance_margin_weight: float = -4.0,
    acceleration_margin_weight: float = -1.0,
    hard_safety_event_weight: float = -20.0,
    acceleration_event_weight: float = -5.0,
    clearance_in_hard_safety_event: bool = True,
) -> ROMBSafetyRewardTerms:
    """Penalize the tail near and beyond the deterministic acceptance limits."""

    head_margin = _bounded_margin(
        head_alignment, soft=head_soft, hard=head_hard, lower_is_bad=True
    )
    clearance_abs = torch.abs(clearance_error_m)
    clearance_margin = _bounded_margin(
        clearance_abs,
        soft=clearance_soft_m,
        hard=clearance_hard_m,
        lower_is_bad=False,
    )
    acceleration_margin = _bounded_margin(
        acceleration_mps2,
        soft=acceleration_soft_mps2,
        hard=acceleration_hard_mps2,
        lower_is_bad=False,
    )
    # Truck reaction is already a termination (via `additional_safety_violation`)
    # and already a hard-event penalty, but until now it had no approach gradient
    # -- the policy was punished for crossing a line it was never pushed away
    # from. Measured 2026-07-25: `truck_raw_reaction` is the dominant remaining
    # termination cause (0.0143 vs 0.0050 for head alignment) and it *rises* with
    # plant stiffness, 0.008-0.019 at h<=0.9 to 0.031-0.036 at h>=1.0. See
    # reports/rom_b_step1_results.md.
    #
    # `truck_reaction_hard_m` mirrors the existing 0.05 m termination threshold
    # and does not move it; only the soft shoulder is new. Weight defaults to 0.0
    # so every existing artifact reproduces bit-exactly.
    if truck_reaction_m is None:
        truck_reaction_margin = torch.zeros_like(acceleration_mps2)
    else:
        truck_reaction_margin = _bounded_margin(
            truck_reaction_m,
            soft=truck_reaction_soft_m,
            hard=truck_reaction_hard_m,
            lower_is_bad=False,
        )
    hard_safety_violation = (
        head_alignment < float(head_hard)
    ) | additional_safety_violation.bool()
    if clearance_in_hard_safety_event:
        hard_safety_violation = hard_safety_violation | (
            clearance_abs > float(clearance_hard_m)
        )
    acceleration_violation = acceleration_mps2 > float(acceleration_hard_mps2)
    head_penalty = float(head_margin_weight) * head_margin.square()
    clearance_penalty = (
        float(clearance_margin_weight) * clearance_margin.square()
    )
    acceleration_penalty = (
        float(acceleration_margin_weight) * acceleration_margin.square()
    )
    truck_reaction_penalty = (
        float(truck_reaction_margin_weight) * truck_reaction_margin.square()
    )
    hard_penalty = (
        float(hard_safety_event_weight) * hard_safety_violation.float()
    )
    acceleration_event_penalty = (
        float(acceleration_event_weight) * acceleration_violation.float()
    )
    reward = (
        head_penalty
        + clearance_penalty
        + acceleration_penalty
        + truck_reaction_penalty
        + hard_penalty
        + acceleration_event_penalty
    )
    return ROMBSafetyRewardTerms(
        reward=torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0),
        head_margin_penalty=head_penalty.detach(),
        clearance_margin_penalty=clearance_penalty.detach(),
        acceleration_margin_penalty=acceleration_penalty.detach(),
        truck_reaction_margin_penalty=truck_reaction_penalty.detach(),
        hard_safety_penalty=hard_penalty.detach(),
        acceleration_event_penalty=acceleration_event_penalty.detach(),
        hard_safety_violation=hard_safety_violation.detach(),
        acceleration_violation=acceleration_violation.detach(),
    )
