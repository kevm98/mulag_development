"""PPO3_1 verge-mowing reward: ten bounded terms with a published budget.

WHY THIS EXISTS
===============
`verge_task/terrain_env_cfg.py` carries 53 weighted reward fields. Isaac
Lab's own locomotion task uses 11. At 53 nobody can reason about the optimum,
and three separate terms turned out to be fighting the task without anyone
noticing for thirteen training campaigns:

* `terrain_action_magnitude_weight = -1.0` on ``mean(action^2)`` charged for the
  SUSTAINED offsets that lane_change_hold, offset_return and boom_sweep
  structurally require. The policy correctly learned to return to neutral --
  the one thing "hold an offset" forbids. Removing it was the first training
  block in a full session that did not degrade (score 1.2058 -> 0.3678).
* The head group (-20 base, -8 far field, x2 tail, plus progress and memory
  terms) dwarfed lateral (3.0 + 3.0), so tracking could never compete.
* Three overlapping motion penalties -- magnitude, rate and acceleration --
  none of which distinguishes "follow the path" from "tilt the head".

THREE RULES, ALL LEARNED THE HARD WAY
=====================================
1. **No Gaussians.** The old reward used ``w * exp(-(err/std)^2)``, which is
   flat beyond ~2 sigma. Measured: at err = 0.50 m the gradient is -2.07 at
   sigma 0.30 and -0.004 at sigma 0.15 -- 520x weaker exactly where the failing
   families live. Every shaping term here is linear or Huber, so the gradient
   never dies where the policy is worst. (The repo had already hit this on the
   yaw reward and fixed it there, then left the same shape in lateral.)

2. **Every term is bounded.** A term that can grow without limit can dominate
   the sum and destabilise the critic. Each contribution below is clipped to a
   stated maximum, so the whole reward lives in a known range and the weights
   mean what they say.

3. **A clip is a Gaussian tail with sharper edges.** Rule 2 was implemented by
   clamping at ``cap = 1.0``, i.e. at ONE scale unit, and that silently
   reintroduced rule 1's defect. Run 2026-08-05_11-51-48, 888 iterations:

       iterations     mean lateral error     fraction under 0.30 m
        0- 200            0.217                     83 %
       200- 400            0.440                     56 %
       400- 600            0.392                     45 %
       600- 888            0.736                     11 %

   ``lateral_scale_m`` was 0.30 with ``cap`` 1.0, so the term saturated at
   0.30 m and its derivative was IDENTICALLY ZERO beyond it. The policy settled
   at 0.734 m -- 2.4x past the cliff -- and ``Episode_Reward/lateral`` logged
   0.0000. It was not failing to learn; it was receiving no signal at all.
   Meanwhile cut height (0.0006 m), head (0.9993) and contact (0.86) all trained
   to well inside their gates, because their gradients were alive.

   So every shaping term must keep a live gradient out to at least **2x its
   acceptance gate**, not to one scale unit. ``scripts/preflight.py`` asserts
   this numerically (check S5) against the thresholds in ``gates.py``, and it is
   the check that would have saved 3.6 GPU-hours. Note the unit test at
   ``test_lateral_gradient_survives_where_the_old_reward_died`` probed
   0.05/0.20/0.30 -- it stopped exactly ON the cliff edge and passed.

THE BUDGET
==========
Positive terms pay for doing the job; negative terms charge for doing it badly.
The budget is published so a reader can see at a glance what the policy is
actually being asked to trade:

    term                weight   max |contribution|   sign
    cut_height            6.0            6.0          +
    lateral              12.0           12.0          +
    ground_contact        4.0            4.0          +
    head_alignment      -12.0           12.0          -
    contact_loss         -3.0            3.0          -
    action_rate          -1.0            1.0          -
    tool_acceleration    -1.0            1.0          -
    contact_force        -1.0            1.0          -
    cylinder_headroom    -2.0            2.0          -
    ------------------------------------------------------
    task total (+)       22.0
    safety total (-)     20.0

Head is deliberately the largest single penalty because head-down is a
precondition for the task rather than a competing objective -- but at 12.0 it is
the same order as the 22.0 of task reward, not 3x it. That ratio is the thing
the old reward got wrong.

WEIGHTS ARE NOT MARGINAL RATES -- COMPARE THE GRADIENTS
=======================================================
The bug this section used to describe was reasoned about in the wrong currency.
A weight only equals a marginal rate for a term that is LINEAR in its metric.
``ground_contact`` (4*c) and ``contact_loss`` (-3*(1-c)) are, so contact's
marginal rate is exactly the sum of their weights. ``lateral`` is not: its
weight is split across scales and divided by each, so at weight 10.0 with the
fine scale at 0.15 m its marginal rate beyond 0.15 m was 10 * 0.6/1.50 = 4.0
per metre, not the 10.0 the old text compared against contact.

Measured twice, both with a HEALTHY critic (explained variance 0.90-0.955, so
neither run was advantage-noise limited):

    run             lateral weight   fine scale   lateral error      total reward
    warmup_prod_v5       10.0          0.15 m    0.189 -> 0.639 m   10.49 -> 11.70
    rebalanced           12.0          0.60 m    0.159 -> 0.334 m   12.49 -> 13.94

Both times total reward ROSE while tracking decayed: PPO was maximising this
reward correctly and the reward's optimum simply was not where the gate is.
Contact's predicted gain in v5, 10.0 * 0.292, matched the measured +2.921 to
three digits, which is what identifies the marginal rate as the mechanism.

WHY CONTACT_LOSS DROPPED FROM -6.0 TO -3.0
==========================================
The old text kept it at -6.0 with this reason: "Contact measured 0.86 against a
0.85 gate -- too thin a margin to weaken the term that earned it." **That gate
no longer exists.** ``ground_contact`` was re-derived from 0.850 to 0.700 in
gates.py against the achievable band measured on the privileged DLS teacher,
because 0.850 was above anything this plant had ever produced. The reason for
protecting contact was a margin against a threshold that has since moved 0.15
away from it -- the student sits at 0.7076 and PPO drives contact to 0.72-0.89
without being asked to. Contact is no longer the binding gate; lateral is, and
it is the only gate the teacher itself fails.

So the rebalance takes it out of contact rather than adding more to lateral.
Contact's marginal rate falls 10.0 -> 7.0 per unit contact fraction, against a
measured effective lateral rate of ~9.0 per metre (below the 12.8 nominal
because the error distribution straddles the 0.60 m knee). Raising lateral again
instead would have been tuning a number until it won; this removes a weight
whose stated justification expired.

NOT INCLUDED, ON PURPOSE
========================
* action MAGNITUDE -- it charges for holding an offset, which is the task.
* progress/shaping terms on every quantity -- they added gradient noise of order
  0.01 against terms of order 5.
* a separate "far field" head term -- the bounded linear head penalty already
  keeps a gradient at every alignment.
"""

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
    # 12.0, not 10.0. At 10.0 the corrected two-scale gradient is 10.67 per
    # metre against contact's 10.0 -- a 6.7% margin, which is inside the error
    # of every estimate feeding this. Widening `lateral_fine_scale_m` cannot buy
    # margin (a wider fine scale is a SHALLOWER one: 0.75 m would drop the
    # gradient to 9.33 and lose to contact again), so the margin has to come
    # from the weight. At 12.0 the operating-band gradient is 12.8 per metre,
    # 28% clear of contact. Budget: task total 22.0 against safety 23.0, still
    # the same order rather than the 3x mismatch the old reward had.
    lateral: float = 12.0
    ground_contact: float = 4.0
    head_alignment: float = -12.0
    # -3.0, not -6.0. See "WHY CONTACT_LOSS DROPPED" above: the -6.0 was held
    # in place to protect a 0.850 contact gate that gates.py has since
    # re-derived to 0.700. Contact's marginal rate is ground_contact +
    # |contact_loss| = 7.0, now below lateral's measured effective rate.
    contact_loss: float = -3.0
    action_rate: float = -1.0
    tool_acceleration: float = -1.0
    contact_force: float = -1.0
    cylinder_headroom: float = -2.0

    # Scales: the error at which a term reaches one unit of its weight.
    #
    # Lateral is deliberately TWO-SCALE. A single scale cannot be both precise
    # near the line and non-vanishing far from it, and picking one costs the
    # other: at 0.30 the gradient died at 0.30 m (see rule 3), while a single
    # coarse scale would blunt the fine tracking that decides the 0.300 m gate.
    #   * coarse (1.50 m) keeps a live gradient across the whole reachable range
    #     -- the failed run wandered to 0.93 m, so 1.50 leaves real headroom.
    #   * fine (0.15 m) supplies the sharp gradient that actually closes the
    #     last few centimetres, at half the gate threshold.
    cut_height_scale_m: float = 0.04
    # Saturates at scale * (cap + 0.5) = 0.12 m = 3x the 0.040 gate. The old
    # cap of 1.0 saturated at 0.06 m, only 1.5x the gate: the same latent cliff
    # as lateral, merely not yet reached because cut height trains to 0.0006 m.
    cut_height_cap: float = 2.5
    lateral_scale_m: float = 1.50
    # 0.60, not 0.15. The fine term exists to sharpen the gradient near zero,
    # but at 0.15 m it saturated at HALF the 0.300 gate -- the same
    # dies-before-the-gate defect this file documents for the coarse term,
    # hiding in the sub-term. Beyond 0.15 m only the coarse term survived, so
    # lateral's marginal value was a flat 10 * 0.6/1.50 = 4.0 per metre, while
    # contact's is 10.0 per unit contact fraction (ground_contact 4*c minus
    # contact_loss 6*(1-c) => slope 10). PPO correctly took that trade.
    #
    # Measured, run warmup_prod, 448 iterations after unfreeze with a HEALTHY
    # critic (explained variance 0.955, so this was not an advantage-noise
    # problem):
    #     lateral        6.720 -> 4.196   (-2.524)  error 0.189 -> 0.639 m
    #     ground_contact 2.380 -> 3.549   (+1.168)  contact 0.595 -> 0.887
    #     contact_loss  -2.429 -> -0.677  (+1.753)
    #     TOTAL         10.490 -> 11.702  (+1.212)  <- reward went UP
    # Contact's predicted gain 10 * 0.292 = +2.92 matches the measured +2.921
    # to three digits, which is what identifies the mechanism as the marginal
    # rate rather than anything stochastic.
    #
    # At 0.60 (= 2x the gate, the rule this file already states) the fine term
    # stays alive across the whole operating band and the mid-range gradient
    # becomes 10 * (0.6/1.50 + 0.4/0.60) = 10.67 per metre, which exceeds
    # contact's 10.0 and so reverses the trade. The cost is a shallower
    # gradient inside 0.15 m (30.7 -> 10.67 per metre); that is the right
    # trade here, because the failure is at 0.6 m, not at 0.05 m.
    lateral_fine_scale_m: float = 0.60
    lateral_fine_fraction: float = 0.40
    head_threshold: float = 0.98
    # TWO-SCALE, for the same reason lateral is. `head_scale` alone at 0.02
    # saturated the penalty at alignment 0.96 while the gate is 0.900, so the
    # ENTIRE failing range carried zero gradient: measured on model_850, p01
    # alignment 0.666 and min 0.237 both returned gradient exactly 0.000. The
    # policy paid a flat -12.0 whether the head was at 0.95 or 0.24, which made
    # collapsing the head a FREE way to buy ground contact -- and it did.
    head_scale: float = 0.02          # fine: sharp right below the threshold
    head_coarse_scale: float = 1.00   # coarse: alive down to alignment ~= -0.02
    head_fine_fraction: float = 0.50
    action_rate_scale: float = 0.25
    # Saturated at 60 m/s^2 against a 120 gate -- dead across the whole failing
    # range. 300 keeps gradient past 2x the gate.
    acceleration_scale_mps2: float = 300.0
    contact_force_limit_n: float = 30_000.0
    # Saturated at 2x the limit (60 kN) while this rigid cutter measures 123 kN,
    # so every real impact sat in the flat region.
    contact_force_scale_n: float = 150_000.0
    headroom_margin_m: float = 0.02
    # Saturated the moment headroom hit 0, so driving PAST a travel limit cost
    # no more than touching it.
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

    # --- task ---------------------------------------------------------------
    # Huber: precise near zero (cut height is the mowing quality), linear far
    # away so a badly-placed cutter still knows which way is better. Normalised
    # by the cap so the contribution stays in [0, w.cut_height] however far out
    # the cap is pushed -- that is what lets the cliff sit at 3x the gate
    # instead of 1.5x without breaking the published bound.
    cut_height = w.cut_height * (
        1.0
        - _huber(cut_height_error, w.cut_height_scale_m, w.cut_height_cap)
        / w.cut_height_cap
    )
    # Two linear scales, blended. The coarse one guarantees a non-zero gradient
    # everywhere the arm can physically get to; the fine one makes the last few
    # centimetres worth chasing. Convex blend, so the total stays in
    # [0, w.lateral] and the sign convention below still holds.
    #
    # Gradient, measured at the weights above:
    #     0.05 m -> 30.7 /m      (coarse 4.0 + fine 26.7)
    #     0.30 m ->  4.0 /m      <- was EXACTLY 0.0 before this fix
    #     0.73 m ->  4.0 /m      <- where the failed run sat, also 0.0 before
    #     1.50 m ->  0.0 /m      (beyond any reachable error)
    coarse = 1.0 - _bounded(lateral_error, w.lateral_scale_m, 1.0)
    fine = 1.0 - _bounded(lateral_error, w.lateral_fine_scale_m, 1.0)
    lateral = w.lateral * (
        (1.0 - w.lateral_fine_fraction) * coarse + w.lateral_fine_fraction * fine
    )
    ground_contact = w.ground_contact * contact

    # --- safety -------------------------------------------------------------
    # Bounded linear in the alignment deficit. No far-field term is needed
    # because this never saturates into a flat plateau.
    head_deficit = torch.clamp(w.head_threshold - head_down_alignment, min=0.0)
    head_alignment = w.head_alignment * (
        w.head_fine_fraction * _bounded(head_deficit, w.head_scale, 1.0)
        + (1.0 - w.head_fine_fraction)
        * _bounded(head_deficit, w.head_coarse_scale, 1.0)
    )

    contact_loss = w.contact_loss * (1.0 - contact)

    # RATE, not magnitude: a held offset has zero rate and so costs nothing,
    # while the jerky motion that tilts the head still costs.
    rate = (applied_action - previous_applied_action).abs().mean(dim=-1)
    action_rate = w.action_rate * _bounded(rate, w.action_rate_scale, 1.0)

    tool_acceleration = w.tool_acceleration * _bounded(
        tool_acceleration_norm, w.acceleration_scale_mps2, 1.0
    )

    excess_force = torch.clamp(contact_force_n.abs() - w.contact_force_limit_n, min=0.0)
    contact_force = w.contact_force * _bounded(
        excess_force, w.contact_force_scale_n, 1.0
    )

    # One barrier for both ends of every cylinder's travel.
    headroom = torch.minimum(
        cylinder_headroom_low_m.min(dim=-1).values,
        cylinder_headroom_high_m.min(dim=-1).values,
    )
    encroachment = torch.clamp(w.headroom_margin_m - headroom, min=0.0)
    cylinder_headroom = w.cylinder_headroom * _bounded(
        encroachment, w.headroom_scale_m, 1.0
    )

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
    )
