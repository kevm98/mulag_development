"""Demonstration-regularized PPO: the teacher stays in the loop after DAgger.

The previous formulation ended imitation before PPO began -- distil, then
fine-tune on reward alone -- and that is the configuration this repository
measured regressing seven times. The difference here is that the privileged DLS
teacher is queried on states the CURRENT POLICY visits, throughout training, so
demonstrations keep stabilising the actor in states PPO wanders into rather than
only in states the teacher itself chose at t=0.

    L(theta) = L_PPO(theta) + lambda_k * L_demo(theta) + beta_k * KL(pi_theta || pi_0)

WHY THE IMITATION MUST BE FILTERED
==================================
An unfiltered demo loss makes the teacher a performance ceiling. On this task it
is worse than that -- it would drag the policy BACKWARDS. Measured 2026-08-10,
6 held-out meshes, 32 envs, on the shipped widened task:

    worst-family lateral error   teacher 0.1039   ROM v3 0.0809   ID/FD v3 0.0456

Both distilled students already beat the privileged teacher on the primary
objective, so pulling them toward it on every state would cost tracking on the
metric the contract scores. Filtering is therefore load-bearing, not a
refinement: imitate only where the teacher is actually the better authority.

THE FILTER, AND WHY IT IS NOT THE OBVIOUS ONE
=============================================
The natural filter is advantage-based, w_t = 1[Q(o,a_teacher) > Q(o,a_pi) + m].
That is not computable here: rsl_rl's PPO carries a STATE-value critic V(o), not
Q(o,a), and obtaining Q(o,a_teacher) counterfactually would require executing
the teacher action to see its return. Two filters are provided instead:

  "state"   imitate where the POLICY IS WEAK, by a task signal (tracking error,
            lost contact). Needs no new critic and no extra rollout. Loses the
            strict "never pull back a better action" guarantee but keeps the
            intent: demonstrations anchor poorly-explored and safety-critical
            states, not states the policy already handles.

  "value"   imitate where the realised one-step TD advantage of the executed
            action is NEGATIVE, i.e. the step went worse than V(o) predicted.
            Uses only quantities PPO already computes. A proxy for "the policy
            is not doing well here", still not a counterfactual on the teacher.

Start with "state". Move to a beta-mixed execution comparison -- run a fraction
of envs teacher-driven and compare achieved per-family return -- once the
teacher-execute path is wired; that measures the thing the advantage filter was
approximating, without needing a Q critic.

NOTHING HERE TOUCHES EVALUATION. `assert_teacher_disabled` guards the scored
paths and must keep doing so: a reported number may never have teacher
influence. This module is training-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DemoScheduleCfg:
    """Annealing for the demonstration and trust-region weights.

    Both decay only when a block is ACCEPTED. Decaying on wall-clock or
    iteration count would relax the anchor while the policy is still failing,
    which is the opposite of what the anchor is for.
    """

    lambda0: float = 1.0
    beta0: float = 0.05
    lambda_decay: float = 0.85
    beta_decay: float = 0.90
    lambda_min: float = 0.0
    beta_min: float = 0.0

    def at(self, accepted_blocks: int) -> tuple[float, float]:
        k = max(int(accepted_blocks), 0)
        lam = max(self.lambda0 * (self.lambda_decay**k), self.lambda_min)
        beta = max(self.beta0 * (self.beta_decay**k), self.beta_min)
        return float(lam), float(beta)


def state_filter(
    *,
    lateral_error_m: torch.Tensor,
    ground_contact: torch.Tensor,
    lateral_threshold_m: float = 0.10,
    require_contact: bool = True,
) -> torch.Tensor:
    """Imitate where the policy is weak: off the line, or off the ground.

    `lateral_threshold_m` defaults to 0.10 because that is roughly where the
    distilled students sit (0.046-0.081) -- above it the policy is doing worse
    than its own distilled baseline and the teacher is the better authority;
    below it the policy is already at or past teacher quality (teacher 0.1039)
    and must not be pulled back.
    """
    weak = lateral_error_m.abs() > float(lateral_threshold_m)
    if require_contact:
        weak = weak | (ground_contact <= 0.0)
    return weak.to(dtype=lateral_error_m.dtype)


def value_filter(advantages: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Imitate where the executed action underperformed the critic's estimate."""
    return (advantages < -abs(float(margin))).to(dtype=advantages.dtype)


def demo_loss(
    *,
    policy_mean: torch.Tensor,
    teacher_action: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean-squared imitation of the teacher's command.

    Weights are per-sample and typically binary. Normalising by their SUM rather
    than by the batch size keeps the loss scale independent of how often the
    filter fires -- otherwise lambda_k would silently mean something different
    as the policy improves and the filter fires less.
    """
    if policy_mean.shape != teacher_action.shape:
        raise ValueError(
            f"policy mean {tuple(policy_mean.shape)} and teacher action "
            f"{tuple(teacher_action.shape)} must match"
        )
    per_sample = ((policy_mean - teacher_action) ** 2).mean(dim=-1)
    w = weights.reshape(-1)
    total = w.sum()
    if float(total) <= 0.0:
        return policy_mean.sum() * 0.0
    return (w * per_sample).sum() / total


def gaussian_kl(
    *,
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, averaged over the batch.

    q is the current policy, p the frozen distilled prior pi_0. This is the
    trust region that stops the actor drifting into a region where the critic
    and the demonstrations are both uninformative.
    """
    var_q = std_q**2
    var_p = std_p**2
    term = (
        torch.log(std_p / std_q.clamp(min=1e-8))
        + (var_q + (mean_q - mean_p) ** 2) / (2.0 * var_p.clamp(min=1e-12))
        - 0.5
    )
    return term.sum(dim=-1).mean()


@dataclass
class BlockVerdict:
    accepted: bool
    reason: str


def accept_block(
    *,
    gates_passed: bool,
    failing: list[str],
    lateral_now: float,
    lateral_reference: float,
    tolerance_m: float = 0.0,
) -> BlockVerdict:
    """Accept a block only on HELD-OUT evidence, never on training return.

    Rising training return is precisely what made the previous 600-iteration run
    uninterpretable: reward went 15,655 -> 17,765 with 21% actor weight movement
    and no held-out number was ever produced. Two conditions, both necessary:
    every safety gate still passes, and tracking did not regress against the
    frozen distilled reference.
    """
    if not gates_passed:
        return BlockVerdict(False, f"gates failing: {sorted(failing)}")
    if lateral_now > lateral_reference + abs(float(tolerance_m)):
        return BlockVerdict(
            False,
            f"lateral regressed {lateral_reference:.4f} -> {lateral_now:.4f}",
        )
    return BlockVerdict(True, f"gates pass, lateral {lateral_now:.4f}")
