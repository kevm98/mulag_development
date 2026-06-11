from __future__ import annotations

from dataclasses import dataclass

import torch


ACTUATOR_ORDER: tuple[str, ...] = (
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
)

ROM_FEATURE_ORDER: tuple[str, ...] = (
    "ValveCmd_1",
    "ValveCmd_2",
    "ValveCmd_3",
    "ValveCmd_4",
    "dP_1",
    "dP_2",
    "dP_3",
    "dP_4",
    "Extension_1",
    "Extension_2",
    "Extension_3",
    "Extension_4",
    "ExtensionVelocity_1",
    "ExtensionVelocity_2",
    "ExtensionVelocity_3",
    "ExtensionVelocity_4",
    "JointPos_1",
    "JointPos_2",
    "JointPos_3",
    "JointPos_4",
    "JointPos_5",
    "JointVel_1",
    "JointVel_2",
    "JointVel_3",
    "JointVel_4",
    "JointVel_5",
)


@dataclass(frozen=True)
class LimitAudit:
    measured_low_amount: torch.Tensor
    measured_high_amount: torch.Tensor
    target_pre_low_amount: torch.Tensor
    target_pre_high_amount: torch.Tensor
    target_post_low_amount: torch.Tensor
    target_post_high_amount: torch.Tensor
    measured_violation: torch.Tensor
    target_pre_violation: torch.Tensor
    target_post_violation: torch.Tensor
    measured_contact: torch.Tensor
    target_pre_contact: torch.Tensor
    target_post_contact: torch.Tensor
    clamped: torch.Tensor
    tested_critical: bool
    passive_true_violation: bool
    target_clamp_warning: bool


def active_actuator_mask(single_actuator: str | None, *, like: torch.Tensor | None = None) -> torch.Tensor:
    mask = torch.ones((1, 4), dtype=torch.float32)
    if single_actuator is not None:
        if single_actuator not in ACTUATOR_ORDER:
            raise ValueError(f"single_actuator must be one of {ACTUATOR_ORDER}, got {single_actuator!r}.")
        mask.zero_()
        mask[:, ACTUATOR_ORDER.index(single_actuator)] = 1.0
    if like is not None:
        mask = mask.to(device=like.device, dtype=like.dtype)
    return mask


def apply_active_mask(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if delta.shape[-1] != 4 or mask.shape[-1] != 4:
        raise ValueError("delta and mask must have four actuator channels.")
    return delta * mask.to(device=delta.device, dtype=delta.dtype)


def passive_hold_target(
    mode: str,
    *,
    fixed_hold: torch.Tensor,
    measured_q: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if mode not in {"fixed", "measured_follow"}:
        raise ValueError("--passive_hold_mode must be 'fixed' or 'measured_follow'.")
    mask = active_mask.to(device=measured_q.device, dtype=torch.bool)
    if mode == "measured_follow":
        return torch.where(mask, fixed_hold.to(measured_q), measured_q)
    return fixed_hold.to(measured_q)


def merge_active_and_passive_targets(
    active_target: torch.Tensor,
    passive_target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    mask = active_mask.to(device=active_target.device, dtype=torch.bool)
    return torch.where(mask, active_target, passive_target.to(active_target))


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).detach().cpu())


def assert_initialized_from_measured(target: torch.Tensor, measured_q: torch.Tensor, *, tolerance: float, label: str) -> None:
    error = max_abs_error(target, measured_q)
    if error > float(tolerance):
        raise RuntimeError(f"FAIL: {label} target initialization mismatch {error} m.")


def compute_limit_audit(
    *,
    measured_q: torch.Tensor,
    target_preclamp: torch.Tensor,
    target_postclamp: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    tolerance: float,
    active_mask: torch.Tensor,
) -> LimitAudit:
    lower_row = lower.reshape(1, 4).to(measured_q)
    upper_row = upper.reshape(1, 4).to(measured_q)
    tol = float(tolerance)
    measured_low = torch.clamp(lower_row - measured_q, min=0.0)
    measured_high = torch.clamp(measured_q - upper_row, min=0.0)
    pre_low = torch.clamp(lower_row - target_preclamp, min=0.0)
    pre_high = torch.clamp(target_preclamp - upper_row, min=0.0)
    post_low = torch.clamp(lower_row - target_postclamp, min=0.0)
    post_high = torch.clamp(target_postclamp - upper_row, min=0.0)
    measured_violation = (measured_low > tol) | (measured_high > tol)
    target_pre_violation = (pre_low > tol) | (pre_high > tol)
    target_post_violation = (post_low > tol) | (post_high > tol)
    measured_contact = ((measured_low > 0.0) & (measured_low <= tol)) | ((measured_high > 0.0) & (measured_high <= tol))
    target_pre_contact = ((pre_low > 0.0) & (pre_low <= tol)) | ((pre_high > 0.0) & (pre_high <= tol))
    target_post_contact = ((post_low > 0.0) & (post_low <= tol)) | ((post_high > 0.0) & (post_high <= tol))
    clamped = torch.ne(target_preclamp, target_postclamp)
    active = active_mask.to(device=measured_q.device, dtype=torch.bool)
    critical = (measured_violation | target_post_violation) & active
    passive_true = (measured_violation | target_post_violation) & ~active
    return LimitAudit(
        measured_low_amount=measured_low,
        measured_high_amount=measured_high,
        target_pre_low_amount=pre_low,
        target_pre_high_amount=pre_high,
        target_post_low_amount=post_low,
        target_post_high_amount=post_high,
        measured_violation=measured_violation,
        target_pre_violation=target_pre_violation,
        target_post_violation=target_post_violation,
        measured_contact=measured_contact,
        target_pre_contact=target_pre_contact,
        target_post_contact=target_post_contact,
        clamped=clamped,
        tested_critical=bool(torch.any(critical).item()),
        passive_true_violation=bool(torch.any(passive_true).item()),
        target_clamp_warning=bool(torch.any(clamped | target_pre_violation | target_pre_contact).item()),
    )


def assert_passive_targets_held(
    target: torch.Tensor,
    passive_target: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    passive = ~active_mask.to(device=target.device, dtype=torch.bool)
    drift = torch.where(passive, torch.abs(target - passive_target.to(target)), torch.zeros_like(target))
    if torch.max(drift).item() > float(tolerance):
        raise RuntimeError(f"FAIL: passive target drift {float(torch.max(drift).item())} m.")
    return drift
