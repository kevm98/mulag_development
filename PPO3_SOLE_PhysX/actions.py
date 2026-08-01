"""Pure action transform for the sole-controller policy.

PPO2 shaped a deterministic cylinder goal produced by the EE cascade and then
handed that goal to ``CylinderDLSController``.  PPO3 has neither: the policy
action IS the cylinder command that reaches the ROM.  Everything that the DLS
used to provide implicitly -- output smoothing, a per-step motion bound, and a
refusal to drive into the extension limits -- has to be stated explicitly here,
because there is no longer a controller downstream to provide it.

The transform is kept free of Isaac/env imports so the bounds it promises can
be tested without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SoleActionOutput:
    """One control step of the sole-controller action chain."""

    raw: torch.Tensor
    """Policy output after NaN scrubbing and clamping to [-1, 1]."""

    filtered: torch.Tensor
    """EMA-smoothed action, before the slew limit."""

    applied: torch.Tensor
    """Command actually handed to the plant, after slew limit and shield."""

    shield_gain: torch.Tensor
    """Per-cylinder authority multiplier from the limit shield; 1.0 = untouched."""


def _broadcast(value, action_dim: int, like: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(action_dim)
    if tensor.numel() != action_dim:
        raise ValueError(f"{name} has {tensor.numel()} values for {action_dim} actions")
    return tensor.reshape(1, action_dim)


def limit_shield(
    action: torch.Tensor,
    *,
    headroom_low_m: torch.Tensor,
    headroom_high_m: torch.Tensor,
    margin_m: float,
    action_to_extension_sign: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fade out commands that point into a nearby extension limit.

    ``headroom_low_m`` is ``measured - learned_min`` and ``headroom_high_m`` is
    ``learned_max - measured``; both are clamped at zero by the caller's physics,
    not here.  A command is scaled by ``headroom / margin`` only in the direction
    that consumes headroom, so the policy always retains full authority to move
    away from a limit.

    This is a shield, not a controller: it can only ever reduce the magnitude of
    the policy's command and never contributes a tracking term of its own.
    ``margin_m <= 0`` disables it and returns the action unchanged.
    """
    if margin_m <= 0.0:
        return action, torch.ones_like(action)
    low = torch.clamp(headroom_low_m / margin_m, 0.0, 1.0)
    high = torch.clamp(headroom_high_m / margin_m, 0.0, 1.0)
    # Policy action is an increment in USD joint coordinates. This asset maps
    # every USD cylinder coordinate to learned extension with a NEGATIVE scale,
    # so a positive action consumes LOW learned headroom. Keep the pure helper
    # general: callers pass the measured mapping sign and identity mappings keep
    # the historical +1 default.
    sign = torch.sign(
        _broadcast(
            action_to_extension_sign,
            int(action.shape[-1]),
            action,
            "action_to_extension_sign",
        )
    )
    if bool((sign == 0.0).any()):
        raise ValueError("action_to_extension_sign cannot contain zero")
    extension_command = action * sign
    gain = torch.where(extension_command > 0.0, high, low)
    gain = torch.where(extension_command == 0.0, torch.ones_like(gain), gain)
    return action * gain, gain


def directional_tilt_shield(
    action: torch.Tensor,
    *,
    signed_tilt: torch.Tensor,
    action_channel: int,
    action_to_tilt_sign: float,
    soft_limit: float,
    hard_limit: float,
    tilt_magnitude: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fade only commands that worsen an observed signed orientation error.

    The shield contributes no corrective command. Below ``soft_limit`` it is
    inactive; between soft and hard it linearly removes authority in the
    direction that increases ``abs(signed_tilt)``; at/above hard it blocks that
    direction completely while preserving full authority to recover.

    ``signed_tilt`` names the direction the shielded channel acts along, so it
    is necessarily a projection. Pass ``tilt_magnitude`` -- the true off-target
    angle -- when the head can leave the cone along an axis that projection
    cannot see; the fade is then driven by the real excursion while the sign
    still selects which command direction is worsening. Without it a caller
    whose ``signed_tilt`` is a single-plane projection has no shield at all
    against excursions square to that plane.
    """
    if action.ndim != 2:
        raise ValueError(f"action must be rank 2, got shape {tuple(action.shape)}")
    channel = int(action_channel)
    if not 0 <= channel < action.shape[1]:
        raise ValueError(f"action_channel {channel} outside width {action.shape[1]}")
    soft = float(soft_limit)
    hard = float(hard_limit)
    if not 0.0 <= soft < hard:
        raise ValueError(f"tilt limits must satisfy 0 <= soft < hard, got {soft}, {hard}")
    response_sign = float(action_to_tilt_sign)
    if response_sign not in (-1.0, 1.0):
        raise ValueError("action_to_tilt_sign must be exactly -1 or +1")

    tilt = signed_tilt.to(device=action.device, dtype=action.dtype).reshape(-1)
    if tilt.shape[0] != action.shape[0]:
        raise ValueError(
            f"signed_tilt batch {tilt.shape[0]} != action batch {action.shape[0]}"
        )
    if tilt_magnitude is None:
        magnitude = tilt.abs()
    else:
        magnitude = (
            tilt_magnitude.to(device=action.device, dtype=action.dtype)
            .reshape(-1)
            .abs()
        )
        if magnitude.shape[0] != action.shape[0]:
            raise ValueError(
                f"tilt_magnitude batch {magnitude.shape[0]} != "
                f"action batch {action.shape[0]}"
            )

    channel_action = action[:, channel]
    predicted_tilt_direction = channel_action * response_sign
    worsening = tilt * predicted_tilt_direction > 0.0
    fade = torch.clamp((hard - magnitude) / (hard - soft), 0.0, 1.0)
    channel_gain = torch.where(worsening, fade, torch.ones_like(fade))
    gain = torch.ones_like(action)
    gain[:, channel] = channel_gain
    return action * gain, gain


def nullspace_head_restoring_overlay(
    action: torch.Tensor,
    *,
    signed_tilt: torch.Tensor,
    action_nullspace: torch.Tensor,
    action_to_tilt_sign: float,
    deadband_rad: float,
    gain: float,
    max_c4_action: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Actively restore head tilt without commanding first-order tool motion.

    ``action_nullspace`` is a four-cylinder action direction whose Cartesian
    cutter Jacobian product is zero and whose c4 component is +1.  Moving along
    it therefore gives c4 orientation authority while c1-c3 cancel c4's XYZ
    translation.  This avoids the head-vs-contact trade produced by correcting
    c4 or ``Messerkopf_joint`` alone.

    The returned second tensor is the additive overlay before the final unit-box
    clamp, allowing the environment to apply its cylinder-limit shield to the
    controller contribution separately from the policy command.
    """
    if action.ndim != 2 or action.shape[1] != 4:
        raise ValueError(f"action must have shape (N, 4), got {tuple(action.shape)}")
    response_sign = float(action_to_tilt_sign)
    if response_sign not in (-1.0, 1.0):
        raise ValueError("action_to_tilt_sign must be exactly -1 or +1")
    deadband = max(float(deadband_rad), 0.0)
    max_action = max(float(max_c4_action), 0.0)
    direction = torch.as_tensor(
        action_nullspace, dtype=action.dtype, device=action.device
    ).reshape(-1)
    if direction.numel() != 4:
        raise ValueError("action_nullspace must contain four values")
    if abs(float(direction[3])) < 1.0e-6:
        raise ValueError("action_nullspace c4 component must be non-zero")
    direction = direction / direction[3]

    tilt = signed_tilt.to(device=action.device, dtype=action.dtype).reshape(-1)
    if tilt.shape[0] != action.shape[0]:
        raise ValueError(
            f"signed_tilt batch {tilt.shape[0]} != action batch {action.shape[0]}"
        )
    error = torch.sign(tilt) * torch.relu(tilt.abs() - deadband)
    c4 = torch.clamp(
        -float(gain) * error / response_sign,
        min=-max_action,
        max=max_action,
    )
    overlay = c4[:, None] * direction[None, :]
    return torch.clamp(action + overlay, -1.0, 1.0), overlay


def invert_sole_chain(
    *,
    target_applied: torch.Tensor,
    previous_applied_action: torch.Tensor,
    filter_beta: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw action that would produce ``target_applied`` through the chain.

    Distillation needs this.  Regressing the actor directly onto the teacher's
    command would systematically undershoot, because the student's output is
    then smoothed by ``sole_action_filter`` on the way to the plant while the
    teacher's label was not.  At ``filter_beta = 0.75`` that is a four-fold
    attenuation of every correction, which is the difference between a student
    that tracks and one that lags the reference by a visible margin.

    In the unsaturated regime the chain is
    ``applied = prev + (1 - beta) * (raw - prev)``, so the exact inverse is
    ``raw = prev + (target - prev) / (1 - beta)``.  Where that lands outside
    [-1, 1] the target is not reachable in one step; the returned mask flags
    those samples so the caller can report the fraction rather than silently
    training toward something the chain cannot emit.
    """
    action_dim = int(target_applied.shape[-1])
    beta = torch.clamp(
        _broadcast(filter_beta, action_dim, target_applied, "filter_beta"), 0.0, 1.0
    )
    if float(beta.max()) >= 1.0:
        raise ValueError(
            "filter_beta must be < 1 during distillation; beta = 1 freezes the "
            "action chain and no raw action can reach the teacher's command"
        )
    raw = previous_applied_action + (target_applied - previous_applied_action) / (
        1.0 - beta
    )
    unreachable = (raw < -1.0) | (raw > 1.0)
    return torch.clamp(raw, -1.0, 1.0), unreachable


def apply_sole_action(
    *,
    policy_action: torch.Tensor,
    previous_applied_action: torch.Tensor,
    filter_beta: float | torch.Tensor,
    max_action_rate: float | torch.Tensor,
    headroom_low_m: torch.Tensor | None = None,
    headroom_high_m: torch.Tensor | None = None,
    limit_shield_margin_m: float = 0.0,
    action_to_extension_sign: float | torch.Tensor = 1.0,
    action_authority: float | torch.Tensor = 1.0,
) -> SoleActionOutput:
    """Turn a raw policy action into the bounded command the ROM receives.

    Order is deliberate.  Smoothing first (so the slew limit sees a signal that
    is already free of single-step spikes), then the slew limit relative to what
    was actually applied last step (so a saturated shield cannot be escaped by
    winding the filter state up), then the shield.
    """
    action_dim = int(policy_action.shape[-1])
    if action_dim != 4:
        raise ValueError(f"Sole-controller action dimension must be 4, got {action_dim}")
    if previous_applied_action.shape != policy_action.shape:
        raise ValueError(
            "previous_applied_action shape "
            f"{tuple(previous_applied_action.shape)} != action shape "
            f"{tuple(policy_action.shape)}"
        )

    raw = torch.clamp(
        torch.nan_to_num(policy_action, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0
    )
    authority = torch.clamp(
        _broadcast(action_authority, action_dim, raw, "action_authority"),
        0.0,
        1.0,
    )
    raw = raw * authority
    effective_previous = previous_applied_action * authority
    beta = torch.clamp(_broadcast(filter_beta, action_dim, raw, "filter_beta"), 0.0, 1.0)
    filtered = beta * effective_previous + (1.0 - beta) * raw

    rate = torch.clamp(
        _broadcast(max_action_rate, action_dim, raw, "max_action_rate"), min=0.0
    )
    step = torch.clamp(filtered - effective_previous, -rate, rate)
    applied = torch.clamp(effective_previous + step, -1.0, 1.0) * authority

    if (
        limit_shield_margin_m > 0.0
        and headroom_low_m is not None
        and headroom_high_m is not None
    ):
        applied, shield_gain = limit_shield(
            applied,
            headroom_low_m=headroom_low_m,
            headroom_high_m=headroom_high_m,
            margin_m=limit_shield_margin_m,
            action_to_extension_sign=action_to_extension_sign,
        )
    else:
        shield_gain = torch.ones_like(applied)

    return SoleActionOutput(
        raw=raw, filtered=filtered, applied=applied, shield_gain=shield_gain
    )
