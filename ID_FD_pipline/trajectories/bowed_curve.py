"""Bowed lateral curve with exact return stroke.

The curve moves laterally (Y) with a forward bow (X) that follows a sine
profile.  The outward and return strokes are the exact reverse of each other,
ensuring closed-path geometry.  Velocity is zero at both endpoints.
"""

from __future__ import annotations

import math

import torch

from .time_scaling import cosine_scaling
from .types import CutterTrajectorySample


def bowed_curve_reference(
    path_distance_m: torch.Tensor,
    *,
    anchor: torch.Tensor,
    lateral_span: float = 0.80,
    forward_bow: float = 0.14,
    lateral_direction: torch.Tensor | None = None,
    forward_direction: torch.Tensor | None = None,
    cut_height: float = 0.05,
) -> CutterTrajectorySample:
    """Generate batched bowed lateral curve references.

    The cutter describes a closed curve: outward along the lateral axis with a
    forward sine bow, then returns along the exact reverse path.  Cosine
    turnaround ensures zero velocity at reversal points.

    Args:
        path_distance_m: accumulated arc-length distance ``(N,)`` [m].
        anchor: truck-frame anchor position ``(N, 3)`` at the curve start.
        lateral_span: lateral stroke length [m] (default 0.80 m).
        forward_bow: peak forward deviation [m] (default 0.14 m).
        lateral_direction: unit vector ``(3,)`` for lateral axis (default Y).
        forward_direction: unit vector ``(3,)`` for forward axis (default X).
        cut_height: Z offset from anchor for ground clearance [m].

    Returns:
        ``CutterTrajectorySample`` with batch size ``N``.
    """
    N = path_distance_m.shape[0]
    device = path_distance_m.device
    if lateral_direction is None:
        lateral_direction = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    else:
        lateral_direction = lateral_direction.to(device, torch.float32)
    lateral_direction = lateral_direction / lateral_direction.norm().clamp(min=1e-8)
    if forward_direction is None:
        forward_direction = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    else:
        forward_direction = forward_direction.to(device, torch.float32)
    forward_direction = forward_direction / forward_direction.norm().clamp(min=1e-8)

    span = float(lateral_span)
    bow = float(forward_bow)
    full_cycle = 2.0 * span  # out + exact return

    phase = torch.remainder(path_distance_m, full_cycle)
    stroke_direction = torch.where(phase < span,
                                   torch.ones(N, device=device),
                                   -torch.ones(N, device=device))
    # Distance within stroke [0, span]
    stroke_dist = torch.where(phase < span, phase, phase - span)
    tau = stroke_dist / max(span, 1e-8)
    tau = tau.clamp(0.0, 1.0)

    h_p, h_v, h_a = cosine_scaling(tau)

    # Lateral: outward 0→span, return span→0
    along_lateral = h_p * span
    along_lateral = torch.where(stroke_direction > 0, along_lateral, span - along_lateral)
    # Forward bow: sin(pi * unit) peaks at midpoint, zero at ends.
    # On return stroke, reverse sign for exact path retrace.
    forward = bow * torch.sin(math.pi * h_p) * stroke_direction

    # Velocity: dlateral/ds = h_v * sign, dforward/ds = bow * pi * cos(pi*h_p) * h_v * sign / span
    vel_lateral = h_v * stroke_direction
    vel_forward = bow * math.pi * torch.cos(math.pi * h_p) * h_v * stroke_direction / max(span, 1e-8)

    position_b = anchor.clone()
    position_b = position_b + along_lateral.unsqueeze(-1) * lateral_direction.unsqueeze(0)
    position_b = position_b + forward.unsqueeze(-1) * forward_direction.unsqueeze(0)
    position_b[:, 2] = anchor[:, 2] + float(cut_height)

    linear_velocity_b = (
        vel_lateral.unsqueeze(-1) * lateral_direction.unsqueeze(0)
        + vel_forward.unsqueeze(-1) * forward_direction.unsqueeze(0)
    )

    # Acceleration: simplified — second derivative of cosine scaling
    acc_lateral = h_a * stroke_direction / max(span, 1e-8)
    acc_forward = (
        -bow * math.pi * math.pi * torch.sin(math.pi * h_p) * (h_v * h_v)
        + bow * math.pi * torch.cos(math.pi * h_p) * h_a
    ) * stroke_direction / max(span * span, 1e-8)

    linear_acceleration_b = (
        acc_lateral.unsqueeze(-1) * lateral_direction.unsqueeze(0)
        + acc_forward.unsqueeze(-1) * forward_direction.unsqueeze(0)
    )

    orientation_w = torch.cat([
        torch.ones(N, 1, device=device),
        torch.zeros(N, 3, device=device),
    ], dim=-1)

    return CutterTrajectorySample(
        position_b=position_b,
        orientation_w=orientation_w,
        linear_velocity_b=linear_velocity_b,
        linear_acceleration_b=linear_acceleration_b,
        angular_velocity_w=torch.zeros(N, 3, device=device),
        path_distance_m=path_distance_m,
        normalized_progress=tau,
        stroke_direction=stroke_direction,
        cylinder_seed=None,
    )
