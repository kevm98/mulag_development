"""Centred lateral line trajectory with cosine turnarounds.

The line is anchored at the caller-supplied ``anchor`` in truck ``base_link``
coordinates and moves along the lateral (Y) direction.  Each outward stroke
covers the full ``span``, followed by an exact return stroke.  Velocity is zero
at both endpoints.
"""

from __future__ import annotations

import math

import torch

from .time_scaling import cosine_scaling
from .types import CutterTrajectorySample


def line_reference(
    path_distance_m: torch.Tensor,
    *,
    anchor: torch.Tensor,
    span: float = 1.00,
    lateral_direction: torch.Tensor | None = None,
    cut_height: float = 0.05,
) -> CutterTrajectorySample:
    """Generate batched centred lateral line references.

    The cutter moves from ``anchor - 0.5*span*lateral`` out to
    ``anchor + 0.5*span*lateral`` and back, with cosine-smooth turnarounds.

    Args:
        path_distance_m: accumulated arc-length distance ``(N,)`` [m].
        anchor: truck-frame anchor position ``(N, 3)``.
        span: total lateral span [m] (default 1.00 m).
        lateral_direction: unit direction vector ``(3,)`` for the lateral axis.
            Defaults to ``(0, 1, 0)`` (truck-frame Y).
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

    half = 0.5 * float(span)
    full_cycle = 2.0 * float(span)  # out + back

    # Phase within one full out-and-back cycle
    phase = torch.remainder(path_distance_m, full_cycle)
    # Determine stroke direction (+1 = outward, -1 = return)
    stroke_direction = torch.where(phase < float(span),
                                   torch.ones(N, device=device),
                                   -torch.ones(N, device=device))
    # distance within current half-stroke [0, span/2]
    half_stroke = torch.where(phase < half,
                              phase,
                              torch.where(phase < float(span),
                                          float(span) - phase,
                                          torch.where(phase < float(span) + half,
                                                      phase - float(span),
                                                      full_cycle - phase)))
    tau = half_stroke / max(half, 1e-8)
    tau = tau.clamp(0.0, 1.0)

    h_p, h_v, h_a = cosine_scaling(tau)

    # Lateral displacement: out = 0→+half, return = half→0 (in reverse phase)
    # During outward stroke: along = h_p * half  (0 → half)
    # During return stroke: along = (1-h_p) * half + half  (half → span) but
    # we simplify: absolute lateral offset from anchor
    outward = (phase < float(span))
    along = torch.where(outward, h_p * half, (1.0 - h_p) * half + half)
    # Map to [-half, +half] centred on anchor
    lateral_offset = along - half

    # Height: fixed cut height from anchor Z
    z_offset = float(cut_height)

    position_b = anchor.clone()
    d = lateral_direction.unsqueeze(0).expand(N, 3)
    position_b = position_b + lateral_offset.unsqueeze(-1) * d
    position_b[:, 2] = anchor[:, 2] + z_offset

    # Velocity: h_v * (half/span) * speed * direction signs
    # The path speed (m/s) comes from the caller's arc-length derivative.
    # Here we provide the unitless lateral velocity scale:
    #   v_lateral = h_v * (half / half) * direction = h_v * direction
    #   But multiplied by sign for stroke direction.
    # Actually: d(position)/d(tau) = h_v * half * lateral_direction (outward)
    #                                or -h_v * half * lateral_direction (return)
    # But the caller scales by d(path_distance)/dt.
    # We provide d(position)/d(path_distance) so:
    #   dp/ds = dp/dtau * dtau/ds
    #   dtau/ds = 1/(half) during each half-stroke
    #   dp/dtau = h_v * half * dir_sign * lateral_direction
    # So dp/ds = h_v * dir_sign * lateral_direction
    vel_scale = h_v * stroke_direction
    linear_velocity_b = vel_scale.unsqueeze(-1) * lateral_direction.unsqueeze(0)

    # Acceleration: h_a / half
    acc_scale = h_a * stroke_direction / max(half, 1e-8)
    linear_acceleration_b = acc_scale.unsqueeze(-1) * lateral_direction.unsqueeze(0)

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
