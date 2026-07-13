"""Drehzapfen-centered mowing arc driven by a **time** profile, not a path re-scaling.

`arc_mowing.py` advanced arc length at a constant rate and then applied quintic scaling on
top of it, so the path was time-scaled twice: the cutter's geometric speed came out
`path_speed * h'(tau)` -- zero at both endpoints, 1.875x path_speed mid-stroke. At the
normalized progress of 0.0511 seen in the 10 s runs that is 1.1 mm of travel on an 0.88 m
arc. Its `linear_velocity_b` was also a derivative with respect to path *distance*, not
time, so anything consuming it as m/s was wrong by a factor of the path speed.

Here the stroke is parameterised by elapsed time through a jerk-limited
accelerate-cruise-decelerate profile, and every returned derivative is with respect to
time: position [m], velocity [m/s], acceleration [m/s^2].
"""

from __future__ import annotations

import math

import torch

from .time_scaling import cruise_profile
from .types import CutterTrajectorySample


def arc_cruise_reference(
    stroke_time_s: torch.Tensor,
    *,
    anchor: torch.Tensor,
    arc_center_xy: tuple[float, float],
    arc_radius_m: float,
    theta_a_deg: float,
    theta_b_deg: float,
    cruise_speed_mps: float = 0.05,
    accel_time_s: float = 0.75,
    decel_time_s: float = 0.75,
    cut_height: float = 0.05,
) -> CutterTrajectorySample:
    """Sample the A->B->A mowing arc at elapsed stroke time.

    Args:
        stroke_time_s: elapsed time within the CURRENT half-stroke ``(N,)`` [s].
            The caller is responsible for stroke bookkeeping (see
            :func:`stroke_state`), which keeps this function pure.
        anchor: ``(N, 3)`` truck-frame anchor; only Z is used.
        cruise_speed_mps: constant cutter speed through the middle of the arc [m/s].
        accel_time_s / decel_time_s: endpoint blend durations [s].

    Returns:
        ``CutterTrajectorySample`` whose linear_velocity_b / linear_acceleration_b are
        true time derivatives.
    """
    device = stroke_time_s.device
    cx, cy = float(arc_center_xy[0]), float(arc_center_xy[1])
    r = float(arc_radius_m)
    theta_a = math.radians(float(theta_a_deg))
    theta_b = math.radians(float(theta_b_deg))
    delta_theta = theta_b - theta_a
    L_arc = r * abs(delta_theta)

    s, v, a, _T = cruise_profile(
        stroke_time_s, length_m=L_arc, cruise_speed_mps=cruise_speed_mps,
        accel_time_s=accel_time_s, decel_time_s=decel_time_s,
    )

    # Arc length -> angle. Constant geometric speed during cruise, by construction.
    dtheta_ds = delta_theta / max(L_arc, 1e-9)          # [rad/m], constant
    theta = theta_a + s * dtheta_ds
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    position = torch.stack([
        cx + r * cos_t,
        cy + r * sin_t,
        anchor[:, 2] + float(cut_height),
    ], dim=-1)

    # dp/dt = dp/dtheta * dtheta/ds * ds/dt
    theta_dot = dtheta_ds * v                            # [rad/s]
    theta_ddot = dtheta_ds * a                           # [rad/s^2]
    velocity = torch.stack([
        -r * sin_t * theta_dot,
        r * cos_t * theta_dot,
        torch.zeros_like(v),
    ], dim=-1)
    acceleration = torch.stack([
        -r * cos_t * theta_dot**2 - r * sin_t * theta_ddot,
        -r * sin_t * theta_dot**2 + r * cos_t * theta_ddot,
        torch.zeros_like(a),
    ], dim=-1)

    n = stroke_time_s.shape[0]
    return CutterTrajectorySample(
        position_b=position,
        orientation_w=torch.cat([torch.ones(n, 1, device=device),
                                 torch.zeros(n, 3, device=device)], dim=-1),
        linear_velocity_b=velocity,
        linear_acceleration_b=acceleration,
        angular_velocity_w=torch.zeros(n, 3, device=device),
        path_distance_m=s,
        normalized_progress=s / max(L_arc, 1e-9),
        stroke_direction=torch.ones(n, device=device),
        cylinder_seed=None,
    )


def stroke_duration(
    *, arc_radius_m: float, theta_a_deg: float, theta_b_deg: float,
    cruise_speed_mps: float, accel_time_s: float = 0.75, decel_time_s: float = 0.75,
) -> tuple[float, float]:
    """Return ``(one_way_seconds, arc_length_m)`` for run-length budgeting.

    A test must be sized from this, never from a fixed 10 s.
    """
    L = float(arc_radius_m) * abs(math.radians(float(theta_b_deg) - float(theta_a_deg)))
    _s, _v, _a, T = cruise_profile(
        torch.zeros(1), length_m=L, cruise_speed_mps=cruise_speed_mps,
        accel_time_s=accel_time_s, decel_time_s=decel_time_s,
    )
    return float(T), float(L)


def stroke_state(elapsed_s: torch.Tensor, one_way_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Split total elapsed time into (time within current half-stroke, direction ±1)."""
    cycle = 2.0 * float(one_way_s)
    phase = torch.remainder(elapsed_s, cycle)
    outward = phase < one_way_s
    stroke_t = torch.where(outward, phase, phase - one_way_s)
    direction = torch.where(outward, torch.ones_like(phase), -torch.ones_like(phase))
    return stroke_t, direction


def arc_cruise_cycle(
    elapsed_s: torch.Tensor,
    *,
    anchor: torch.Tensor,
    arc_center_xy: tuple[float, float],
    arc_radius_m: float,
    theta_a_deg: float,
    theta_b_deg: float,
    cruise_speed_mps: float = 0.05,
    accel_time_s: float = 0.75,
    decel_time_s: float = 0.75,
    cut_height: float = 0.05,
) -> CutterTrajectorySample:
    """Full A->B->A cycling reference at total elapsed time. B->A is the exact reverse."""
    one_way_s, _L = stroke_duration(
        arc_radius_m=arc_radius_m, theta_a_deg=theta_a_deg, theta_b_deg=theta_b_deg,
        cruise_speed_mps=cruise_speed_mps, accel_time_s=accel_time_s, decel_time_s=decel_time_s,
    )
    stroke_t, direction = stroke_state(elapsed_s, one_way_s)
    outward = direction > 0
    # The return stroke is the outward stroke played backwards: sample at (T - t) and
    # negate the time derivatives. Endpoint velocity is zero on both, so the reversal is
    # continuous in position, velocity AND acceleration.
    t_eff = torch.where(outward, stroke_t, one_way_s - stroke_t)
    sample = arc_cruise_reference(
        t_eff, anchor=anchor, arc_center_xy=arc_center_xy, arc_radius_m=arc_radius_m,
        theta_a_deg=theta_a_deg, theta_b_deg=theta_b_deg, cruise_speed_mps=cruise_speed_mps,
        accel_time_s=accel_time_s, decel_time_s=decel_time_s, cut_height=cut_height,
    )
    flip = torch.where(outward, 1.0, -1.0).reshape(-1, 1)
    sample.linear_velocity_b = sample.linear_velocity_b * flip
    sample.linear_acceleration_b = sample.linear_acceleration_b * flip
    sample.stroke_direction = direction
    sample.normalized_progress = torch.where(
        outward, sample.normalized_progress, 1.0 - sample.normalized_progress
    )
    return sample
