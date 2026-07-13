"""Drehzapfen-centered slewing arc for stop-and-mow mowing.

The cutter traces a constant-radius circular arc around the arm slew centre
(Drehzapfen) in the truck base_link frame.  The outward (A→B) and return
(B→A) strokes are exact reverses.  Minimum-jerk time scaling ensures smooth
turnarounds with zero velocity/acceleration at endpoints.

Geometry:
    x(u) = cx + r * cos(theta(u))
    y(u) = cy + r * sin(theta(u))
    z(u) = z_nominal

with theta(u) = theta_A + u * (theta_B - theta_A) and u = h(tau) for A→B,
u = 1 - h(tau) for B→A.
"""

from __future__ import annotations

import math

import torch

from .time_scaling import quintic_scaling
from .types import CutterTrajectorySample


def arc_mowing_reference(
    path_distance_m: torch.Tensor,
    *,
    anchor: torch.Tensor,
    arc_center_xy: tuple[float, float] = (4.327, -3.224),
    arc_radius_m: float = 1.286,
    theta_a_deg: float = -26.7,
    theta_b_deg: float = 9.3,
    cut_height: float = 0.05,
) -> CutterTrajectorySample:
    """Generate batched Drehzapfen-centered slewing arc references.

    The cutter follows a circular arc in the truck-frame XY plane centered at
    ``arc_center_xy`` with radius ``arc_radius_m``, from angle ``theta_a_deg``
    to ``theta_b_deg`` and back.  Minimum-jerk (quintic) time scaling produces
    zero velocity and acceleration at both reversal points.

    Args:
        path_distance_m: accumulated arc-length distance ``(N,)`` [m].
        anchor: truck-frame anchor position ``(N, 3)`` — used only for Z.
        arc_center_xy: ``(cx, cy)`` of the Drehzapfen slew centre in truck frame.
        arc_radius_m: constant horizontal radius [m].
        theta_a_deg: start angle [degrees] (A endpoint).
        theta_b_deg: end angle [degrees] (B endpoint).
        cut_height: Z offset from anchor for ground clearance [m].

    Returns:
        ``CutterTrajectorySample`` with batch size ``N``.
    """
    N = path_distance_m.shape[0]
    device = path_distance_m.device

    cx = float(arc_center_xy[0])
    cy = float(arc_center_xy[1])
    r = float(arc_radius_m)
    theta_a = math.radians(float(theta_a_deg))
    theta_b = math.radians(float(theta_b_deg))
    delta_theta = theta_b - theta_a  # signed angular span

    # One-way arc length and full cycle
    L_arc = r * abs(delta_theta)
    L_cycle = 2.0 * L_arc

    # Phase within the full A→B→A cycle
    phase = torch.remainder(path_distance_m, L_cycle)
    stroke_direction = torch.where(phase < L_arc,
                                   torch.ones(N, device=device),
                                   -torch.ones(N, device=device))
    # Distance within current half-stroke [0, L_arc]
    stroke_dist = torch.where(phase < L_arc, phase, phase - L_arc)
    tau = stroke_dist / max(L_arc, 1e-8)
    tau = tau.clamp(0.0, 1.0)

    # Minimum-jerk time scaling: h(0)=0, h(1)=1, h'(0)=h'(1)=h''(0)=h''(1)=0
    h_p, h_v, h_a = quintic_scaling(tau)

    # A→B: u = h(tau),  B→A: u = 1 - h(tau)
    u = torch.where(stroke_direction > 0, h_p, 1.0 - h_p)

    # Angular position
    theta = theta_a + u * delta_theta
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Cartesian position in truck frame
    x_pos = cx + r * cos_t
    y_pos = cy + r * sin_t
    z_pos = anchor[:, 2] + float(cut_height)

    position_b = torch.stack([x_pos, y_pos, z_pos], dim=-1)

    # ---- Analytical velocity ----
    # d(position)/d(tau) = d(position)/d(theta) * d(theta)/du * du/d(h) * dh/d(tau)
    # d(position)/d(theta) = r * (-sin(theta), cos(theta))
    # d(theta)/du = delta_theta
    # du/dh = +1 (A→B) or -1 (B→A)
    # dh/d(tau) = h_v
    # So: dp/dtau = r * (-sin(theta), cos(theta)) * delta_theta * sign * h_v
    # And: dp/ds = dp/dtau * dtau/ds = dp/dtau / L_arc
    sign = stroke_direction
    dtheta_du = delta_theta
    # Angular velocity w.r.t. path distance: dtheta/ds = dtheta/du * du/dh * dh/dtau * dtau/ds
    # = delta_theta * sign * h_v * (1/L_arc)
    dtheta_ds = dtheta_du * sign * h_v / max(L_arc, 1e-8)

    vel_x = -r * sin_t * dtheta_ds
    vel_y = r * cos_t * dtheta_ds
    vel_z = torch.zeros(N, device=device)

    linear_velocity_b = torch.stack([vel_x, vel_y, vel_z], dim=-1)

    # ---- Analytical acceleration ----
    # d^2(position)/ds^2: chain rule through theta
    # d^2x/ds^2 = -r * cos(theta) * (dtheta/ds)^2 - r * sin(theta) * d^2theta/ds^2
    # d^2theta/ds^2 = delta_theta * sign * h_a / L_arc^2
    d2theta_ds2 = dtheta_du * sign * h_a / max(L_arc * L_arc, 1e-8)

    acc_x = -r * cos_t * dtheta_ds * dtheta_ds - r * sin_t * d2theta_ds2
    acc_y = -r * sin_t * dtheta_ds * dtheta_ds + r * cos_t * d2theta_ds2
    acc_z = torch.zeros(N, device=device)

    linear_acceleration_b = torch.stack([acc_x, acc_y, acc_z], dim=-1)

    orientation_w = torch.cat([
        torch.ones(N, 1, device=device),
        torch.zeros(N, 3, device=device),
    ], dim=-1)

    # Normalized progress: 0 at start of A→B, 1 at end of A→B; 0 at start of B→A, 1 at end of B→A
    normalized_progress = tau

    return CutterTrajectorySample(
        position_b=position_b,
        orientation_w=orientation_w,
        linear_velocity_b=linear_velocity_b,
        linear_acceleration_b=linear_acceleration_b,
        angular_velocity_w=torch.zeros(N, 3, device=device),
        path_distance_m=path_distance_m,
        normalized_progress=normalized_progress,
        stroke_direction=stroke_direction,
        cylinder_seed=None,
    )
