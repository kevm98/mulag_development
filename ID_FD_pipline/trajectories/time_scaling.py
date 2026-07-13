"""Smooth time-scaling functions for trajectory parameterization.

All functions map a normalised progress ``tau ∈ [0, 1]`` to a position scale
``h(tau) ∈ [0, 1]`` with ``h(0)=0, h(1)=1`` and zero velocity/acceleration at
endpoints for the cosine and quintic variants.
"""

from __future__ import annotations

import math

import torch


def cosine_scaling(tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Cosine time scaling with zero endpoint velocity.

    .. math::
        h(\tau)   &= 0.5 \cdot (1 - \cos(\pi \cdot \tau))   \\
        h'(\tau)  &= 0.5 \cdot \pi \cdot \sin(\pi \cdot \tau)  \\
        h''(\tau) &= 0.5 \cdot \pi^2 \cdot \cos(\pi \cdot \tau)

    Args:
        tau: normalised progress ``(N,)`` in ``[0, 1]``.

    Returns:
        Tuple ``(position_scale, velocity_scale, acceleration_scale)``, each
        ``(N,)``.
    """
    pi = math.pi
    cos_tau = torch.cos(pi * tau)
    sin_tau = torch.sin(pi * tau)
    h_p = 0.5 * (1.0 - cos_tau)
    h_v = 0.5 * pi * sin_tau
    h_a = 0.5 * pi * pi * cos_tau
    return h_p, h_v, h_a


def cruise_profile(
    t: torch.Tensor,
    *,
    length_m: float,
    cruise_speed_mps: float,
    accel_time_s: float = 0.75,
    decel_time_s: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    r"""Jerk-limited accelerate–cruise–decelerate profile in **time**.

    Returns physical ``(s, v, a, T)``: arc-length [m], speed [m/s], acceleration [m/s²]
    along the path, and the total stroke duration [s].

    Velocity ramps with a smoothstep, so acceleration is continuous and jerk is bounded:

    .. math::
        v(\tau) = v_c (3\tau^2 - 2\tau^3), \quad
        s(\tau) = v_c t_a (\tau^3 - \tfrac{1}{2}\tau^4), \quad
        a(\tau) = \frac{v_c}{t_a}(6\tau - 6\tau^2)

    which integrates to exactly ``0.5 * v_c * t_a`` of blend distance, as a trapezoid does.

    This replaces applying quintic scaling across the *whole* stroke. That produced a
    geometric speed of ``path_speed * h'(tau)`` — zero at both endpoints and 1.875x
    ``path_speed`` mid-stroke — so at the observed progress of 0.0511 the cutter had moved
    1.1 mm on an 0.88 m arc. It also double-scaled an already arc-length-parameterised
    path.

    If the stroke is too short to reach cruise speed the profile degrades to a symmetric
    triangle (cruise speed is reduced, endpoints still have zero velocity).

    Args:
        t: elapsed stroke time ``(N,)`` [s].
        length_m: stroke length [m].
        cruise_speed_mps: requested constant-speed value [m/s].
        accel_time_s: acceleration blend duration [s].
        decel_time_s: deceleration blend duration [s].

    Returns:
        ``(s, v, a, total_time_s)``.
    """
    v_c = float(cruise_speed_mps)
    t_a = max(float(accel_time_s), 1e-6)
    t_d = max(float(decel_time_s), 1e-6)
    length = max(float(length_m), 1e-9)

    # Blend distances; shrink cruise speed if the stroke cannot fit both blends.
    if 0.5 * v_c * t_a + 0.5 * v_c * t_d > length:
        v_c = length / (0.5 * t_a + 0.5 * t_d)
    d_a = 0.5 * v_c * t_a
    d_d = 0.5 * v_c * t_d
    d_c = max(length - d_a - d_d, 0.0)
    t_c = d_c / max(v_c, 1e-9)
    total = t_a + t_c + t_d

    t = t.clamp(min=0.0, max=total)
    s = torch.zeros_like(t)
    v = torch.zeros_like(t)
    a = torch.zeros_like(t)

    # --- accelerate ---
    m = t < t_a
    if m.any():
        tau = (t[m] / t_a).clamp(0.0, 1.0)
        s[m] = v_c * t_a * (tau**3 - 0.5 * tau**4)
        v[m] = v_c * (3.0 * tau**2 - 2.0 * tau**3)
        a[m] = (v_c / t_a) * (6.0 * tau - 6.0 * tau**2)

    # --- cruise ---
    m = (t >= t_a) & (t < t_a + t_c)
    if m.any():
        s[m] = d_a + v_c * (t[m] - t_a)
        v[m] = v_c
        a[m] = 0.0

    # --- decelerate (mirror of the accel blend) ---
    m = t >= t_a + t_c
    if m.any():
        tau = ((total - t[m]) / t_d).clamp(0.0, 1.0)  # 1 -> 0
        s[m] = length - v_c * t_d * (tau**3 - 0.5 * tau**4)
        v[m] = v_c * (3.0 * tau**2 - 2.0 * tau**3)
        a[m] = -(v_c / t_d) * (6.0 * tau - 6.0 * tau**2)

    return s, v, a, float(total)


def quintic_scaling(tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Quintic (minimum-jerk) time scaling with zero endpoint velocity and acceleration.

    .. math::
        h(\tau)   &= 10\tau^3 - 15\tau^4 + 6\tau^5   \\
        h'(\tau)  &= 30\tau^2 - 60\tau^3 + 30\tau^4  \\
        h''(\tau) &= 60\tau - 180\tau^2 + 120\tau^3

    Args:
        tau: normalised progress ``(N,)`` in ``[0, 1]``.

    Returns:
        Tuple ``(position_scale, velocity_scale, acceleration_scale)``, each
        ``(N,)``.
    """
    t2 = tau * tau
    t3 = t2 * tau
    t4 = t3 * tau
    t5 = t4 * tau
    h_p = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    h_v = 30.0 * t2 - 60.0 * t3 + 30.0 * t4
    h_a = 60.0 * tau - 180.0 * t2 + 120.0 * t3
    return h_p, h_v, h_a
