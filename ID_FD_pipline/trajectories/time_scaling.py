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
