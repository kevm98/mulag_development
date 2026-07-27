"""Smooth time scaling used by the validated head-down arc controller."""

from .time_scaling import cosine_scaling, quintic_scaling

__all__ = [
    "cosine_scaling",
    "quintic_scaling",
]
