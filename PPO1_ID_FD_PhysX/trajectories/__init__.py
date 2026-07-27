"""Smooth time scaling used by the validated head-down arc controller."""

from .time_scaling import cosine_scaling, quintic_scaling
from .workspace_shapes import SUPPORTED_SHAPES, generate_workspace_trajectory

__all__ = [
    "cosine_scaling",
    "quintic_scaling",
    "SUPPORTED_SHAPES",
    "generate_workspace_trajectory",
]
