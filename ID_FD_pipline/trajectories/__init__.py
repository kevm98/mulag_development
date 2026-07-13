"""Shared trajectory generators for stop-and-mow controller and PPO environment.

All generators are batch-compatible, deterministic, side-effect free, and
produce :class:`CutterTrajectorySample` instances.  Positions are in the truck
``base_link`` frame; the caller anchors them at the current station.
"""

from .types import CutterTrajectorySample
from .time_scaling import cosine_scaling, quintic_scaling
from .line import line_reference
from .arc_mowing import arc_mowing_reference
from .bowed_curve import bowed_curve_reference
from .sine_sweep import sine_sweep_reference
from .sampler import TrajectorySampler

__all__ = [
    "CutterTrajectorySample",
    "cosine_scaling",
    "quintic_scaling",
    "line_reference",
    "arc_mowing_reference",
    "bowed_curve_reference",
    "sine_sweep_reference",
    "TrajectorySampler",
]
