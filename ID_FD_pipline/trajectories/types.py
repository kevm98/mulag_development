"""Typed trajectory samples for batched cutter-space reference generation.

Every sample carries the nominal cutter pose, velocity, acceleration, and
auxiliary state needed by the controller cascade.  All positions are in the
truck ``base_link`` frame by convention; no world-frame coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CutterTrajectorySample:
    """Single batched sample of a cutter reference trajectory.

    All tensors have leading batch dimension ``(N,)``.

    ``position_b`` and ``linear_velocity_b`` are in the truck ``base_link``
    frame.  ``orientation_w`` and ``angular_velocity_w`` are world-frame
    quaternions / angular velocities (currently unused for position-only
    control, but included for future orientation tracking).
    """

    position_b: torch.Tensor          # (N, 3)  cutter tip in truck base_link
    orientation_w: torch.Tensor       # (N, 4)  world-frame quaternion (identity by default)

    linear_velocity_b: torch.Tensor   # (N, 3)  desired cutter velocity in base_link
    linear_acceleration_b: torch.Tensor  # (N, 3)  desired cutter acceleration in base_link
    angular_velocity_w: torch.Tensor  # (N, 3)  world-frame angular velocity (zeros by default)

    path_distance_m: torch.Tensor     # (N,)    accumulated arc length [m]
    normalized_progress: torch.Tensor # (N,)    progress in [0, 1] within current stroke
    stroke_direction: torch.Tensor    # (N,)    +1 = outward, -1 = return

    cylinder_seed: torch.Tensor | None  # (N, 4) or None  mapped IK seed

    @classmethod
    def zeros(cls, batch_size: int, device: torch.device) -> "CutterTrajectorySample":
        """Return a zero-filled sample for masking / initialization."""
        return cls(
            position_b=torch.zeros(batch_size, 3, device=device),
            orientation_w=torch.cat([
                torch.ones(batch_size, 1, device=device),
                torch.zeros(batch_size, 3, device=device),
            ], dim=-1),
            linear_velocity_b=torch.zeros(batch_size, 3, device=device),
            linear_acceleration_b=torch.zeros(batch_size, 3, device=device),
            angular_velocity_w=torch.zeros(batch_size, 3, device=device),
            path_distance_m=torch.zeros(batch_size, device=device),
            normalized_progress=torch.zeros(batch_size, device=device),
            stroke_direction=torch.ones(batch_size, device=device),
            cylinder_seed=None,
        )

    def to(self, device: torch.device) -> "CutterTrajectorySample":
        """Move all tensors to *device*."""
        return CutterTrajectorySample(
            position_b=self.position_b.to(device),
            orientation_w=self.orientation_w.to(device),
            linear_velocity_b=self.linear_velocity_b.to(device),
            linear_acceleration_b=self.linear_acceleration_b.to(device),
            angular_velocity_w=self.angular_velocity_w.to(device),
            path_distance_m=self.path_distance_m.to(device),
            normalized_progress=self.normalized_progress.to(device),
            stroke_direction=self.stroke_direction.to(device),
            cylinder_seed=self.cylinder_seed.to(device) if self.cylinder_seed is not None else None,
        )
