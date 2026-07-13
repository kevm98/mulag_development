"""Reference manager: trajectory sampling + ground correction + cylinder seeds.

Wraps a :class:`~trajectories.sampler.TrajectorySampler` and a
:class:`GroundFollower` to produce the complete nominal cutter reference at
every step.  Cylinder IK seeds are computed from the cached ground Jacobian
pseudoinverse when available.
"""

from __future__ import annotations

import torch

from ID_FD_pipline.trajectories.sampler import TrajectorySampler
from ID_FD_pipline.trajectories.types import CutterTrajectorySample
from .ground_following import GroundFollower


class ReferenceManager:
    """Combined trajectory, ground, and cylinder-seed reference provider.

    One instance per environment batch.  The caller advances the sampler and
    applies corrections.
    """

    def __init__(
        self,
        sampler: TrajectorySampler,
        ground_follower: GroundFollower | None = None,
        *,
        ground_pinv: torch.Tensor | None = None,
        head_down_cylinder_seed: float | None = None,
    ):
        self.sampler = sampler
        self.ground_follower = ground_follower or GroundFollower()
        self.ground_pinv = ground_pinv
        self.head_down_seed = head_down_cylinder_seed

    @torch.no_grad()
    def step(self) -> CutterTrajectorySample:
        """Advance the sampler and return the next nominal sample."""
        self.sampler.advance()
        return self.sampler.sample()

    @torch.no_grad()
    def sample(self) -> CutterTrajectorySample:
        """Return the current nominal sample without advancing."""
        return self.sampler.sample()

    @torch.no_grad()
    def corrected_sample(
        self,
        cutter_world_z: torch.Tensor | None = None,
    ) -> CutterTrajectorySample:
        """Return the nominal sample with ground-height correction applied.

        Args:
            cutter_world_z: ``(N,)`` current cutter world-Z [m] for height correction.
                If None, no correction is applied.

        Returns:
            ``CutterTrajectorySample`` with corrected ``position_b``.
        """
        nominal = self.sampler.sample()
        corrected = CutterTrajectorySample(
            position_b=nominal.position_b.clone(),
            orientation_w=nominal.orientation_w.clone(),
            linear_velocity_b=nominal.linear_velocity_b.clone(),
            linear_acceleration_b=nominal.linear_acceleration_b.clone(),
            angular_velocity_w=nominal.angular_velocity_w.clone(),
            path_distance_m=nominal.path_distance_m.clone(),
            normalized_progress=nominal.normalized_progress.clone(),
            stroke_direction=nominal.stroke_direction.clone(),
            cylinder_seed=nominal.cylinder_seed,
        )

        if cutter_world_z is not None and self.ground_follower is not None:
            h_correction = self.ground_follower.correction(cutter_world_z)
            corrected.position_b[:, 2] = corrected.position_b[:, 2] + h_correction

        return corrected

    @torch.no_grad()
    def compute_cylinder_seed(
        self,
        goal_local: torch.Tensor,
        anchor_seed: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute a cylinder IK seed from the ground Jacobian pseudoinverse.

        Args:
            goal_local: ``(N, 3)`` desired cutter position in truck frame.
            anchor_seed: ``(N, 4)`` cylinder configuration at the anchor.

        Returns:
            ``(N, 4)`` cylinder seed, or None if no pseudoinverse is available.
        """
        if self.ground_pinv is None:
            return None
        pinv = self.ground_pinv.to(device=goal_local.device, dtype=goal_local.dtype)
        delta_ee = goal_local[:, :3] - self.sampler.anchor[:, :3]
        delta_cyl = delta_ee @ pinv.T
        seed = anchor_seed.clone()
        seed[:, :3] = anchor_seed[:, :3] + delta_cyl
        if self.head_down_seed is not None and float(self.head_down_seed) > 0:
            seed[:, 3] = float(self.head_down_seed)
        return seed

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset sampler progress."""
        self.sampler.reset(env_ids)
