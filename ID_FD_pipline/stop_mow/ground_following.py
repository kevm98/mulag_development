"""Ground-following height correction for the cutter.

Computes deterministic height offsets to keep the cutter near a target ground
clearance.  Initially uses the cutter world-Z only; later phases may add
left/right height sensors for uneven terrain.
"""

from __future__ import annotations

import torch


class GroundFollower:
    """Deterministic ground-height correction using cutter world-Z.

    Does NOT mutate the nominal path.  Produces a height correction that the
    caller adds to the nominal goal.
    """

    def __init__(
        self,
        target_clearance_m: float = 0.05,
        max_correction_m: float = 0.12,
        correction_gain: float = 0.5,
    ):
        self.target_clearance = float(target_clearance_m)
        self.max_correction = float(max_correction_m)
        self.correction_gain = float(correction_gain)

    @torch.no_grad()
    def height_error(self, cutter_world_z: torch.Tensor) -> torch.Tensor:
        """Height error: positive = cutter too high.

        Args:
            cutter_world_z: ``(N,)`` cutter bottom world-Z [m].

        Returns:
            ``(N,)`` height error [m]. Positive means cutter needs to go DOWN.
        """
        return cutter_world_z - self.target_clearance

    @torch.no_grad()
    def correction(self, cutter_world_z: torch.Tensor) -> torch.Tensor:
        """Compute bounded truck-frame Z correction.

        Args:
            cutter_world_z: ``(N,)`` cutter bottom world-Z [m].

        Returns:
            ``(N,)`` correction to add to the nominal goal Z [m].
            Positive = move goal upward to compensate for cutter being too low.
        """
        error = self.height_error(cutter_world_z)
        # negative error = cutter too low -> positive correction = move goal up
        correction = -self.correction_gain * error
        return correction.clamp(-self.max_correction, self.max_correction)
