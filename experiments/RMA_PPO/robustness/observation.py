"""Instantiate the frozen verge PPO's 66-D observation on the CPU surrogate.

The frozen policy is the verge-mowing PPO4_1 actor.  Its observation contract
(``verge_task/terrain_observations.build_verge_observation``) is

    0:9    height_profile           14:  terrain, NOT modelled by the surrogate
    9      cut_height_error             terrain, NOT modelled
    10     lateral_error                terrain, NOT modelled
    11     cutter_vertical_velocity     tool kinematics, NOT modelled
    12:16  measured_cylinder_extensions        <- surrogate state
    16:20  measured_cylinder_velocities        <- surrogate state
    20:24  cylinder_headroom_low  (/span)      <- surrogate state + limits
    24:28  cylinder_headroom_high (/span)      <- surrogate state + limits
    28:32  previous_applied_action             <- surrogate
    32:36  current_valve_commands              <- surrogate (applied valve)
    36:40  rom_delta_extension                 <- surrogate (SD delta)
    40:44  commanded_target_lead               <- surrogate (see below)
    44     head_down_alignment          tool pose, NOT modelled
    45     head_down_alignment_rate     tool pose, NOT modelled
    46     truck_speed_mps              chassis, NOT modelled
    47     scan_valid_fraction          sensor, NOT modelled
    48:50  head_down_axis_xy            tool pose, NOT modelled
    50     cutter_contact_force         contact, NOT modelled
    51:54  path_preview                 path, NOT modelled
    54:58  cylinder tracking residual   USD-space quantity, NOT reproducible
    58:62  actuator prediction residual        <- surrogate
    62:66  normalized cylinder effort   joint torque, NOT modelled

Channels the surrogate genuinely produces are computed exactly.  Every other
channel is held at the frozen policy's OWN observation-normalizer mean, i.e.
at standard score zero.  That is the least-informative in-distribution value
available; the previous surrogate used arbitrary constants (headroom = 0.10
against a training mean of 0.762 +/- 0.007, a standard score of -99), which
put the policy far outside its own input support before any stress was
applied.

Reference injection.  A tracking-only surrogate has no terrain, so the task
command cannot enter through the height profile.  It is injected through
``commanded_target_lead`` (40:44), the channel that already means "where I
have been told to go, minus where I am".  This is a proxy, declared here and
held IDENTICAL at every stress level, so it cannot bias the robustness curve.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path("/Volumes/DATA/rrlab_mulag")
MAPPING_PATH = REPO / "mulag_plant/runtime/cylinder_coordinate_mapping.json"

# Channels the surrogate computes from its own state; everything else is
# imputed at the frozen normalizer mean.
COMPUTED_CHANNELS = tuple(range(12, 44)) + tuple(range(58, 62))


def cylinder_learned_limits() -> tuple[torch.Tensor, torch.Tensor]:
    """Learned-space (Simscape extension) travel limits for the four cylinders.

    Reconstructed through the repository's own coordinate adapter.  Verified
    against the frozen checkpoint: feeding the checkpoint's mean extensions
    through the headroom formula reproduces its channel 20:28 means to four
    decimal places on all four cylinders.
    """
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from mulag_plant.runtime.cylinder_coordinate_adapter import CylinderCoordinateMapping
    from mulag_plant.runtime.cylinder_space_scaling import CylinderSpaceScaling

    raw = json.loads(MAPPING_PATH.read_text())
    cylinders = raw["cylinders"]
    names = sorted(cylinders, key=lambda k: cylinders[k]["rom_index"])
    mapping = CylinderCoordinateMapping.from_json(
        MAPPING_PATH, mode="nominal_sign_offset", allow_provisional_nominal_reference=True
    )
    usd_limits = torch.tensor(
        [[[cylinders[n]["isaac_min"], cylinders[n]["isaac_max"]] for n in names]],
        dtype=torch.float64,
    )
    scaling = CylinderSpaceScaling.from_coordinate_mapping(
        mapping=mapping, cylinder_names=names,
        dof_indices=torch.arange(4), usd_limits=usd_limits,
    )
    return (scaling.learned_min.float().reshape(1, 4),
            scaling.learned_max.float().reshape(1, 4))


class VergeObservationAdapter:
    """Build the frozen 66-D verge observation from surrogate state."""

    def __init__(self, obs_mean: torch.Tensor):
        self.obs_mean = obs_mean.detach().reshape(1, -1).clone()
        if self.obs_mean.shape[-1] != 66:
            raise ValueError("the frozen verge contract is 66-D")
        self.learned_min, self.learned_max = cylinder_learned_limits()
        self.span = torch.clamp(self.learned_max - self.learned_min, min=1.0e-6)

    def build(
        self,
        *,
        extension: torch.Tensor,
        velocity: torch.Tensor,
        prev_action: torch.Tensor,
        applied_valve: torch.Tensor,
        rom_delta: torch.Tensor,
        target_lead: torch.Tensor,
        prediction_residual: torch.Tensor,
    ) -> torch.Tensor:
        n = extension.shape[0]
        obs = self.obs_mean.expand(n, -1).clone()
        obs[:, 12:16] = extension
        obs[:, 16:20] = velocity
        obs[:, 20:24] = torch.clamp(extension - self.learned_min, min=0.0) / self.span
        obs[:, 24:28] = torch.clamp(self.learned_max - extension, min=0.0) / self.span
        obs[:, 28:32] = prev_action
        obs[:, 32:36] = applied_valve
        obs[:, 36:40] = rom_delta
        obs[:, 40:44] = target_lead
        obs[:, 58:62] = torch.clamp(prediction_residual / self.span, -5.0, 5.0)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
