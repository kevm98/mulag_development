"""Observation for verge mowing.

The trajectory-library preview is gone. PPO2 previewed a *stored path* two
numbers wide (an XY delta 0.25 s ahead), which only works when the ground is a
known geometry from a fixed library. Here the reference is the ground itself, so
the policy is given a ray-cast height profile of the verge ahead of the cutter.

That substitution is the whole generalization argument: nothing in the
observation identifies which verge this is, so a policy that can hold cut height
over the training strip has no way to be doing it by recall.
"""

from __future__ import annotations

import torch

# Along-track samples of the verge profile ahead of the cutter.
HEIGHT_PROFILE_BINS = 9
PATH_PREVIEW_BINS = 3

VERGE_OBS_DIM = HEIGHT_PROFILE_BINS + 44


def nonfinite_row_mask(*tensors: torch.Tensor | None) -> torch.Tensor:
    """Return one flag per environment when any raw state tensor is non-finite.

    Observation construction deliberately scrubs NaN/Inf so a malformed value
    cannot reach the actor.  The environment still needs the *raw* corruption
    mask, however, so it can terminate and recover that simulation instead of
    silently training on a zero-filled observation.
    """
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        raise ValueError("nonfinite_row_mask requires at least one tensor")
    rows = int(present[0].shape[0])
    mask = torch.zeros(rows, dtype=torch.bool, device=present[0].device)
    for tensor in present:
        if tensor.ndim == 0 or int(tensor.shape[0]) != rows:
            raise ValueError(
                f"raw state tensor shape {tuple(tensor.shape)} does not have {rows} rows"
            )
        values = tensor.to(device=mask.device).reshape(rows, -1)
        mask |= ~torch.isfinite(values).all(dim=-1)
    return mask


#: Extra columns exposing ACTUATOR MODEL ERROR. See build_verge_observation.
MODEL_ERROR_CHANNELS = 12

#: Extra columns exposing the HEAD JOINT FRAME. See build_verge_observation.
HEAD_FRAME_CHANNELS = 2

HEAD_FRAME_CHANNEL_NAMES = ("head_joint_signed_tilt", "arm_slew_rad")

#: Extra columns exposing CONTACT STATE. See build_verge_observation.
CONTACT_STATE_CHANNELS = 6

#: Names of the contact-state columns, in order. Referenced by the migration
#: utility and by tests, so the ordering is a contract and not a comment.
CONTACT_STATE_CHANNEL_NAMES = (
    "contact_boolean",
    "log_contact_force",
    "contact_force_rate",
    "airborne_gap",
    "contact_acquired",
    "contact_loss_steps",
)


def verge_observation_dim(
    *,
    profile_bins: int = HEIGHT_PROFILE_BINS,
    model_error: bool = False,
    model_error_history: int = 1,
    contact_state: bool = False,
    head_frame: bool = False,
) -> int:
    extra = (
        MODEL_ERROR_CHANNELS * max(int(model_error_history), 1) if model_error else 0
    )
    if contact_state:
        extra += CONTACT_STATE_CHANNELS
    if head_frame:
        extra += HEAD_FRAME_CHANNELS
    return profile_bins + 44 + extra


def override_cfg(cfg, name: str, value):
    """Set a config field, refusing to invent one.

    ``configclass`` objects accept arbitrary attribute assignment, so a
    misspelled override is not an error -- it is a silent no-op that produces
    a sweep row identical to its baseline and looks like "this knob does
    nothing". That happened here: an ``action_scale`` sweep was written as
    ``sole_action_scale`` and returned three bit-identical rows on both
    actuators before the duplication gave it away.

    Every CLI override goes through this, so the next such typo fails at
    startup naming the field instead of costing a run.
    """
    if not hasattr(cfg, name):
        raise AttributeError(
            f"{type(cfg).__name__} has no field {name!r}; refusing to create "
            "it, because an invented config field is silently ignored by the "
            "environment and produces a result identical to the baseline"
        )
    setattr(cfg, name, value)
    return value


def observation_space_for_cfg(cfg) -> int:
    """Observation width implied by a config, including the trailing channel.

    Seven entry points (training, evaluation, distillation, four probes and
    two readiness checks) each rebuilt this from the same three flags. Adding
    an eighth flag to six of seven is how an entry point silently starts
    constructing a different-width network than the one training produced, so
    the derivation lives here and every caller asks for it.
    """
    return (
        verge_observation_dim(
            model_error=bool(getattr(cfg, "terrain_model_error_observation", False)),
            model_error_history=int(
                getattr(cfg, "terrain_model_error_history", 1)
            ),
            contact_state=bool(getattr(cfg, "terrain_contact_observation", False)),
            head_frame=bool(getattr(cfg, "terrain_head_frame_observation", False)),
        )
        + 1
    )


def collapse_scan_width(scan: torch.Tensor, *, bins: int = HEIGHT_PROFILE_BINS) -> torch.Tensor:
    """Collapse a path-ordered ``(along_track, width)`` ray grid by median.

    The ray caster must use ``ordering="yx"`` so each contiguous group contains
    the lateral samples at one along-track position. The returned first column
    is the ground under the cutter and the remaining columns look ahead.
    """
    if scan.ndim != 2:
        raise ValueError(f"Expected a 2-D ray scan, got shape {tuple(scan.shape)}.")
    if bins <= 0 or scan.shape[1] % bins != 0:
        raise ValueError(
            f"Ray count {scan.shape[1]} must be divisible by {bins} profile bins."
        )
    width_samples = scan.shape[1] // bins
    return scan.reshape(scan.shape[0], bins, width_samples).median(dim=2).values


def build_verge_observation(
    *,
    height_profile: torch.Tensor,
    cut_height_error: torch.Tensor,
    lateral_error: torch.Tensor,
    cutter_vertical_velocity: torch.Tensor,
    measured_cylinder_extensions: torch.Tensor,
    measured_cylinder_velocities: torch.Tensor,
    cylinder_headroom_low: torch.Tensor,
    cylinder_headroom_high: torch.Tensor,
    previous_applied_action: torch.Tensor,
    current_valve_commands: torch.Tensor,
    rom_delta_extension: torch.Tensor,
    commanded_target_lead: torch.Tensor,
    head_down_alignment: torch.Tensor,
    head_down_alignment_rate: torch.Tensor,
    truck_speed_mps: torch.Tensor,
    scan_valid_fraction: torch.Tensor,
    head_down_axis_xy: torch.Tensor,
    cutter_contact_force_normalized: torch.Tensor,
    path_preview: torch.Tensor,
    model_error: torch.Tensor | None = None,
    model_error_history: int = 1,
    contact_state: torch.Tensor | None = None,
    head_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the verge-mowing observation.

    Layout (with the default 9 profile bins, total 54):

        0:9    height_profile -- verge height ahead of the cutter, relative to
               the cutter, nearest bin first. Positive means ground below.
        9      cut_height_error = measured cut height - target
        10     lateral_error    = cutter y - verge centre line
        11     cutter_vertical_velocity
        12:16  measured_cylinder_extensions
        16:20  measured_cylinder_velocities
        20:24  cylinder_headroom_low   (normalized)
        24:28  cylinder_headroom_high  (normalized)
        28:32  previous_applied_action
        32:36  current_valve_commands (predicted inside the ROM; plant state)
        36:40  rom_delta_extension
        40:44  commanded_target_lead
        44     head_down_alignment
        45     head_down_alignment_rate
        46     truck_speed_mps

        47     scan_valid_fraction
        48:50  signed world-XY components of the cutter's local +Z axis
        50     filtered cutter-to-terrain interaction force, normalized
        51:54  red-path lateral offsets 0.5, 1.0 and 2.0 m ahead,
               relative to the current path target

    With ``model_error`` supplied, 12 further columns are appended:

        54:58  cylinder tracking residual = measured extension - commanded
               target. Under ROM-A this is near zero because the ROM reaches
               what it is told; under a LEARNED actuator it is the model's
               standing error.
        58:62  actuator prediction residual = realized delta extension - the
               delta the actuator model predicted for the PREVIOUS step. This
               is literally "how wrong was the model last step".
        62:66  normalized cylinder effort -- the hydraulic load the actuator is
               working against, which is what a learned model gets wrong first
               when the tool is in contact.

    Why these are needed at all: the policy is a memoryless MLP. It already
    sees the model's PREDICTION (rom_delta_extension) and the CURRENT measured
    extension, but computing the error requires differencing position across
    two steps and comparing against the previous prediction. A feedforward
    network cannot form that from one observation, so model error is invisible
    to it no matter how much it trains. Measured 2026-08-01: p01 head alignment
    sat at 0.79-0.80 across ten PPO blocks and was unmoved by feedforward
    source or levelling gain -- the policy had no observable evidence of the
    error it was being asked to correct.

    With ``contact_state`` supplied, 6 further columns are appended AFTER the
    model-error block (so a checkpoint migrated to the model-error width keeps
    every column meaning what it meant):

        +0  contact_boolean     -- the physical PhysX contact bit
        +1  log_contact_force   -- log1p(F / force_scale)
        +2  contact_force_rate  -- (F - F_prev) / dt, normalized and clipped
        +3  airborne_gap        -- geometric gap above the support height,
                                   normalized, zero while supported
        +4  contact_acquired    -- has this episode ever had real support
        +5  contact_loss_steps  -- consecutive missing-contact steps, normalized

    Why the log, specifically. The pre-existing
    ``cutter_contact_force_normalized`` column clips ``F / 1000`` at 10, i.e.
    at 10 kN. Measured on the held-out verge the in-contact force runs at
    19.7 kN mean with a 119 kN maximum on ROM-A and 170 kN on ID/FD
    (reports/contact_safe_v1/prefix_ppo3_s_curve_seed17.json), so EVERY force
    the policy has ever had to react to sits on the flat part of that channel:
    stable support and a 119 kN impact are the same number in the observation.
    A policy cannot regulate a quantity it cannot see, and it cannot anticipate
    an impact whose approach is invisible -- which is why the force RATE is
    here too. The clipped column is retained rather than replaced so migrated
    checkpoints keep their learned weights on it.

    ``scan_valid_fraction`` lets the policy tell a measured
    profile from a fallback one. A sensor that sees nothing returns the strip's
    base height, which is indistinguishable from genuinely flat ground unless the
    validity is also visible. The two signed axis components are appended after
    the legacy 48-D prefix so old terrain checkpoints can be migrated by adding
    zero input weights. The physical contact-force channel is appended last so
    later PPO can distinguish actual terrain support from a geometric height
    estimate. Alignment alone is an unsigned cosine and cannot tell PPO which
    direction cylinder 4 must move to recover the world-down pose.
    """
    columns = [
        height_profile,
        cut_height_error.reshape(-1, 1),
        lateral_error.reshape(-1, 1),
        cutter_vertical_velocity.reshape(-1, 1),
        measured_cylinder_extensions,
        measured_cylinder_velocities,
        cylinder_headroom_low,
        cylinder_headroom_high,
        previous_applied_action,
        current_valve_commands,
        rom_delta_extension,
        commanded_target_lead,
        head_down_alignment.reshape(-1, 1),
        head_down_alignment_rate.reshape(-1, 1),
        truck_speed_mps.reshape(-1, 1),
        scan_valid_fraction.reshape(-1, 1),
        head_down_axis_xy,
        cutter_contact_force_normalized.reshape(-1, 1),
        path_preview,
    ]
    if model_error is not None:
        columns.append(model_error)
    if contact_state is not None:
        columns.append(contact_state)
    if head_frame is not None:
        columns.append(head_frame)
    obs = torch.nan_to_num(torch.cat(columns, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    expected = (
        verge_observation_dim(
            profile_bins=height_profile.shape[-1],
            model_error=model_error is not None,
            # Previously omitted, so any history depth above 1 raised a
            # spurious dimension mismatch and the depth-3 stack the
            # model-error docstring recommends could never be enabled.
            model_error_history=int(model_error_history),
            contact_state=contact_state is not None,
            head_frame=head_frame is not None,
        )
        + 1
    )
    if obs.shape[-1] != expected:
        raise RuntimeError(
            f"Verge observation dim mismatch: got {obs.shape[-1]}, expected {expected}."
        )
    return obs
