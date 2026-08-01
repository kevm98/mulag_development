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


def verge_observation_dim(
    *,
    profile_bins: int = HEIGHT_PROFILE_BINS,
    model_error: bool = False,
    model_error_history: int = 1,
) -> int:
    extra = (
        MODEL_ERROR_CHANNELS * max(int(model_error_history), 1) if model_error else 0
    )
    return profile_bins + 44 + extra


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
    obs = torch.nan_to_num(torch.cat(columns, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    expected = (
        verge_observation_dim(
            profile_bins=height_profile.shape[-1],
            model_error=model_error is not None,
        )
        + 1
    )
    if obs.shape[-1] != expected:
        raise RuntimeError(
            f"Verge observation dim mismatch: got {obs.shape[-1]}, expected {expected}."
        )
    return obs
