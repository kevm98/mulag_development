#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Evaluate a PPO-ROM checkpoint on held-out terrain and eight gates."""

from __future__ import annotations

import argparse
import cProfile
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402
from PPO_ROM.path_shapes import (  # noqa: E402
    apply_trajectory_shapes,
    parse_trajectory_shapes,
    require_enough_environments,
)
from verge_task.vehicle_speed import configure_path_horizon  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", nargs="?", default=None)
p.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument(
    "--steps",
    type=int,
    default=2808,
    help="Maximum control steps; the 2 km/h, 26 m path completes at about 2808.",
)
p.add_argument(
    "--natural-visuals",
    "--natural_visuals",
    dest="natural_visuals",
    action="store_true",
    help=(
        "Use the lightweight two-lane road, matte grass/soil skin and sparse "
        "grass clumps. Visual-only: no textures, collision or observation changes."
    ),
)
p.add_argument(
    "--stage",
    type=int,
    default=6,
    help="Curriculum stage; it must match the checkpoint (final task: 6).",
)
p.add_argument("--meshes", "--mesh", dest="meshes", default="1337,17,909")
p.add_argument("--report", default="")
p.add_argument(
    "--profile-out",
    default="",
    help=argparse.SUPPRESS,
)
p.add_argument(
    "--trace-out",
    "--trace_out",
    dest="trace_out",
    default="",
    help=(
        "Write env-0 per-step EE and arm-joint tracking data to a compressed "
        "NPZ file. Use with exactly one mesh."
    ),
)
p.add_argument(
    "--trajectory-shape",
    "--trajectory_shape",
    "--path-family",
    dest="trajectory_shape",
    default="",
    help="Use one cutter tracking-line shape in every environment.",
)
p.add_argument(
    "--trajectory-shapes",
    "--trajectory_shapes",
    "--path-families",
    dest="trajectory_shapes",
    default="",
    help="Comma-separated shapes, or 'all', distributed across environments.",
)
p.add_argument(
    "--vehicle-speed-kmh",
    "--vehicle_speed_kmh",
    # The promotion contract and the post-training eval both name their speeds
    # `unimog_speed_2` .. `unimog_speed_7`, so accept that spelling directly
    # rather than making the caller translate it at every call site.
    "--unimog-speed",
    "--unimog_speed",
    dest="vehicle_speed_kmh",
    type=float,
    default=None,
    help=(
        "Pin one scripted truck speed in km/h; default samples the task's "
        "2--7 km/h range. --unimog_speed is the same option under the name the "
        "gate contract uses."
    ),
)
p.add_argument(
    "--vehicle-speed-min-kmh",
    "--vehicle_speed_min_kmh",
    dest="vehicle_speed_min_kmh",
    type=float,
    default=None,
    help="Minimum constant truck speed sampled once at reset.",
)
p.add_argument(
    "--vehicle-speed-max-kmh",
    "--vehicle_speed_max_kmh",
    dest="vehicle_speed_max_kmh",
    type=float,
    default=None,
    help="Maximum constant truck speed sampled once at reset.",
)
p.add_argument(
    "--episode-length-s",
    "--episode_length_s",
    dest="episode_length_s",
    type=float,
    default=None,
    help="Override episode duration; the 2--7 km/h range uses about 47.3 seconds.",
)
p.add_argument(
    "--cylinder_target_max_lead_per_axis_m",
    default="",
    help=(
        "Per-cylinder anti-windup lead caps in metres, comma separated, one "
        "per cylinder (e.g. '0.030,0.030,0.030,0.02'). The task default is now "
        "0.030 on c1-c3: measured 2026-08-14 over meshes 1337/17/909, a 0.05 "
        "cap let a cylinder chase its own target into its stop and pinned "
        "headroom at exactly 0.0000 above 4 km/h, while 0.030 holds "
        "0.037-0.039 m flat from 2 to 7 km/h and clears every gate. An earlier "
        "note here credited 0.05 with halving boom_sweep lateral error; that "
        "predates boom_sweep's re-authoring to a pure-XY sweep and no longer "
        "reproduces. Relaxing the c4 cap still collapses head alignment."
    ),
)
p.add_argument(
    "--cylinder_target_leak",
    dest="cylinder_target_leak",
    type=float,
    default=None,
    help=(
        "Leaky-integration coefficient for the cylinder reference target. Only "
        "has an effect when the task integrates from the previous target "
        "(cylinder_target_integrates_from_previous_target=True, which the "
        "ID-FD and ID(AAM)-FD tasks set and ROM does not). At 0.0 the "
        "integrator is unleaked, so a persistent action bias accumulates until "
        "a cylinder reaches its stop; a small leak pulls the target back "
        "toward the measured joint position so the bias settles at a bounded "
        "offset of delta/leak instead."
    ),
)
p.add_argument(
    "--cylinder_target_limit_inset_m",
    dest="cylinder_target_limit_inset_m",
    type=float,
    default=None,
    help=(
        "Stroke-limit reference governor, in metres of learned extension. The "
        "commanded cylinder target is clamped this far inside each stop, so the "
        "reference cannot ask for the last of the travel. The cylinder_headroom "
        "gate wants a standing 0.005 m, which the plain soft-limit clamp does "
        "not provide -- it permits sitting exactly on the stop. 0 disables."
    ),
)
p.add_argument(
    "--cylinder_target_integrates_from_previous_target",
    dest="integrates_from_prev",
    choices=("true", "false"),
    default=None,
    help=(
        "Integrate the cylinder reference target from the PREVIOUS target "
        "(true) or from the MEASURED joint position (false). This is the one "
        "structural difference between the ROM actuator and the ID-FD/AAM ones: "
        "ROM integrates from measured and holds cylinder 3 flat at a stroke "
        "fraction of 0.34 across the whole 26 m path, while the ID/FD tasks "
        "integrate from the previous target and walk it out to 0.39."
    ),
)
p.add_argument(
    "--cylinder_target_lead_high_speed_per_axis_m",
    dest="lead_high_speed",
    default="",
    help=(
        "Comma-separated per-cylinder lead caps used ABOVE the schedule speed. "
        "Empty = single-value behaviour. The cap trades lateral tracking "
        "against impact energy and which binds depends on speed: 0.040 clears "
        "lateral at 4 km/h but fails hard impacts at 5-7, while 0.030 does the "
        "reverse."
    ),
)
p.add_argument(
    "--cylinder_target_lead_schedule_speed_kmh",
    dest="lead_schedule_speed",
    type=float,
    default=None,
    help="Truck speed in km/h above which the high-speed lead row applies.",
)
p.add_argument(
    "--online_aam_latent_mode",
    dest="online_aam_latent_mode",
    choices=('estimated', 'privileged', 'zero', 'mean', 'shuffled', 'frozen'),
    default=None,
    help=(
        "Degrade only the two PPO latent observation columns to test whether "
        "the policy uses them; the AAM-conditioned ID adapter remains on the "
        "same estimated latent in every arm. "
        "'shuffled' is the decisive arm: same marginal distribution as "
        "the deployed estimate, so no out-of-distribution penalty, but "
        "every environment receives another environment's context."
        " 'privileged' is diagnostic-only: PPO receives frozen true z while "
        "the controller remains on estimated z_hat."
        " 'frozen' latches the live estimate at "
        "--online_aam_latent_freeze_tick and holds it: same content and "
        "magnitude as 'estimated', differing only in that it stops updating, "
        "so it isolates ONLINE adaptation from static conditioning."
    ),
)
p.add_argument(
    "--online_aam_latent_freeze_tick",
    type=int,
    default=None,
    help="tick at which latent_mode='frozen' latches the estimate. UNITS: 50 ms MODEL ticks (20 Hz), not 60 Hz control steps -- the AAM counter only advances when the model clock samples.",
)
p.add_argument(
    "--online_aam_controller_latent_mode",
    choices=("estimated", "true", "zero", "mean", "shuffled", "frozen"),
    default=None,
    help=(
        "Controller latent override. Promotion requires estimated; true "
        "evaluates the privileged teacher's native plant. zero/shuffled/frozen "
        "exist for the routing ablation (baseline / ID-only / PPO-only / full) "
        "and are NOT valid for promotion evidence -- promote_system.py "
        "rejects anything but estimated, deliberately."
    ),
)
p.add_argument(
    "--rom_factor_perturbation",
    default=None,
    metavar="FACTOR=LO:HI[,FACTOR=LO:HI...]",
    help=(
        "ROM plant perturbation via system_factors, e.g. 'seal=0.0:0.4'. The "
        "ROM produces the valve rather than consuming it, so valve "
        "effectiveness does not apply there."
    ),
)
p.add_argument(
    "--rom_factor_step",
    default=None,
    metavar="FACTOR=VALUE",
    help="mid-episode ROM factor change, e.g. 'seal=0.9'",
)
p.add_argument(
    "--plant_alpha_range",
    default=None,
    metavar="LO:HI",
    help="ID-FD valve effectiveness range, e.g. '0.7:1.0' (applied at the FD boundary)",
)
p.add_argument(
    "--plant_delay_steps_range",
    default=None,
    metavar="LO:HI",
    help="ID-FD transport delay in control steps, e.g. '0:3'",
)
p.add_argument(
    "--plant_step_time_s",
    type=float,
    default=None,
    help="episode time at which the plant perturbation steps (both plants)",
)
p.add_argument(
    "--plant_step_alpha_scale",
    type=float,
    default=None,
    help="ID-FD: alpha is multiplied by this at --plant_step_time_s",
)
p.add_argument(
    "--adaptation_trace_path",
    default=None,
    help=(
        "write a per-timestep npz (z_hat, errors, valve, true theta) for the "
        "adaptation figures; distance-binned reports cannot resolve the AAM "
        "warmup or a mid-episode step"
    ),
)
p.add_argument(
    "--c3_physical_extension_reserve_m",
    dest="c3_physical_extension_reserve_m",
    type=float,
    default=None,
    help=(
        "Metres of LEARNED extension held back from cylinder 3's stroke by the "
        "PhysX solver, while the cylinder_headroom gate keeps scoring the "
        "original stroke. A mechanical reserve, not controller behaviour. c3's "
        "coordinate scale is negative, so this trims the USD LOWER limit."
    ),
)
p.add_argument(
    "--dls_null_space_centering_gain",
    dest="dls_null_space_centering_gain",
    type=float,
    default=None,
    help=(
        "Spend the redundant DOF of the 3x4 deployment IK solve on centering "
        "cylinder strokes instead of leaving it at minimum-norm. Affects the "
        "reset pose only -- the run-time teacher solves a square 3x3 system and "
        "has no null space. 0 reproduces minimum-norm exactly."
    ),
)
p.add_argument(
    "--teacher",
    action="store_true",
    help=(
        "Score the privileged DLS teacher instead of a checkpoint. The teacher "
        "is not a checkpoint -- it is a runtime env mode, so no policy is "
        "loaded and every environment is driven by the teacher. Use this to "
        "anchor gate thresholds in what the plant can actually achieve."
    ),
)
p.add_argument(
    "--teacher-initial-vertical-offset-m",
    type=float,
    default=None,
    help="Diagnostic override for the privileged teacher's initial vertical offset.",
)
p.add_argument(
    "--teacher-max-vertical-offset-m",
    type=float,
    default=None,
    help="Diagnostic lower-travel bound for privileged teacher calibration.",
)
p.add_argument(
    "--teacher-max-upward-offset-m",
    type=float,
    default=None,
    help="Diagnostic upper bound for privileged teacher force relief.",
)
p.add_argument(
    "--teacher-max-raw-action",
    type=float,
    default=None,
    help="Diagnostic normalized action cap for the privileged teacher.",
)
p.add_argument(
    "--null",
    action="store_true",
    help=(
        "Score the null (zero-action) policy. Gate thresholds must separate a "
        "working controller from doing nothing, and gates.py's own rule "
        "demands >=3 meshes -- this measures that baseline on the same "
        "contract instead of a single pinned mesh."
    ),
)
p.add_argument(
    "--tracking-amplitude",
    "--tracking_amplitude",
    dest="tracking_amplitude",
    type=float,
    default=None,
    help="Pin the path-amplitude randomization band to one scale.",
)
p.add_argument(
    "--task",
    default="",
    help=(
        "Gym task to score. Empty = PPO3_1 (ROM-A). Set "
        "RRLAB-Mulag-Verge-IDFD-v0 to score the ID/FD pipeline on the "
        "IDENTICAL terrain, reward and gates -- that shared path is what makes "
        "a PPO3_1 vs PPO4_1 number a statement about the actuator rather than "
        "about two diverged copies of a script."
    ),
)
p.add_argument(
    "--factor_domain",
    default="",
    choices=("", "nominal", "hydraulic", "mechanical", "combined_train", "combined_test"),
    help="Enable the latched deployment factor manifest for this domain.",
)
p.add_argument(
    "--factor_key",
    default="",
    help="Explicit factor key/run id, e.g. flow_min, flow_max, or combined_test:102.",
)
p.add_argument("--factor_manifest", default="")
p.add_argument(
    "--online-aam-residual-scale",
    "--online_aam_residual_scale",
    dest="online_aam_residual_scale",
    type=float,
    default=None,
    help=(
        "Diagnostic override for PPO-ID(AAM)-FD latent residual actuation. "
        "The AAM remains in the observation; zero isolates residual-actuation "
        "damage without changing the 68-D policy contract."
    ),
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

try:
    selected_shapes = parse_trajectory_shapes(
        single=a.trajectory_shape, multiple=a.trajectory_shapes
    )
    require_enough_environments(selected_shapes, a.num_envs)
except ValueError as exc:
    p.error(str(exc))

meshes = tuple(int(part) for part in a.meshes.split(",") if part.strip())
if not meshes:
    p.error("--meshes must name at least one terrain seed")
if a.adaptation_trace_path and len(meshes) > 1:
    # One path, one file: a second mesh would overwrite the first and leave a
    # trace that silently describes only the last mesh scored.
    p.error(
        "--adaptation_trace_path writes a single file; pass one mesh per "
        "process (score_per_mesh.sh already does this via TRACE_DIR)"
    )
if a.trace_out and len(meshes) != 1:
    p.error("--trace-out requires exactly one mesh")
if a.profile_out and len(meshes) != 1:
    p.error("--profile-out requires exactly one mesh")
if a.checkpoint is None and not a.teacher and not a.null:
    p.error("evaluate_gates.py requires a checkpoint path (or --teacher / --null)")
if a.checkpoint is not None and (a.teacher or a.null):
    p.error("--teacher/--null score a runtime mode, not a checkpoint; pass one or the other")
if a.teacher and a.null:
    p.error("--teacher and --null are mutually exclusive")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO_ROM  # noqa: F401,E402
from PPO_ROM.gates import GATES, check_gates  # noqa: E402

if a.task.startswith("RRLAB-Mulag-Verge-ID-AAM-FD"):
    import importlib

    _pkg = importlib.import_module("PPO_ID_AAM_FD")
    TASK_ID = a.task
    EXPECTED_ACTUATOR = "id_aam_fd"
elif a.task.startswith("RRLAB-Mulag-Verge-IDFD"):
    import importlib

    _pkg = importlib.import_module("PPO_ID_FD")
    TASK_ID = a.task
    EXPECTED_ACTUATOR = "idfd"
else:
    _pkg = PPO_ROM
    TASK_ID = a.task or PPO_ROM.TERRAIN_TASK_ID
    EXPECTED_ACTUATOR = "rom"
print(f"[PPO3-1-GATES] task={TASK_ID} actuator={EXPECTED_ACTUATOR}", flush=True)
from PPO_ROM.terrain_env_cfg import (  # noqa: E402
    TerrainC4AuthorityPPORunnerCfg,
    apply_c4_authority_stage,
)
from verge_task.actuators import assert_checkpoint_actuator  # noqa: E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)


@torch.no_grad()
def safety_component_masks(raw) -> dict[str, torch.Tensor]:
    """Split the raw safety predicate into promotion-relevant causes."""
    cfg = raw.cfg
    cylinder_q = raw.robot.data.joint_pos[:, raw._cylinder_joint_ids]
    limits = raw.robot.data.soft_joint_pos_limits[:, raw._cylinder_joint_ids]
    margin = float(getattr(cfg, "cylinder_limit_margin_m", 0.002))
    components = {
        "cylinder_limit": (
            (cylinder_q < limits[..., 0] - margin)
            | (cylinder_q > limits[..., 1] + margin)
        ).any(dim=-1),
        "nonfinite": raw._terrain_nonfinite_mask(),
        "truck_drift": (
            raw._truck_post_constraint_drift_xy
            > float(getattr(cfg, "truck_drift_termination_m", 0.02))
        )
        | (
            raw._truck_drift_xy
            > float(getattr(cfg, "truck_raw_reaction_termination_m", 0.05))
        ),
        "head_alignment": raw._measure_head_down_alignment()
        < float(getattr(cfg, "head_down_termination_alignment", 0.98)),
        "clearance": raw.cut_height_error().abs()
        > float(getattr(cfg, "cutter_clearance_hard_band_m", 0.20)),
    }
    warmup = (raw._preposition_counter > 0) | (raw._settle_counter > 0)
    components = {name: value & ~warmup for name, value in components.items()}
    components["union"] = torch.stack(tuple(components.values())).any(dim=0)
    return components


_Z_HAT_LIVE: dict = {}

FACTOR_FIELD_NAMES = (
    "flow",
    "compliance",
    "seal",
    "pressure",
    "mass_scale",
    "com_z",
)


def _per_env_headroom(raw, headroom_cyl_steps, active_steps) -> list[dict]:
    """One row per environment: worst headroom, and the factor draw it ran under.

    ``headroom_cyl_steps`` is (steps, envs, cylinders) and ``active_steps`` is
    (steps, envs); an environment parked at the 26 m endpoint must not count,
    which is why the inactive entries are replaced by +inf rather than masked
    away (the mask would ragged the per-environment reduction).
    """
    gate = 0.005
    active = active_steps.unsqueeze(-1)
    masked = torch.where(
        active, headroom_cyl_steps, torch.full_like(headroom_cyl_steps, float("inf"))
    )
    per_env_min = masked.amin(dim=0)
    worst_over_cylinders = headroom_cyl_steps.amin(dim=-1)
    below = (worst_over_cylinders < gate) & active_steps
    active_counts = active_steps.float().sum(dim=0).clamp(min=1.0)
    below_fraction = below.float().sum(dim=0) / active_counts

    keys = list(getattr(raw, "_deployment_factor_keys", []) or [])
    values = getattr(raw, "_deployment_factor_values", None)
    value_rows = values.detach().cpu().tolist() if values is not None else []

    rows = []
    for env_index in range(per_env_min.shape[0]):
        mins = [float(v) for v in per_env_min[env_index]]
        row = {
            "env": int(env_index),
            "factor_key": keys[env_index] if env_index < len(keys) else None,
            "min_headroom_per_cylinder_m": mins,
            "min_headroom_m": min(mins),
            "binding_cylinder": int(min(range(len(mins)), key=lambda i: mins[i])),
            "fraction_steps_below_gate": float(below_fraction[env_index]),
        }
        if env_index < len(value_rows):
            row["factors"] = dict(
                zip(FACTOR_FIELD_NAMES, [float(v) for v in value_rows[env_index]])
            )
        rows.append(row)
    return rows


_EFFORT_LIMITS = None


def _resolve_effort_limits(raw):
    """Per-cylinder effort_limit_sim in the cylinder-joint order."""
    global _EFFORT_LIMITS
    limits = []
    for name in (
        "drehzapfen_cylinder", "ausleger_i_cylinder",
        "ausleger_ii_cylinder", "messerkopf_schwenk_cylinder",
    ):
        actuator = raw.robot.actuators.get(name)
        value = getattr(actuator, "effort_limit_sim", None) if actuator else None
        limits.append(float(torch.as_tensor(value).flatten()[0]) if value is not None else float('inf'))
    _EFFORT_LIMITS = torch.tensor(limits, device=raw.device).reshape(1, 4)
    return _EFFORT_LIMITS


def _headroom_diagnostic(
    measured_headroom,
    target_headroom,
    overshoot,
    torque_ratio,
    measured_x_steps,
    target_x_steps,
    active_steps,
) -> dict:
    """Three tests that separate a target problem from an authority problem."""
    gate = 0.005
    worst = measured_headroom.amin(dim=-1)
    binding = measured_headroom.argmin(dim=-1)
    breach = (worst < gate) & active_steps
    out = {"gate": gate, "breach_steps": int(breach.sum())}
    if not bool(breach.any()):
        return out
    index = binding[breach].reshape(-1, 1)

    # --- test 1: is the TARGET also at its stop when the PLANT is? ---
    target_at_breach = target_headroom[breach].gather(1, index).reshape(-1)
    out["measured_headroom_at_breach_mean"] = float(worst[breach].mean())
    out["target_headroom_at_breach_mean"] = float(target_at_breach.mean())
    out["target_headroom_at_breach_p95"] = float(
        torch.quantile(target_at_breach, 0.95)
    )
    out["fraction_target_also_below_gate"] = float(
        (target_at_breach < gate).float().mean()
    )
    out["overshoot_at_breach_mean"] = float(
        overshoot[breach].gather(1, index).mean()
    )

    # --- test 2: is the drive saturated at those steps? ---
    if torque_ratio is not None:
        ratio_at_breach = torque_ratio[breach].gather(1, index).reshape(-1)
        out["torque_ratio_at_breach_mean"] = float(ratio_at_breach.mean())
        out["torque_ratio_at_breach_p95"] = float(
            torch.quantile(ratio_at_breach, 0.95)
        )
        out["fraction_drive_saturated_at_breach"] = float(
            (ratio_at_breach > 0.98).float().mean()
        )
        healthy = (~breach) & active_steps
        if bool(healthy.any()):
            out["torque_ratio_healthy_mean"] = float(
                torque_ratio[healthy].abs().amax(dim=-1).mean()
            )

    # --- test 3: does the plant follow the target at all? ---
    # d(measured)/d(target) per cylinder over consecutive steps. Near zero is
    # the saturation signature: the target moves and the joint does not.
    d_meas = measured_x_steps[1:] - measured_x_steps[:-1]
    d_targ = target_x_steps[1:] - target_x_steps[:-1]
    live = active_steps[1:] & (d_targ.abs().amax(dim=-1) > 1.0e-6)
    if bool(live.any()):
        num = (d_meas[live] * d_targ[live]).sum(dim=0)
        den = (d_targ[live] * d_targ[live]).sum(dim=0).clamp(min=1.0e-12)
        out["tracking_gain_per_cylinder"] = [float(v) for v in (num / den)]
    breach_env = breach.any(dim=0)
    for label, mask in (("breach_envs", breach_env), ("healthy_envs", ~breach_env)):
        if not bool(mask.any()):
            continue
        sel = live & mask.reshape(1, -1)
        if not bool(sel.any()):
            continue
        num = (d_meas[sel] * d_targ[sel]).sum(dim=0)
        den = (d_targ[sel] * d_targ[sel]).sum(dim=0).clamp(min=1.0e-12)
        out[f"tracking_gain_{label}"] = [float(v) for v in (num / den)]
    out["binding_cylinder_counts"] = [
        int((binding[breach] == i).sum()) for i in range(measured_headroom.shape[-1])
    ]
    return out


@torch.no_grad()
def roll_mesh(raw, policy, steps: int, mesh: int, *, teacher: bool = False) -> dict:
    """Actor-mean rollout on one held-out mesh, collecting the 8 gate metrics.

    With ``teacher=True`` the DLS teacher drives every environment. The mask is
    set AFTER reset because ``_teacher_execute_mask`` is allocated at env
    construction and reset is the point the rollout actually starts from.
    """
    obs, _ = raw.reset(seed=int(a.seed) + mesh)
    if teacher:
        raw.set_teacher_execute_mask(
            torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        )
    errors, accelerations, alignments = [], [], []
    laterals, contacts = [], []
    _resolve_effort_limits(raw)
    headrooms = []
    headrooms_per_cylinder = []
    headroom_lows = []
    headroom_highs = []
    stroke_fractions = []
    target_headrooms = []
    target_overshoots = []
    torque_ratios = []
    measured_xs = []
    target_xs = []
    travels, scan_validities = [], []
    cutter_x_positions, ground_heights, cutter_clearances = [], [], []
    actions, termination_events, hard_impact_events = [], [], []
    active_steps = []
    safety_steps: dict[str, list[torch.Tensor]] = {
        "cylinder_limit": [],
        "nonfinite": [],
        "truck_drift": [],
        "head_alignment": [],
        "clearance": [],
        "union": [],
    }
    trace = None
    if a.trace_out:
        trace = {
            "time_s": [],
            "ee_actual_m": [],
            "ee_reference_m": [],
            "ee_error_m": [],
            "joint_actual_rad": [],
            "joint_reference_rad": [],
            "joint_error_rad": [],
            "travel_m": [],
            "cut_height_error_m": [],
            "lateral_error_m": [],
        }
    _report_every = max(int(steps) // 10, 1)
    _t0 = time.time()
    for _step in range(int(steps)):
        if _step and _step % _report_every == 0:
            _elapsed = time.time() - _t0
            print(
                f"[PPO3-1-GATES] mesh {mesh}: step {_step}/{int(steps)} "
                f"({100.0 * _step / int(steps):.0f}%) {_elapsed:.0f}s elapsed, "
                f"~{_elapsed / _step * (int(steps) - _step):.0f}s left",
                flush=True,
            )
        action = policy(obs)
        obs, _, terminated, truncated, _ = raw.step(action)
        # DirectRLEnv resets inside ``step`` before returning observations. Do
        # not start a second episode in the same held-out rollout once the
        # complete vector reaches the 26 m endpoint.
        if bool(truncated.all()):
            break
        # Faster replicas remain parked at the 26 m endpoint while slower
        # replicas finish. Their parked arm is not part of the active rollout
        # and must not count as a tracking/contact failure in the aggregate.
        active = ~raw._truck_path_complete
        active_steps.append(active)
        if trace is not None:
            ee_actual = raw._compute_ee_pos_local()
            ee_reference = raw._original_reference_tool
            joint_actual = raw.robot.data.joint_pos[:, raw._arm_joint_ids]
            # ``_joint_ref`` is already in the ordered five-arm-joint contract;
            # unlike ``robot.data.joint_pos``, it is not indexed by articulation
            # DOF IDs.  Indexing it with ``_arm_joint_ids`` corrupts GPU memory
            # when those physical IDs are larger than four.
            joint_reference = raw._joint_ref
            trace["time_s"].append(float((_step + 1) * raw.step_dt))
            trace["ee_actual_m"].append(ee_actual[0].detach().cpu().tolist())
            trace["ee_reference_m"].append(ee_reference[0].detach().cpu().tolist())
            trace["ee_error_m"].append(
                float(torch.linalg.norm(ee_reference[0] - ee_actual[0]))
            )
            trace["joint_actual_rad"].append(joint_actual[0].detach().cpu().tolist())
            trace["joint_reference_rad"].append(
                joint_reference[0].detach().cpu().tolist()
            )
            trace["joint_error_rad"].append(
                (joint_reference[0] - joint_actual[0]).detach().cpu().tolist()
            )
            trace["travel_m"].append(float(raw._truck_travel_m[0]))
            trace["cut_height_error_m"].append(float(raw.cut_height_error()[0]))
            trace["lateral_error_m"].append(float(raw.verge_lateral_error()[0]))
        termination_events.append(terminated & ~truncated)
        actions.append(raw._sole_applied.clone())
        errors.append(raw.cut_height_error().abs())
        accelerations.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )
        alignments.append(-raw._head_down_axis_world()[:, 2])
        laterals.append(raw.verge_lateral_error().abs())
        contacts.append(raw.cutter_ground_contact().float())
        measured_x, _ = raw._refresh_cylinder_state()
        headroom_low, headroom_high = raw._cylinder_headroom(measured_x)
        # The gate is the min over all four cylinders, which cannot say which
        # one is against its stop. Keep the un-collapsed (envs, 4) tensor as
        # well, plus the position in each stroke, so a headroom failure can be
        # attributed to a cylinder and to an end of travel.
        headroom_per_cylinder = torch.minimum(headroom_low, headroom_high)
        headrooms.append(headroom_per_cylinder.amin(dim=-1))
        headrooms_per_cylinder.append(headroom_per_cylinder)
        headroom_lows.append(headroom_low)
        headroom_highs.append(headroom_high)
        stroke_fractions.append(
            (measured_x - raw._learned_min)
            / (raw._learned_max - raw._learned_min).clamp(min=1.0e-6)
        )
        # The gate measures the MEASURED joint. Track the COMMANDED target in the
        # same space as well: if the target keeps its clearance while the plant
        # does not, the cylinder is being driven or pushed past what was asked
        # for, and clamping the reference cannot buy headroom.
        target_x = raw._cylinder_adapter.scaling.usd_q_to_learned_x(
            raw._cylinder_pos_target
        )
        target_low, target_high = raw._cylinder_headroom(target_x)
        aam = getattr(raw, '_online_aam', None)
        if aam is not None and getattr(aam, 'z_hat', None) is not None:
            z = aam.z_hat.detach()
            # Do not collapse the latent axis into this check.  A constant
            # vector [a, b] repeated in every environment has a non-zero
            # scalar std when a != b, but that does not show that the online
            # estimate responds to history or randomized plant factors.
            across_env_std = (
                float(z.std(dim=0, unbiased=False).amax())
                if z.ndim == 2 and z.shape[0] > 1
                else 0.0
            )
            previous = _Z_HAT_LIVE.get('previous')
            temporal_delta = (
                float((z - previous).abs().amax())
                if isinstance(previous, torch.Tensor)
                and previous.shape == z.shape
                else 0.0
            )
            _Z_HAT_LIVE['previous'] = z.clone()
            _Z_HAT_LIVE['across_env_std'] = max(
                _Z_HAT_LIVE.get('across_env_std', 0.0), across_env_std
            )
            _Z_HAT_LIVE['temporal_delta'] = max(
                _Z_HAT_LIVE.get('temporal_delta', 0.0), temporal_delta
            )
            _Z_HAT_LIVE['live'] = (
                _Z_HAT_LIVE['across_env_std'] > 1.0e-6
                or _Z_HAT_LIVE['temporal_delta'] > 1.0e-6
            )
        target_headrooms.append(torch.minimum(target_low, target_high))
        target_overshoots.append(measured_x - target_x)
        measured_xs.append(measured_x.clone())
        target_xs.append(target_x.clone())
        # Drive authority. effort_limit_sim is what PhysX enforces;
        # effort_limit is None on these actuators.
        try:
            applied = raw.robot.data.applied_torque[:, raw._cylinder_joint_ids]
            torque_ratios.append((applied.abs() / _EFFORT_LIMITS).clamp(max=9.9))
        except Exception:
            pass
        travels.append(raw._truck_travel_m.clone())
        scan_validities.append(raw.scan_valid_fraction())
        cutter_x_positions.append(
            raw._compute_cutter_pos_w()[:, 0] - raw.scene.env_origins[:, 0]
        )
        ground_heights.append(raw.ground_height_under_cutter())
        cutter_clearances.append(raw.cutter_height_above_ground())
        hard_impact_events.append(raw._hard_impact_event.clone())
        components = safety_component_masks(raw)
        for name, values in components.items():
            safety_steps[name].append(values)

    rollout_steps = max(len(errors), 1)
    error_steps = torch.stack(errors)
    lateral_steps = torch.stack(laterals)
    contact_steps = torch.stack(contacts)
    active_steps = torch.stack(active_steps)
    alignment_steps = torch.stack(alignments)
    alignment_values = alignment_steps[active_steps]
    headroom_steps = torch.stack(headrooms)
    headroom_cyl_steps = torch.stack(headrooms_per_cylinder)
    headroom_low_steps = torch.stack(headroom_lows)
    headroom_high_steps = torch.stack(headroom_highs)
    stroke_fraction_steps = torch.stack(stroke_fractions)
    target_headroom_steps = torch.stack(target_headrooms)
    target_overshoot_steps = torch.stack(target_overshoots)
    measured_x_steps = torch.stack(measured_xs)
    target_x_steps = torch.stack(target_xs)
    torque_ratio_steps = torch.stack(torque_ratios) if torque_ratios else None
    travel_steps = torch.stack(travels)
    scan_steps = torch.stack(scan_validities)
    cutter_x_steps = torch.stack(cutter_x_positions)
    ground_height_steps = torch.stack(ground_heights)
    cutter_clearance_steps = torch.stack(cutter_clearances)
    action_steps = torch.stack(actions)
    termination_steps = torch.stack(termination_events)
    hard_impact_steps = torch.stack(hard_impact_events)
    safety_tensors = {
        name: torch.stack(values) for name, values in safety_steps.items()
    }
    error = error_steps[active_steps]

    distance_rows = []
    active_length = float(raw.cfg.terrain_tracking_active_length_m)
    bin_edges = torch.linspace(0.0, active_length, 9, device=raw.device)
    for index in range(len(bin_edges) - 1):
        lower = float(bin_edges[index])
        upper = float(bin_edges[index + 1])
        if index == len(bin_edges) - 2:
            mask = (travel_steps >= lower) & (travel_steps <= upper)
        else:
            mask = (travel_steps >= lower) & (travel_steps < upper)
        mask &= active_steps
        if not bool(mask.any()):
            continue
        row = {
            "start_m": lower,
            "end_m": upper,
            "samples": int(mask.sum()),
            "mean_abs_cut_height_error_m": float(error_steps[mask].mean()),
            "mean_abs_lateral_error_m": float(lateral_steps[mask].mean()),
            "ground_contact_fraction": float(contact_steps[mask].mean()),
            "p01_head_alignment": float(
                torch.quantile(torch.stack(alignments)[mask], 0.01)
            ),
            "p01_cylinder_headroom_m": float(
                torch.quantile(headroom_steps[mask], 0.01)
            ),
            "p01_cylinder_headroom_per_cylinder_m": [
                float(value)
                for value in torch.quantile(
                    headroom_cyl_steps[mask], 0.01, dim=0
                )
            ],
            "p01_headroom_low_per_cylinder_m": [
                float(value)
                for value in torch.quantile(headroom_low_steps[mask], 0.01, dim=0)
            ],
            "p01_headroom_high_per_cylinder_m": [
                float(value)
                for value in torch.quantile(headroom_high_steps[mask], 0.01, dim=0)
            ],
            "mean_stroke_fraction_per_cylinder": [
                float(value) for value in stroke_fraction_steps[mask].mean(dim=0)
            ],
            "p01_target_headroom_per_cylinder_m": [
                float(value)
                for value in torch.quantile(target_headroom_steps[mask], 0.01, dim=0)
            ],
            "p99_abs_target_overshoot_per_cylinder_m": [
                float(value)
                for value in torch.quantile(
                    target_overshoot_steps[mask].abs(), 0.99, dim=0
                )
            ],
            "scan_valid_fraction": float(scan_steps[mask].mean()),
            "ground_height_mean_m": float(ground_height_steps[mask].mean()),
            "ground_height_min_m": float(ground_height_steps[mask].min()),
            "ground_height_max_m": float(ground_height_steps[mask].max()),
            "cutter_height_above_ground_mean_m": float(
                cutter_clearance_steps[mask].mean()
            ),
            "cutter_x_min_m": float(cutter_x_steps[mask].min()),
            "cutter_x_max_m": float(cutter_x_steps[mask].max()),
            "hard_impact_event_fraction": float(hard_impact_steps[mask].float().mean()),
            "episode_termination_rate": float(termination_steps[mask].float().mean()),
            "mean_abs_action": [
                float(value)
                for value in action_steps[mask].abs().mean(dim=0)
            ],
            "action_saturation_fraction": float(
                (action_steps[mask].abs() >= 0.99).float().mean()
            ),
            "safety_component_rates": {
                name: float(values[mask].float().mean())
                for name, values in safety_tensors.items()
                if name != "union"
            },
            "safety_violation_rate": float(
                safety_tensors["union"][mask].float().mean()
            ),
        }
        distance_rows.append(row)

    family_rows = []
    for family in sorted(set(raw._tracking_path_family_names)):
        family_mask = torch.as_tensor(
            [name == family for name in raw._tracking_path_family_names],
            dtype=torch.bool,
            device=raw.device,
        )
        family_active = active_steps & family_mask.unsqueeze(0)
        family_rows.append(
            {
                "family": family,
                "mean_abs_cut_height_error_m": float(
                    error_steps[family_active & family_mask.unsqueeze(0)].mean()
                ),
                "mean_abs_lateral_error_m": float(
                    lateral_steps[family_active & family_mask.unsqueeze(0)].mean()
                ),
                "ground_contact_fraction": float(
                    contact_steps[family_active & family_mask.unsqueeze(0)].mean()
                ),
            }
        )

    amplitude_rows = []
    env_amplitudes = tuple(getattr(raw, "_terrain_amplitude_scales", ()) or ())
    for amplitude in sorted(set(env_amplitudes)):
        amplitude_mask = torch.as_tensor(
            [value == amplitude for value in env_amplitudes],
            dtype=torch.bool,
            device=raw.device,
        )
        if not bool(amplitude_mask.any()):
            continue
        amplitude_active = active_steps & amplitude_mask.unsqueeze(0)
        amplitude_rows.append(
            {
                "amplitude_scale": float(amplitude),
                "environments": int(amplitude_mask.sum()),
                "mean_abs_lateral_error_m": float(
                    lateral_steps[amplitude_active].mean()
                ),
                "mean_abs_cut_height_error_m": float(
                    error_steps[amplitude_active].mean()
                ),
                "ground_contact_fraction": float(
                    contact_steps[amplitude_active].mean()
                ),
            }
        )

    result = {
        "mesh": mesh,
        "steps": int(rollout_steps),
        "num_envs": int(raw.num_envs),
        "control_dt_s": float(raw.step_dt),
        "episode_length_s": float(raw.cfg.episode_length_s),
        "mow_strip_start_x_m": float(raw.cfg.mow_strip_start_x_m),
        "mow_strip_length_m": float(raw.cfg.mow_strip_length_m),
        "tracking_reference_x_m": float(raw.cfg.terrain_tracking_reference_x_m),
        "tracking_active_length_m": float(raw.cfg.terrain_tracking_active_length_m),
        "max_truck_travel_m": float(travel_steps.max()),
        "cutter_x_min_m": float(cutter_x_steps.min()),
        "cutter_x_max_m": float(cutter_x_steps.max()),
        "scan_valid_fraction": float(scan_steps[active_steps].mean()),
        "episode_termination_rate": float(termination_steps[active_steps].float().mean()),
        "episode_termination_fraction": float(termination_steps.sum())
        / max(raw.num_envs, 1),
        "safety_component_rates": {
            name: float(values[active_steps].float().mean())
            for name, values in safety_tensors.items()
            if name != "union"
        },
        "distance_results": distance_rows,
        "amplitude_results": amplitude_rows,
        "mean_abs_cut_height_error_m": float(error.mean()),
        "worst_family_mean_abs_lateral_error_m": max(
            row["mean_abs_lateral_error_m"] for row in family_rows
        ),
        "worst_family_ground_contact_fraction": min(
            row["ground_contact_fraction"] for row in family_rows
        ),
        "p01_head_alignment": float(torch.quantile(alignment_values, 0.01)),
        "hard_impact_step_fraction": float(hard_impact_steps[active_steps].sum())
        / max(int(active_steps.sum()), 1),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(torch.stack(accelerations)[active_steps], 0.95)
        ),
        "min_cylinder_headroom_m": float(headroom_steps[active_steps].min()),
        "p01_cylinder_headroom_m": float(
            torch.quantile(headroom_steps[active_steps], 0.01)
        ),
        "headroom_diagnostic": _headroom_diagnostic(
            headroom_cyl_steps, target_headroom_steps, target_overshoot_steps,
            torque_ratio_steps, measured_x_steps, target_x_steps, active_steps,
        ),
        "cylinder_names": list(raw.cfg.cylinder_joint_names),
        # Per-ENVIRONMENT headroom, joined to that environment's deployment
        # factor draw. The gate is a p01 over environments and steps together,
        # so it cannot say whether a headroom failure is a few weak-actuator
        # draws or every environment equally. Those two have opposite fixes:
        # the first is a plant-context problem and belongs to the online
        # adaptation, the second is structural to the trajectory.
        "per_env_headroom": _per_env_headroom(
            raw, headroom_cyl_steps, active_steps
        ),
        "cylinder_learned_min": [float(v) for v in raw._learned_min.reshape(-1)],
        "cylinder_learned_max": [float(v) for v in raw._learned_max.reshape(-1)],
        "p01_cylinder_headroom_per_cylinder_m": [
            float(v)
            for v in torch.quantile(headroom_cyl_steps[active_steps], 0.01, dim=0)
        ],
        "min_cylinder_headroom_per_cylinder_m": [
            float(v) for v in headroom_cyl_steps[active_steps].amin(dim=0)
        ],
        "safety_violation_rate": float(
            safety_tensors["union"][active_steps].float().mean()
        ),
        # Compatibility alias for reports produced before the statistic was
        # correctly distinguished from actual episode terminations.
        "safety_termination_rate": float(
            safety_tensors["union"][active_steps].float().mean()
        ),
        "path_family_results": family_rows,
    }
    if trace is not None:
        trace_path = Path(a.trace_out).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            time_s=np.asarray(trace["time_s"], dtype=np.float64),
            ee_actual_m=np.asarray(trace["ee_actual_m"], dtype=np.float64),
            ee_reference_m=np.asarray(trace["ee_reference_m"], dtype=np.float64),
            ee_error_m=np.asarray(trace["ee_error_m"], dtype=np.float64),
            joint_actual_rad=np.asarray(trace["joint_actual_rad"], dtype=np.float64),
            joint_reference_rad=np.asarray(
                trace["joint_reference_rad"], dtype=np.float64
            ),
            joint_error_rad=np.asarray(trace["joint_error_rad"], dtype=np.float64),
            travel_m=np.asarray(trace["travel_m"], dtype=np.float64),
            cut_height_error_m=np.asarray(
                trace["cut_height_error_m"], dtype=np.float64
            ),
            lateral_error_m=np.asarray(trace["lateral_error_m"], dtype=np.float64),
            mesh=np.asarray(mesh, dtype=np.int64),
            pipeline=np.asarray(EXPECTED_ACTUATOR),
            trajectory_shape=np.asarray(raw._tracking_path_family_names[0]),
            tracking_amplitude=np.asarray(
                float(getattr(raw, "_terrain_amplitude_scales", (np.nan,))[0])
            ),
            control_dt_s=np.asarray(float(raw.step_dt)),
        )
        print(f"[PPO3-1-GATES] wrote trace {trace_path}", flush=True)
    return result


def assert_speed_channel_normalized(payload: dict, speed_kmh: float | None) -> None:
    """Refuse to score a policy whose speed input is not actually normalized.

    The empirical normalizer travels inside ``actor_state_dict``, so a
    checkpoint resumed from a fixed-speed run carries that run's zero-variance
    speed statistic.  RSL-RL divides by ``std + 1e-2``, which turns the channel
    into a x100 amplifier at every speed except the one it was frozen at: the
    policy then posts pristine gate margins at that single speed and collapses
    everywhere else.  Scoring it silently is how eleven hours of retraining got
    attributed to RL rather than to an input transform.
    """

    from verge_task.terrain_observations import TRUCK_SPEED_OBS_INDEX

    state = payload.get("actor_state_dict") or {}
    std = state.get("obs_normalizer._std")
    mean = state.get("obs_normalizer._mean")
    if std is None or mean is None:
        return
    index = TRUCK_SPEED_OBS_INDEX
    if index >= std.shape[-1]:
        return
    channel_std = float(std[0, index])
    channel_mean = float(mean[0, index])
    if channel_std >= 1.0e-3:
        return
    scored_mps = (float(speed_kmh) / 3.6) if speed_kmh is not None else channel_mean
    feature = (scored_mps - channel_mean) / (channel_std + 1.0e-2)
    if abs(feature) <= 1.0:
        return
    raise RuntimeError(
        "checkpoint's frozen observation normalizer has a degenerate truck-speed "
        f"channel (index {index}: mean {channel_mean:.4f} m/s, std {channel_std:.6f}). "
        f"Scoring at {scored_mps * 3.6:.2f} km/h feeds the actor a normalized "
        f"value of {feature:+.1f} where every other channel is order 1, so the "
        "result would measure the normalizer, not the policy. Retrain or reseed "
        "the speed channel to the sampled range first "
        "(verge_task.trajectory_regularized.reseed_frozen_speed_normalizer)."
    )


def main() -> None:
    if a.teacher or a.null:
        checkpoint_path = None
    else:
        checkpoint_path = Path(a.checkpoint).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        infos = payload.get("infos") or {}
        source = str(infos.get("source", ""))
        schema = str(infos.get("schema", ""))
        if schema not in {"terrain_teacher_student_v1", "terrain_shared_v2"} or source not in {
            "PPO3 terrain teacher distillation v1",
            "PPO3 terrain PPO",
            "PPO3-1 terrain PPO",
            "PPO4-1 terrain PPO",
        }:
            raise RuntimeError(
                "evaluate_gates accepts only supported terrain policy checkpoints; "
                f"got source={source!r}, schema={schema!r}"
            )
        assert_checkpoint_actuator(
            infos, EXPECTED_ACTUATOR, context="PPO3-1 gate evaluation"
        )
        assert_speed_channel_normalized(payload, a.vehicle_speed_kmh)

    per_mesh: dict[str, dict] = {}
    for mesh in meshes:
        env_cfg = parse_env_cfg(
            TASK_ID,
            device=a.device,
            num_envs=a.num_envs,
        )
        apply_c4_authority_stage(env_cfg, int(a.stage))
        apply_trajectory_shapes(env_cfg, selected_shapes)
        if a.cylinder_target_max_lead_per_axis_m:
            override_cfg(
                env_cfg,
                "cylinder_target_max_lead_per_axis_m",
                tuple(
                    float(part)
                    for part in a.cylinder_target_max_lead_per_axis_m.split(",")
                    if part.strip()
                ),
            )
        if a.cylinder_target_leak is not None:
            override_cfg(env_cfg, "cylinder_target_leak", float(a.cylinder_target_leak))
        if a.integrates_from_prev is not None:
            override_cfg(
                env_cfg,
                "cylinder_target_integrates_from_previous_target",
                a.integrates_from_prev == "true",
            )
        if a.lead_high_speed:
            override_cfg(
                env_cfg,
                "cylinder_target_lead_high_speed_per_axis_m",
                tuple(float(v) for v in a.lead_high_speed.split(",") if v.strip()),
            )
        if a.lead_schedule_speed is not None:
            override_cfg(
                env_cfg,
                "cylinder_target_lead_schedule_speed_kmh",
                float(a.lead_schedule_speed),
            )
        if a.online_aam_latent_mode is not None:
            if EXPECTED_ACTUATOR != "id_aam_fd":
                p.error("--online_aam_latent_mode requires the PPO-ID(AAM)-FD task")
            override_cfg(
                env_cfg, "online_aam_policy_latent_mode", str(a.online_aam_latent_mode)
            )
        if a.online_aam_controller_latent_mode is not None:
            if EXPECTED_ACTUATOR != "id_aam_fd":
                p.error(
                    "--online_aam_controller_latent_mode requires the "
                    "PPO-ID(AAM)-FD task"
                )
            override_cfg(
                env_cfg,
                "online_aam_latent_mode",
                str(a.online_aam_controller_latent_mode),
            )
        if a.online_aam_latent_freeze_tick is not None:
            override_cfg(
                env_cfg, "online_aam_latent_freeze_tick",
                int(a.online_aam_latent_freeze_tick),
            )
        if a.rom_factor_perturbation is not None:
            ranges = {}
            for part in str(a.rom_factor_perturbation).split(","):
                name, _, span = part.partition("=")
                lo, _, hi = span.partition(":")
                ranges[name.strip()] = (float(lo), float(hi))
            override_cfg(env_cfg, "rom_factor_perturbation_enabled", True)
            override_cfg(env_cfg, "rom_factor_perturbation_ranges", ranges)
        if a.rom_factor_step is not None:
            name, _, value = str(a.rom_factor_step).partition("=")
            override_cfg(env_cfg, "rom_factor_perturbation_enabled", True)
            override_cfg(
                env_cfg, "rom_factor_perturbation_step_values",
                {name.strip(): float(value)},
            )
        if a.plant_alpha_range is not None:
            lo, _, hi = str(a.plant_alpha_range).partition(":")
            override_cfg(env_cfg, "plant_perturbation_enabled", True)
            override_cfg(env_cfg, "plant_perturbation_alpha_range", (float(lo), float(hi)))
        if a.plant_delay_steps_range is not None:
            lo, _, hi = str(a.plant_delay_steps_range).partition(":")
            override_cfg(env_cfg, "plant_perturbation_enabled", True)
            override_cfg(
                env_cfg, "plant_perturbation_delay_steps_range", (int(lo), int(hi))
            )
        if a.plant_step_time_s is not None:
            override_cfg(env_cfg, "plant_perturbation_step_time_s", float(a.plant_step_time_s))
            override_cfg(
                env_cfg, "rom_factor_perturbation_step_time_s", float(a.plant_step_time_s)
            )
        if a.plant_step_alpha_scale is not None:
            override_cfg(
                env_cfg, "plant_perturbation_step_alpha_scale",
                float(a.plant_step_alpha_scale),
            )
        if a.adaptation_trace_path is not None:
            override_cfg(env_cfg, "adaptation_trace_path", str(a.adaptation_trace_path))
        if a.c3_physical_extension_reserve_m is not None:
            override_cfg(
                env_cfg,
                "c3_physical_extension_reserve_m",
                float(a.c3_physical_extension_reserve_m),
            )
        if a.dls_null_space_centering_gain is not None:
            override_cfg(
                env_cfg,
                "dls_null_space_centering_gain",
                float(a.dls_null_space_centering_gain),
            )
        if a.cylinder_target_limit_inset_m is not None:
            override_cfg(
                env_cfg,
                "cylinder_target_limit_inset_m",
                float(a.cylinder_target_limit_inset_m),
            )
        env_cfg.observation_space = observation_space_for_cfg(env_cfg)
        if a.tracking_amplitude is not None:
            override_cfg(
                env_cfg, "terrain_tracking_amplitude_scale",
                float(a.tracking_amplitude),
            )
            override_cfg(
                env_cfg, "terrain_tracking_amplitude_scales",
                (float(a.tracking_amplitude),),
            )
        if a.vehicle_speed_kmh is not None:
            env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_kmh)
            env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_kmh)
        elif a.vehicle_speed_min_kmh is not None or a.vehicle_speed_max_kmh is not None:
            if (
                a.vehicle_speed_min_kmh is None
                or a.vehicle_speed_max_kmh is None
                or not 0.0 < a.vehicle_speed_min_kmh <= a.vehicle_speed_max_kmh
            ):
                raise ValueError(
                    "speed range must provide 0 < min <= max"
                )
            env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_min_kmh)
            env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_max_kmh)
        env_cfg.vehicle_speed_per_episode = True
        if a.episode_length_s is not None:
            env_cfg.episode_length_s = float(a.episode_length_s)
        else:
            configure_path_horizon(env_cfg)
        env_cfg.use_eval_terrain = True
        env_cfg.terrain_base_seed = int(a.seed)
        env_cfg.terrain_eval_seed_override = mesh
        env_cfg.enable_contact_tracking_demo = False
        env_cfg.enable_visual_grid_floor = False
        env_cfg.enable_road_scenery = bool(a.natural_visuals)
        env_cfg.mow_strip_use_vertex_palette = bool(a.natural_visuals)
        env_cfg.enable_road_edge_markings = False
        env_cfg.enable_verge_grass_tufts = bool(a.natural_visuals)
        env_cfg.randomize_start_phase = False
        env_cfg.randomize_trajectory_shape = False
        env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
        if a.factor_domain or a.factor_key:
            env_cfg.deployment_factor_enabled = True
            env_cfg.deployment_factor_domain = a.factor_domain or "nominal"
            env_cfg.deployment_factor_key = str(a.factor_key)
            if a.factor_manifest:
                env_cfg.deployment_factor_manifest = str(Path(a.factor_manifest).resolve())
            env_cfg.deployment_factor_seed = int(a.seed)
        if a.online_aam_residual_scale is not None:
            if EXPECTED_ACTUATOR != "id_aam_fd":
                p.error("--online-aam-residual-scale requires the PPO-ID(AAM)-FD task")
            env_cfg.online_aam_residual_scale = float(a.online_aam_residual_scale)
        if a.teacher:
            env_cfg.sole_teacher_mode = "execute"
            if a.teacher_initial_vertical_offset_m is not None:
                env_cfg.contact_demo_initial_vertical_offset_m = float(
                    a.teacher_initial_vertical_offset_m
                )
            if a.teacher_max_vertical_offset_m is not None:
                env_cfg.contact_demo_max_vertical_offset_m = float(
                    a.teacher_max_vertical_offset_m
                )
            if a.teacher_max_upward_offset_m is not None:
                env_cfg.contact_demo_max_upward_offset_m = float(
                    a.teacher_max_upward_offset_m
                )
            if a.teacher_max_raw_action is not None:
                env_cfg.terrain_teacher_max_raw_action = float(
                    a.teacher_max_raw_action
                )
        configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))

        env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
        raw = env.unwrapped
        if a.teacher or a.null:
            zeros = torch.zeros(
                (raw.num_envs, int(env_cfg.action_space)), device=a.device
            )

            def policy(_obs, _zeros=zeros):
                return _zeros
        else:
            if EXPECTED_ACTUATOR == "id_aam_fd":
                from PPO_ID_AAM_FD.env_cfg import PpoIDAAMFDPPORunnerCfg
                runner_cfg = PpoIDAAMFDPPORunnerCfg()
            elif EXPECTED_ACTUATOR == "idfd":
                from PPO_ID_FD.env_cfg import Ppo41IDFDPPORunnerCfg
                runner_cfg = Ppo41IDFDPPORunnerCfg()
            else:
                runner_cfg = TerrainC4AuthorityPPORunnerCfg()
            runner_cfg.seed = int(a.seed)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
            runner = OnPolicyRunner(
                wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device
            )
            runner.load(
                str(checkpoint_path),
                load_cfg={
                    "actor": True,
                    "critic": True,
                    "optimizer": False,
                    "iteration": True,
                    "rnd": False,
                },
            )
            policy = runner.get_inference_policy(device=a.device)
        profiler = cProfile.Profile() if a.profile_out else None
        if profiler is not None:
            profiler.enable()
        try:
            per_mesh[str(mesh)] = roll_mesh(
                raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
            )
        finally:
            if profiler is not None:
                profiler.disable()
                profile_path = Path(a.profile_out).expanduser().resolve()
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                profiler.dump_stats(profile_path)
                print(f"[PPO3-1-GATES] wrote profile {profile_path}", flush=True)
        # Flush the per-step adaptation trace before the env is torn down.
        # Setting adaptation_trace_path only ARMS the recorder; without this
        # call the run looks configured, produces no npz, and the adaptation
        # figures would silently have nothing to plot.
        saved = None
        if getattr(raw, "save_adaptation_trace", None) is not None:
            saved = raw.save_adaptation_trace(
                mesh=int(mesh),
                policy_latent_mode=str(
                    getattr(env_cfg, "online_aam_policy_latent_mode", "")
                ),
                controller_latent_mode=str(
                    getattr(env_cfg, "online_aam_latent_mode", "")
                ),
                latent_freeze_tick=int(
                    getattr(env_cfg, "online_aam_latent_freeze_tick", 0)
                ),
            )
        if getattr(a, "adaptation_trace_path", None):
            if saved is None:
                raise SystemExit(
                    "--adaptation_trace_path was set but no trace was written; "
                    "the recorder never armed (is the task one that records?)"
                )
            print(f"[PPO3-1-GATES] wrote adaptation trace {saved}", flush=True)
        env.close()

    if len(per_mesh) >= 3:
        result = check_gates(per_mesh)
    else:
        result = {
            "meshes": len(per_mesh),
            "metrics": {},
            "gates": {},
            "margins": {},
            "passed": False,
            "failing": [],
            "thin_margins": [],
            "note": "fewer than 3 meshes; per-mesh metrics saved, gates not evaluated",
        }
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subject": (
            "dls_teacher"
            if a.teacher
            else "null_zero_action" if a.null else "policy_checkpoint"
        ),
        "sole_teacher_mode": "execute" if a.teacher else "off",
        "cylinder_target_max_lead_per_axis_m": list(
            getattr(env_cfg, "cylinder_target_max_lead_per_axis_m", ()) or ()
        ),
        "lead_cap_overridden": bool(a.cylinder_target_max_lead_per_axis_m),
        "cylinder_target_integrates_from_previous_target": bool(
            getattr(env_cfg, "cylinder_target_integrates_from_previous_target", False)
        ),
        "cylinder_target_lead_high_speed_per_axis_m": list(
            getattr(env_cfg, "cylinder_target_lead_high_speed_per_axis_m", ()) or ()
        ),
        "cylinder_target_lead_schedule_speed_kmh": float(
            getattr(env_cfg, "cylinder_target_lead_schedule_speed_kmh", 4.0)
        ),
        # Guard against a disconnected latent. A patch once left the AAM
        # update unreachable, so z_hat held its initial value and every
        # ablation arm scored byte-identically while the gates still
        # passed. Record whether z_hat actually varies across envs/steps.
        "z_hat_is_live": bool(_Z_HAT_LIVE.get("live", False)),
        "z_hat_std_observed": float(_Z_HAT_LIVE.get("across_env_std", 0.0)),
        "z_hat_across_env_std_observed": float(
            _Z_HAT_LIVE.get("across_env_std", 0.0)
        ),
        "z_hat_temporal_delta_observed": float(
            _Z_HAT_LIVE.get("temporal_delta", 0.0)
        ),
        "online_aam_latent_mode": str(
            getattr(env_cfg, "online_aam_policy_latent_mode", "estimated")
        ),
        "online_aam_controller_latent_mode": str(
            getattr(env_cfg, "online_aam_latent_mode", "estimated")
        ),
        "c3_physical_extension_reserve_m": float(
            getattr(env_cfg, "c3_physical_extension_reserve_m", 0.0)
        ),
        "cylinder_target_leak": float(getattr(env_cfg, "cylinder_target_leak", 0.0)),
        "dls_null_space_centering_gain": float(
            getattr(env_cfg, "dls_null_space_centering_gain", 0.0)
        ),
        "cylinder_target_limit_inset_m": float(
            getattr(env_cfg, "cylinder_target_limit_inset_m", 0.0)
        ),
        "cylinder_target_integrates_from_previous_target": bool(
            getattr(env_cfg, "cylinder_target_integrates_from_previous_target", False)
        ),
        "vehicle_speed_min_kmh": float(
            getattr(env_cfg, "vehicle_speed_min_kmh", 0.0)
        ),
        "vehicle_speed_max_kmh": float(
            getattr(env_cfg, "vehicle_speed_max_kmh", 0.0)
        ),
        "factor_domain": str(
            getattr(env_cfg, "deployment_factor_domain", "nominal")
        ),
        "factor_key": str(getattr(env_cfg, "deployment_factor_key", "")),
        "online_aam_residual_scale": float(
            getattr(env_cfg, "online_aam_residual_scale", 0.0)
        ),
        "online_aam_checkpoint": str(
            Path(getattr(env_cfg, "online_aam_checkpoint", "")).expanduser().resolve()
        ) if getattr(env_cfg, "online_aam_checkpoint", "") else "",
        "online_aam_adapter_checkpoint": str(
            Path(getattr(env_cfg, "online_aam_adapter_checkpoint", "")).expanduser().resolve()
        ) if getattr(env_cfg, "online_aam_adapter_checkpoint", "") else "",
        "online_aam_history_lag": int(
            getattr(env_cfg, "online_aam_history_lag", 0)
        ),
        "online_aam_smoother_window": int(
            getattr(env_cfg, "online_aam_smoother_window", 0)
        ),
        "deployment_factor_enabled": bool(
            getattr(env_cfg, "deployment_factor_enabled", False)
        ),
        "deployment_factor_manifest": str(
            Path(getattr(env_cfg, "deployment_factor_manifest", "")).expanduser().resolve()
        ) if getattr(env_cfg, "deployment_factor_manifest", "") else "",
        "evaluation_seed": int(a.seed),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "stage": int(a.stage),
        "actuator": EXPECTED_ACTUATOR,
        "action_width": int(env_cfg.action_space),
        "meshes": sorted(per_mesh),
        "per_mesh": per_mesh,
        "gates": result,
    }
    printable = {
        "checkpoint": str(checkpoint_path),
        "gates": result,
        "per_mesh_metrics": report["per_mesh"],
    }
    print(json.dumps(printable, indent=2), flush=True)

    if a.report:
        path = Path(a.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"[PPO3-1-GATES] wrote {path}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
