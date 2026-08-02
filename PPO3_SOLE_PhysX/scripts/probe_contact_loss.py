#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Event-level diagnosis of cutter contact loss for PPO3 (ROM-A) and PPO4 (ID/FD).

This is a MEASUREMENT script, not a fix. It exists because "the cutter lifts"
is a visual claim, and every previous attempt in this repository to act on a
visual claim without an event trace produced a null result (see
``reports/FINAL_REPORT.md`` and ``HANDOVER_PPO3_PPO4.md``).

For every step of a rollout it records the full contact/actuator state, then
finds every transition ``contact -> no contact`` that occurs after support was
acquired, and classifies it into exactly one of five mechanisms:

    impact_ejection   high force sample, then separation with the cutter
                      moving UPWARD -- the tool is thrown off the verge
    actuator_lag      commanded target lead or model residual grows before
                      separation -- the plant did not follow the command
    saturation        cylinder headroom collapses or the limit shield engages
    invalid_sensing   the height scan stopped seeing the verge
    path_geometry     none of the above; the reference itself left the ground

The classifier is ordered, so an event is attributed to the first mechanism
whose evidence is present. That ordering is recorded in the report, and every
event also carries its raw window so a different ordering can be re-derived
from the artifact without re-running the simulation.

Both actuators run through this identically (``--actuator {rom,idfd}``), so a
PPO3-vs-PPO4 difference in the event mix is attributable to the actuator.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True)
p.add_argument("--actuator", default="rom", choices=("rom", "idfd"))
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=960)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--eval_terrain", action="store_true")
p.add_argument("--path_family", default="s_curve")
p.add_argument("--terrain_height_scale", type=float, default=0.0)
p.add_argument(
    "--eval_terrain_seed",
    type=int,
    default=0,
    help=(
        "Move the HELD-OUT mesh off its pinned seed 1337. --seed cannot do "
        "this: a held-out rollout is deterministic and every --seed returns "
        "identical numbers. Zero keeps the gate mesh."
    ),
)
p.add_argument(
    "--drive_substep_interpolation", default="", choices=("", "true", "false")
)
p.add_argument("--arm_stiffness_scale", type=float, default=None)
# Phase 6 impact-energy sweep knobs. All of them change the PLANT or the
# command chain, not the policy, so a sweep row is directly comparable with
# the baseline row produced by the same checkpoint.
p.add_argument("--action_scale", type=float, default=None)
p.add_argument("--max_action_rate", type=float, default=None)
p.add_argument("--cylinder_target_max_lead_m", type=float, default=None)
p.add_argument("--cylinder_target_leak", type=float, default=None)
p.add_argument(
    "--contact_observation", default="", choices=("", "true", "false")
)
p.add_argument("--model_error_history", type=int, default=None)
p.add_argument(
    "--window",
    type=int,
    default=12,
    help="Steps of context stored on each side of a contact-loss event.",
)
p.add_argument(
    "--max_event_windows",
    type=int,
    default=60,
    help="Cap on stored full windows; aggregates always cover every event.",
)
p.add_argument(
    "--acquire_steps",
    type=int,
    default=3,
    help="Consecutive contact steps required before support counts as acquired.",
)
p.add_argument(
    "--hard_impact_force_n",
    type=float,
    default=30000.0,
    help="Force above which a sample is called a hard impact.",
)
p.add_argument(
    "--takeoff_velocity_mps",
    type=float,
    default=0.02,
    help="Upward cutter velocity at separation that makes it a takeoff.",
)
p.add_argument(
    "--visible_gap_m",
    type=float,
    default=0.005,
    help="Geometric airborne gap that counts as a visible lift.",
)
p.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO3_SOLE_PhysX  # noqa: F401,E402
from PPO3_SOLE_PhysX.distill import assert_teacher_disabled  # noqa: E402
from PPO3_SOLE_PhysX.actuators import (  # noqa: E402
    assert_checkpoint_actuator,
    resolve_actuator,
)
from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from PPO3_SOLE_PhysX.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)

TASK_ID, RUNNER_CFG_CLS, ACTUATOR_LABEL = resolve_actuator(a.actuator)


def build_env():
    env_cfg = parse_env_cfg(TASK_ID, device=a.device, num_envs=a.num_envs)
    apply_terrain_curriculum_stage(env_cfg, int(a.stage))
    env_cfg.action_space = 5 if env_cfg.sole_head_joint_action else 4
    if env_cfg.sole_head_joint_action:
        env_cfg.terrain_messerkopf_level_gain = 0.0
        env_cfg.terrain_messerkopf_level_quadratic_gain = 0.0
    env_cfg.sole_ik_hint = False
    if a.contact_observation:
        env_cfg.terrain_contact_observation = a.contact_observation == "true"
    if a.model_error_history is not None:
        env_cfg.terrain_model_error_history = int(a.model_error_history)
    if a.action_scale is not None:
        env_cfg.sole_action_scale = float(a.action_scale)
    if a.max_action_rate is not None:
        env_cfg.sole_max_action_rate = float(a.max_action_rate)
    if a.cylinder_target_max_lead_m is not None:
        env_cfg.cylinder_target_max_lead_m = float(a.cylinder_target_max_lead_m)
    if a.cylinder_target_leak is not None:
        env_cfg.cylinder_target_leak = float(a.cylinder_target_leak)
    # This script measures the excursion; terminating on it would truncate the
    # very interval being sized.
    env_cfg.terrain_terminate_on_contact_loss = False
    env_cfg.terrain_terminate_on_hard_impact = False
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.use_eval_terrain = bool(a.eval_terrain)
    env_cfg.terrain_eval_seed_override = int(a.eval_terrain_seed)
    env_cfg.terrain_base_seed = int(a.seed)
    env_cfg.terrain_height_scale_override = float(a.terrain_height_scale)
    env_cfg.enable_contact_tracking_demo = False
    env_cfg.tracking_path_family = str(a.path_family)
    env_cfg.enable_visual_grid_floor = False
    env_cfg.enable_road_scenery = False
    # Measure the mechanism, do not let a safety latch censor it: the point of
    # this script is to see the excursion, and a termination would hide the
    # steps after separation, which is exactly the part being diagnosed.
    env_cfg.defer_safety_terminations = True
    env_cfg.head_down_termination_alignment = -1.0
    if a.arm_stiffness_scale is not None:
        env_cfg.terrain_arm_stiffness_scale = float(a.arm_stiffness_scale)
    if a.drive_substep_interpolation:
        env_cfg.terrain_kinematic_drive_substep_interpolation = (
            a.drive_substep_interpolation == "true"
        )
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    assert_teacher_disabled(env_cfg, context="PPO3/PPO4 contact-loss probe")
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    return env, env.unwrapped


#: Scalar channels stored per step per environment. Order is the report's
#: ``channels`` key, so a stored window can be read without this source file.
CHANNELS = (
    "contact",
    "contact_force_n",
    "cutter_world_z_m",
    "ground_height_m",
    "cutter_height_above_ground_m",
    "airborne_gap_m",
    "cutter_vertical_velocity_mps",
    "cutter_vertical_acceleration_mps2",
    "scan_valid_fraction",
    "path_progress",
    "head_alignment",
    "action_rate",
    "shield_gain",
    "min_cylinder_headroom_m",
    "max_abs_target_lead_m",
    "max_abs_valve_cmd",
    "max_abs_predicted_delta_m",
    "max_abs_realized_delta_m",
    "max_abs_model_residual_m",
    "tool_acceleration_mps2",
)

#: Per-cylinder channels, stored as four columns each.
VECTOR_CHANNELS = (
    "raw_action",
    "applied_action",
    "cylinder_x_m",
    "cylinder_xdot_mps",
    "cylinder_target_m",
    "target_lead_m",
)


def rollout(raw, policy, steps: int) -> dict:
    obs, _ = raw.reset(seed=int(a.seed))
    dt = max(float(raw.step_dt), 1.0e-6)
    reference = float(
        getattr(raw.cfg, "cutter_contact_reference_height_m", -0.30)
    )

    scalars = {name: [] for name in CHANNELS}
    vectors = {name: [] for name in VECTOR_CHANNELS}

    previous_z = None
    previous_vz = None
    previous_x = None
    previous_predicted = None
    strip_length = float(raw._mow_strip_cfg().length_m)

    for _ in range(steps):
        action = policy(obs)
        obs, _, _, _, _ = raw.step(action)

        contact = raw.cutter_ground_contact()
        force = raw.cutter_terrain_contact_force_n()
        cutter_z = raw._cutter_world_z()
        ground_z = raw.ground_height_under_cutter()
        height_above = cutter_z - ground_z
        # The calibrated contact reference is where the real head sits when the
        # verge supports it, so the gap above THAT -- not above the raw mesh --
        # is what a viewer sees as lift. Identical definition to
        # ``cut_height_error``'s airborne branch, kept separate so this stays a
        # geometric quantity even while the sensor says contact.
        airborne_gap = torch.relu(height_above - reference)

        vz = (
            (cutter_z - previous_z) / dt
            if previous_z is not None
            else torch.zeros_like(cutter_z)
        )
        az = (
            (vz - previous_vz) / dt
            if previous_vz is not None
            else torch.zeros_like(cutter_z)
        )
        previous_z = cutter_z.detach().clone()
        previous_vz = vz.detach().clone()

        x, xdot = raw._refresh_cylinder_state()
        low, high = raw._cylinder_headroom(x)
        lead = raw._commanded_target_lead(x)
        realized = (
            x - previous_x if previous_x is not None else torch.zeros_like(x)
        )
        predicted = raw._fd_delta_corrected.detach().clone()
        residual = (
            realized - previous_predicted
            if previous_predicted is not None
            else torch.zeros_like(x)
        )
        previous_x = x.detach().clone()
        previous_predicted = predicted

        scalars["contact"].append(contact.float())
        scalars["contact_force_n"].append(force)
        scalars["cutter_world_z_m"].append(cutter_z)
        scalars["ground_height_m"].append(ground_z)
        scalars["cutter_height_above_ground_m"].append(height_above)
        scalars["airborne_gap_m"].append(airborne_gap)
        scalars["cutter_vertical_velocity_mps"].append(vz)
        scalars["cutter_vertical_acceleration_mps2"].append(az)
        scalars["scan_valid_fraction"].append(raw.scan_valid_fraction())
        scalars["path_progress"].append(
            torch.clamp(raw._truck_travel_m / max(strip_length, 1.0e-6), 0.0, 1.0)
        )
        scalars["head_alignment"].append(raw._measure_head_down_alignment())
        scalars["action_rate"].append(
            (raw._sole_applied - raw._prev_sole_applied).abs().amax(dim=-1)
        )
        scalars["shield_gain"].append(raw._sole_shield_gain.amin(dim=-1))
        scalars["min_cylinder_headroom_m"].append(
            torch.minimum(low, high).amin(dim=-1)
        )
        scalars["max_abs_target_lead_m"].append(lead.abs().amax(dim=-1))
        scalars["max_abs_valve_cmd"].append(
            raw._current_valve_cmd.abs().amax(dim=-1)
        )
        scalars["max_abs_predicted_delta_m"].append(predicted.abs().amax(dim=-1))
        scalars["max_abs_realized_delta_m"].append(realized.abs().amax(dim=-1))
        scalars["max_abs_model_residual_m"].append(residual.abs().amax(dim=-1))
        scalars["tool_acceleration_mps2"].append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )

        vectors["raw_action"].append(raw._sole_raw.detach().clone())
        vectors["applied_action"].append(raw._sole_applied.detach().clone())
        vectors["cylinder_x_m"].append(x.detach().clone())
        vectors["cylinder_xdot_mps"].append(xdot.detach().clone())
        vectors["cylinder_target_m"].append(
            raw._cylinder_pos_target.detach().clone()
        )
        vectors["target_lead_m"].append(lead.detach().clone())

    return {
        "scalars": {
            name: torch.stack(values).detach().cpu()
            for name, values in scalars.items()
        },
        "vectors": {
            name: torch.stack(values).detach().cpu()
            for name, values in vectors.items()
        },
    }


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.flatten().float(), q))


def analyse(series: dict, raw) -> dict:
    scalars = series["scalars"]
    vectors = series["vectors"]
    contact = scalars["contact"] > 0.5
    steps, num_envs = contact.shape

    force = scalars["contact_force_n"]
    gap = scalars["airborne_gap_m"]
    vz = scalars["cutter_vertical_velocity_mps"]
    scan = scalars["scan_valid_fraction"]
    lead = scalars["max_abs_target_lead_m"]
    headroom = scalars["min_cylinder_headroom_m"]
    shield = scalars["shield_gain"]
    residual = scalars["max_abs_model_residual_m"]

    acquire_steps = max(int(a.acquire_steps), 1)
    hard_force = float(a.hard_impact_force_n)
    takeoff_v = float(a.takeoff_velocity_mps)
    visible_gap = float(a.visible_gap_m)

    # Reference distributions taken from stable contact only. "Stable" is
    # deliberately narrow: in contact, not moving vertically, not near a
    # cylinder limit, not shielded. Force targets derived from all in-contact
    # samples would be inflated by the very impacts this is diagnosing.
    stable = (
        contact
        & (vz.abs() < 0.02)
        & (headroom > 0.02)
        & (shield > 0.999)
        & (scalars["tool_acceleration_mps2"] < 120.0)
    )
    stable_force = force[stable]
    in_contact_force = force[contact]

    # Force-rate is a per-environment finite difference of the force channel.
    force_rate = torch.zeros_like(force)
    force_rate[1:] = (force[1:] - force[:-1]) / max(float(raw.step_dt), 1.0e-6)

    events = []
    windows = []
    half = max(int(a.window), 1)

    acquired_step = torch.full((num_envs,), -1, dtype=torch.long)
    for env in range(num_envs):
        run = 0
        acquired = False
        for step in range(steps):
            if bool(contact[step, env]):
                run += 1
                if not acquired and run >= acquire_steps:
                    acquired = True
                    acquired_step[env] = step
            else:
                if acquired and run > 0:
                    # A contact -> no-contact transition after acquisition.
                    previous_force = float(force[step - 1, env])
                    this_gap = float(gap[step, env])
                    this_vz = float(vz[step, env])
                    lo = max(step - half, 0)
                    hi = min(step + half + 1, steps)
                    pre_lo = max(step - half, 0)

                    pre_scan = float(scan[pre_lo:step, env].min()) if step > pre_lo else 1.0
                    pre_lead = lead[pre_lo:step, env]
                    lead_growth = (
                        float(pre_lead[-1] - pre_lead.min())
                        if pre_lead.numel() > 0
                        else 0.0
                    )
                    pre_headroom = (
                        float(headroom[pre_lo:step, env].min()) if step > pre_lo else 1.0
                    )
                    pre_shield = (
                        float(shield[pre_lo:step, env].min()) if step > pre_lo else 1.0
                    )
                    pre_force_max = (
                        float(force[pre_lo:step, env].max()) if step > pre_lo else 0.0
                    )
                    pre_residual = (
                        float(residual[pre_lo:step, env].max()) if step > pre_lo else 0.0
                    )

                    takeoff = this_vz > takeoff_v
                    high_force = pre_force_max >= float(
                        _percentile(in_contact_force, 0.95)
                    )
                    # Ordered attribution. Impact/ejection first because it is
                    # the only mechanism the repository has already measured
                    # (94.4 % of separations leave the ground moving upward),
                    # and because the others are all conditions that would also
                    # have to produce an upward departure to matter.
                    if takeoff and high_force:
                        kind = "impact_ejection"
                    elif pre_scan < 0.80:
                        kind = "invalid_sensing"
                    elif pre_headroom < 0.005 or pre_shield < 0.999:
                        kind = "saturation"
                    elif lead_growth > 0.002 or pre_residual > 0.002:
                        kind = "actuator_lag"
                    elif takeoff:
                        kind = "impact_ejection"
                    else:
                        kind = "path_geometry"

                    airborne_run = 0
                    peak_gap = 0.0
                    probe = step
                    while probe < steps and not bool(contact[probe, env]):
                        airborne_run += 1
                        peak_gap = max(peak_gap, float(gap[probe, env]))
                        probe += 1

                    event = {
                        "env": int(env),
                        "step": int(step),
                        "kind": kind,
                        "path_family": str(raw._tracking_path_family_names[env]),
                        "terrain_height_scale": float(
                            raw._terrain_height_scales[env]
                        ),
                        "path_progress": float(scalars["path_progress"][step, env]),
                        "previous_contact_force_n": previous_force,
                        "peak_force_before_n": pre_force_max,
                        "cutter_vertical_velocity_mps": this_vz,
                        "airborne_gap_m": this_gap,
                        "peak_airborne_gap_m": peak_gap,
                        "airborne_steps": int(airborne_run),
                        "scan_valid_fraction_before": pre_scan,
                        "target_lead_growth_m": lead_growth,
                        "min_headroom_before_m": pre_headroom,
                        "min_shield_gain_before": pre_shield,
                        "max_model_residual_before_m": pre_residual,
                        "head_alignment": float(scalars["head_alignment"][step, env]),
                        "hard_impact_before": bool(pre_force_max > hard_force),
                        "visible_lift": bool(peak_gap > visible_gap),
                        "takeoff": bool(takeoff),
                    }
                    events.append(event)
                    if len(windows) < int(a.max_event_windows):
                        windows.append(
                            {
                                "env": int(env),
                                "step": int(step),
                                "kind": kind,
                                "first_step": int(lo),
                                "scalars": {
                                    name: [
                                        round(float(v), 6)
                                        for v in scalars[name][lo:hi, env].tolist()
                                    ]
                                    for name in CHANNELS
                                },
                                "vectors": {
                                    name: [
                                        [round(float(v), 6) for v in row]
                                        for row in vectors[name][lo:hi, env].tolist()
                                    ]
                                    for name in VECTOR_CHANNELS
                                },
                            }
                        )
                run = 0

    acquired_mask = acquired_step >= 0
    post_contact_num = 0
    post_contact_den = 0
    max_airborne_run = 0
    for env in range(num_envs):
        start = int(acquired_step[env])
        if start < 0:
            continue
        window = contact[start:, env]
        post_contact_num += int(window.sum())
        post_contact_den += int(window.numel())
        run = 0
        for value in window.tolist():
            if value:
                run = 0
            else:
                run += 1
                max_airborne_run = max(max_airborne_run, run)

    kinds = {}
    for event in events:
        kinds[event["kind"]] = kinds.get(event["kind"], 0) + 1

    families = {}
    for event in events:
        key = event["path_family"]
        families[key] = families.get(key, 0) + 1

    return {
        "steps": int(steps),
        "num_envs": int(num_envs),
        "scored_samples": int(steps * num_envs),
        "contact_acquired_fraction": float(acquired_mask.float().mean()),
        "mean_acquisition_step": float(
            acquired_step[acquired_mask].float().mean()
        )
        if bool(acquired_mask.any())
        else float("nan"),
        "ground_contact_fraction": float(contact.float().mean()),
        "post_acquisition_contact_fraction": (
            post_contact_num / post_contact_den if post_contact_den else float("nan")
        ),
        "contact_loss_event_count": len(events),
        "contact_loss_events_per_1000_samples": 1000.0
        * len(events)
        / max(steps * num_envs, 1),
        "max_consecutive_airborne_steps": int(max_airborne_run),
        "event_kind_counts": kinds,
        "event_path_family_counts": families,
        "takeoff_event_count": sum(1 for e in events if e["takeoff"]),
        "impact_ejection_event_count": kinds.get("impact_ejection", 0),
        "hard_impact_event_count": sum(
            1 for e in events if e["hard_impact_before"]
        ),
        "visible_lift_event_count": sum(1 for e in events if e["visible_lift"]),
        "upward_separation_fraction": (
            sum(1 for e in events if e["takeoff"]) / len(events)
            if events
            else float("nan")
        ),
        "airborne_gap": {
            "max_m": float(gap.max()),
            "p95_m": _percentile(gap, 0.95),
            "mean_m": float(gap.mean()),
            "max_event_peak_m": max(
                (e["peak_airborne_gap_m"] for e in events), default=0.0
            ),
        },
        "cutter_vertical_velocity": {
            "max_positive_mps": float(vz.max()),
            "p95_abs_mps": _percentile(vz.abs(), 0.95),
            "mean_at_separation_mps": (
                sum(e["cutter_vertical_velocity_mps"] for e in events) / len(events)
                if events
                else float("nan")
            ),
        },
        "contact_force_all_samples": {
            "mean_n": float(force.mean()),
            "p95_n": _percentile(force, 0.95),
            "max_n": float(force.max()),
        },
        "contact_force_in_contact": {
            "count": int(in_contact_force.numel()),
            "mean_n": float(in_contact_force.mean())
            if in_contact_force.numel()
            else float("nan"),
            "p50_n": _percentile(in_contact_force, 0.50),
            "p95_n": _percentile(in_contact_force, 0.95),
            "max_n": float(in_contact_force.max())
            if in_contact_force.numel()
            else float("nan"),
        },
        # These three are what Phase 4's force-regulation reward must be built
        # from. Taken from STABLE contact only, so an impact cannot set the
        # target the policy is then paid to reach.
        "contact_force_stable": {
            "count": int(stable_force.numel()),
            "mean_n": float(stable_force.mean())
            if stable_force.numel()
            else float("nan"),
            "std_n": float(stable_force.std())
            if stable_force.numel() > 1
            else float("nan"),
            "p05_n": _percentile(stable_force, 0.05),
            "p50_n": _percentile(stable_force, 0.50),
            "p95_n": _percentile(stable_force, 0.95),
            "p99_n": _percentile(stable_force, 0.99),
            "max_n": float(stable_force.max())
            if stable_force.numel()
            else float("nan"),
        },
        "force_rate": {
            "mean_abs_nps": float(force_rate.abs().mean()),
            "p95_abs_nps": _percentile(force_rate.abs(), 0.95),
            "max_nps": float(force_rate.max()),
            "min_nps": float(force_rate.min()),
        },
        "target_lead": {
            "p95_m": _percentile(lead, 0.95),
            "max_m": float(lead.max()),
        },
        "shield_active_fraction": float((shield < 0.999).float().mean()),
        "min_cylinder_headroom_m": float(headroom.min()),
        "min_scan_valid_fraction": float(scan.min()),
        "events": events,
        "event_windows": windows,
        "channels": list(CHANNELS),
        "vector_channels": list(VECTOR_CHANNELS),
    }


def main() -> None:
    env, raw = build_env()
    runner_cfg = RUNNER_CFG_CLS()
    runner_cfg.seed = int(a.seed)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device)
    checkpoint_path = Path(a.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    infos = payload.get("infos") or {}
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
    assert_checkpoint_actuator(infos, a.actuator, context="contact-loss probe")
    policy = runner.get_inference_policy(device=a.device)

    series = rollout(raw, policy, int(a.steps))
    summary = analyse(series, raw)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "actuator": str(a.actuator),
        "actuator_label": ACTUATOR_LABEL,
        "stage": int(a.stage),
        "seed": int(a.seed),
        "path_family": str(a.path_family),
        "terrain": "held_out" if a.eval_terrain else "training",
        "terrain_height_scale_override": float(a.terrain_height_scale),
        "arm_stiffness_scale": float(raw.cfg.terrain_arm_stiffness_scale),
        "head_joint_stiffness_scale": float(
            raw.cfg.terrain_head_joint_stiffness_scale
        ),
        "drive_substep_interpolation": bool(
            raw.cfg.terrain_kinematic_drive_substep_interpolation
        ),
        "teacher_mode": str(raw.cfg.sole_teacher_mode),
        "direct_target_demo": bool(raw.cfg.enable_contact_tracking_demo),
        "classifier": {
            "order": [
                "impact_ejection",
                "invalid_sensing",
                "saturation",
                "actuator_lag",
                "path_geometry",
            ],
            "acquire_steps": int(a.acquire_steps),
            "hard_impact_force_n": float(a.hard_impact_force_n),
            "takeoff_velocity_mps": float(a.takeoff_velocity_mps),
            "visible_gap_m": float(a.visible_gap_m),
        },
        "path_families_realized": sorted(set(raw._tracking_path_family_names)),
        "height_scales_realized": sorted(
            set(float(v) for v in raw._terrain_height_scales)
        ),
        "summary": summary,
    }

    printable = {k: v for k, v in summary.items() if k not in {"events", "event_windows"}}
    print(json.dumps(printable, indent=2), flush=True)

    if a.report:
        path = Path(a.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"[PPO3-SOLE] wrote {path}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
