#!/usr/bin/env python3
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
    choices=('estimated', 'privileged', 'zero', 'mean', 'shuffled', 'frozen', 'lpf'),
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
        " 'lpf' band-limits the same estimate at "
        "--online_aam_policy_latent_lpf_hz, holding its content and removing "
        "only its high-frequency movement. With 'estimated' (unlimited) and "
        "'frozen' (zero) it forms a bandwidth axis, which is what separates "
        "the latent's temporal variation from the plant identity it carries; "
        "the two cannot be told apart by mean/shuffled alone, because those "
        "change content and variability at once."
    ),
)
p.add_argument(
    "--online_aam_policy_latent_lpf_hz",
    type=float,
    default=None,
    help=(
        "Cutoff in Hz for --online_aam_latent_mode lpf. Applies to the two PPO "
        "latent observation columns only; the ID adapter keeps the full-rate "
        "estimate. UNITS: Hz against the 60 Hz control clock. The raw latent "
        "is a 20 Hz staircase, so cutoffs above about 10 Hz are indistinguishable "
        "from 'estimated'."
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
    "--actuator_authority_alpha",
    type=float,
    default=None,
    help=(
        "fraction of the realised cylinder motion that reaches the PhysX "
        "target, applied identically on every stack after the actuator model. "
        "This is the common stress axis: the deployment factors are inert on "
        "the ID/FD plant and clamped on the ROM, so they do not provide one. "
        "1.0 is unperturbed; alpha -> 0 must make the reference untrackable."
    ),
)
p.add_argument(
    "--actuator_authority_alpha_after",
    type=float,
    default=None,
    help="authority factor from --actuator_authority_step_time_s onwards",
)
p.add_argument(
    "--actuator_authority_step_time_s",
    type=float,
    default=None,
    help=(
        "episode time at which the authority factor steps. A sham arm sets "
        "--actuator_authority_alpha_after equal to --actuator_authority_alpha "
        "so the schedule fires and the plant does not change."
    ),
)
p.add_argument(
    "--actuator_stress_deadband_m",
    type=float,
    default=None,
    help=(
        "displacement-domain deadband in METRES, applied at the same boundary "
        "as --actuator_authority_alpha and therefore on every stack. This is "
        "the delta_x domain, so valve-domain severities do not transfer."
    ),
)
p.add_argument(
    "--actuator_stress_lag_tau_s",
    type=float,
    default=None,
    help="first-order lag on the realised cylinder increment, seconds",
)
p.add_argument(
    "--actuator_stress_delay_steps",
    type=int,
    default=None,
    help=(
        "pure transport delay on the realised cylinder increment, in control "
        "ticks. This is the only stress axis with no analytic breaking point: "
        "shifting a command in time leaves its magnitude unchanged, so it "
        "never drives the required command past the actuator limits."
    ),
)
p.add_argument(
    "--actuator_stress_rate_limit_m",
    type=float,
    default=None,
    help="max change in the realised cylinder increment per tick, metres",
)
p.add_argument(
    "--deployment_mass_step_time_s",
    type=float,
    default=None,
    help=(
        "episode time at which the tool mass steps. The mechanical factors are "
        "the only deployment factors that reach every stack unclamped, so this "
        "is the NATIVE mid-episode plant change -- no emulator, no proxy."
    ),
)
p.add_argument(
    "--deployment_mass_step_scale",
    type=float,
    default=None,
    help=(
        "tool mass after the step, as a multiple of its baseline. A sham arm "
        "sets this to 1.0 so the schedule fires and the plant is left alone."
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
    "--video",
    action="store_true",
    help=(
        "Record an mp4 of the rollout instead of opening a viewer. This box is "
        "headless (no DISPLAY), so the live Isaac window used on the Legion "
        "machine cannot open here; recording is the equivalent. Forces "
        "--enable_cameras."
    ),
)
p.add_argument(
    "--video_length",
    type=int,
    default=600,
    help="Frames to record. 600 at 60 Hz is ~10 s of simulated motion.",
)
p.add_argument(
    "--video_dir",
    default="outputs/videos",
)
p.add_argument(
    "--residual_checkpoint",
    default=None,
    help=(
        "Score a frozen base policy plus a history-conditioned residual. The "
        "residual lives in the actor's `mlp`, so this reuses roll_mesh and every "
        "measurement convention in this file rather than reimplementing them -- "
        "a standalone evaluator was written and got the frozen base 29x wrong "
        "twice, first by mis-handling parked steps and then by omitting rsl_rl's "
        "normaliser epsilon."
    ),
)
p.add_argument(
    "--residual_history",
    type=int,
    default=20,
)
p.add_argument(
    "--residual_alpha",
    type=float,
    default=0.5,
)
p.add_argument(
    "--residual_latent_dim",
    type=int,
    default=8,
)
p.add_argument(
    "--residual_no_history",
    action="store_true",
    help="Zero the history: same capacity and weights, but cannot adapt online.",
)
p.add_argument(
    "--online_grad_lr",
    type=float,
    default=0.0,
    help=(
        "Online gradient adaptation of the policy's final layer, per control "
        "step, against the measured actuator prediction residual (observation "
        "columns 58:62). 0 disables it. See "
        "reports/PREREG_ONLINE_GRAD_2026-09-05.md."
    ),
)
p.add_argument(
    "--online_grad_sham",
    action="store_true",
    help=(
        "Sham control: identical update magnitude and parameter motion, with "
        "the gradient sign randomised per environment. Separates the direction "
        "of the gradient from the fact of perturbing the weights."
    ),
)
p.add_argument(
    "--task_bias_y",
    type=float,
    default=0.0,
    help=(
        "Task-space reference offset on the tracking line, metres. The policy "
        "is shown lateral error measured against a line displaced by this much "
        "(observation column 10 becomes e_lat - b_Y); the SCORED error is still "
        "measured against the true line. Everything downstream of the "
        "observation -- policy, FD, plant -- is untouched. See "
        "reports/PREREG_ORACLE_PHI_2026-09-06.md."
    ),
)
p.add_argument(
    "--task_bias_z",
    type=float,
    default=0.0,
    help=(
        "Task-space reference offset on the cut-height target, metres. "
        "cut_height_error is relu(height - reference) and zero in contact, so "
        "raising the reference by b_Z makes the observed error relu(e_cut - "
        "b_Z): positive lifts the cutter, negative pushes it down. Staying "
        "inside the relu keeps the channel non-negative, i.e. inside the "
        "distribution the frozen policy was trained on. The SCORED error uses "
        "the nominal reference."
    ),
)
p.add_argument(
    "--paired_task_bias",
    default="",
    help=(
        "Sweep task-space offsets paired per environment inside ONE process, "
        "as 'bY:bZ,bY:bZ,...' in metres. The first entry is the baseline and "
        "should be 0:0. Pairing on the environment cancels the terrain, path "
        "and plant draw, which across-process comparison cannot."
    ),
)
p.add_argument(
    "--adapt_trace",
    default="",
    help=(
        "Record the per-step adaptation trace to this .json: true payload, the "
        "online estimate, and the task errors, averaged across environments. "
        "This is the figure the adaptive-control literature actually asks for "
        "-- estimated parameter against true parameter over time, and tracking "
        "error through the transient -- rather than another table of episode "
        "aggregates."
    ),
)
p.add_argument(
    "--obs_dump",
    default="",
    help=(
        "Record (features, true payload) pairs to this .npz for fitting an "
        "online payload estimator. Features are the 66 TASK columns only; the "
        "four context columns are never included, because an estimator trained "
        "on the answer is the failure the whole design exists to avoid "
        "(rma_adaptation.assert_no_context_leakage). The label is read from "
        "the env's own context, not from the observation, so it stays correct "
        "under MULAG_CONTEXT_ABLATE."
    ),
)
p.add_argument(
    "--obs_dump_every",
    type=int,
    default=10,
    help="Record one row every N control steps. 900 steps at 10 gives 90 rows per env.",
)
p.add_argument(
    "--paired_smoothing",
    default="",
    help=(
        "Sweep the command-smoothing knobs paired per environment inside ONE "
        "process, as 'base,beta:rate,beta:rate,...'. The first entry must be "
        "the literal 'base', which leaves the configured values untouched so "
        "the sweep is anchored to the frozen baseline measured in the same "
        "process. beta is sole_action_filter (filtered = beta*previous + "
        "(1-beta)*raw, so HIGHER is gentler) and rate is sole_max_action_rate "
        "(the per-step clamp, so LOWER is gentler). Both are read from cfg "
        "every step, so changing them between rollouts needs no rebuild. "
        "Motivated by RESULT_rootcause_contact_2026-09-06.md: the heavy-payload "
        "failure is impact and rebound with command authority to spare, and "
        "these are the two knobs that act on impact rather than on position."
    ),
)
p.add_argument(
    "--valve_probe_amp",
    type=float,
    default=0.0,
    help=(
        "Add an independent per-channel PRBS of this amplitude to the frozen "
        "policy's valve command, so the valve-to-cutter map can be identified. "
        "Closed-loop commands are too correlated to identify it, which is the "
        "finding of Section 4.4. Use with --trace_out and one mesh."
    ),
)
p.add_argument(
    "--valve_probe_hold",
    type=int,
    default=10,
    help="Control steps each PRBS level is held, matching the Chapter 4 probe.",
)
p.add_argument(
    "--online_grad_lag",
    type=int,
    default=10,
    help=(
        "Control steps between a valve command and its effect on the cutter. "
        "The probe of 2026-09-06 measures 10 steps, 167 ms at 60 Hz, matching "
        "the command-to-pressure delay of Section 7.3.1. The update attributes "
        "the current error to the command issued this many steps ago; 0 "
        "reproduces the uncompensated behaviour."
    ),
)
p.add_argument(
    "--online_grad_beta",
    type=float,
    default=0.02,
    help=(
        "EMA coefficient on the gradient signal. A loop closed around 167 ms of "
        "dead time is only stable well below 1/(2*pi*0.167) = 0.95 Hz, so the "
        "update is smoothed over roughly 1/beta steps. 1.0 disables smoothing."
    ),
)
p.add_argument(
    "--online_grad_trust",
    type=float,
    default=0.05,
    help="Max absolute deviation of an adapted weight from its frozen value.",
)
p.add_argument(
    "--arms_per_env",
    default="",
    help=(
        "Comma-separated context arms assigned PER ENVIRONMENT inside ONE "
        "rollout, e.g. 'privileged,mean,shuffled,zero'. Use this, not "
        "--paired_arms: running one arm per rollout back-to-back on the same "
        "env is BIASED. An A/A control with three identical arms measured "
        "1.1-2.4 mm differences with CIs excluding zero, and an order-"
        "counterbalanced mass sweep showed no effect ascending and a REVERSED "
        "sign descending. Per-env assignment steps every arm in the same "
        "physics at the same time, so there is no sequence to bias it."
    ),
)
p.add_argument(
    "--paired_mass_scale",
    default="",
    help=(
        "Comma-separated cutter mass scales scored IN ONE PROCESS on the same "
        "environments, e.g. '1.0,1.5,2.5,4.0'. G1 on the MECHANICAL axis. "
        "Actuator axes are measured dead in this pipeline -- 1.07 s of "
        "transport delay does not degrade tracking and weakening the valve "
        "IMPROVES it -- while the only privileged-context win all night came "
        "from mass_scale/com_z, the sole factors that reach the FD plant. "
        "This finds whether a wider payload range costs a fixed policy enough "
        "to give an estimator something to recover."
    ),
)
p.add_argument(
    "--paired_alpha",
    default="",
    help=(
        "Comma-separated valve-effectiveness values scored IN ONE PROCESS on "
        "the same environments, e.g. '1.0,0.6,0.4,0.25'. G1 on the authority "
        "axis: delay was measured non-degrading to 16 steps, so authority is "
        "the next candidate for a perturbation box that actually costs the "
        "baseline something. Watch action_saturation alongside the error -- a "
        "degradation that arrives WITH saturation is the actuator running out "
        "of command, not a recoverable adaptation opportunity."
    ),
)
p.add_argument(
    "--paired_delay_steps",
    default="",
    help=(
        "Comma-separated transport delays in control steps to score IN ONE "
        "PROCESS on the same environments, e.g. '0,2,4,8'. This is the G1 "
        "degradation gate: the question is whether a fixed no-context policy "
        "gets MEASURABLY WORSE as the plant moves. Adaptation can only recover "
        "headroom that a degradation creates, so if no severity degrades the "
        "baseline there is nothing for any estimator to win back. Paired "
        "per-env like --paired_arms, because across-process means at one seed "
        "disagree by 22-74% and cannot resolve this."
    ),
)
p.add_argument(
    "--paired_arms",
    default="",
    help=(
        "Comma-separated online_aam_policy_latent_mode arms to score IN ONE "
        "PROCESS on the same environments, e.g. "
        "'privileged,mean,shuffled,zero'. The latent mode is read from cfg on "
        "every observation build, so arms can be switched on a live env; each "
        "arm re-resets from the same seed, so they share terrain, path and "
        "factor draw and the per-env differences are paired. This is the only "
        "valid way to measure an oracle gap here -- separate processes at the "
        "same seed disagree by more than any adaptation effect being sought."
    ),
)
p.add_argument(
    "--action_scale",
    type=float,
    default=None,
    help=(
        "Evaluate at a different per-step action scale in metres. Required "
        "whenever the checkpoint was TRAINED at a non-default scale: the "
        "policy's action is q_ref = q_now + action * action_scale, so scoring "
        "a 0.012-trained policy at the 0.02 default silently changes the plant "
        "contract under it. The per-axis row is re-derived, not just the "
        "scalar, because cylinder_action_scale_per_axis_m is computed FROM "
        "action_scale by apply_c4_authority_stage and would otherwise stay at "
        "the stage's value."
    ),
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
    "--plant_step_delay_steps",
    type=int,
    default=None,
    help=(
        "ID-FD: transport delay becomes this many control steps at "
        "--plant_step_time_s. Measured dose-response: delay 2 -> 8 costs +22% "
        "lateral error, where a valve-effectiveness step of x0.6 costs 0.2% "
        "and is therefore useless as an adaptation perturbation."
    ),
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
p.add_argument(
    "--tracking_reference_step_m",
    type=float,
    default=None,
    help=(
        "Metres of lateral step added to the COMMANDED tracking line at "
        "--tracking_reference_step_time_s. Unlike a plant perturbation this "
        "changes what the controller is asked to do, so the response latency "
        "it produces is a property of the control stack."
    ),
)
p.add_argument(
    "--tracking_reference_step_time_s",
    type=float,
    default=None,
    help="When the commanded-line step fires, in seconds of scored travel.",
)
p.add_argument(
    "--online_aam_adapter_checkpoint",
    default=None,
    help=(
        "Bounded ID adapter to load. Required when changing the latent source: "
        "an adapter trained on estimated latents refuses a privileged one, "
        "because it would then consume a distribution it never saw."
    ),
)
p.add_argument(
    "--deployment_factor_scale",
    type=float,
    default=None,
    help=(
        "Magnitude of the deployment-factor randomisation as a multiple of the "
        "manifest draw: 0 is the nominal plant, 1 reproduces the manifest, 2 "
        "doubles every deviation. Direction is fixed, so a sweep varies only "
        "how far the plant is from nominal and not which plants are visited."
    ),
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

if getattr(a, "video", False):
    # Cameras must be enabled BEFORE the app launches; the render pipeline is
    # built at startup and cannot be turned on afterwards.
    a.enable_cameras = True
launcher = AppLauncher(a)
simulation_app = launcher.app

# Base-observation columns the residual's history encoder read; filled in
# when a residual policy is built, empty for non-residual arms.
RESIDUAL_HISTORY_COLS: list[int] = []

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

    _pkg = importlib.import_module("PPO_AAM_FD")
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
# Gate 0 adaptation-site counters, stashed before env.close() tears down the
# CUDA tensors they live on, and read back when the report is assembled.
_GATE0_READOUT: dict = {}
# Policy-side latent statistics and command-reversal rate. The controller-side
# z_hat above is identical in every policy-latent arm by construction, so it
# cannot serve as the mechanism variable for a bandwidth comparison.
_Z_POLICY_LIVE: dict = {}
_CMD_LIVE: dict = {}

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


def _build_probe_policy(runner, raw, env_cfg):
    """Frozen policy plus an independent per-channel PRBS on the valve command.

    Section 4.4 measures that a task controller's own commands are close to
    redundant with the state, so they cannot identify the actuator map. The same
    argument applies to identifying the valve-to-cutter geometry AF needs, so
    the probe is injected rather than inferred from natural motion.

    The probe is written to PROBE_LOG alongside the trace so the regression uses
    the injected component, which is independent of the state by construction.
    """
    base_policy = runner.get_inference_policy(device=a.device)
    amp = float(a.valve_probe_amp)
    hold = max(int(a.valve_probe_hold), 1)
    n = int(raw.num_envs)
    state = {"k": 0, "probe": torch.zeros((n, 4), device=a.device)}

    def probe_policy(obs):
        if state["k"] % hold == 0:
            state["probe"] = (
                torch.randint(0, 2, (n, 4), device=a.device).float() * 2.0 - 1.0
            ) * amp
        state["k"] += 1
        action = base_policy(obs)
        PROBE_LOG.append(state["probe"].detach().cpu().clone())
        return torch.clamp(action + state["probe"], -1.0, 1.0)

    print(f"[PPO3-1-GATES] valve probe: amp={amp} hold={hold}", flush=True)
    return probe_policy


PROBE_LOG = []

# Filled when --obs_dump is given; features never include the context columns.
# A always-present dict rather than a rebindable None, so no call site needs a
# `global` declaration and it cannot depend on whether that site is a function.
OBS_DUMP = {"on": False, "x": [], "y": []}

# Per-step adaptation trace: true payload, online estimate, task errors.
ADAPT_TRACE = {"on": False, "t": [], "true_payload": [], "est_payload": [],
               "cut_mm": [], "lateral_mm": [], "contact": [],
               # Individual environments: the periodic plant phase-offsets each
               # one, so only a single environment shows the square wave.
               "true_env0": [], "est_env0": [], "true_env1": [], "est_env1": [],
               "true_env2": [], "est_env2": []}


# Mutable so --paired_task_bias can change the offset between rollouts without
# rebuilding the policy: a rebuilt policy would also reset rsl_rl's normaliser
# bookkeeping, and the arms would then differ by more than the offset.
TASK_BIAS = {"y": 0.0, "z": 0.0}

# Observation columns carrying the two scored task errors. Fixed at the front of
# build_verge_observation's layout (col 9 cut height, col 10 lateral) and ahead
# of every optional block, so they do not move with the model-error, contact,
# head-frame or context appends.
CUT_ERROR_COL = 9
LATERAL_ERROR_COL = 10


def _build_task_bias_policy(runner, raw, env_cfg):
    """Frozen policy driven against a displaced task-space reference.

    This is the oracle probe of PREREG_ORACLE_PHI_2026-09-06: it asks whether a
    two-parameter correction EXISTS that improves both scored channels at a
    given payload, before any estimator is built to find it online. Nothing is
    learned and nothing adapts; the offset is a constant supplied on the
    command line.

    The offset is applied to the observation only:

        cut     relu(e_cut - b_Z)   -- exactly what raising the contact
                                       reference by b_Z produces, since
                                       e_cut = relu(h - reference) and is zero
                                       in contact. Staying inside the relu
                                       keeps the channel non-negative, which is
                                       the only sign the frozen policy has ever
                                       seen on it.
        lateral e_lat - b_Y         -- exactly what displacing the tracking
                                       line by b_Y produces, since
                                       e_lat = cutter_y - line_y.

    The scored quantities are read from the environment, not from the
    observation, so they remain the true errors against the true reference.
    """
    base_policy = runner.get_inference_policy(device=a.device)

    def task_bias_policy(obs):
        base = obs["policy"] if isinstance(obs, dict) else obs
        b_y = float(TASK_BIAS["y"])
        b_z = float(TASK_BIAS["z"])
        if b_y == 0.0 and b_z == 0.0:
            # Bit-for-bit the frozen base, so the b=0 arm is the instrument
            # check of invariant 1 rather than an approximation to it.
            return base_policy(obs)
        shifted = base.clone()
        shifted[:, CUT_ERROR_COL] = torch.relu(
            shifted[:, CUT_ERROR_COL] - b_z
        )
        shifted[:, LATERAL_ERROR_COL] = shifted[:, LATERAL_ERROR_COL] - b_y
        if isinstance(obs, dict):
            payload = dict(obs)
            payload["policy"] = shifted
            return base_policy(payload)
        return base_policy(shifted)

    print(
        f"[PPO3-1-GATES] task-space reference offset: b_Y={a.task_bias_y} m "
        f"b_Z={a.task_bias_z} m (observation only; scoring unchanged)",
        flush=True,
    )
    return task_bias_policy


ONLINE_GRAD_STATS = {"steps": 0, "quiet": 0.0, "reverted": 0, "resid": 0.0}


def _build_online_grad_policy(runner, raw, env_cfg):
    """Frozen policy whose final layer is updated by gradient descent at runtime.

    This is an analytic policy gradient: the measured task error is mapped into
    action space through the plant geometry and backpropagated into the policy.

    An earlier version descended on the actuator prediction residual instead.
    That was an objective mismatch and the ladder showed it cleanly: the
    residual fell monotonically with the learning rate while cut-height error
    rose from 15.5 mm to 56.7 mm. Shrinking a model residual by moving the
    *command* makes the plant do what the model predicted, which is the opposite
    of making the cutter track. Model error is corrected by moving the model.

    The signal here is the task error itself. `verge_lateral_error` is the
    cutter's Y offset from the tracking line and `cut_height_error` its Z
    contact gap, so rows 1 and 2 of the cascade's 3x4 cylinder-to-cutter
    Jacobian carry them into cylinder space and the action-sign vector into
    action space:

        g = e_lat * Ja[1] + e_cut * Ja[2]

    The surrogate `sum_i u_i * g_i` has gradient `sum_i g_i * du_i/dtheta`, so
    one descent step moves the command the way the geometry says reduces the
    measured error. Positive error on either channel means that axis must move
    negative, which is the sign this produces.

    No reward, no rollout, no credit assignment, no privileged input.
    """
    from tensordict import TensorDict

    actor = runner.alg.actor
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    linears = [m for m in actor.mlp.modules() if isinstance(m, torch.nn.Linear)]
    if not linears:
        raise RuntimeError("online gradient adaptation found no Linear layer in the actor")
    head = linears[-1]
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    frozen = {"w": head.weight.detach().clone(), "b": head.bias.detach().clone()}
    trust = float(a.online_grad_trust)
    opt = torch.optim.SGD([head.weight, head.bias], lr=float(a.online_grad_lr))

    # Fixed 3x4 map from learned cylinder extension to cutter XYZ, carried into
    # policy-action coordinates. Fail loudly rather than descend on a wrong
    # direction.
    env_obj = raw
    for _ in range(4):
        if hasattr(env_obj, "_cascade"):
            break
        env_obj = getattr(env_obj, "unwrapped", getattr(env_obj, "env", None))
        if env_obj is None:
            break
    if env_obj is None or not hasattr(env_obj, "_cascade"):
        raise RuntimeError(
            "online gradient adaptation needs the cutter Jacobian and could not "
            "reach `_cascade` on the environment"
        )
    _ja = (
        env_obj._cascade.jacobian.to(device=a.device, dtype=torch.float32)
        * env_obj._action_to_learned_sign.reshape(1, 4).to(
            device=a.device, dtype=torch.float32
        )
    )
    jac_row_y = _ja[1].reshape(1, 4)
    jac_row_z = _ja[2].reshape(1, 4)
    print(f"[PPO3-1-GATES] task-error Jacobian rows y={_ja[1].tolist()} z={_ja[2].tolist()}", flush=True)

    n = int(raw.num_envs)
    # Sham sign is drawn once per environment and held, so the arm perturbs the
    # weights exactly as much as the real one but in an uninformative direction.
    sham_sign = (torch.randint(0, 2, (n, 4), device=a.device).float() * 2.0 - 1.0)
    from collections import deque

    lag = max(int(a.online_grad_lag), 0)
    beta = float(a.online_grad_beta)
    obs_buf = deque(maxlen=lag + 1)
    g_ema = torch.zeros((n, 4), device=a.device)
    # `resid` accumulates the mean absolute prediction residual, which is the
    # quantity the update actually minimises. The learning rate is selected on
    # it and never on a scored gate metric.
    stats = ONLINE_GRAD_STATS

    print(
        f"[PPO3-1-GATES] online gradient adaptation: lr={a.online_grad_lr} "
        f"trust={trust} params={head.weight.numel() + head.bias.numel()} "
        f"mode={'SHAM' if a.online_grad_sham else 'live'}",
        flush=True,
    )

    def online_grad_policy(obs):
        base = obs["policy"] if isinstance(obs, dict) else obs
        td = TensorDict({"policy": base}, batch_size=[base.shape[0]])
        obs_buf.append(base.detach().clone())
        with torch.enable_grad():
            action = actor(td)
            # Channel 9 is the cut-height (Z) error and channel 10 the lateral
            # (Y) error, both raw: rsl_rl normalises inside the actor, so this
            # observation is untouched.
            e_cut = base[:, 9:10].detach()
            e_lat = base[:, 10:11].detach()
            stats["resid"] += float(
                (e_cut.abs().mean() + e_lat.abs().mean()).detach().cpu()
            )
            d = e_lat * jac_row_y + e_cut * jac_row_z
            if a.online_grad_sham:
                d = d * sham_sign[: d.shape[0]]
            # Low-pass the signal: the loop is closed around 167 ms of dead time
            # and is only stable well below its bandwidth.
            g_ema.mul_(1.0 - beta).add_(d.detach(), alpha=beta)
            stats["steps"] += 1
            # Attribute the error now to the command issued `lag` steps ago,
            # which is the one that produced it. Without this the update fights
            # its own past actions, which the probe identifies as the most
            # likely cause of the uncompensated version degrading.
            if len(obs_buf) > lag:
                old = obs_buf[0]
                u_old = actor(TensorDict({"policy": old}, batch_size=[old.shape[0]]))
                live = (u_old.detach().abs() > 0.05).float()
                stats["quiet"] += float((1.0 - live).mean().detach().cpu())
                loss = (u_old * g_ema.detach() * live).sum(dim=-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            head.weight.copy_(frozen["w"] + (head.weight - frozen["w"]).clamp(-trust, trust))
            head.bias.copy_(frozen["b"] + (head.bias - frozen["b"]).clamp(-trust, trust))
            if not torch.isfinite(head.weight).all() or not torch.isfinite(head.bias).all():
                head.weight.copy_(frozen["w"])
                head.bias.copy_(frozen["b"])
                stats["reverted"] += 1
        return action.detach()

    online_grad_policy.stats = stats
    return online_grad_policy


@torch.no_grad()
def _build_residual_policy(runner, raw, env_cfg):
    """Frozen base + history residual, scored through this file's own rollout.

    The history is kept in this closure, which is correct HERE and would be
    wrong in training: roll_mesh calls the policy exactly once per environment
    step, whereas PPO calls the actor many times per update epoch and would
    advance a closure-held buffer out of sync with the simulation.

    rsl_rl's actor performs the observation normalisation, so this code never
    touches it. That is deliberate: hand-rolling it produced a frozen base 29x
    off the reference twice.
    """
    from tensordict import TensorDict

    from verge_task.residual_adapter import ResidualMLP

    payload = torch.load(a.residual_checkpoint, map_location=a.device, weights_only=False)
    sd = payload["actor_state_dict"]
    base_width = int(sd["mlp.base_mlp.0.weight"].shape[1])
    action_dim = int(env_cfg.action_space)
    base_dim = base_width - 4 if base_width in (70, 72) else base_width
    history = int(a.residual_history)
    # The column spec travels IN the checkpoint (ResidualMLP registers it as a
    # buffer), so a starved-encoder residual is evaluated on exactly the columns
    # it was trained on. Checkpoints written before that buffer existed read the
    # full base observation, which is what they were trained on.
    if "mlp.history_cols" in sd:
        hist_cols = sd["mlp.history_cols"].to(a.device).long()
    else:
        hist_cols = torch.arange(base_dim, dtype=torch.long, device=a.device)
    global RESIDUAL_HISTORY_COLS
    RESIDUAL_HISTORY_COLS = [int(c) for c in hist_cols.tolist()]
    feature = int(hist_cols.numel()) + action_dim
    wide = base_width + history * feature

    import torch.nn as nn
    base_mlp = nn.Sequential(
        nn.Linear(sd["mlp.base_mlp.0.weight"].shape[1], sd["mlp.base_mlp.0.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.2.weight"].shape[1], sd["mlp.base_mlp.2.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.4.weight"].shape[1], sd["mlp.base_mlp.4.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.6.weight"].shape[1], sd["mlp.base_mlp.6.weight"].shape[0]),
    ).to(a.device)
    residual = ResidualMLP(
        base_mlp, base_dim=base_dim, base_width=base_width, action_dim=action_dim,
        history=history, latent_dim=int(a.residual_latent_dim),
        alpha=float(a.residual_alpha), history_cols=hist_cols.cpu(),
    ).to(a.device)

    actor = runner.alg.actor
    actor.mlp = residual
    # Widen the actor's normaliser to the residual's observation and load the
    # checkpoint's statistics, so rsl_rl normalises exactly as it did in training.
    norm = actor.obs_normalizer
    for name in ("_mean", "_var", "_std"):
        if hasattr(norm, name):
            src = sd[f"obs_normalizer.{name}"].to(a.device)
            setattr(norm, name, src.clone())
    missing, unexpected = actor.load_state_dict(sd, strict=False)
    lost = [k for k in missing
            if k.startswith("mlp.") and k != "mlp.history_cols"]
    if lost:
        raise RuntimeError(f"residual weights did not load: missing {lost[:4]}")
    actor.eval()
    print(
        f"[PPO3-1-GATES] residual policy: base_width={base_width} history={history} "
        f"alpha={a.residual_alpha} wide={wide} "
        f"cols={feature - action_dim}+{action_dim}={feature}/step "
        f"history={'ZEROED' if a.residual_no_history else 'live'}",
        flush=True,
    )

    n = int(raw.num_envs)
    state = {
        "hist": torch.zeros((n, history, feature), device=a.device),
        "prev": torch.zeros((n, action_dim), device=a.device),
    }

    def residual_policy(obs):
        base = obs["policy"] if isinstance(obs, dict) else obs
        step = torch.nan_to_num(
            torch.cat((base[:, hist_cols], state["prev"]), dim=-1), nan=0.0,
            posinf=0.0, neginf=0.0)
        state["hist"] = torch.roll(state["hist"], shifts=-1, dims=1)
        state["hist"][:, -1, :] = step
        flat = state["hist"].flatten(1)
        if a.residual_no_history:
            flat = torch.zeros_like(flat)
        td = TensorDict({"policy": torch.cat((base, flat), dim=-1)},
                        batch_size=[base.shape[0]])
        with torch.no_grad():
            action = actor(td)
        state["prev"] = torch.nan_to_num(action.detach().clone(), nan=0.0)
        return action

    return residual_policy


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
    _frames = [] if getattr(a, "video", False) else None
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
        if ADAPT_TRACE["on"]:
            _true = getattr(raw, "_mass_step_scale_now", None)
            _est = getattr(raw, "_payload_ema", None)
            ADAPT_TRACE["t"].append(float(_step) * float(raw.step_dt))
            # True payload as a multiple of base; the estimate is carried in
            # normalised context units, so invert (m - 1) / 0.5 to compare them
            # on the same axis.
            # Fleet MEAN washes out the signal: the periodic plant gives each
            # environment its own phase, so at any instant about half are light
            # and half heavy and the mean is flat. Record individual
            # environments as well, which is what the figure actually needs.
            ADAPT_TRACE["true_payload"].append(
                float(_true.mean()) if _true is not None else float("nan")
            )
            ADAPT_TRACE["est_payload"].append(
                float((_est * 0.5 + 1.0).mean()) if _est is not None else float("nan")
            )
            for _k in (0, 1, 2):
                ADAPT_TRACE[f"true_env{_k}"].append(
                    float(_true[_k]) if _true is not None else float("nan")
                )
                ADAPT_TRACE[f"est_env{_k}"].append(
                    float(_est[_k] * 0.5 + 1.0) if _est is not None else float("nan")
                )
            ADAPT_TRACE["cut_mm"].append(float(raw.cut_height_error().abs().mean()) * 1000.0)
            ADAPT_TRACE["lateral_mm"].append(float(raw.verge_lateral_error().abs().mean()) * 1000.0)
            ADAPT_TRACE["contact"].append(float(raw.cutter_ground_contact().float().mean()))
        if OBS_DUMP["on"] and _step % max(int(a.obs_dump_every), 1) == 0:
            _base = obs["policy"] if isinstance(obs, dict) else obs
            # The label comes from the ENV, not the observation: under
            # MULAG_CONTEXT_ABLATE the observation's context is zeroed or
            # permuted, and a label read from there would be the ablation.
            _true = raw._rma_context_observation()[:, 3]
            OBS_DUMP["x"].append(_base[:, :66].detach().cpu().clone())
            OBS_DUMP["y"].append(_true.detach().cpu().clone())
        action = policy(obs)
        obs, _, terminated, truncated, _ = raw.step(action)
        if _frames is not None and len(_frames) < int(a.video_length):
            # Captured HERE, not via gym.wrappers.RecordVideo: this rollout
            # drives `raw` (the unwrapped env) directly, so a wrapper placed
            # around `env` never sees a step and silently records nothing.
            _frame = raw.render()
            if _frame is not None:
                _frames.append(_frame)
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
            # What the ACTOR reads, after the policy latent mode is applied.
            # Read-only: _policy_latent_observation never advances filter state,
            # which is advanced once per control step inside the ID pipeline.
            # Welford accumulation gives the per-environment temporal variance.
            try:
                zp = raw._policy_latent_observation().detach()
            except Exception:
                zp = None
            if zp is not None:
                n = int(_Z_POLICY_LIVE.get('n', 0)) + 1
                mean = _Z_POLICY_LIVE.get('mean')
                m2 = _Z_POLICY_LIVE.get('m2')
                if not isinstance(mean, torch.Tensor) or mean.shape != zp.shape:
                    mean = torch.zeros_like(zp)
                    m2 = torch.zeros_like(zp)
                    n = 1
                delta = zp - mean
                mean = mean + delta / float(n)
                m2 = m2 + delta * (zp - mean)
                _Z_POLICY_LIVE['n'] = n
                _Z_POLICY_LIVE['mean'] = mean
                _Z_POLICY_LIVE['m2'] = m2
                # ACROSS-ENVIRONMENT spread of the latent the actor reads.
                # Temporal variance alone cannot separate `privileged` from
                # `zero`: both are constant within an episode, so both report
                # 0. They differ in whether environments see DIFFERENT values
                # -- privileged carries a distinct plant code per env, zero is
                # identically zero everywhere. Without this, a broken
                # privileged arm is indistinguishable from a policy that
                # ignores the latent, and Gate 1 cannot be interpreted.
                if zp.ndim == 2 and zp.shape[0] > 1:
                    _Z_POLICY_LIVE['across_env_std'] = max(
                        float(_Z_POLICY_LIVE.get('across_env_std', 0.0)),
                        float(zp.std(dim=0, unbiased=False).amax()),
                    )
                _Z_POLICY_LIVE['abs_max'] = max(
                    float(_Z_POLICY_LIVE.get('abs_max', 0.0)),
                    float(zp.abs().amax()),
                )
        # Command reversal rate, the second mechanism variable. A reversal is a
        # sign change of the commanded increment, counted per axis per step and
        # scaled to the control rate, matching experiments/E1 control effort.
        u_now = getattr(raw, '_current_valve_cmd', None)
        if isinstance(u_now, torch.Tensor):
            u_now = u_now.detach()
            prev_u = _CMD_LIVE.get('prev_u')
            if isinstance(prev_u, torch.Tensor) and prev_u.shape == u_now.shape:
                sign_now = torch.sign(u_now - prev_u)
                prev_sign = _CMD_LIVE.get('prev_sign')
                if (
                    isinstance(prev_sign, torch.Tensor)
                    and prev_sign.shape == sign_now.shape
                ):
                    _CMD_LIVE['reversals'] = _CMD_LIVE.get('reversals', 0.0) + float(
                        (sign_now != prev_sign).float().mean()
                    )
                    _CMD_LIVE['steps'] = int(_CMD_LIVE.get('steps', 0)) + 1
                _CMD_LIVE['prev_sign'] = sign_now
            _CMD_LIVE['prev_u'] = u_now.clone()
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
        # Per-ENVIRONMENT mean error, the unit a paired arm comparison needs.
        # Whole-run means are not comparable across processes: two identical
        # invocations of this script at the same seed were measured 22% apart
        # (38.75-67.43 mm over six nominally-equivalent runs), so an across-run
        # difference smaller than that carries no information. Pairing on the
        # environment cancels the terrain, path and factor draw it shares.
        "per_env_lateral_m": [
            float(v) for v in (
                (lateral_steps * active_steps).sum(dim=0)
                / active_steps.sum(dim=0).clamp_min(1)
            )
        ],
        "per_env_cut_error_m": [
            float(v) for v in (
                (error_steps.abs() * active_steps).sum(dim=0)
                / active_steps.sum(dim=0).clamp_min(1)
            )
        ],
        # Per-environment ground contact, the clearance guard that pairs with
        # the two error channels. Needed per env because the Z reference offset
        # can trade contact for cut-height error and the aggregate hides it.
        "per_env_contact_fraction": [
            float(v) for v in (
                (contact_steps * active_steps).sum(dim=0)
                / active_steps.sum(dim=0).clamp_min(1)
            )
        ],
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
    if PROBE_LOG:
        # The injected probe, saved beside the report so the regression uses the
        # component that is independent of the state by construction.
        probe_path = Path(str(a.report)).expanduser().resolve().with_suffix(".probe.npz")
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probe_path, probe=torch.stack(PROBE_LOG).numpy()
        )
        print(f"[PPO3-1-GATES] wrote {probe_path}", flush=True)
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
    if _frames:
        # Written here rather than through gym.wrappers.RecordVideo: this rollout
        # drives the UNWRAPPED env, so a recording wrapper never sees a step.
        import os as _os

        import imageio.v2 as _imageio

        _dir = _os.path.join(str(a.video_dir), f"mesh{mesh}")
        _os.makedirs(_dir, exist_ok=True)
        _path = _os.path.join(_dir, f"rollout_mesh{mesh}.mp4")
        _imageio.mimsave(_path, _frames, fps=30, macro_block_size=1)
        print(f"[PPO3-1-GATES] wrote {_path} ({len(_frames)} frames)", flush=True)

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
    if a.adapt_trace:
        ADAPT_TRACE["on"] = True
        print(f"[PPO3-1-GATES] adaptation trace -> {a.adapt_trace}", flush=True)
    if a.obs_dump:
        OBS_DUMP["on"] = True
        print(f"[PPO3-1-GATES] obs dump -> {a.obs_dump} "
              f"(every {a.obs_dump_every} steps, 66 task columns, no context)",
              flush=True)
    for mesh in meshes:
        env_cfg = parse_env_cfg(
            TASK_ID,
            device=a.device,
            num_envs=a.num_envs,
        )
        apply_c4_authority_stage(env_cfg, int(a.stage))
        if a.action_scale is not None:
            # MUST come after apply_c4_authority_stage, never before and never
            # followed by another stage call: that chain reaches
            # apply_sole_curriculum_stage, which ASSIGNS action_scale for the
            # stage, so an override on either side of it is silently reverted.
            # The per-axis row is then re-derived here by hand -- exactly the
            # tail of apply_c4_authority_stage -- because it is computed from
            # action_scale and would otherwise keep the stage's value while the
            # scalar carried ours, mixing two contracts in one run.
            override_cfg(env_cfg, "action_scale", float(a.action_scale))
            _base_scale = max(float(env_cfg.action_scale), 1.0e-9)
            override_cfg(
                env_cfg,
                "cylinder_action_scale_per_axis_m",
                (
                    _base_scale,
                    _base_scale,
                    _base_scale,
                    _base_scale * float(getattr(env_cfg, "c4_action_scale_gain", 3.0)),
                ),
            )
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
            if (
                str(a.online_aam_latent_mode) == "lpf"
                and a.online_aam_policy_latent_lpf_hz is None
            ):
                p.error(
                    "--online_aam_latent_mode lpf requires "
                    "--online_aam_policy_latent_lpf_hz"
                )
            override_cfg(
                env_cfg, "online_aam_policy_latent_mode", str(a.online_aam_latent_mode)
            )
        if a.online_aam_policy_latent_lpf_hz is not None:
            if EXPECTED_ACTUATOR != "id_aam_fd":
                p.error(
                    "--online_aam_policy_latent_lpf_hz requires the "
                    "PPO-ID(AAM)-FD task"
                )
            if str(a.online_aam_latent_mode) != "lpf":
                p.error(
                    "--online_aam_policy_latent_lpf_hz is only read by "
                    "--online_aam_latent_mode lpf; passing it to another arm "
                    "would record a cutoff the run never applied"
                )
            if not (float(a.online_aam_policy_latent_lpf_hz) > 0.0):
                p.error("--online_aam_policy_latent_lpf_hz must be > 0")
            override_cfg(
                env_cfg,
                "online_aam_policy_latent_lpf_hz",
                float(a.online_aam_policy_latent_lpf_hz),
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
        for name in ("actuator_authority_alpha", "actuator_authority_alpha_after",
                     "actuator_authority_step_time_s",
                     "deployment_mass_step_time_s", "deployment_mass_step_scale"):
            value = getattr(a, name, None)
            if value is not None:
                override_cfg(env_cfg, name, float(value))
        # Displacement-domain stress. The chain is only installed when at least
        # one axis is set, and it is a bit-for-bit identity at nominal, so an
        # unstressed run is byte-identical whether or not these flags exist.
        stress_axes = {
            "actuator_stress_deadband_m": float,
            "actuator_stress_lag_tau_s": float,
            "actuator_stress_delay_steps": int,
            "actuator_stress_rate_limit_m": float,
        }
        stress_set = False
        for name, cast in stress_axes.items():
            value = getattr(a, name, None)
            if value is not None:
                override_cfg(env_cfg, name, cast(value))
                stress_set = stress_set or cast(value) != cast(0)
        if stress_set:
            override_cfg(env_cfg, "actuator_stress_enabled", True)
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
        if a.deployment_factor_scale is not None:
            override_cfg(env_cfg, "deployment_factor_scale",
                         float(a.deployment_factor_scale))
        if a.online_aam_adapter_checkpoint is not None:
            override_cfg(env_cfg, "online_aam_adapter_checkpoint",
                         str(a.online_aam_adapter_checkpoint))
        if a.tracking_reference_step_m is not None:
            override_cfg(env_cfg, "tracking_reference_step_m",
                         float(a.tracking_reference_step_m))
        if a.tracking_reference_step_time_s is not None:
            override_cfg(env_cfg, "tracking_reference_step_time_s",
                         float(a.tracking_reference_step_time_s))
        if a.plant_step_time_s is not None:
            override_cfg(env_cfg, "plant_perturbation_step_time_s", float(a.plant_step_time_s))
            override_cfg(
                env_cfg, "rom_factor_perturbation_step_time_s", float(a.plant_step_time_s)
            )
        if a.plant_step_delay_steps is not None:
            override_cfg(env_cfg, "plant_perturbation_enabled", True)
            override_cfg(
                env_cfg, "plant_perturbation_step_delay_steps",
                int(a.plant_step_delay_steps),
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

        # ---- viewport camera, RENDERING ONLY -------------------------------
        # Touches env_cfg.viewer and nothing else, so no measurement path is
        # affected; it only decides where the recording camera sits. Driven by
        # environment variables so the CLI signature is unchanged.
        #
        # The default viewer eye (7.5, 7.5, 7.5) looks at the world origin, but
        # the truck STARTS at x = -25.4 m and drives +26 m, so with the default
        # the machine is a distant speck for most of the run. MULAG_VIEW_FOLLOW=1
        # pins the camera to the truck (origin_type "asset_root", asset "robot")
        # so the eye/lookat below become an OFFSET that rides along with it.
        import os as _osv
        if _osv.environ.get("MULAG_VIEW_EYE") or _osv.environ.get("MULAG_VIEW_FOLLOW"):
            def _triple(name, default):
                raw_v = _osv.environ.get(name)
                if not raw_v:
                    return default
                parts = tuple(float(v) for v in raw_v.split(","))
                if len(parts) != 3:
                    raise SystemExit(f"{name} takes three comma-separated numbers")
                return parts
            viewer = getattr(env_cfg, "viewer", None)
            if viewer is not None:
                viewer.eye = _triple("MULAG_VIEW_EYE", viewer.eye)
                viewer.lookat = _triple("MULAG_VIEW_LOOKAT", viewer.lookat)
                if _osv.environ.get("MULAG_VIEW_FOLLOW"):
                    viewer.origin_type = "asset_root"
                    viewer.asset_name = _osv.environ.get("MULAG_VIEW_ASSET", "robot")
                    viewer.env_index = int(_osv.environ.get("MULAG_VIEW_ENV", "0"))
                res = _osv.environ.get("MULAG_VIEW_RES")
                if res:
                    w, h = (int(v) for v in res.split(","))
                    viewer.resolution = (w, h)
                print(
                    f"[PPO3-1-GATES] viewer eye={viewer.eye} lookat={viewer.lookat} "
                    f"origin={viewer.origin_type} asset={viewer.asset_name} "
                    f"res={viewer.resolution}",
                    flush=True,
                )

        env = gym.make(
            TASK_ID, cfg=env_cfg,
            render_mode="rgb_array" if getattr(a, "video", False) else None,
        )
        raw = env.unwrapped
        if a.teacher or a.null:
            zeros = torch.zeros(
                (raw.num_envs, int(env_cfg.action_space)), device=a.device
            )

            def policy(_obs, _zeros=zeros):
                return _zeros
        else:
            if EXPECTED_ACTUATOR == "id_aam_fd":
                from PPO_AAM_FD.env_cfg import PpoIDAAMFDPPORunnerCfg
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
            # A FiLM-trained checkpoint carries mlp.film / mlp.head / mlp.tail
            # instead of the plain mlp.N keys, so the actor must be wrapped
            # BEFORE load or the state_dict does not match. Detected from the
            # checkpoint itself rather than a flag, so a FiLM checkpoint cannot
            # be scored as a plain one by forgetting to pass something.
            _peek = torch.load(str(checkpoint_path), map_location="cpu",
                               weights_only=False)
            _sd = _peek.get("model_state_dict") or _peek.get("actor_state_dict") or {}
            if any(k.startswith("mlp.film.") for k in _sd):
                from verge_task.film_adapter import FiLMWrap

                _nctx = int(env_cfg.observation_space) - 66
                runner.alg.actor.mlp = FiLMWrap(
                    runner.alg.actor.mlp, n_ctx=_nctx
                ).to(a.device)
                print(f"[PPO3-1-GATES] FiLM checkpoint detected: actor wrapped "
                      f"before load, n_ctx={_nctx}", flush=True)
            del _peek, _sd
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
            if getattr(a, "residual_checkpoint", None):
                policy = _build_residual_policy(runner, raw, env_cfg)
            elif float(getattr(a, "valve_probe_amp", 0.0)) > 0.0:
                policy = _build_probe_policy(runner, raw, env_cfg)
            elif abs(float(getattr(a, "online_grad_lr", 0.0))) > 0.0:
                policy = _build_online_grad_policy(runner, raw, env_cfg)
            elif (
                float(getattr(a, "task_bias_y", 0.0)) != 0.0
                or float(getattr(a, "task_bias_z", 0.0)) != 0.0
                or getattr(a, "paired_task_bias", "")
            ):
                TASK_BIAS["y"] = float(a.task_bias_y)
                TASK_BIAS["z"] = float(a.task_bias_z)
                policy = _build_task_bias_policy(runner, raw, env_cfg)
        profiler = cProfile.Profile() if a.profile_out else None
        if profiler is not None:
            profiler.enable()
        try:
            if a.arms_per_env:
                arms = [x.strip() for x in a.arms_per_env.split(",") if x.strip()]
                # `arms[i % n]` ALIASES the arm against the terrain: height scale
                # is `scales[env_index % 4]` and path family is
                # `families[env_index % 11]`, so with 4 arms every arm is locked
                # to one relief scale for the whole run. Measured on an A/A where
                # all arms were the same tensor by construction, that produced a
                # 1.0-2.5 mm spread ordered by relief, not by context -- larger
                # than the effects under test.
                #
                # 11 families x 4 scales are coprime, so a block of 44 covers
                # every (family, scale) pair exactly once. Assigning arms in
                # blocks of 44 therefore gives every arm an identical terrain
                # census; with MULAG_SHARED_TERRAIN=1 the seed is shared too, so
                # env j and env j+44 differ ONLY by arm and the comparison is
                # exactly paired.
                # DERIVE the cycle; do not hardcode it. It is
                # lcm(#path families, #height scales), and the height-scale count
                # is a CURRICULUM setting: stage 4 training runs six scales
                # (0.35..1.15) while the gate config runs four. A hardcoded 44
                # silently re-aliases arm against terrain the moment that count
                # changes -- which is the defect this block exists to fix.
                from math import lcm

                from verge_task.terrain import TRACKING_PATH_FAMILIES

                scales = tuple(
                    getattr(raw, "_terrain_height_scales", None)
                    or getattr(env_cfg, "terrain_height_scales", (0.5, 0.7, 0.85, 1.0))
                )
                n_scales = len(set(scales))
                n_families = len(TRACKING_PATH_FAMILIES)
                CYCLE = lcm(n_families, n_scales)
                block = CYCLE * len(arms)
                print(
                    f"[PPO3-1-GATES] terrain cycle = lcm({n_families} families, "
                    f"{n_scales} height scales) = {CYCLE}; block = {block}",
                    flush=True,
                )
                if raw.num_envs % block:
                    raise SystemExit(
                        f"--arms_per_env with {len(arms)} arms needs num_envs to be a "
                        f"multiple of {block} = lcm({n_families} path families, "
                        f"{n_scales} height scales) x {len(arms)} arms, so each arm gets "
                        f"a balanced terrain census; got {raw.num_envs}."
                    )
                modes = [arms[(i // CYCLE) % len(arms)] for i in range(raw.num_envs)]
                raw._per_env_arm_modes = modes
                print(f"[PPO3-1-GATES] arms per env: {arms} "
                      f"({raw.num_envs // len(arms)} envs each)", flush=True)
                result = roll_mesh(
                    raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
                )
                raw._per_env_arm_modes = None
                lat = result["per_env_lateral_m"]
                cut = result["per_env_cut_error_m"]
                result["arms_per_env"] = {
                    arm: {
                        "env_indices": [i for i, m in enumerate(modes) if m == arm],
                        "per_env_lateral_m": [lat[i] for i, m in enumerate(modes) if m == arm],
                        "per_env_cut_error_m": [cut[i] for i, m in enumerate(modes) if m == arm],
                    }
                    for arm in arms
                }
                per_mesh[str(mesh)] = result
            elif a.paired_task_bias:
                # The oracle probe. One process, one plant draw, one terrain:
                # every offset is rolled against the SAME environments, so the
                # comparison is paired and the across-run spread that makes
                # marginal effects unreadable (invariant 19) does not enter.
                pairs = []
                for item in a.paired_task_bias.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    b_y, b_z = (float(v) for v in item.split(":"))
                    pairs.append((b_y, b_z))
                if pairs and pairs[0] != (0.0, 0.0):
                    raise SystemExit(
                        "--paired_task_bias must start with the 0:0 baseline, "
                        "so every offset is read against the frozen base "
                        "measured in the same process; got "
                        f"{pairs[0][0]}:{pairs[0][1]}."
                    )
                res = {}
                for b_y, b_z in pairs:
                    TASK_BIAS["y"] = b_y
                    TASK_BIAS["z"] = b_z
                    print(
                        f"[PPO3-1-GATES] paired task bias: b_Y={b_y} b_Z={b_z}",
                        flush=True,
                    )
                    res[f"{b_y}:{b_z}"] = roll_mesh(
                        raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
                    )
                TASK_BIAS["y"] = float(a.task_bias_y)
                TASK_BIAS["z"] = float(a.task_bias_z)
                base = res[f"{pairs[0][0]}:{pairs[0][1]}"]
                base["paired_task_bias"] = {
                    k: {
                        "per_env_lateral_m": r["per_env_lateral_m"],
                        "per_env_cut_error_m": r["per_env_cut_error_m"],
                        # The guard. cut_height_error is relu'd and zero in
                        # contact, so an offset that buries the cutter scores a
                        # perfect cut height by destroying clearance. A Z win
                        # is only real if contact does not run away with it.
                        "per_env_contact_fraction": r["per_env_contact_fraction"],
                        "worst_family_ground_contact_fraction": r[
                            "worst_family_ground_contact_fraction"
                        ],
                        "safety_violation_rate": r["safety_violation_rate"],
                    }
                    for k, r in res.items()
                }
                per_mesh[str(mesh)] = base
            elif a.paired_smoothing:
                items = [x.strip() for x in a.paired_smoothing.split(",") if x.strip()]
                if not items or items[0] != "base":
                    raise SystemExit(
                        "--paired_smoothing must start with the literal 'base' so "
                        "the sweep is anchored to the configured baseline measured "
                        f"in this same process; got {items[:1]}."
                    )
                beta0 = float(getattr(env_cfg, "sole_action_filter", 0.75))
                rate0 = float(getattr(env_cfg, "sole_max_action_rate", 0.20))
                print(
                    f"[PPO3-1-GATES] smoothing baseline: beta={beta0} rate={rate0}",
                    flush=True,
                )
                res = {}
                for item in items:
                    if item == "base":
                        beta, rate = beta0, rate0
                    else:
                        beta, rate = (float(v) for v in item.split(":"))
                    # Written onto the cfg the env actually reads each step.
                    raw.cfg.sole_action_filter = beta
                    raw.cfg.sole_max_action_rate = rate
                    print(
                        f"[PPO3-1-GATES] paired smoothing: beta={beta} rate={rate}",
                        flush=True,
                    )
                    res[f"{beta}:{rate}"] = roll_mesh(
                        raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
                    )
                raw.cfg.sole_action_filter = beta0
                raw.cfg.sole_max_action_rate = rate0
                base = res[f"{beta0}:{rate0}"]
                base["paired_smoothing"] = {
                    "baseline_key": f"{beta0}:{rate0}",
                    "arms": {
                        k: {
                            "per_env_lateral_m": r["per_env_lateral_m"],
                            "per_env_cut_error_m": r["per_env_cut_error_m"],
                            "per_env_contact_fraction": r["per_env_contact_fraction"],
                            "worst_family_ground_contact_fraction": r[
                                "worst_family_ground_contact_fraction"
                            ],
                            "safety_violation_rate": r["safety_violation_rate"],
                            "p95_tool_acceleration_mps2": r[
                                "p95_tool_acceleration_mps2"
                            ],
                            "hard_impact_step_fraction": r["hard_impact_step_fraction"],
                        }
                        for k, r in res.items()
                    },
                }
                per_mesh[str(mesh)] = base
            elif a.paired_mass_scale:
                levels = [float(x) for x in a.paired_mass_scale.split(",") if x.strip()]
                view = raw.robot.root_physx_view
                body = int(raw._deployment_factor_body_id)
                baseline = raw._deployment_factor_baseline_masses
                idx_all = torch.arange(raw.num_envs, dtype=torch.long)
                res = {}
                for lv in levels:
                    masses = baseline.clone()
                    masses[:, body] = baseline[:, body] * lv
                    view.set_masses(masses, idx_all)
                    print(f"[PPO3-1-GATES] paired mass_scale: {lv}", flush=True)
                    res[str(lv)] = roll_mesh(
                        raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
                    )
                view.set_masses(baseline, idx_all)
                base = res[str(levels[0])]
                base["paired_mass_scale"] = {
                    k: {"per_env_lateral_m": r["per_env_lateral_m"],
                        "per_env_cut_error_m": r["per_env_cut_error_m"]}
                    for k, r in res.items()
                }
                per_mesh[str(mesh)] = base
            elif a.paired_alpha:
                levels = [float(x) for x in a.paired_alpha.split(",") if x.strip()]
                from PPO_AAM_FD.plant_perturbation import (
                    PlantPerturbation, PlantPerturbationCfg,
                )
                res = {}
                for lv in levels:
                    raw._plant_perturbation = PlantPerturbation(
                        PlantPerturbationCfg(
                            enabled=lv < 1.0,
                            alpha_range=(lv, lv),
                            delay_steps_range=(0, 0),
                        ),
                        num_envs=raw.num_envs, device=raw.device,
                        control_dt=raw._control_dt,
                    )
                    print(f"[PPO3-1-GATES] paired alpha: {lv}", flush=True)
                    res[str(lv)] = roll_mesh(
                        raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
                    )
                base = res[str(levels[0])]
                base["paired_alpha"] = {
                    k: {"per_env_lateral_m": r["per_env_lateral_m"],
                        "per_env_cut_error_m": r["per_env_cut_error_m"]}
                    for k, r in res.items()
                }
                per_mesh[str(mesh)] = base
            elif a.paired_delay_steps:
                # G1 degradation envelope, paired. The perturbation lives in
                # PlantPerturbation, which samples per-env at construction, so
                # the delay is re-driven here by rebuilding its config between
                # rollouts on the SAME env instance.
                levels = [
                    int(x.strip()) for x in a.paired_delay_steps.split(",") if x.strip()
                ]
                from PPO_AAM_FD.plant_perturbation import (
                    PlantPerturbation, PlantPerturbationCfg,
                )
                stress_results = {}
                for level in levels:
                    raw._plant_perturbation = PlantPerturbation(
                        PlantPerturbationCfg(
                            enabled=level > 0,
                            alpha_range=(1.0, 1.0),
                            delay_steps_range=(level, level),
                        ),
                        num_envs=raw.num_envs,
                        device=raw.device,
                        control_dt=raw._control_dt,
                    )
                    print(f"[PPO3-1-GATES] paired delay: {level}", flush=True)
                    stress_results[str(level)] = roll_mesh(
                        raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
                    )
                base = stress_results[str(levels[0])]
                base["paired_delay"] = {
                    k: {
                        "per_env_lateral_m": r["per_env_lateral_m"],
                        "per_env_cut_error_m": r["per_env_cut_error_m"],
                    }
                    for k, r in stress_results.items()
                }
                per_mesh[str(mesh)] = base
            elif a.paired_arms:
                # Score every arm on the SAME live env. `_flatten_observations`
                # re-reads online_aam_policy_latent_mode on each build, so the
                # arm can be switched between rollouts without rebuilding, and
                # each rollout re-resets from the identical seed. The result is
                # a paired per-environment comparison, which is the only kind
                # this harness can support: separate processes at one seed were
                # measured 22% apart, far above any oracle gap worth finding.
                arms = [x.strip() for x in a.paired_arms.split(",") if x.strip()]
                previous_mode = getattr(raw.cfg, "online_aam_policy_latent_mode", None)
                arm_results = {}
                for arm in arms:
                    override_cfg(raw.cfg, "online_aam_policy_latent_mode", arm)
                    print(f"[PPO3-1-GATES] paired arm: {arm}", flush=True)
                    arm_results[arm] = roll_mesh(
                        raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
                    )
                if previous_mode is not None:
                    override_cfg(
                        raw.cfg, "online_aam_policy_latent_mode", previous_mode
                    )
                base = arm_results[arms[0]]
                base["paired_arms"] = {
                    arm: {
                        "per_env_lateral_m": r["per_env_lateral_m"],
                        "per_env_cut_error_m": r["per_env_cut_error_m"],
                        "mean_abs_cut_height_error_m": r["mean_abs_cut_height_error_m"],
                        "worst_family_mean_abs_lateral_error_m": r[
                            "worst_family_mean_abs_lateral_error_m"
                        ],
                    }
                    for arm, r in arm_results.items()
                }
                per_mesh[str(mesh)] = base
            else:
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
        # Read the Gate 0 counters while the runtime is still alive: they are
        # CUDA tensors, and env.close() tears the simulation down underneath
        # them. Cumulative over the process, so this is the whole mesh.
        _aam_rt = getattr(raw, "_online_aam", None)
        if _aam_rt is not None and hasattr(_aam_rt, "gate0_instrumentation"):
            try:
                _GATE0_READOUT["gate0"] = _aam_rt.gate0_instrumentation()
            except Exception as exc:  # instrumentation must never kill a run
                _GATE0_READOUT["gate0"] = {"error": repr(exc)}
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
    # Gate 0 readout. The counters live on the online-AAM runtime and are
    # cumulative over the process, so this is the whole run for this mesh.
    # Recorded unconditionally: the repair plan requires the fallback rate to
    # be reported beside every latent result, and a result whose fallback rate
    # was never looked at cannot be interpreted afterwards.
    gate0_readout = _GATE0_READOUT.get("gate0")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gate0_adaptation_site": gate0_readout,
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
        # The bandwidth axis. Recorded on every arm, not only the lpf ones, so
        # a sweep summariser can order the arms without parsing their names.
        "online_aam_policy_latent_lpf_hz": float(
            getattr(env_cfg, "online_aam_policy_latent_lpf_hz", 0.0)
        ),
        # Mechanism variables. The first is the temporal variance of the latent
        # AS THE ACTOR READS IT, which is what a bandwidth sweep varies; the
        # second is how often the commanded increment changes sign.
        "policy_latent_temporal_var": float(
            (
                _Z_POLICY_LIVE["m2"] / max(int(_Z_POLICY_LIVE.get("n", 1)) - 1, 1)
            ).mean()
        ) if isinstance(_Z_POLICY_LIVE.get("m2"), torch.Tensor) else 0.0,
        # Separates a live privileged arm from a dead one; see the accumulation
        # site for why temporal variance alone is not sufficient.
        "policy_latent_across_env_std": float(
            _Z_POLICY_LIVE.get("across_env_std", 0.0)
        ),
        "policy_latent_abs_max": float(_Z_POLICY_LIVE.get("abs_max", 0.0)),
        "command_reversals_per_s": (
            float(_CMD_LIVE["reversals"]) / float(_CMD_LIVE["steps"])
            / max(float(getattr(env_cfg.sim, "dt", 1.0 / 180.0))
                  * float(getattr(env_cfg, "decimation", 3)), 1e-9)
        ) if int(_CMD_LIVE.get("steps", 0)) > 0 else 0.0,
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
        # Which base-observation columns the residual's history encoder read,
        # recovered from the checkpoint. Records the starved-encoder arm in the
        # artefact rather than only in the launch command.
        "residual_checkpoint": str(getattr(a, "residual_checkpoint", "") or ""),
        "residual_no_history": bool(getattr(a, "residual_no_history", False)),
        "online_grad_lr": float(getattr(a, "online_grad_lr", 0.0)),
        "online_grad_sham": bool(getattr(a, "online_grad_sham", False)),
        "online_grad_trust": float(getattr(a, "online_grad_trust", 0.0)),
        "online_grad_lag": int(getattr(a, "online_grad_lag", 0)),
        "online_grad_beta": float(getattr(a, "online_grad_beta", 1.0)),
        "online_grad_diag": (
            {
                "mean_abs_task_error": ONLINE_GRAD_STATS["resid"] / max(ONLINE_GRAD_STATS["steps"], 1),
                "quiet_axis_fraction": ONLINE_GRAD_STATS["quiet"] / max(ONLINE_GRAD_STATS["steps"], 1),
                "nonfinite_reverts": ONLINE_GRAD_STATS["reverted"],
                "steps": ONLINE_GRAD_STATS["steps"],
            }
            if ONLINE_GRAD_STATS["steps"] else {}
        ),
        "residual_history_cols": RESIDUAL_HISTORY_COLS,
        "deployment_factor_enabled": bool(
            getattr(env_cfg, "deployment_factor_enabled", False)
        ),
        "action_scale": float(getattr(env_cfg, "action_scale", 0.0)),
        "cylinder_action_scale_per_axis_m": list(
            getattr(env_cfg, "cylinder_action_scale_per_axis_m", ()) or ()
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

    if ADAPT_TRACE["on"] and ADAPT_TRACE["t"]:
        _tp = Path(a.adapt_trace)
        _tp.parent.mkdir(parents=True, exist_ok=True)
        _tp.write_text(json.dumps(
            {k: v for k, v in ADAPT_TRACE.items() if k != "on"}, indent=1))
        print(f"[PPO3-1-GATES] wrote adaptation trace {_tp} "
              f"({len(ADAPT_TRACE['t'])} steps)", flush=True)

    if OBS_DUMP["on"] and OBS_DUMP["x"]:
        import numpy as _np

        _dump = Path(a.obs_dump)
        _dump.parent.mkdir(parents=True, exist_ok=True)
        _x = torch.cat(OBS_DUMP["x"], dim=0).numpy()
        _y = torch.cat(OBS_DUMP["y"], dim=0).numpy()
        # Record the layout so a temporal model can reshape the dump back into
        # (steps, envs, features). Windows are only reconstructible when the
        # sampling stride is 1; at any other stride the rows are not contiguous
        # in time and a convolution over them would be meaningless.
        _np.savez_compressed(
            _dump, x=_x, y=_y,
            n_envs=_np.int32(int(raw.num_envs)),
            stride=_np.int32(int(a.obs_dump_every)),
        )
        print(f"[PPO3-1-GATES] wrote obs dump {_dump} "
              f"x={_x.shape} y={_y.shape} n_envs={int(raw.num_envs)} "
              f"stride={int(a.obs_dump_every)}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
