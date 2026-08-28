#!/usr/bin/env python3
"""Train or continue the final sole-controller PPO-ROM policy."""

from __future__ import annotations

import argparse
import copy
import json
import sys
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
p.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stages", default="6")
p.add_argument(
    "--trajectory-shape",
    "--trajectory_shape",
    "--path-family",
    dest="trajectory_shape",
    default="",
    help="Train every environment on one cutter tracking-line shape.",
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
    "--stage-iterations",
    "--stage_iterations",
    dest="stage_iterations",
    type=int,
    default=400,
)
p.add_argument(
    "--save-interval",
    "--save_interval",
    dest="save_interval",
    type=int,
    default=50,
)
p.add_argument(
    "--num-steps-per-env",
    "--num_steps_per_env",
    dest="num_steps_per_env",
    type=int,
    default=None,
    help="PPO rollout length; default uses the runner configuration.",
)
p.add_argument("--learning_rate", type=float, default=None)
p.add_argument("--entropy_coef", type=float, default=None)
p.add_argument(
    "--gamma",
    type=float,
    default=None,
    help="GAE discount factor. The recovery probe uses 0.97 for the long 2 km/h episode.",
)
p.add_argument(
    "--lam",
    type=float,
    default=None,
    help="GAE lambda. Leave at the task default unless running a planned ablation.",
)
p.add_argument(
    "--demo_regularized",
    action="store_true",
    help=(
        "Keep the DLS teacher in label-only mode on policy-visited states and "
        "add filtered demonstration loss plus KL to the frozen prior. The "
        "teacher never executes; PPO remains the sole plant controller."
    ),
)
p.add_argument(
    "--trajectory_regularized",
    action="store_true",
    help=(
        "Use one PPO optimizer step containing reward PPO, forward Gaussian "
        "KL to a frozen copy of the resumed warm actor, and strict future "
        "disagreement credit. The reference never executes and DLS stays off."
    ),
)
p.add_argument(
    "--trajectory_local_coef",
    type=float,
    default=0.05,
    help="Forward reference-KL coefficient for --trajectory_regularized.",
)
p.add_argument(
    "--trajectory_future_coef",
    type=float,
    default=0.0,
    help="Optional future-disagreement coefficient; production keeps this off.",
)
p.add_argument(
    "--trajectory_max_update_kl",
    type=float,
    default=0.01,
    help="Maximum old-policy to new-policy KL; larger updates are rolled back atomically.",
)
p.add_argument(
    "--trajectory_max_reference_kl",
    type=float,
    default=0.05,
    help="Maximum frozen-reference to live-policy KL before atomic rollback.",
)
p.add_argument(
    "--trajectory_clip_param",
    type=float,
    default=0.10,
    help="PPO ratio/value clip used by trajectory-regularized PPO.",
)
p.add_argument(
    "--trajectory_learning_epochs",
    type=int,
    default=3,
    help="PPO epochs per rollout for trajectory-regularized PPO.",
)
p.add_argument(
    "--trajectory_actor_learning_rate",
    type=float,
    default=1.0e-5,
    help="Fixed actor learning rate for trajectory-regularized PPO.",
)
p.add_argument(
    "--trajectory_critic_learning_rate",
    type=float,
    default=1.0e-4,
    help="Fixed critic learning rate; warmed Adam moments are preserved.",
)
p.add_argument(
    "--demo_lambda",
    type=float,
    default=0.25,
    help="Filtered demonstration-loss weight for --demo_regularized.",
)
p.add_argument(
    "--demo_kl_beta",
    type=float,
    default=0.02,
    help="Frozen-prior Gaussian KL weight for --demo_regularized.",
)
p.add_argument(
    "--demo_max_kl",
    type=float,
    default=0.25,
    help=(
        "Hard post-update KL cap to the frozen distilled prior. A positive "
        "value projects an actor update back toward the prior when the "
        "penalty alone is insufficient."
    ),
)
p.add_argument(
    "--demo_lateral_threshold",
    type=float,
    default=0.10,
    help="Only weak/off-contact policy states receive teacher imitation.",
)
p.add_argument(
    "--desired_kl",
    type=float,
    default=None,
    help="Adaptive-schedule KL setpoint (inherited default 0.02; the "
    "teacher_student fine-tune regime used 0.008).",
)
p.add_argument(
    "--std_clamp",
    default=None,
    help="'min,max' bounds clamped onto the per-axis exploration std after "
    "every PPO update (e.g. '0.03,0.15'), mirroring skrl's clip_log_std. "
    "Without it the entropy bonus can inflate std without limit.",
)
p.add_argument("--init_std", type=float, default=None)
p.add_argument(
    "--fixed_std",
    type=float,
    default=None,
    help=(
        "Set every Gaussian action std to this value after resume and remove "
        "the std parameter from gradient updates."
    ),
)
p.add_argument("--resume", default=None, help="66x4 checkpoint to resume from.")
p.add_argument(
    "--from_scratch",
    action="store_true",
    help="Explicit ablation: initialize a random terrain actor.",
)
p.add_argument(
    "--reset_optimizer_on_resume",
    action="store_true",
    help="Load actor/critic weights but discard stale PPO optimizer moments.",
)
p.add_argument(
    "--reset_actor_optimizer_on_resume",
    action="store_true",
    help=(
        "Discard only actor Adam moments after loading a checkpoint. Use when "
        "an accepted checkpoint becomes a new frozen-reference block; critic "
        "weights and optimizer calibration are retained."
    ),
)
p.add_argument(
    "--reset_critic_head_on_resume",
    action="store_true",
    help="Zero the resumed critic output head after a reward/plant change.",
)
p.add_argument(
    "--resume_std",
    type=float,
    default=None,
    help="Exploration std forced on ALL actions after resume.",
)
p.add_argument(
    "--resume_c4_std",
    type=float,
    default=None,
    help="Exploration std forced on the head-pitch (c4) action after resume.",
)
p.add_argument(
    "--freeze_actor_iterations",
    type=int,
    default=0,
    help=(
        "Critic warm-up: hold the actor (mean net AND exploration std) exactly "
        "at its resumed weights for the first N iterations while the critic "
        "learns against the actor's rollouts (requires_grad=False, so the "
        "shared Adam skips actor params). The adaptive KL scheduler is "
        "suspended for the window and the learning rate reset at unfreeze. "
        "Measured context: with a reset critic head the value loss stays at "
        "6k-26k and the policy descends its own reward; this is the standard "
        "remedy."
    ),
)
p.add_argument(
    "--obs_normalizer",
    choices=("freeze", "adapt"),
    default="freeze",
    help="Freeze when continuing the selected policy; adapt is an ablation.",
)
p.add_argument(
    "--task",
    default="",
    help="Gym task override; empty selects the final PPO-ROM task.",
)
p.add_argument("--checkpoint_out", default="")
p.add_argument("--log_dir", default="")
p.add_argument(
    "--cylinder_target_lead_c4_m",
    type=float,
    default=None,
    help="Override the c4 lead cap (metres).",
)
p.add_argument(
    "--c4_action_scale_gain",
    type=float,
    default=None,
    help="Override the c4 action-scale multiplier vs c1-c3.",
)
p.add_argument(
    "--until_converged",
    action="store_true",
    help="Repeat the final curriculum stage until all shared gates pass.",
)
p.add_argument("--convergence_check_interval", type=int, default=50)
p.add_argument("--convergence_steps", type=int, default=300)
p.add_argument(
    "--vehicle_speed_kmh",
    type=float,
    default=None,
    help=(
        "Pin one scripted truck speed. If omitted, PPO-ROM keeps the task's "
        "2--7 km/h per-episode range."
    ),
)
p.add_argument(
    "--vehicle_speed_min_kmh",
    type=float,
    default=None,
    help="With --vehicle_speed_max_kmh: per-episode uniform truck-speed "
    "randomization, the teacher_student repo's regime (constant within an "
    "episode, resampled at reset) adapted to this asset's kinematic drive. "
    "The periodic mid-episode resampler inherited from the PPO2 obstacle "
    "task is disabled so the speed really is per-episode. Evaluation "
    "scripts stay pinned at their own speed and are unaffected.",
)
p.add_argument("--vehicle_speed_max_kmh", type=float, default=None)
p.add_argument(
    "--vehicle_speed_sampling",
    choices=("uniform", "active_time_balanced"),
    default="uniform",
    help=(
        "Episode reset distribution. active_time_balanced uses p(v) proportional "
        "to v to offset the shorter active path duration of faster replicas."
    ),
)
p.add_argument(
    "--speed_segment_duration_s",
    type=float,
    default=0.0,
    help=(
        "Training-only equal-duration path segment. Zero keeps the full 26 m "
        "deployment episode used by evaluation."
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

stages = tuple(int(part) for part in a.stages.split(",") if part.strip())
if not stages or any(stage not in (1, 2, 3, 4, 5, 6) for stage in stages):
    p.error("--stages must be a comma-separated subset of 1,2,3,4,5,6")
if a.resume and a.from_scratch:
    p.error("--resume and --from_scratch are mutually exclusive")
if a.demo_regularized and a.trajectory_regularized:
    p.error("--demo_regularized and --trajectory_regularized are mutually exclusive")
if a.reset_actor_optimizer_on_resume and not a.trajectory_regularized:
    p.error("--reset_actor_optimizer_on_resume requires --trajectory_regularized")
if a.trajectory_regularized and not a.resume:
    p.error("--trajectory_regularized requires a resumed warm reference checkpoint")
if a.trajectory_regularized:
    if min(a.trajectory_local_coef, a.trajectory_future_coef) < 0.0:
        p.error("trajectory coefficients must be non-negative")
    if a.trajectory_local_coef == 0.0 and a.trajectory_future_coef == 0.0:
        p.error("at least one trajectory coefficient must be positive")
    if min(a.trajectory_max_update_kl, a.trajectory_max_reference_kl) <= 0.0:
        p.error("trajectory KL limits must be positive")
    if not 0.0 < a.trajectory_clip_param < 1.0:
        p.error("--trajectory_clip_param must be in (0, 1)")
    if a.trajectory_learning_epochs <= 0:
        p.error("--trajectory_learning_epochs must be positive")
    if min(a.trajectory_actor_learning_rate, a.trajectory_critic_learning_rate) <= 0.0:
        p.error("trajectory actor/critic learning rates must be positive")
if a.fixed_std is not None and a.fixed_std <= 0.0:
    p.error("--fixed_std must be positive")
if a.fixed_std is not None and a.std_clamp is not None:
    p.error("--fixed_std and --std_clamp are mutually exclusive")
if (a.vehicle_speed_min_kmh is None) != (a.vehicle_speed_max_kmh is None):
    p.error(
        "--vehicle_speed_min_kmh and --vehicle_speed_max_kmh must be given "
        "together; one alone would silently pin the speed"
    )
if a.vehicle_speed_min_kmh is not None:
    if not 0.0 < a.vehicle_speed_min_kmh <= a.vehicle_speed_max_kmh:
        p.error("truck speed range must satisfy 0 < min <= max")
if a.speed_segment_duration_s < 0.0:
    p.error("--speed_segment_duration_s must be non-negative")
if a.until_converged:
    raise SystemExit(
        "--until_converged scores the legacy 26-gate contract, whose "
        "ground_contact 0.90 / worst_family_contact 0.85 / "
        "hard_impact_event_count 0 thresholds the DLS teacher itself fails. "
        "It cannot pass. Train for a fixed --stage_iterations and score the "
        "result with scripts/evaluate_gates.py, which implements the 8-gate "
        "multi-mesh contract."
    )

std_clamp_bounds = None
if a.std_clamp is not None:
    try:
        std_lo, std_hi = (float(part) for part in a.std_clamp.split(","))
    except ValueError:
        p.error("--std_clamp must be 'min,max', e.g. '0.03,0.15'")
    if not 0.0 < std_lo < std_hi:
        p.error("--std_clamp must satisfy 0 < min < max")
    std_clamp_bounds = (std_lo, std_hi)
if a.resume is None and not a.from_scratch:
    p.error(
        "production ppo3_1 training requires a seed checkpoint (--resume) or "
        "an explicitly labelled --from_scratch ablation"
    )

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO_ROM  # noqa: F401,E402  (registers the gym task)

if a.task.startswith("RRLAB-Mulag-Verge-IDFD"):
    import importlib

    importlib.import_module("PPO_ID_FD")
    TASK_ID = a.task
    EXPECTED_ACTUATOR = "idfd"
else:
    TASK_ID = a.task or PPO_ROM.TERRAIN_TASK_ID
    EXPECTED_ACTUATOR = "rom"
print(f"[PPO3-1-SOLE] task={TASK_ID} actuator={EXPECTED_ACTUATOR}", flush=True)
from PPO_ROM.terrain_env_cfg import (  # noqa: E402
    TerrainC4AuthorityPPORunnerCfg,  # noqa: F401  (kept for compatibility)
    apply_c4_authority_stage,
)
from verge_task.terrain_env_cfg import (  # noqa: E402
    configure_terrain_gpu_physx_memory,
)
from verge_task.distill import assert_teacher_disabled  # noqa: E402
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)
from verge_task.terrain_convergence import summarize_convergence  # noqa: E402
from verge_task.actuators import resolve_actuator  # noqa: E402


def apply_plant_overrides(cfg) -> None:
    """Re-assert the CLI c4-authority and plant overrides on top of a stage."""
    if a.cylinder_target_lead_c4_m is not None:
        override_cfg(cfg, "cylinder_target_lead_c4_m", float(a.cylinder_target_lead_c4_m))
    if a.c4_action_scale_gain is not None:
        override_cfg(cfg, "c4_action_scale_gain", float(a.c4_action_scale_gain))


@torch.inference_mode()
def convergence_rollout(raw, policy, steps: int) -> dict:
    """Deterministic rollout used as the stop condition for persistent PPO."""
    obs, _ = raw.reset(seed=int(a.seed) + 100_003)
    errors, accelerations, alignments = [], [], []
    laterals, scan_validity, contacts = [], [], []
    wheel_support, wheel_forces = [], []
    truck_heading_errors = []
    terminations = 0
    for _ in range(int(steps)):
        action = policy(obs)
        obs, _, terminated, _, _ = raw.step(action)
        errors.append(raw.cut_height_error().abs())
        accelerations.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )
        alignments.append(raw._measure_head_down_alignment())
        laterals.append(raw.verge_lateral_error().abs())
        scan_validity.append(raw.scan_valid_fraction())
        contacts.append(raw.cutter_ground_contact().float())
        wheel_support.append(raw.wheel_road_contact_fraction())
        wheel_forces.append(raw.wheel_road_contact_forces_n())
        truck_heading_errors.append(raw._truck_heading_error_rad)
        terminations += int(terminated.sum())

    error_steps = torch.stack(errors)
    lateral_steps = torch.stack(laterals)
    contact_steps = torch.stack(contacts)
    error = error_steps.reshape(-1)
    metrics = {
        "mean_abs_cut_height_error_m": float(error.mean()),
        "p95_abs_cut_height_error_m": float(torch.quantile(error, 0.95)),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(torch.cat(accelerations), 0.95)
        ),
        "min_head_alignment": float(torch.cat(alignments).min()),
        "p01_head_alignment": float(torch.quantile(torch.cat(alignments), 0.01)),
        "mean_abs_lateral_error_m": float(lateral_steps.mean()),
        "scan_valid_fraction": float(torch.cat(scan_validity).mean()),
        "ground_contact_fraction": float(contact_steps.mean()),
        "min_wheel_road_contact_duty_cycle": float(
            (
                torch.stack(wheel_forces)
                > float(getattr(raw.cfg, "road_contact_force_threshold_n", 10.0))
            ).float().mean(dim=(0, 1)).min()
        ),
        "max_truck_post_constraint_drift_m": float(
            raw._max_truck_post_constraint_drift.max()
        ),
        "max_truck_heading_error_deg": float(
            torch.rad2deg(torch.cat(truck_heading_errors).max())
        ),
        "safety_termination_rate": terminations / max(steps * raw.num_envs, 1),
    }
    family_rows = []
    for family in sorted(set(raw._tracking_path_family_names)):
        family_mask = torch.as_tensor(
            [name == family for name in raw._tracking_path_family_names],
            dtype=torch.bool,
            device=raw.device,
        )
        family_rows.append(
            {
                "family": family,
                "num_envs": int(family_mask.sum()),
                "mean_abs_cut_height_error_m": float(
                    error_steps[:, family_mask].mean()
                ),
                "mean_abs_lateral_error_m": float(
                    lateral_steps[:, family_mask].mean()
                ),
                "ground_contact_fraction": float(
                    contact_steps[:, family_mask].mean()
                ),
            }
        )
    metrics.update(
        {
            "worst_family_mean_abs_cut_height_error_m": max(
                row["mean_abs_cut_height_error_m"] for row in family_rows
            ),
            "worst_family_mean_abs_lateral_error_m": max(
                row["mean_abs_lateral_error_m"] for row in family_rows
            ),
            "worst_family_ground_contact_fraction": min(
                row["ground_contact_fraction"] for row in family_rows
            ),
        }
    )
    result = summarize_convergence(metrics)
    result["path_family_results"] = family_rows
    return result


def main() -> None:
    resume_path = Path(a.resume).resolve() if a.resume else None
    resume_payload = None
    resume_source = "random ablation"
    teacher_handoff = False
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        infos = resume_payload.get("infos") or {}
        resume_source = str(infos.get("source", ""))
        schema = str(infos.get("schema", ""))
        # "PPO4-1 terrain PPO" is what THIS script writes when --task selects the
        # ID-FD env (see checkpoint_source below), so leaving it out meant the
        # ID-FD pipeline could warm-start from the distilled student but never
        # resume from its own output: the first run succeeded, and every retry
        # after a failed gate died here in seconds while the supervisor kept
        # re-scoring the same stale checkpoint in a loop.
        #
        # This allowlist is not what keeps a ROM policy out of an ID-FD run.
        # `assert_checkpoint_actuator` immediately below compares the stored
        # actuator against the task-derived EXPECTED_ACTUATOR and rejects the
        # mismatch, and the observation/action widths are checked after that.
        if schema not in {"terrain_teacher_student_v1", "terrain_shared_v2"} or resume_source not in {
            "PPO3 terrain teacher distillation v1",
            "PPO3 terrain PPO",
            "PPO3-1 terrain PPO",
            "PPO4-1 terrain PPO",
        }:
            raise RuntimeError(
                "checkpoint is outside the supported ppo3_1 terrain contracts "
                f"(source={resume_source!r}, schema={schema!r})"
            )
        teacher_handoff = resume_source == "PPO3 terrain teacher distillation v1"
        from verge_task.actuators import assert_checkpoint_actuator

        assert_checkpoint_actuator(
            infos, EXPECTED_ACTUATOR, context="PPO3-1 training resume"
        )

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=a.device,
        num_envs=a.num_envs,
    )
    apply_plant_overrides(env_cfg)
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.enable_contact_tracking_demo = False
    if resume_payload is not None:
        resume_infos = resume_payload.get("infos") or {}
        checkpoint_width = int(resume_infos.get("observation_width", -1))
        if checkpoint_width != int(env_cfg.observation_space):
            raise RuntimeError(
                f"checkpoint observation width {checkpoint_width} does not match "
                f"the ppo3_1 contract {env_cfg.observation_space}; migrate with "
                "verge_task/scripts/migrate_shared_policy.py --mode "
                "widen_observation --target_observations 66"
            )
        checkpoint_action_width = int(resume_infos.get("action_width", 4))
        if checkpoint_action_width != int(env_cfg.action_space):
            raise RuntimeError(
                f"checkpoint action width {checkpoint_action_width} does not "
                f"match environment width {env_cfg.action_space}"
            )
    if a.vehicle_speed_min_kmh is not None:
        env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_min_kmh)
        env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_max_kmh)
        env_cfg.vehicle_speed_resample_step_threshold = 10**9
    elif a.vehicle_speed_kmh is not None:
        env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_kmh)
        env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_kmh)
    env_cfg.vehicle_speed_per_episode = True
    env_cfg.vehicle_speed_sampling = str(a.vehicle_speed_sampling)
    env_cfg.vehicle_training_segment_duration_s = float(
        a.speed_segment_duration_s
    )
    configure_path_horizon(env_cfg)
    env_cfg.terrain_base_seed = int(a.seed)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))

    _, runner_cfg_cls, actuator_label = resolve_actuator(EXPECTED_ACTUATOR)
    runner_cfg = runner_cfg_cls()
    checkpoint_source = (
        "PPO3-1 terrain PPO"
        if EXPECTED_ACTUATOR == "rom"
        else "PPO4-1 terrain PPO"
    )
    runner_cfg.seed = int(a.seed)
    runner_cfg.max_iterations = int(a.stage_iterations)
    runner_cfg.save_interval = int(a.save_interval)
    if a.num_steps_per_env is not None:
        runner_cfg.num_steps_per_env = int(a.num_steps_per_env)
    if a.gamma is not None:
        if not 0.0 < a.gamma <= 1.0:
            p.error("--gamma must satisfy 0 < gamma <= 1")
        runner_cfg.algorithm.gamma = float(a.gamma)
    if a.lam is not None:
        if not 0.0 < a.lam <= 1.0:
            p.error("--lam must satisfy 0 < lam <= 1")
        runner_cfg.algorithm.lam = float(a.lam)
    if a.learning_rate is not None:
        runner_cfg.algorithm.learning_rate = float(a.learning_rate)
    if a.entropy_coef is not None:
        runner_cfg.algorithm.entropy_coef = float(a.entropy_coef)
    if a.desired_kl is not None:
        runner_cfg.algorithm.desired_kl = float(a.desired_kl)
    if a.init_std is not None:
        runner_cfg.actor.distribution_cfg.init_std = float(a.init_std)
    if a.trajectory_regularized:
        runner_cfg.algorithm.schedule = "fixed"
        runner_cfg.algorithm.learning_rate = float(a.trajectory_critic_learning_rate)
        runner_cfg.algorithm.clip_param = float(a.trajectory_clip_param)
        runner_cfg.algorithm.num_learning_epochs = int(a.trajectory_learning_epochs)

    apply_c4_authority_stage(env_cfg, stages[0])
    apply_trajectory_shapes(env_cfg, selected_shapes)
    if a.speed_segment_duration_s > 0.0:
        phases = (0.0, 0.25, 0.50, 0.75)
        env_cfg.tracking_path_family_schedule = tuple(
            family for phase in phases for family in selected_shapes
        )
        env_cfg.tracking_path_start_phase_schedule = tuple(
            phase for phase in phases for _ in selected_shapes
        )
    apply_plant_overrides(env_cfg)
    configure_path_horizon(env_cfg)
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    env_cfg.randomize_trajectory_shape = False
    env_cfg.randomize_start_phase = False
    requested_speed_kmh = (
        float(env_cfg.vehicle_speed_min_kmh),
        float(env_cfg.vehicle_speed_max_kmh),
    )
    if a.demo_regularized:
        env_cfg.sole_teacher_mode = "label"
    else:
        assert_teacher_disabled(env_cfg, context="PPO3-1 PPO training")
    env = gym.make(
        TASK_ID, cfg=env_cfg, render_mode=None
    )
    raw = env.unwrapped
    built_speed_kmh = (
        float(raw.cfg.vehicle_speed_min_kmh),
        float(raw.cfg.vehicle_speed_max_kmh),
    )
    if any(
        abs(built - want) > 1.0e-6
        for built, want in zip(built_speed_kmh, requested_speed_kmh)
    ):
        raise RuntimeError(
            "truck speed changed during env construction: requested "
            f"{requested_speed_kmh} km/h, built {built_speed_kmh} km/h. The "
            "deployment trajectory's truck_motion_mode overwrote it. Training "
            "on a parked truck while distillation and acceptance drive is the "
            "defect this check exists to stop."
        )
    print(
        f"[PPO3-1-DRIVE] truck speed verified at {built_speed_kmh[0]:.2f}-"
        f"{built_speed_kmh[1]:.2f} km/h after construction "
        f"(trajectory={raw.cfg.trajectory_shape})",
        flush=True,
    )
    raw.reset(seed=int(a.seed))
    reset_obs = raw._get_observations()["policy"]
    if tuple(reset_obs.shape) != (raw.num_envs, int(raw.cfg.observation_space)):
        raise RuntimeError(
            f"reset observation shape {tuple(reset_obs.shape)} violates the "
            "ppo3_1 contract"
        )
    if not bool(torch.isfinite(reset_obs).all()):
        raise RuntimeError("non-finite reset observation; refusing to start PPO")

    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    log_dir = (
        Path(a.log_dir).expanduser().resolve()
        if a.log_dir
        else ROOT
        / "logs/rsl_rl"
        / runner_cfg.experiment_name
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=str(log_dir), device=a.device
    )

    def configure_trajectory_optimizer() -> None:
        """Split actor/critic rates without discarding loaded Adam moments."""

        old_optimizer = runner.alg.optimizer
        if len(old_optimizer.param_groups) == 2:
            old_optimizer.param_groups[0]["lr"] = float(a.trajectory_actor_learning_rate)
            old_optimizer.param_groups[1]["lr"] = float(a.trajectory_critic_learning_rate)
            runner.alg.learning_rate = float(a.trajectory_actor_learning_rate)
            return
        if len(old_optimizer.param_groups) != 1:
            raise RuntimeError(
                f"cannot split optimizer with {len(old_optimizer.param_groups)} parameter groups"
            )

        template = {
            key: value
            for key, value in old_optimizer.param_groups[0].items()
            if key not in {"params", "lr"}
        }
        new_optimizer = type(old_optimizer)(
            [
                {
                    "params": list(runner.alg.actor.parameters()),
                    "lr": float(a.trajectory_actor_learning_rate),
                },
                {
                    "params": list(runner.alg.critic.parameters()),
                    "lr": float(a.trajectory_critic_learning_rate),
                },
            ]
        )
        for group in new_optimizer.param_groups:
            group.update(template)
        for parameter, state in old_optimizer.state.items():
            new_optimizer.state[parameter] = state
        runner.alg.optimizer = new_optimizer
        runner.alg.learning_rate = float(a.trajectory_actor_learning_rate)

    resume_infos_for_optimizer = (resume_payload or {}).get("infos") or {}
    if a.trajectory_regularized and bool(
        resume_infos_for_optimizer.get("trajectory_regularized", False)
    ):
        configure_trajectory_optimizer()

    startup_manifest = {
        "schema": "terrain_shared_v2",
        "actuator": EXPECTED_ACTUATOR,
        "actuator_label": actuator_label,
        "task": TASK_ID,
        "experiment": str(runner_cfg.experiment_name),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "initialization_source": resume_source,
        "resume": str(resume_path) if resume_path is not None else None,
        "teacher_mode": str(raw.cfg.sole_teacher_mode),
        "teacher_enabled_during_ppo": bool(a.demo_regularized),
        "teacher_labels_enabled": bool(a.demo_regularized),
        "teacher_execution_during_ppo": False,
        "demo_regularized": bool(a.demo_regularized),
        "demo_lambda": float(a.demo_lambda) if a.demo_regularized else None,
        "demo_kl_beta": float(a.demo_kl_beta) if a.demo_regularized else None,
        "demo_max_kl": float(a.demo_max_kl) if a.demo_regularized else None,
        "demo_lateral_threshold": (
            float(a.demo_lateral_threshold) if a.demo_regularized else None
        ),
        "trajectory_regularized": bool(a.trajectory_regularized),
        "trajectory_reference": "frozen_resumed_warm_actor" if a.trajectory_regularized else None,
        "trajectory_reference_checkpoint": (
            str(resume_path) if a.trajectory_regularized and resume_path is not None else None
        ),
        "reset_actor_optimizer_on_resume": bool(a.reset_actor_optimizer_on_resume),
        "trajectory_local_coef": (
            float(a.trajectory_local_coef) if a.trajectory_regularized else None
        ),
        "trajectory_future_coef": (
            float(a.trajectory_future_coef) if a.trajectory_regularized else None
        ),
        "trajectory_max_update_kl": (
            float(a.trajectory_max_update_kl) if a.trajectory_regularized else None
        ),
        "trajectory_max_reference_kl": (
            float(a.trajectory_max_reference_kl) if a.trajectory_regularized else None
        ),
        "trajectory_actor_learning_rate": (
            float(a.trajectory_actor_learning_rate) if a.trajectory_regularized else None
        ),
        "trajectory_critic_learning_rate": (
            float(a.trajectory_critic_learning_rate) if a.trajectory_regularized else None
        ),
        "num_envs": int(raw.num_envs),
        "observation_width": int(raw.cfg.observation_space),
        "action_width": int(raw.cfg.action_space),
        "c4_authority": raw.c4_authority_report(),
        "truck_speed_kmh": list(built_speed_kmh),
        "vehicle_speed_sampling": str(
            getattr(raw.cfg, "vehicle_speed_sampling", "uniform")
        ),
        "speed_segment_duration_s": float(
            getattr(raw.cfg, "vehicle_training_segment_duration_s", 0.0)
        ),
        "trajectory_shape": str(raw.cfg.trajectory_shape),
        "obs_normalizer": str(a.obs_normalizer),
        "path_families": sorted(set(raw._tracking_path_family_names)),
        "terrain_base_seed": int(raw._mow_strip_cfg().seed),
        "terrain_height_scales": sorted(
            set(float(value) for value in raw._terrain_height_scales)
        ),
        "learning_rate": float(runner_cfg.algorithm.learning_rate),
        "entropy_coef": float(runner_cfg.algorithm.entropy_coef),
        "gamma": float(runner_cfg.algorithm.gamma),
        "lam": float(runner_cfg.algorithm.lam),
        "desired_kl": float(runner_cfg.algorithm.desired_kl),
        "schedule": str(runner_cfg.algorithm.schedule),
        "clip_param": float(runner_cfg.algorithm.clip_param),
        "num_learning_epochs": int(runner_cfg.algorithm.num_learning_epochs),
        "std_clamp": list(std_clamp_bounds) if std_clamp_bounds else None,
        "fixed_std": float(a.fixed_std) if a.fixed_std is not None else None,
        "freeze_actor_iterations": int(a.freeze_actor_iterations),
        "seed": int(a.seed),
        "stages": list(stages),
    }
    (log_dir / "run_manifest.json").write_text(
        json.dumps(startup_manifest, indent=2, sort_keys=True)
    )
    print("[PPO3-1-RUN] " + json.dumps(startup_manifest, sort_keys=True), flush=True)

    if resume_path is not None:
        if teacher_handoff:
            load_cfg = {
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }
        elif a.reset_optimizer_on_resume:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            }
        else:
            load_cfg = None
        runner.load(str(resume_path), load_cfg=load_cfg)
        if teacher_handoff or a.reset_critic_head_on_resume:
            critic_output = runner.alg.critic.mlp[-1]
            if not isinstance(critic_output, torch.nn.Linear):
                raise RuntimeError("terrain critic output is not Linear")
            with torch.no_grad():
                critic_output.weight.zero_()
                critic_output.bias.zero_()
        if a.resume_std is not None:
            distribution = runner.alg.actor.distribution
            with torch.no_grad():
                distribution.std_param[:] = float(a.resume_std)
            print(
                f"[PPO3-1-VERGE] set exploration std to {float(a.resume_std)} "
                "on all actions after resume",
                flush=True,
            )
        if a.resume_c4_std is not None:
            distribution = runner.alg.actor.distribution
            with torch.no_grad():
                distribution.std_param[3] = float(a.resume_c4_std)
            print(
                f"[PPO3-1-VERGE] set c4 exploration std to {float(a.resume_c4_std)} "
                "after resume",
                flush=True,
            )
        if a.learning_rate is None:
            resumed_lr = float(runner.alg.optimizer.param_groups[0]["lr"])
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = resumed_lr
            runner.alg.learning_rate = resumed_lr
        else:
            runner.alg.learning_rate = float(a.learning_rate)
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = float(a.learning_rate)
        if a.trajectory_regularized:
            configure_trajectory_optimizer()
            if a.reset_actor_optimizer_on_resume:
                for parameter in runner.alg.actor.parameters():
                    runner.alg.optimizer.state.pop(parameter, None)
                print(
                    "[PPO3-1-OPTIMIZER] reset actor moments at new reference block; "
                    "critic moments retained",
                    flush=True,
                )
            print(
                "[PPO3-1-OPTIMIZER] fixed split rates: "
                f"actor={a.trajectory_actor_learning_rate:g} "
                f"critic={a.trajectory_critic_learning_rate:g}; loaded moments retained",
                flush=True,
            )
        print(f"[PPO3-1-SOLE] resumed from {a.resume}", flush=True)
        if teacher_handoff:
            print(
                "[PPO3-1-TEACHER-HANDOFF] loaded actor+normalizer only; "
                "teacher is off, critic and PPO optimizer are new",
                flush=True,
            )
        if a.demo_regularized:
            print(
                "[PPO3-1-DEMO] label-only DLS teacher enabled for filtered "
                "training targets; PPO remains the sole plant controller",
                flush=True,
            )
    else:
        bias = torch.as_tensor(
            tuple(getattr(raw.cfg, "terrain_initial_action_bias", (0.0, 0.0, 0.0, 0.0))),
            dtype=torch.float32,
            device=a.device,
        )
        output_layer = runner.alg.get_policy().mlp[-1]
        if not isinstance(output_layer, torch.nn.Linear) or output_layer.bias is None:
            raise RuntimeError("terrain actor output layer is not a biased Linear layer")
        with torch.no_grad():
            output_layer.bias.copy_(bias)
        print(f"[PPO3-1-VERGE] initialized actor action bias={bias.tolist()}", flush=True)

    if a.fixed_std is not None:
        distribution = runner.alg.actor.distribution
        if not hasattr(distribution, "std_param"):
            raise RuntimeError(
                f"--fixed_std requires scalar Gaussian std_param, got {type(distribution).__name__}"
            )
        with torch.no_grad():
            distribution.std_param.fill_(float(a.fixed_std))
        distribution.std_param.requires_grad_(False)
        print(
            f"[PPO3-1-STD] fixed every action std at {float(a.fixed_std):g}; "
            "std is not trainable",
            flush=True,
        )

    trajectory_reference_actor = None
    if a.trajectory_regularized:
        from verge_task.trajectory_regularized import (
            ReferenceTrajectoryPPOCfg,
            install_reference_trajectory_ppo,
        )

        trajectory_reference_actor = copy.deepcopy(runner.alg.actor).to(a.device)
        trajectory_reference_actor.eval()
        for parameter in trajectory_reference_actor.parameters():
            parameter.requires_grad_(False)
        install_reference_trajectory_ppo(
            runner.alg,
            trajectory_reference_actor,
            ReferenceTrajectoryPPOCfg(
                local_coef=float(a.trajectory_local_coef),
                future_coef=float(a.trajectory_future_coef),
                max_update_kl=float(a.trajectory_max_update_kl),
                max_reference_kl=float(a.trajectory_max_reference_kl),
            ),
        )
        print(
            "[PPO3-1-TRAJECTORY] single-step reference PPO installed: "
            f"local={a.trajectory_local_coef:g} future={a.trajectory_future_coef:g} "
            f"update_kl<={a.trajectory_max_update_kl:g} "
            f"reference_kl<={a.trajectory_max_reference_kl:g}; "
            "DLS off, PPO sole controller",
            flush=True,
        )

    demo_prior_actor = None
    demo_teacher_steps: list[torch.Tensor] = []
    demo_weight_steps: list[torch.Tensor] = []
    if a.demo_regularized:
        from verge_task.demo_regularized import (
            demo_loss,
            gaussian_kl,
            state_filter,
        )

        demo_prior_actor = copy.deepcopy(runner.alg.actor).to(a.device)
        demo_prior_actor.eval()
        for parameter in demo_prior_actor.parameters():
            parameter.requires_grad_(False)

        original_process_env_step = runner.alg.process_env_step

        def process_env_step_with_demo(obs, rewards, dones, extras):
            demo_teacher_steps.append(raw.terrain_teacher_raw_action.detach().clone())
            demo_weight_steps.append(
                state_filter(
                    lateral_error_m=raw.verge_lateral_error().detach(),
                    ground_contact=raw.cutter_ground_contact().detach(),
                    lateral_threshold_m=float(a.demo_lateral_threshold),
                    require_contact=True,
                )
            )
            return original_process_env_step(obs, rewards, dones, extras)

        runner.alg.process_env_step = process_env_step_with_demo
        print(
            "[PPO3-1-DEMO] collecting policy-visited teacher labels with "
            "state/contact filtering; teacher execution remains disabled",
            flush=True,
        )

    undiagnosed_alg_update = runner.alg.update

    def update_with_diagnostics(*args, **kwargs):
        stats = None
        try:
            storage = runner.alg.storage
            with torch.no_grad():
                returns = storage.returns.flatten().float()
                values = storage.values.flatten().float()
                return_var = returns.var()
                if float(return_var) > 1.0e-8:
                    stats = {
                        "Diagnostics/explained_variance": float(
                            1.0 - (returns - values).var() / return_var
                        ),
                        "Diagnostics/return_mean": float(returns.mean()),
                        "Diagnostics/return_std": float(return_var.sqrt()),
                    }
        except Exception as exc:  # noqa: BLE001 - never fail the training step
            print(f"[PPO3-1-DIAG] skipped: {exc}", flush=True)

        if stats is not None and not update_with_diagnostics.bias_initialized:
            update_with_diagnostics.bias_initialized = True
            try:
                head = runner.alg.critic.mlp[-1]
                if isinstance(head, torch.nn.Linear) and head.bias is not None:
                    before = float(head.bias.detach().mean())
                    if abs(before) < 1.0e-6:  # only when it really was zeroed
                        gamma = float(runner.alg.gamma)
                        mean_reward = float(runner.alg.storage.rewards.mean())
                        target = mean_reward / max(1.0 - gamma, 1.0e-6)
                        with torch.no_grad():
                            head.bias.fill_(target)
                        print(
                            f"[PPO3-1-WARMSTART] critic output bias {before:.4f} "
                            f"-> {target:.1f}  (mean reward {mean_reward:.3f} / "
                            f"(1-gamma={gamma:g})); the stored bootstrapped "
                            f"return was only "
                            f"{stats['Diagnostics/return_mean']:.1f}",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001 - never fail the run
                print(f"[PPO3-1-WARMSTART] skipped: {exc}", flush=True)

        result = undiagnosed_alg_update(*args, **kwargs)

        writer = getattr(getattr(runner, "logger", None), "writer", None) or getattr(
            runner, "writer", None
        )
        if stats and writer is not None:
            try:
                step = int(runner.current_learning_iteration)
                for tag, value in stats.items():
                    writer.add_scalar(tag, value, step)
            except Exception as exc:  # noqa: BLE001
                print(f"[PPO3-1-DIAG] write skipped: {exc}", flush=True)
        elif stats and not update_with_diagnostics.warned:
            update_with_diagnostics.warned = True
            print(
                "[PPO3-1-DIAG] WARNING: no TensorBoard writer found; "
                "explained_variance will NOT be recorded for this run",
                flush=True,
            )
        return result

    update_with_diagnostics.warned = False
    update_with_diagnostics.bias_initialized = False

    runner.alg.update = update_with_diagnostics
    print(
        "[PPO3-1-VERGE] logging Diagnostics/explained_variance -- the scale-free "
        "critic health metric (raw value loss scales with return magnitude)",
        flush=True,
    )

    if std_clamp_bounds is not None:
        clamp_distribution = runner.alg.actor.distribution
        if not hasattr(clamp_distribution, "std_param"):
            raise RuntimeError(
                "--std_clamp requires a scalar-std Gaussian actor "
                f"(got {type(clamp_distribution).__name__})"
            )
        original_alg_update = runner.alg.update

        def update_with_std_clamp(*args, **kwargs):
            result = original_alg_update(*args, **kwargs)
            with torch.no_grad():
                clamp_distribution.std_param.clamp_(*std_clamp_bounds)
            return result

        runner.alg.update = update_with_std_clamp
        print(
            f"[PPO3-1-VERGE] exploration std clamped to {list(std_clamp_bounds)} "
            "after every PPO update",
            flush=True,
        )

    if a.demo_regularized:
        original_alg_update_with_demo = runner.alg.update

        def update_with_demo_regularization(*args, **kwargs):
            storage = runner.alg.storage
            rollout_observations = storage.observations.clone()
            teacher_steps = list(demo_teacher_steps)
            weight_steps = list(demo_weight_steps)
            demo_teacher_steps.clear()
            demo_weight_steps.clear()

            result = original_alg_update_with_demo(*args, **kwargs)
            if (
                demo_prior_actor is None
                or not teacher_steps
                or len(teacher_steps) != int(storage.num_transitions_per_env)
                or not any(parameter.requires_grad for parameter in runner.alg.actor.parameters())
            ):
                return result

            teacher_actions = torch.stack(teacher_steps, dim=0).flatten(0, 1)
            weights = torch.stack(weight_steps, dim=0).flatten(0, 1)
            observations = rollout_observations.flatten(0, 1)

            runner.alg.actor(observations, stochastic_output=True)
            policy_mean = runner.alg.actor.distribution.mean
            with torch.no_grad():
                demo_prior_actor(observations, stochastic_output=True)
                prior_mean = demo_prior_actor.distribution.mean
                prior_std = demo_prior_actor.distribution.std
            current_std = runner.alg.actor.distribution.std
            filtered_demo = demo_loss(
                policy_mean=policy_mean,
                teacher_action=teacher_actions,
                weights=weights,
            )
            prior_kl = gaussian_kl(
                mean_q=policy_mean,
                std_q=current_std,
                mean_p=prior_mean,
                std_p=prior_std,
            )
            regularizer = (
                float(a.demo_lambda) * filtered_demo
                + float(a.demo_kl_beta) * prior_kl
            )
            runner.alg.optimizer.zero_grad()
            regularizer.backward()
            torch.nn.utils.clip_grad_norm_(
                runner.alg.actor.parameters(), runner.alg.max_grad_norm
            )
            runner.alg.optimizer.step()
            if std_clamp_bounds is not None:
                with torch.no_grad():
                    runner.alg.actor.distribution.std_param.clamp_(*std_clamp_bounds)

            pre_projection_kl = float(prior_kl.detach())
            post_projection_kl = prior_kl.detach()
            projection_alpha = 0.0
            max_demo_kl = float(a.demo_max_kl)
            if max_demo_kl > 0.0 and pre_projection_kl > max_demo_kl:
                current_parameters = {
                    name: parameter.detach().clone()
                    for name, parameter in runner.alg.actor.named_parameters()
                }
                prior_parameters = dict(demo_prior_actor.named_parameters())
                if set(current_parameters) != set(prior_parameters):
                    raise RuntimeError(
                        "frozen prior and live actor parameters do not match "
                        "for the demonstration trust-region projection"
                    )

                def set_blend(alpha: float) -> None:
                    with torch.no_grad():
                        for name, parameter in runner.alg.actor.named_parameters():
                            parameter.copy_(
                                (1.0 - alpha) * current_parameters[name]
                                + alpha * prior_parameters[name]
                            )

                def measure_prior_kl() -> torch.Tensor:
                    with torch.no_grad():
                        runner.alg.actor(observations, stochastic_output=True)
                        current_mean_now = runner.alg.actor.distribution.mean
                        current_std_now = runner.alg.actor.distribution.std
                        demo_prior_actor(observations, stochastic_output=True)
                        prior_mean_now = demo_prior_actor.distribution.mean
                        prior_std_now = demo_prior_actor.distribution.std
                        return gaussian_kl(
                            mean_q=current_mean_now,
                            std_q=current_std_now,
                            mean_p=prior_mean_now,
                            std_p=prior_std_now,
                        ).detach()

                lo, hi = 0.0, 1.0
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    set_blend(mid)
                    if float(measure_prior_kl()) > max_demo_kl:
                        lo = mid
                    else:
                        hi = mid
                set_blend(hi)
                projection_alpha = hi
                post_projection_kl = measure_prior_kl()

            result = dict(result)
            result.update(
                {
                    "demo": float(filtered_demo.detach()),
                    "demo_kl": float(post_projection_kl),
                    "demo_kl_pre_projection": pre_projection_kl,
                    "demo_projection_alpha": projection_alpha,
                    "demo_weight_fraction": float(weights.mean().detach()),
                }
            )
            return result

        runner.alg.update = update_with_demo_regularization
        print(
            f"[PPO3-1-DEMO] regularizer active: lambda={float(a.demo_lambda):g} "
            f"beta={float(a.demo_kl_beta):g} "
            f"max_kl={float(a.demo_max_kl):g} "
            f"lateral_threshold={float(a.demo_lateral_threshold):g}m",
            flush=True,
        )

    original_runner_save = runner.save

    def save_with_terrain_provenance(path: str, infos=None) -> None:
        checkpoint_infos = {
            "source": checkpoint_source,
            "schema": "terrain_shared_v2",
            "actuator": EXPECTED_ACTUATOR,
            "actuator_label": actuator_label,
            "task": TASK_ID,
            "experiment": str(runner_cfg.experiment_name),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iteration": int(runner.current_learning_iteration),
            "num_envs": int(raw.num_envs),
            "observation_width": int(raw.cfg.observation_space),
            "action_width": int(raw.cfg.action_space),
            "head_joint_action": bool(raw.cfg.sole_head_joint_action),
            "c4_authority": raw.c4_authority_report(),
            "initialization_source": resume_source,
            # Empty on a healthy resume; non-empty records that this run had to
            # widen a frozen input statistic the resumed policy shipped with.
            "normalizer_repairs": list(_normalizer_repairs),
            "teacher_enabled_during_ppo": bool(a.demo_regularized),
            "teacher_labels_enabled": bool(a.demo_regularized),
            "teacher_execution_during_ppo": False,
            "demo_regularized": bool(a.demo_regularized),
            "demo_lambda": float(a.demo_lambda) if a.demo_regularized else None,
            "demo_kl_beta": float(a.demo_kl_beta) if a.demo_regularized else None,
            "demo_max_kl": float(a.demo_max_kl) if a.demo_regularized else None,
            "trajectory_regularized": bool(a.trajectory_regularized),
            "trajectory_reference": (
                "frozen_resumed_warm_actor" if a.trajectory_regularized else None
            ),
            "trajectory_local_coef": (
                float(a.trajectory_local_coef) if a.trajectory_regularized else None
            ),
            "trajectory_future_coef": (
                float(a.trajectory_future_coef) if a.trajectory_regularized else None
            ),
            "trajectory_max_update_kl": (
                float(a.trajectory_max_update_kl) if a.trajectory_regularized else None
            ),
            "trajectory_max_reference_kl": (
                float(a.trajectory_max_reference_kl) if a.trajectory_regularized else None
            ),
            "trajectory_actor_learning_rate": (
                float(a.trajectory_actor_learning_rate) if a.trajectory_regularized else None
            ),
            "trajectory_critic_learning_rate": (
                float(a.trajectory_critic_learning_rate) if a.trajectory_regularized else None
            ),
            "gamma": float(runner.alg.gamma),
            "lam": float(runner.alg.lam),
            "schedule": str(runner.alg.schedule),
            "clip_param": float(runner.alg.clip_param),
            "fixed_std": float(a.fixed_std) if a.fixed_std is not None else None,
            "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
            "terrain_base_seed": int(raw._mow_strip_cfg().seed),
            "path_families": sorted(set(raw._tracking_path_family_names)),
            "terrain_height_scales": sorted(
                set(float(value) for value in raw._terrain_height_scales)
            ),
        }
        if infos:
            checkpoint_infos.update(infos)
        original_runner_save(path, infos=checkpoint_infos)

    runner.save = save_with_terrain_provenance

    def observation_normalizers() -> list:
        """Every EmpiricalNormalization the actor and critic own."""
        return [
            mod
            for owner in (runner.alg.actor, runner.alg.critic)
            for mod in owner.modules()
            if type(mod).__name__ == "EmpiricalNormalization"
        ]

    _frozen_normalizers = observation_normalizers() if a.obs_normalizer == "freeze" else []
    _normalizer_snapshot = None
    # Only the freeze path can inherit a stale statistic; `adapt` re-estimates.
    _normalizer_repairs: list[str] = []
    if _frozen_normalizers:
        from verge_task.trajectory_regularized import (
            assert_normalizers_unchanged,
            freeze_normalizers_across_train_mode,
            reseed_frozen_speed_normalizer,
        )

        # Repair BEFORE the snapshot: a normalizer resumed from a fixed-speed
        # run encodes that speed as a zero-variance constant, and freezing it
        # over a randomized range multiplies the channel by 100 rather than
        # normalizing it.  Once snapshotted, the drift guard would preserve the
        # damage for the whole run.
        _normalizer_repairs = reseed_frozen_speed_normalizer(
            runner.alg,
            float(raw.cfg.vehicle_speed_min_kmh),
            float(raw.cfg.vehicle_speed_max_kmh),
        )
        for _repair in _normalizer_repairs:
            print(f"[PPO3-1-NORM] reseeded {_repair}", flush=True)
        if not _normalizer_repairs:
            print(
                "[PPO3-1-NORM] resumed normalizer already covers the sampled "
                "speed range; no reseed needed",
                flush=True,
            )

        _normalizer_snapshot = freeze_normalizers_across_train_mode(runner.alg)
        update_before_normalizer_guard = runner.alg.update

        def update_with_frozen_normalizer_guard(*args, **kwargs):
            assert_normalizers_unchanged(
                _normalizer_snapshot, runner.alg.actor, runner.alg.critic
            )
            result = update_before_normalizer_guard(*args, **kwargs)
            assert_normalizers_unchanged(
                _normalizer_snapshot, runner.alg.actor, runner.alg.critic
            )
            counts = [float(module.count) for module in _frozen_normalizers]
            result = dict(result)
            result["normalizer_actor_count"] = counts[0] if counts else 0.0
            result["normalizer_critic_count"] = counts[1] if len(counts) > 1 else 0.0
            result["normalizer_drift"] = 0.0
            return result

        runner.alg.update = update_with_frozen_normalizer_guard
    if _frozen_normalizers:
        print(
            f"[PPO3-1-NORM] froze {len(_frozen_normalizers)} observation "
            "normalizer(s) for the ENTIRE run; the running mean/std are part of "
            "the resumed policy, not statistics to re-estimate",
            flush=True,
        )
    else:
        print(
            "[PPO3-1-NORM] observation normalizer left ADAPTING -- ablation "
            "only; this is the configuration measured to degrade a distilled "
            "actor with no meaningful weight change",
            flush=True,
        )

    def critic_warmup(freeze: int) -> None:
        """Critic warm-up with a frozen actor.

        With the actor's requires_grad=False, loss.backward() leaves its grads
        None and the shared Adam step() skips those parameters entirely, so the
        actor (mean net AND exploration std -- the distribution is an actor
        submodule) cannot move while the critic learns against its rollouts.
        The obs-normalizer buffers are unaffected and keep adapting.

        The adaptive KL scheduler is suspended for the window: the normalizer
        keeps moving between rollout and recompute, so the update KL of a
        frozen actor is tiny but NONZERO, and ``adaptive`` would ratchet the
        shared learning rate x1.5 per minibatch to its 1e-2 cap. The first
        unfrozen updates would then fire at ~30x the configured rate --
        exactly the student-destroying jump this warm-up exists to prevent.
        At unfreeze the schedule is restored and the learning rate reset to
        the configured value.
        """
        schedule = runner.alg.schedule
        base_lr = float(runner.alg.learning_rate)
        runner.alg.schedule = "fixed"
        runner.alg.actor.requires_grad_(False)
        _norm_modules = observation_normalizers()
        for _mod in _norm_modules:
            _mod.eval()  # update() is a no-op unless self.training
        print(
            f"[PPO3-1-WARMUP] froze {len(_norm_modules)} observation "
            "normalizer(s) for the window; a moving normalizer changes a "
            "frozen actor's behaviour",
            flush=True,
        )
        runner.learn(num_learning_iterations=freeze, init_at_random_ep_len=False)
        warmup_checkpoint = log_dir / "warmup_checkpoint.pt"
        warmup_iteration = int(runner.current_learning_iteration)
        resume_iteration = warmup_iteration + 1
        warmup_metadata = {
            "checkpoint_kind": "critic_warmup_complete",
            "warmup_iterations": int(freeze),
            "warmup_actor_frozen": True,
            "warmup_iteration": int(resume_iteration),
            "teacher_enabled_during_ppo": bool(a.demo_regularized),
            "teacher_labels_enabled": bool(a.demo_regularized),
            "teacher_execution_during_ppo": False,
            "demo_regularized": bool(a.demo_regularized),
            "demo_max_kl": float(a.demo_max_kl) if a.demo_regularized else None,
            "trajectory_regularized": bool(a.trajectory_regularized),
            "trajectory_reference": (
                "frozen_resumed_warm_actor" if a.trajectory_regularized else None
            ),
            "trajectory_local_coef": (
                float(a.trajectory_local_coef) if a.trajectory_regularized else None
            ),
            "trajectory_future_coef": (
                float(a.trajectory_future_coef) if a.trajectory_regularized else None
            ),
            "trajectory_max_update_kl": (
                float(a.trajectory_max_update_kl) if a.trajectory_regularized else None
            ),
            "trajectory_max_reference_kl": (
                float(a.trajectory_max_reference_kl) if a.trajectory_regularized else None
            ),
            "trajectory_actor_learning_rate": (
                float(a.trajectory_actor_learning_rate) if a.trajectory_regularized else None
            ),
            "trajectory_critic_learning_rate": (
                float(a.trajectory_critic_learning_rate) if a.trajectory_regularized else None
            ),
            "schedule": str(runner.alg.schedule),
            "clip_param": float(runner.alg.clip_param),
            "fixed_std": float(a.fixed_std) if a.fixed_std is not None else None,
            "gamma": float(runner.alg.gamma),
            "lam": float(runner.alg.lam),
        }
        runner.current_learning_iteration = resume_iteration
        try:
            runner.save(str(warmup_checkpoint), infos=warmup_metadata)
        finally:
            runner.current_learning_iteration = warmup_iteration
        (log_dir / "warmup_checkpoint.json").write_text(
            json.dumps(
                {
                    **warmup_metadata,
                    "checkpoint": str(warmup_checkpoint),
                    "iteration": int(resume_iteration),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"[PPO3-1-WARMUP] saved resumable checkpoint {warmup_checkpoint} "
            f"at resume boundary iteration {resume_iteration}",
            flush=True,
        )
        runner.alg.actor.requires_grad_(True)
        if a.obs_normalizer == "adapt":
            for _mod in _norm_modules:
                _mod.train()
        runner.alg.schedule = schedule
        runner.alg.learning_rate = base_lr
        for param_group in runner.alg.optimizer.param_groups:
            param_group["lr"] = base_lr
        print(
            f"[PPO3-1-WARMUP] actor frozen for {freeze} iterations "
            f"(critic-only updates at fixed lr); actor unfrozen, schedule "
            f"{schedule!r} restored, lr reset to {base_lr}",
            flush=True,
        )

    final_stage = stages[-1]
    last_convergence_result = None
    for stage in stages:
        apply_c4_authority_stage(raw.cfg, stage)
        apply_plant_overrides(raw.cfg)
        configure_path_horizon(raw.cfg)
        raw.cfg.trajectory_shape = str(raw.cfg.terrain_deployment_trajectory_shape)
        raw.cfg.randomize_trajectory_shape = False
        raw.cfg.randomize_start_phase = False
        if float(raw.cfg.vehicle_speed_min_kmh) != requested_speed_kmh[0] or float(
            raw.cfg.vehicle_speed_max_kmh
        ) != requested_speed_kmh[1]:
            raise RuntimeError(
                f"stage {stage} changed the truck speed to "
                f"({raw.cfg.vehicle_speed_min_kmh}, {raw.cfg.vehicle_speed_max_kmh})"
                f" km/h; the run requested {requested_speed_kmh} km/h"
            )
        print(
            f"[PPO3-1-VERGE] === stage {stage} === "
            f"truck={raw.cfg.vehicle_speed_min_kmh:.2f}-"
            f"{raw.cfg.vehicle_speed_max_kmh:.2f} km/h per episode "
            f"c4_lead={raw.cfg.cylinder_target_max_lead_per_axis_m[3]:.3f} m "
            f"c4_scale={raw.cfg.cylinder_action_scale_per_axis_m[3]:.4f} m/unit "
            f"filter={raw.cfg.sole_action_filter:.2f} "
            + (
                f"until convergence (checks every {a.convergence_check_interval} iterations)"
                if a.until_converged and stage == final_stage
                else f"for {a.stage_iterations} iterations"
            ),
            flush=True,
        )
        warmed_up = 0
        if stage == stages[0] and a.freeze_actor_iterations > 0:
            warmed_up = int(a.freeze_actor_iterations)
            critic_warmup(warmed_up)
        if not (a.until_converged and stage == final_stage):
            remaining_total = int(a.stage_iterations) - warmed_up
            if remaining_total > 0:
                runner.learn(
                    num_learning_iterations=remaining_total,
                    init_at_random_ep_len=False,
                )
            continue

        while True:
            runner.learn(
                num_learning_iterations=int(a.convergence_check_interval),
                init_at_random_ep_len=False,
            )
            policy = runner.get_inference_policy(device=a.device)
            result = convergence_rollout(raw, policy, int(a.convergence_steps))
            last_convergence_result = result
            runner.alg.train_mode()
            result.update(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "iteration": int(runner.current_learning_iteration),
                    "num_envs": int(raw.num_envs),
                    "stage": int(stage),
                }
            )
            convergence_report = log_dir / "convergence_latest.json"
            convergence_report.write_text(json.dumps(result, indent=2))
            runner.save(
                str(log_dir / f"model_{runner.current_learning_iteration}.pt"),
                infos={"convergence": result},
            )
            print(
                "[PPO3-1-CONVERGENCE] " + json.dumps(result, sort_keys=True),
                flush=True,
            )
            if result["passed"]:
                converged_path = log_dir / "model_converged.pt"
                runner.save(
                    str(converged_path),
                    infos={"convergence": result},
                )
                print(
                    f"[PPO3-1-CONVERGENCE] PASSED; wrote {converged_path}",
                    flush=True,
                )
                break

    if a.checkpoint_out:
        checkpoint_out = Path(a.checkpoint_out).expanduser().resolve()
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_info = {
            "source": checkpoint_source,
            "schema": "terrain_shared_v2",
            "actuator": EXPECTED_ACTUATOR,
            "actuator_label": actuator_label,
            "task": TASK_ID,
            "experiment": str(runner_cfg.experiment_name),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iteration": int(runner.current_learning_iteration),
            "num_envs": int(raw.num_envs),
            "stage": int(final_stage),
            "seed": int(a.seed),
            "learning_rate": float(runner.alg.optimizer.param_groups[0]["lr"]),
            "convergence": last_convergence_result,
            "observation_width": int(raw.cfg.observation_space),
            "action_width": int(raw.cfg.action_space),
            "head_joint_action": bool(raw.cfg.sole_head_joint_action),
            "c4_authority": raw.c4_authority_report(),
            "initialization_source": resume_source,
            # Empty on a healthy resume; non-empty records that this run had to
            # widen a frozen input statistic the resumed policy shipped with.
            "normalizer_repairs": list(_normalizer_repairs),
            "teacher_enabled_during_ppo": bool(a.demo_regularized),
            "teacher_labels_enabled": bool(a.demo_regularized),
            "teacher_execution_during_ppo": False,
            "demo_regularized": bool(a.demo_regularized),
            "demo_max_kl": float(a.demo_max_kl) if a.demo_regularized else None,
            "trajectory_regularized": bool(a.trajectory_regularized),
            "trajectory_reference": (
                "frozen_resumed_warm_actor" if a.trajectory_regularized else None
            ),
            "trajectory_local_coef": (
                float(a.trajectory_local_coef) if a.trajectory_regularized else None
            ),
            "trajectory_future_coef": (
                float(a.trajectory_future_coef) if a.trajectory_regularized else None
            ),
            "trajectory_max_update_kl": (
                float(a.trajectory_max_update_kl) if a.trajectory_regularized else None
            ),
            "trajectory_max_reference_kl": (
                float(a.trajectory_max_reference_kl) if a.trajectory_regularized else None
            ),
            "trajectory_actor_learning_rate": (
                float(a.trajectory_actor_learning_rate) if a.trajectory_regularized else None
            ),
            "trajectory_critic_learning_rate": (
                float(a.trajectory_critic_learning_rate) if a.trajectory_regularized else None
            ),
            "schedule": str(runner.alg.schedule),
            "clip_param": float(runner.alg.clip_param),
            "fixed_std": float(a.fixed_std) if a.fixed_std is not None else None,
            "gamma": float(runner.alg.gamma),
            "lam": float(runner.alg.lam),
            "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
            "terrain_base_seed": int(raw._mow_strip_cfg().seed),
            "path_families": sorted(set(raw._tracking_path_family_names)),
            "terrain_height_scales": sorted(
                set(float(value) for value in raw._terrain_height_scales)
            ),
        }
        runner.save(str(checkpoint_out), infos=checkpoint_info)
        manifest = checkpoint_out.with_suffix(".json")
        manifest_row = {"checkpoint": str(checkpoint_out), **checkpoint_info}
        manifest.write_text(json.dumps(manifest_row, indent=2))
        print(
            "[PPO3-1-CHECKPOINT] " + json.dumps(manifest_row, sort_keys=True),
            flush=True,
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
