#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Train the PPO3 verge-following sole controller over ROM-A.

Production training starts from the terrain teacher-student initializer and
walks the authority curriculum in ``env_cfg.apply_sole_curriculum_stage``. A
random start is available only as an explicitly labelled ablation. A stage advances only
after ``--stage_iterations`` PPO iterations; there is deliberately no automatic
promotion gate here, because a sole controller that has not yet learned to move
produces meaningless gate statistics.  Use ``scripts/evaluate.py`` between
stages for that.
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
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument(
    "--stages",
    default="1,2,3,4",
    help=(
        "Curriculum stages, in order. Stage 5 is stage 4's arm contract on the "
        "rough-lawn verge and is opt-in, so it is not in the default schedule."
    ),
)
p.add_argument("--stage_iterations", type=int, default=500)
p.add_argument(
    "--until_converged",
    action="store_true",
    help="Repeat the final curriculum stage until all terrain gates pass.",
)
p.add_argument("--convergence_check_interval", type=int, default=50)
p.add_argument("--convergence_steps", type=int, default=300)
p.add_argument("--save_interval", type=int, default=50)
p.add_argument("--learning_rate", type=float, default=None)
p.add_argument(
    "--fixed_learning_rate",
    action="store_true",
    help="Disable adaptive-KL learning-rate changes (useful for convergence fine-tuning).",
)
p.add_argument("--init_std", type=float, default=None)
p.add_argument("--entropy_coef", type=float, default=None)
p.add_argument("--vehicle_speed_kmh", type=float, default=4.0)
p.add_argument(
    "--head_joint_action",
    action="store_true",
    default=None,
    help="Use a fifth policy output as the hydraulic head-tilt residual.",
)
p.add_argument(
    "--legacy_four_action",
    dest="head_joint_action",
    action="store_false",
    help="Compatibility only: run the retired 54x4 PPO3 interface.",
)
p.add_argument(
    "--head_action_to_c4_gain",
    type=float,
    default=None,
    help="Override the q5 residual gain mixed into cylinder-4 action.",
)
p.add_argument(
    "--drive_substep_interpolation",
    default="",
    choices=("", "true", "false"),
    help=(
        "Override terrain_kinematic_drive_substep_interpolation. Use true for "
        "new runs trained against the corrected per-substep terrain dynamics."
    ),
)
p.add_argument("--trajectory_library_path", default="")
p.add_argument("--command_rom_checkpoint", default="")
p.add_argument(
    "--tail_focus",
    action="store_true",
    help=(
        "Training-only oversampling of measured head-tail path families and "
        "relief scales. Held-out evaluation is unchanged."
    ),
)
p.add_argument(
    "--head_tail_risk",
    action="store_true",
    help="Enable worst-2-percent head weighting and hard-gate failure memory.",
)
# --- contact-safe experiment (fix/ppo3-ppo4-zero-contact-loss) -------------
p.add_argument(
    "--contact_focus",
    action="store_true",
    help=(
        "Training-only oversampling of the MEASURED contact-loss region: "
        "s_curve, full relief, late path phases. Held-out evaluation is "
        "unchanged. See reports/CONTACT_SAFE_V1_REPORT.md."
    ),
)
p.add_argument(
    "--contact_stage",
    default="",
    choices=("", "A", "B", "C"),
    help=(
        "Contact-loss grace curriculum: A=4 steps, B=2, C=1. Strict "
        "evaluation tolerates none. Empty keeps the config default."
    ),
)
p.add_argument(
    "--contact_observation",
    default="",
    choices=("", "true", "false"),
    help="Override terrain_contact_observation (changes the observation width).",
)
p.add_argument(
    "--contact_safety_reward",
    default="",
    choices=("", "true", "false"),
    help="Override terrain_contact_safety_reward (the seven-term contact group).",
)
p.add_argument(
    "--contact_termination",
    default="",
    choices=("", "true", "false"),
    help="Override terrain_terminate_on_contact_loss.",
)
p.add_argument("--contact_loss_weight", type=float, default=None)
p.add_argument("--contact_gap_weight", type=float, default=None)
p.add_argument("--contact_force_target_n", type=float, default=None)
p.add_argument("--contact_force_soft_limit_n", type=float, default=None)
p.add_argument("--lift_terminal_reward", type=float, default=None)
p.add_argument("--model_error_history", type=int, default=None)
# --- Phase 6 impact-energy sweep knobs ------------------------------------
p.add_argument("--action_scale", type=float, default=None)
p.add_argument("--max_action_rate", type=float, default=None)
p.add_argument("--arm_stiffness_scale", type=float, default=None)
p.add_argument(
    "--cylinder_target_max_lead_m",
    type=float,
    default=None,
    help=(
        "Anti-windup bound on how far the commanded PhysX target may lead the "
        "measured cylinder. Shared by ROM-A and ID/FD -- both read it in "
        "mulag_upper_level_env._run_cylinder_id_pipeline. Measured saturated "
        "at the inherited 25 mm on BOTH actuators before the fix."
    ),
)
p.add_argument("--cylinder_target_leak", type=float, default=None)
p.add_argument(
    "--height_schedule",
    default="",
    help=(
        "Comma-separated training relief scales, e.g. '0.50,0.70'. Empty keeps "
        "the wide default (0.35..1.15). The held-out gate tuple is never "
        "changed by this."
    ),
)
p.add_argument(
    "--sole_ik_hint",
    action="store_true",
    help=(
        "ABLATION ONLY. Put the cascade's cylinder error into the observation "
        "to measure what the Jacobian is worth. Never report as a "
        "sole-controller result."
    ),
)
p.add_argument("--sole_limit_shield_margin_m", type=float, default=None)
p.add_argument("--resume", default=None, help="Checkpoint to continue from.")
p.add_argument(
    "--from_scratch",
    action="store_true",
    help=(
        "Explicit ablation: initialize a random terrain actor. Production "
        "teacher-student PPO requires --resume from distill_terrain.py."
    ),
)
p.add_argument(
    "--reset_optimizer_on_resume",
    action="store_true",
    help="Load actor/critic weights but discard stale PPO optimizer moments.",
)
p.add_argument(
    "--reset_critic_head_on_resume",
    action="store_true",
    help="Zero the resumed critic output head after loading changed reward semantics.",
)
p.add_argument(
    "--resume_c4_std",
    type=float,
    default=None,
    help="Optional exploration std for the head-pitch action after resume.",
)
p.add_argument(
    "--resume_q5_std",
    type=float,
    default=None,
    help="Optional exploration std for the fifth hydraulic head-residual action.",
)
p.add_argument(
    "--resume_std",
    type=float,
    default=None,
    help=(
        "Optional exploration std for ALL four actions after resume. "
        "--init_std cannot do this on a resume: it sets the distribution at "
        "construction and the checkpoint overwrites it on load."
    ),
)
p.add_argument(
    "--checkpoint_out",
    default="",
    help=(
        "Write the final actor/critic/optimizer state to this exact path and a "
        "JSON manifest beside it. Intended for checkpointed training blocks."
    ),
)
p.add_argument(
    "--log_dir",
    default="",
    help="Run-specific TensorBoard/checkpoint directory (default: timestamped).",
)
p.add_argument(
    "--zero_action_probe",
    action="store_true",
    help="Step 100 zero actions before training and report plant motion.",
)
p.add_argument("--probe_only", action="store_true")
p.add_argument(
    "--actuator",
    default="rom",
    choices=("rom", "idfd"),
    help=(
        "Which actuator converts the cylinder command into motion. "
        "rom = PPO3 command ROM; idfd = PPO4 ID->FD. Everything else about "
        "training is identical, which is the point."
    ),
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

stages = tuple(int(part) for part in a.stages.split(",") if part.strip())
if not stages or any(stage not in (1, 2, 3, 4, 5) for stage in stages):
    p.error("--stages must be a comma-separated subset of 1,2,3,4,5")
if a.stage_iterations <= 0 or a.save_interval <= 0:
    p.error("stage iterations and save interval must be positive")
if a.learning_rate is not None and a.learning_rate <= 0.0:
    p.error("--learning_rate must be positive")
if a.resume_std is not None and a.resume_std <= 0.0:
    p.error("--resume_std must be positive")
if a.init_std is not None and a.init_std <= 0.0:
    p.error("--init_std must be positive")
if a.resume_c4_std is not None and a.resume_c4_std <= 0.0:
    p.error("--resume_c4_std must be positive")
if a.resume_q5_std is not None and a.resume_q5_std <= 0.0:
    p.error("--resume_q5_std must be positive")
if a.resume_q5_std is not None and a.head_joint_action is False:
    p.error("--resume_q5_std requires --head_joint_action")
if a.convergence_check_interval <= 0 or a.convergence_steps <= 0:
    p.error("convergence interval and steps must be positive")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO3_SOLE_PhysX  # noqa: F401,E402  (registers the gym task)
from PPO3_SOLE_PhysX.distill import assert_teacher_disabled  # noqa: E402
from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    TerrainSolePolicyPPORunnerCfg,  # noqa: F401  (default actuator)
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from PPO3_SOLE_PhysX.actuators import resolve_actuator  # noqa: E402
from PPO3_SOLE_PhysX.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)

# One task, two actuators. Resolved once here so every downstream use --
# env construction, runner config, checkpoint tagging -- cannot disagree.
TASK_ID, RUNNER_CFG_CLS, ACTUATOR_LABEL = resolve_actuator(a.actuator)
print(f"[PPO-TERRAIN] actuator={a.actuator} ({ACTUATOR_LABEL}) task={TASK_ID}", flush=True)
from PPO3_SOLE_PhysX.terrain_convergence import summarize_convergence  # noqa: E402


def apply_plant_overrides(cfg) -> None:
    """Re-assert the CLI plant/command overrides on top of a curriculum stage.

    ``apply_terrain_curriculum_stage`` -> ``apply_sole_curriculum_stage`` writes
    ``action_scale`` and ``sole_max_action_rate`` for the stage it is given, and
    it runs both at startup and again on every stage transition. An override
    applied only once, before that, is silently reverted the moment the
    curriculum is applied -- so this is called after EVERY such call rather
    than once during construction.
    """
    if a.action_scale is not None:
        override_cfg(cfg, "action_scale", float(a.action_scale))
    if a.max_action_rate is not None:
        override_cfg(cfg, "sole_max_action_rate", float(a.max_action_rate))
    if a.arm_stiffness_scale is not None:
        override_cfg(cfg, "terrain_arm_stiffness_scale", float(a.arm_stiffness_scale))
    if a.cylinder_target_max_lead_m is not None:
        override_cfg(
            cfg, "cylinder_target_max_lead_m", float(a.cylinder_target_max_lead_m)
        )
    if a.cylinder_target_leak is not None:
        override_cfg(cfg, "cylinder_target_leak", float(a.cylinder_target_leak))


@torch.no_grad()
def zero_action_probe(raw, steps: int = 100) -> dict[str, float]:
    """Measure what a zero command does to the plant.

    PPO2 could assert that a zero action reproduces the teacher bit-exactly.
    PPO3 has no teacher, so the meaningful invariant is the opposite one: a zero
    command must leave the arm essentially where it is. Large motion here means
    the target integrator is drifting under gravity and the policy would be
    learning to fight a bias rather than to track.
    """
    raw.reset(seed=int(a.seed))
    start = raw._compute_ee_pos_local().detach().clone()
    zero = torch.zeros(
        (raw.num_envs, int(raw.cfg.action_space)), device=raw.device
    )
    max_drift = 0.0
    for _ in range(steps):
        raw.step(zero)
        drift = float(
            torch.linalg.norm(raw._compute_ee_pos_local() - start, dim=-1).max()
        )
        max_drift = max(max_drift, drift)
    return {
        "steps": float(steps),
        "max_ee_drift_m": max_drift,
        "scan_valid_fraction": float(raw.scan_valid_fraction().mean()),
        "mean_abs_cut_height_error_m": float(raw.cut_height_error().abs().mean()),
        "max_abs_applied_action": float(raw._sole_applied.abs().max()),
        "max_abs_cylinder_lead_m": float(
            raw._commanded_target_lead(raw._refresh_cylinder_state()[0]).abs().max()
        ),
    }


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
    result["terrain"] = "training_validation"
    result["path_family_results"] = family_rows
    return result


def main() -> None:
    resume_path = Path(a.resume).resolve() if a.resume else None
    if resume_path is None and not a.from_scratch:
        raise RuntimeError(
            "Production terrain PPO requires a teacher-student initializer. Run "
            "scripts/distill_terrain.py first and pass its checkpoint with "
            "--resume. Use --from_scratch only for an explicitly labelled ablation."
        )
    if resume_path is not None and a.from_scratch:
        raise ValueError("--resume and --from_scratch are mutually exclusive")
    resume_payload = None
    resume_source = "random ablation"
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        infos = resume_payload.get("infos") or {}
        resume_source = str(infos.get("source", ""))
        from PPO3_SOLE_PhysX.actuators import assert_checkpoint_actuator
        assert_checkpoint_actuator(infos, a.actuator, context="PPO3/PPO4 resume")
        schema = str(infos.get("schema", ""))
        if schema not in {"terrain_teacher_student_v1", "terrain_shared_v2"} or resume_source not in {
            "PPO3 terrain teacher distillation v1",
            "PPO3 terrain PPO",
        }:
            raise RuntimeError(
                "Checkpoint is outside the supported terrain policy contracts "
                f"(source={resume_source!r}, schema={schema!r}). All pre-correction "
                "terrain PPO and legacy stored-path distillation checkpoints are "
                "intentionally invalid."
            )

    env_cfg = parse_env_cfg(
        TASK_ID, device=a.device, num_envs=a.num_envs
    )
    if a.head_joint_action is not None:
        env_cfg.sole_head_joint_action = bool(a.head_joint_action)
    env_cfg.action_space = 5 if env_cfg.sole_head_joint_action else 4
    if env_cfg.sole_head_joint_action:
        env_cfg.terrain_messerkopf_level_gain = 0.0
        env_cfg.terrain_messerkopf_level_quadratic_gain = 0.0
    if a.head_action_to_c4_gain is not None:
        if a.head_action_to_c4_gain <= 0.0:
            raise ValueError("--head_action_to_c4_gain must be positive")
        env_cfg.sole_head_action_to_c4_gain = float(a.head_action_to_c4_gain)
    if a.sole_ik_hint:
        raise ValueError("--sole_ik_hint is not part of the terrain observation contract")
    env_cfg.sole_ik_hint = False
    # --- contact-safe overrides, applied BEFORE the observation width is
    # derived so a widened observation and the network built for it cannot
    # disagree. ---
    if a.contact_observation:
        env_cfg.terrain_contact_observation = a.contact_observation == "true"
    if a.contact_safety_reward:
        env_cfg.terrain_contact_safety_reward = a.contact_safety_reward == "true"
    if a.contact_termination:
        env_cfg.terrain_terminate_on_contact_loss = a.contact_termination == "true"
        env_cfg.terrain_terminate_on_hard_impact = a.contact_termination == "true"
    if a.contact_stage:
        # Stage A tolerates a burst of missed samples so early exploration is
        # not deleted by sensor noise; C is one step from strict.
        env_cfg.terrain_contact_loss_grace_steps = {
            "A": 4, "B": 2, "C": 1
        }[a.contact_stage]
    if a.contact_loss_weight is not None:
        env_cfg.terrain_contact_loss_weight = float(a.contact_loss_weight)
    if a.contact_gap_weight is not None:
        env_cfg.terrain_contact_gap_weight = float(a.contact_gap_weight)
    if a.contact_force_target_n is not None:
        env_cfg.terrain_contact_force_target_n = float(a.contact_force_target_n)
    if a.contact_force_soft_limit_n is not None:
        env_cfg.terrain_contact_force_soft_limit_n = float(
            a.contact_force_soft_limit_n
        )
    if a.lift_terminal_reward is not None:
        env_cfg.terrain_lift_terminal_reward = float(a.lift_terminal_reward)
    if a.model_error_history is not None:
        if a.model_error_history < 1:
            raise ValueError("--model_error_history must be >= 1")
        override_cfg(env_cfg, "terrain_model_error_history", int(a.model_error_history))
    apply_plant_overrides(env_cfg)
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.enable_contact_tracking_demo = False
    resume_load_path = resume_path
    if resume_payload is not None:
        resume_infos = resume_payload.get("infos") or {}
        checkpoint_width = int(resume_infos.get("observation_width", -1))
        if checkpoint_width != int(env_cfg.observation_space):
            raise RuntimeError(
                f"checkpoint observation width {checkpoint_width} does not match "
                f"the frozen terrain contract {env_cfg.observation_space}; automatic "
                "migration is disabled for production training"
            )
        checkpoint_action_width = int(resume_infos.get("action_width", 4))
        if checkpoint_action_width != int(env_cfg.action_space):
            raise RuntimeError(
                f"checkpoint action width {checkpoint_action_width} does not match "
                f"environment width {env_cfg.action_space}; q4/q5 checkpoints are "
                "not shape-compatible"
            )
        if bool(resume_infos.get("migration_requires_optimizer_reset", False)) and not bool(
            a.reset_optimizer_on_resume
        ):
            raise RuntimeError(
                "migrated shared-policy checkpoints require "
                "--reset_optimizer_on_resume"
            )
    env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_kmh)
    env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_kmh)
    if a.drive_substep_interpolation:
        env_cfg.terrain_kinematic_drive_substep_interpolation = (
            a.drive_substep_interpolation == "true"
        )
    env_cfg.terrain_base_seed = int(a.seed)
    if a.trajectory_library_path:
        env_cfg.trajectory_library_path = str(Path(a.trajectory_library_path).resolve())
    if a.command_rom_checkpoint:
        env_cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).resolve())
    if a.sole_limit_shield_margin_m is not None:
        env_cfg.sole_limit_shield_margin_m = float(a.sole_limit_shield_margin_m)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))

    runner_cfg = RUNNER_CFG_CLS()
    # RSL-RL seeds policy sampling from the runner config, not from raw.reset().
    # The block supervisor varies this seed so separate resumed processes do not
    # replay the same stochastic update sequence from checkpoints that omit RNG
    # state.
    runner_cfg.seed = int(a.seed)
    runner_cfg.max_iterations = int(a.stage_iterations)
    runner_cfg.save_interval = int(a.save_interval)
    if a.learning_rate is not None:
        runner_cfg.algorithm.learning_rate = float(a.learning_rate)
    if a.fixed_learning_rate:
        runner_cfg.algorithm.schedule = "fixed"
    if a.entropy_coef is not None:
        runner_cfg.algorithm.entropy_coef = float(a.entropy_coef)
    if a.init_std is not None:
        runner_cfg.actor.distribution_cfg.init_std = float(a.init_std)

    apply_terrain_curriculum_stage(env_cfg, stages[0])
    apply_plant_overrides(env_cfg)
    # Training-only relief spread. The held-out gate terrain deliberately keeps
    # the narrower cfg tuple so scores stay comparable; PPO sees 0.35 (near-flat)
    # and 1.15 (rougher than anything scored) so it generalizes past the gates
    # rather than to them. Per-scale tracking is logged as
    # height_scale/<scale>_* so an unservable scale is visible, not averaged.
    env_cfg.terrain_height_scales = (0.35, 0.50, 0.70, 0.85, 1.00, 1.15)
    if a.tail_focus:
        # CUDA tail probe (seed 17, 64 x 960) localized all 144 samples below
        # alignment 0.95 to offset_return/lane_change_hold at relief 0.85/1.0.
        # Keep one copy of every family to prevent forgetting, then repeat the
        # two failure families six more times each (~61% hard-family exposure).
        env_cfg.tracking_path_family_schedule = (
            "boom_sweep",
            "double_s_curve",
            "fourier_meander",
            "gaussian_slalom",
            "lane_change_hold",
            "lateral_chirp",
            "lateral_sine",
            "offset_return",
            "random_spline",
            "s_curve",
            "straight",
            *("offset_return",) * 6,
            *("lane_change_hold",) * 6,
        )
        # The two hard families fail as q approaches 0.7, but with the authored
        # reference all parallel environments encounter that transition in the
        # same PPO rollout. Keep one canonical copy of every family, then make
        # the repeated hard-family copies begin across the measured failure
        # window. This supplies recovery examples in every episode without
        # changing the held-out mesh or teleporting the physical truck.
        hard_start_phases = (0.55, 0.60, 0.65, 0.70, 0.72, 0.75)
        env_cfg.tracking_path_start_phase_schedule = (
            *(-1.0 for _ in range(11)),
            *hard_start_phases,
            *hard_start_phases,
        )
        env_cfg.terrain_height_scales = (0.85, 1.0, 1.0, 1.0)
    if a.contact_focus:
        # The measured contact-loss region, not a guessed one. The pre-fix
        # probe put all 428 ROM-A events (and all 458 ID/FD events) on s_curve
        # at relief 1.0 on the held-out mesh, concentrated late in the path
        # (reports/contact_safe_v1/prefix_*_s_curve_seed*.json).
        #
        # Keep one copy of every family so the policy does not forget the ten
        # it already handles, then repeat s_curve seven more times (~39 % hard-
        # family exposure) with start phases spread across the late window so
        # every rollout contains the final mound/hollow region rather than
        # reaching it once, simultaneously, in all parallel environments.
        env_cfg.tracking_path_family_schedule = (
            "boom_sweep",
            "double_s_curve",
            "fourier_meander",
            "gaussian_slalom",
            "lane_change_hold",
            "lateral_chirp",
            "lateral_sine",
            "offset_return",
            "random_spline",
            "s_curve",
            "straight",
            *("s_curve",) * 7,
        )
        late_phases = (0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 0.75)
        env_cfg.tracking_path_start_phase_schedule = (
            *(-1.0 for _ in range(11)),
            *late_phases,
        )
        # Height schedule is supplied by the caller through --height_schedule
        # so the three-block ramp (easy -> full -> balanced) is explicit in the
        # driver script rather than hidden in a flag.
    if a.head_tail_risk:
        env_cfg.terrain_head_tail_fraction = 0.02
        env_cfg.terrain_head_tail_multiplier = 2.0
        env_cfg.terrain_head_tail_memory_trigger = 0.95
        env_cfg.terrain_head_tail_memory_steps = 30
        env_cfg.terrain_head_tail_memory_penalty = -1.0
    # Training-only path widening, same rationale and same pattern as the relief
    # spread above: the gated held-out path deliberately keeps the cfg default so
    # scores stay comparable, while PPO is exercised over a wider lateral band
    # than it is ever scored on.
    #
    # The families author 0.10-0.24 m of swing against 0.74 m of available
    # half-strip, so the arm was working a third of the room the verge offers
    # (mean peak-to-peak 0.202 m). 1.8x takes that to 0.355 m.
    #
    # Measured cost on the SHIPPED checkpoint, which was never trained for it
    # (128 envs x 960 held-out steps): 1.2x -> 14/17, 1.4x -> 13/17,
    # 1.8x -> 9/17. Lateral error stays well inside its 0.30 gate at every scale
    # (0.081/0.092/0.118), so reach is not the limit -- head alignment is
    # (0.9500 -> 0.9155/0.8928/0.3379). A wider path demands more slew and slew
    # tilts the head, so this widening AMPLIFIES the outstanding head coupling.
    # Expect a policy trained here to need the head driven as a real joint
    # rather than the 0.76 feedforward with a bounded correction.
    # Tail-focus must reproduce the measured failure geometry. The diagnostic
    # tail is on the held-out 1.0x paths; applying the generic 1.8x widening
    # changes the problem and previously drove head alignment as low as 0.338.
    # Contact focus is the same argument: the contact-loss events were measured
    # at 1.0x, so widening the path would train against a different failure.
    env_cfg.terrain_tracking_amplitude_scale = (
        1.0 if (a.tail_focus or a.contact_focus) else 1.8
    )
    if a.height_schedule:
        scales = tuple(
            float(part) for part in a.height_schedule.split(",") if part.strip()
        )
        if not scales or any(value <= 0.0 for value in scales):
            raise ValueError("--height_schedule must be positive comma-separated scales")
        env_cfg.terrain_height_scales = scales

    # The inherited trajectory exists only for reset deployment. Scored terrain
    # episodes always use the sensed verge reference and a fixed deployment.
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    env_cfg.randomize_trajectory_shape = False
    env_cfg.randomize_start_phase = False
    # PPO always runs teacher-free. Distillation is scripts/distill_terrain.py, and it
    # hands over a checkpoint, not a teacher.
    assert_teacher_disabled(env_cfg, context="PPO3 PPO training")
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    # DirectRLEnv construction leaves the asset at its spawn pose. Without an
    # explicit reset, PPO's first rollout starts before the production arm
    # deployment/settle/ROM warm-up and sees ~1.1 m cut-height error. Every
    # training run must enter through the same verified reset path as probes.
    raw.reset(seed=int(a.seed))
    if bool(getattr(raw.cfg, "enable_contact_tracking_demo", False)):
        raise RuntimeError("direct-target contact demo must be disabled for PPO")
    reset_obs = raw._get_observations()["policy"]
    if tuple(reset_obs.shape) != (raw.num_envs, int(raw.cfg.observation_space)):
        raise RuntimeError(
            f"reset observation shape {tuple(reset_obs.shape)} violates the frozen contract"
        )
    if not bool(torch.isfinite(reset_obs).all()):
        raise RuntimeError("non-finite reset observation; refusing to start PPO")
    reset_cut_height = raw.cutter_height_above_ground()
    reset_cut_error = raw.cut_height_error().abs()
    reset_ground = raw._cutter_world_z() - reset_cut_height
    print(
        "[PPO3-RESET] "
        f"mean_abs_cut_error={float(reset_cut_error.mean()):.4f} m "
        f"p95_abs_cut_error={float(torch.quantile(reset_cut_error, 0.95)):.4f} m "
        f"mean_cut_height={float(reset_cut_height.mean()):.4f} m "
        f"mean_cutter_z={float(raw._cutter_world_z().mean()):.4f} m "
        f"mean_ground_z={float(reset_ground.mean()):.4f} m "
        f"scan_valid={float(raw.scan_valid_fraction().mean()):.3f} "
        f"ride_height={float(raw.truck_ride_height().mean()):.4f} m "
        f"mean_lateral_error={float(raw.verge_lateral_error().abs().mean()):.4f} m "
        f"min_head_alignment={float(raw._measure_head_down_alignment().min()):.4f}",
        flush=True,
    )

    if a.zero_action_probe or a.probe_only:
        report = zero_action_probe(raw)
        print(f"[PPO3-SOLE] zero-action probe: {report}", flush=True)
        if report["max_abs_applied_action"] > 1.0e-7:
            raise RuntimeError(
                "zero action produced a non-zero plant command; the action "
                "chain is not a fixed point at rest"
            )
        if a.probe_only:
            env.close()
            simulation_app.close()
            return

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
    startup_manifest = {
        "schema": "terrain_shared_v2",
        "actuator": str(a.actuator),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "initialization_source": resume_source,
        "resume": str(resume_path) if resume_path is not None else None,
        "teacher_mode": str(raw.cfg.sole_teacher_mode),
        "direct_target_demo": bool(raw.cfg.enable_contact_tracking_demo),
        "num_envs": int(raw.num_envs),
        "observation_width": int(raw.cfg.observation_space),
        "action_width": int(raw.cfg.action_space),
        "head_joint_action": bool(raw.cfg.sole_head_joint_action),
        "head_action_to_c4_gain": float(raw.cfg.sole_head_action_to_c4_gain),
        "automatic_head_level_gain": float(raw.cfg.terrain_messerkopf_level_gain),
        "path_families": sorted(set(raw._tracking_path_family_names)),
        "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
        "terrain_base_seed": int(raw._mow_strip_cfg().seed),
        "terrain_height_scales": sorted(
            set(float(value) for value in raw._terrain_height_scales)
        ),
        "tail_focus": bool(a.tail_focus),
        "head_tail_risk": bool(a.head_tail_risk),
        "actor_parameters": int(
            sum(parameter.numel() for parameter in runner.alg.actor.parameters())
        ),
        "critic_parameters": int(
            sum(parameter.numel() for parameter in runner.alg.critic.parameters())
        ),
        "num_steps_per_env": int(runner_cfg.num_steps_per_env),
        "learning_rate": float(runner_cfg.algorithm.learning_rate),
        "entropy_coef": float(runner_cfg.algorithm.entropy_coef),
        "seed": int(a.seed),
        "stages": list(stages),
    }
    (log_dir / "run_manifest.json").write_text(
        json.dumps(startup_manifest, indent=2, sort_keys=True)
    )
    print("[PPO3-RUN] " + json.dumps(startup_manifest, sort_keys=True), flush=True)
    if resume_path is not None:
        teacher_handoff = resume_source == "PPO3 terrain teacher distillation v1"
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
        runner.load(str(resume_load_path), load_cfg=load_cfg)
        if teacher_handoff:
            # PPO learns a new value function and optimizer. Only the deployable
            # actor (including its observation normalizer) crosses the boundary.
            critic_output = runner.alg.critic.mlp[-1]
            if not isinstance(critic_output, torch.nn.Linear):
                raise RuntimeError("terrain critic output is not Linear")
            with torch.no_grad():
                critic_output.weight.zero_()
                critic_output.bias.zero_()
        if a.reset_critic_head_on_resume:
            critic_output = runner.alg.critic.mlp[-1]
            if not isinstance(critic_output, torch.nn.Linear):
                raise RuntimeError("terrain critic output is not Linear")
            with torch.no_grad():
                critic_output.weight.zero_()
                critic_output.bias.zero_()
        # Applied before --resume_c4_std so the head-pitch axis can still be set
        # independently on top of a global reduction.
        if a.resume_std is not None:
            distribution = runner.alg.actor.distribution
            if not hasattr(distribution, "std_param"):
                raise RuntimeError("terrain actor distribution has no std_param")
            with torch.no_grad():
                distribution.std_param[:] = float(a.resume_std)
            print(
                f"[PPO3-VERGE] set exploration std to {float(a.resume_std)} "
                f"on all {int(raw.cfg.action_space)} actions after resume",
                flush=True,
            )
        if a.resume_c4_std is not None:
            distribution = runner.alg.actor.distribution
            if not hasattr(distribution, "std_param"):
                raise RuntimeError("terrain actor distribution has no std_param")
            with torch.no_grad():
                distribution.std_param[3] = float(a.resume_c4_std)
        if a.resume_q5_std is not None:
            distribution = runner.alg.actor.distribution
            if not hasattr(distribution, "std_param"):
                raise RuntimeError("terrain actor distribution has no std_param")
            with torch.no_grad():
                distribution.std_param[4] = float(a.resume_q5_std)
        # rsl_rl restores the optimizer param-group LR but not PPO's separate
        # ``learning_rate`` scalar. On the next adaptive-KL update that stale
        # scalar overwrites the restored optimizer LR (3e-4 instead of the
        # checkpoint's 1.52e-5 in the measured run), causing a large policy
        # jump immediately after resume. Keep both representations in sync.
        if a.learning_rate is None:
            resumed_lr = float(runner.alg.optimizer.param_groups[0]["lr"])
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = resumed_lr
        else:
            resumed_lr = float(a.learning_rate)
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = resumed_lr
        runner.alg.learning_rate = resumed_lr
        print(f"[PPO3-SOLE] resumed from {a.resume}", flush=True)
        if teacher_handoff:
            print(
                "[PPO3-TEACHER-HANDOFF] loaded actor+normalizer only; "
                "teacher is off, critic and PPO optimizer are new",
                flush=True,
            )
        print(f"[PPO3-SOLE] resumed optimizer learning_rate={resumed_lr:.6g}", flush=True)
        if a.reset_optimizer_on_resume:
            print("[PPO3-SOLE] reset optimizer moments on resume", flush=True)
        if a.reset_critic_head_on_resume:
            print("[PPO3-SOLE] reset critic output head on resume", flush=True)
        if a.resume_c4_std is not None:
            print(
                f"[PPO3-SOLE] set resumed c4 exploration std={a.resume_c4_std:.6g}",
                flush=True,
            )
        if a.resume_q5_std is not None:
            print(
                f"[PPO3-SOLE] set resumed q5 exploration std={a.resume_q5_std:.6g}",
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
        # The bias is authored per cylinder. When Messerkopf_joint is driven as a
        # real 5th action it starts unbiased: the reset seeds its reference from
        # the inherited coupling, so zero action already holds the validated
        # deployment pose and any bias here would push the head off it.
        width = int(output_layer.bias.numel())
        if bias.numel() < width:
            bias = torch.cat(
                [bias, torch.zeros(width - bias.numel(), device=bias.device)]
            )
        elif bias.numel() > width:
            raise RuntimeError(
                f"terrain_initial_action_bias has {bias.numel()} entries but the "
                f"actor emits {width} actions"
            )
        with torch.no_grad():
            output_layer.bias.copy_(bias)
        print(f"[PPO3-VERGE] initialized actor action bias={bias.tolist()}", flush=True)

    final_stage = stages[-1]
    last_convergence_result = None
    for stage in stages:
        apply_terrain_curriculum_stage(raw.cfg, stage)
        apply_plant_overrides(raw.cfg)
        raw.cfg.trajectory_shape = str(raw.cfg.terrain_deployment_trajectory_shape)
        raw.cfg.randomize_trajectory_shape = False
        raw.cfg.randomize_start_phase = False
        print(
            f"[PPO3-VERGE] === stage {stage} === "
            f"truck={raw.cfg.vehicle_speed_min_kmh:.2f} km/h "
            f"action_scale={raw.cfg.action_scale:.4f} m "
            f"rate={raw.cfg.sole_max_action_rate:.3f}/step "
            f"filter={raw.cfg.sole_action_filter:.2f} "
            + (
                f"until convergence (checks every {a.convergence_check_interval} iterations)"
                if a.until_converged and stage == final_stage
                else f"for {a.stage_iterations} iterations"
            ),
            flush=True,
        )
        if not (a.until_converged and stage == final_stage):
            runner.learn(
                num_learning_iterations=int(a.stage_iterations),
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
            checkpoint = log_dir / f"model_{runner.current_learning_iteration}.pt"
            runner.save(
                str(checkpoint),
                infos={
                    "source": "PPO3 terrain PPO",
                    "schema": "terrain_shared_v2",
                    "actuator": str(a.actuator),
                    "observation_width": int(raw.cfg.observation_space),
                    "action_width": int(raw.cfg.action_space),
                    "head_joint_action": bool(raw.cfg.sole_head_joint_action),
                    "head_action_to_c4_gain": float(
                        raw.cfg.sole_head_action_to_c4_gain
                    ),
                    "convergence": result,
                },
            )
            print(
                "[PPO3-CONVERGENCE] " + json.dumps(result, sort_keys=True),
                flush=True,
            )
            if result["passed"]:
                converged_path = log_dir / "model_converged.pt"
                runner.save(
                    str(converged_path),
                    infos={
                        "source": "PPO3 terrain PPO",
                        "schema": "terrain_shared_v2",
                        "actuator": str(a.actuator),
                        "observation_width": int(raw.cfg.observation_space),
                        "action_width": int(raw.cfg.action_space),
                        "head_joint_action": bool(raw.cfg.sole_head_joint_action),
                        "head_action_to_c4_gain": float(
                            raw.cfg.sole_head_action_to_c4_gain
                        ),
                        "convergence": result,
                    },
                )
                print(
                    f"[PPO3-CONVERGENCE] PASSED; wrote {converged_path}",
                    flush=True,
                )
                break

    if a.checkpoint_out:
        checkpoint_out = Path(a.checkpoint_out).expanduser().resolve()
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_info = {
            "source": "PPO3 terrain PPO",
            "schema": "terrain_shared_v2",
            # Every save path must carry this. The supervisor evaluates THIS
            # checkpoint, and an untagged one is read as 'rom', which made the
            # held-out evaluator refuse an ID/FD block and killed the run.
            "actuator": str(a.actuator),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iteration": int(runner.current_learning_iteration),
            "num_envs": int(raw.num_envs),
            "stage": int(final_stage),
            "seed": int(a.seed),
            "learning_rate": float(runner.alg.optimizer.param_groups[0]["lr"]),
            "fixed_learning_rate": bool(a.fixed_learning_rate),
            "convergence": last_convergence_result,
            "observation_width": int(raw.cfg.observation_space),
            "action_width": int(raw.cfg.action_space),
            "head_joint_action": bool(raw.cfg.sole_head_joint_action),
            "head_action_to_c4_gain": float(raw.cfg.sole_head_action_to_c4_gain),
            "initialization_source": resume_source,
            "teacher_enabled_during_ppo": False,
            "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
            "terrain_base_seed": int(raw._mow_strip_cfg().seed),
            "path_families": sorted(set(raw._tracking_path_family_names)),
            "terrain_height_scales": sorted(
                set(float(value) for value in raw._terrain_height_scales)
            ),
            "optimizer_reset_on_resume": bool(a.reset_optimizer_on_resume),
            "critic_head_reset_on_resume": bool(a.reset_critic_head_on_resume),
            "resume_c4_std": a.resume_c4_std,
            "resume_q5_std": a.resume_q5_std,
            "head_tail_risk": bool(a.head_tail_risk),
        }
        runner.save(str(checkpoint_out), infos=checkpoint_info)
        manifest = checkpoint_out.with_suffix(".json")
        manifest_row = {"checkpoint": str(checkpoint_out), **checkpoint_info}
        manifest.write_text(json.dumps(manifest_row, indent=2))
        print(
            "[PPO3-CHECKPOINT] " + json.dumps(manifest_row, sort_keys=True),
            flush=True,
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
