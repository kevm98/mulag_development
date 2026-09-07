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
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=64)
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
p.add_argument("--gamma", type=float, default=None)
p.add_argument("--lam", type=float, default=None)
p.add_argument(
    "--demo_regularized",
    action="store_true",
    help=(
        "Query the DLS teacher only for labels on policy-visited weak states, "
        "then add filtered imitation and frozen-prior KL after each PPO update. "
        "The policy remains the sole executing controller."
    ),
)
p.add_argument("--demo_lambda", type=float, default=0.25)
p.add_argument("--demo_kl_beta", type=float, default=0.02)
p.add_argument("--demo_max_kl", type=float, default=0.25)
p.add_argument("--demo_lateral_threshold", type=float, default=0.10)
p.add_argument(
    "--latent_teacher_regularized",
    action="store_true",
    help=(
        "RMA handoff for ID-AAM-FD only: the executing actor receives live "
        "estimated z_hat, while a frozen copy of the resumed actor supplies "
        "action labels from the same observations with privileged true z."
    ),
)
p.add_argument("--latent_teacher_lambda", type=float, default=0.5)
p.add_argument("--latent_teacher_kl_beta", type=float, default=0.05)
p.add_argument("--latent_teacher_max_kl", type=float, default=0.05)
p.add_argument("--vehicle_speed_kmh", type=float, default=4.0)
p.add_argument(
    "--vehicle_speed_min_kmh",
    type=float,
    default=None,
    help="Minimum constant truck speed sampled once per episode.",
)
p.add_argument(
    "--vehicle_speed_max_kmh",
    type=float,
    default=None,
    help="Maximum constant truck speed sampled once per episode.",
)
p.add_argument(
    "--vehicle_speed_sampling",
    choices=("uniform", "active_time_balanced"),
    default="uniform",
    help=(
        "Episode reset distribution. active_time_balanced uses p(v) proportional "
        "to v so fast replicas contribute as many active path transitions as "
        "slow replicas before parking at 26 m."
    ),
)
p.add_argument(
    "--speed_segment_duration_s",
    type=float,
    default=0.0,
    help=(
        "Use equal-duration training segments at every sampled speed. The "
        "held-out evaluator keeps the complete 26 m path."
    ),
)
p.add_argument(
    "--head_joint_action",
    action="store_true",
    default=None,
    help="Use a fifth policy output as a direct Messerkopf_joint position target.",
)
p.add_argument(
    "--legacy_four_action",
    dest="head_joint_action",
    action="store_false",
    help="Use the production four-cylinder action interface.",
)
p.add_argument(
    "--head_action_to_c4_gain",
    type=float,
    default=None,
    help="Deprecated and ignored: direct q5 is never mixed into cylinder 4.",
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
    "--lateral_focus",
    action="store_true",
    help=(
        "Oversample lane_change_hold and offset_return without changing their "
        "start phase; preserves the physical lead-in/deployment contract."
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
    "--boom_focus",
    action="store_true",
    help=(
        "Stage B recovery distribution: HALF the environments are boom_sweep "
        "at late path phases (0.55-0.95), the measured worst contact region "
        "(251 consecutive airborne steps, 1.18 m gap, while passing all 17 "
        "historical gates). The other half keeps one copy of every family to "
        "prevent forgetting. Supersedes --contact_focus, which oversampled "
        "s_curve -- the second-worst region."
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
p.add_argument(
    "--contact_latch",
    default="",
    choices=("", "true", "false"),
    help=(
        "Latch a visible lift for the rest of the episode instead of "
        "terminating. Termination cannot do this job here: resets are "
        "all-or-nothing, and a 0.0039 per-step termination rate at 256 envs "
        "collapsed the mean episode length to 10 of 960 steps."
    ),
)
p.add_argument("--contact_latch_release_steps", type=int, default=None)
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
# --- lateral/contact reward balance ---------------------------------------
# Training-only: these change the reward, never the observation width nor the
# plant, so the evaluator neither needs nor accepts them. Measured motivation:
# the shipped balance puts ~64 units of weight on the contact family against
# ~6.5 on lateral, and `lateral_tracking_std_m` equals the 0.30 m gate, so the
# reward barely separates a passing 0.30 m from a failing 0.48 m. The policy
# duly bought contact 0.83 -> 0.95 by giving back lateral 0.30 -> 0.48 on the
# large-excursion families, while the privileged teacher reached 0.18 m in the
# same action space. Sharpening lateral moves the equilibrium, it does not
# move any gate.
p.add_argument("--contact_tail_fraction", type=float, default=None)
p.add_argument("--contact_tail_multiplier", type=float, default=None)
p.add_argument("--lateral_tail_fraction", type=float, default=None)
p.add_argument("--lateral_tail_multiplier", type=float, default=None)
p.add_argument("--head_penalty_weight", type=float, default=None, help='Training-only override of terrain_head_penalty_weight (default -20.0). Targeted alternative to generic motion penalties: measured 2026-08-05, action-magnitude 0 gives the best tracking ever (worst lat 0.3732) but head falls to 0.7060, while a -2.0 rate penalty restores head to 0.9232 and wrecks tracking (0.5207) because it damps the lateral transition lane_change_hold needs. Charging for head tilt specifically avoids both.')
p.add_argument("--rate_penalty_weight", type=float, default=None, help='Training-only override of sole_rate_penalty_weight. Prefer this to the action MAGNITUDE penalty for this task: a held lateral offset has zero rate and so costs nothing, while the jerky motion that tilts the head still costs. Measured 2026-08-05: magnitude penalty 0.0 improved worst-family lateral 0.3980 -> 0.3732 and mean lateral 0.1931 -> 0.1814 (both best ever) but let head alignment fall 0.9195 -> 0.7060.')
p.add_argument("--action_magnitude_weight", type=float, default=None, help='Training-only override of terrain_action_magnitude_weight. Holding a SUSTAINED lateral offset -- what lane_change_hold, offset_return and boom_sweep require -- needs a sustained non-zero action, and this penalty charges for it every step. Measured: the untrained policy holds lane_change_hold at 0.128 m and every trained one lands at 0.51-0.66.')
p.add_argument("--lateral_tracking_std_m", type=float, default=None)
p.add_argument("--lateral_tracking_weight", type=float, default=None)
p.add_argument("--dense_lateral_error_weight", type=float, default=None)
p.add_argument(
    "--floating_cutter",
    action="store_true",
    help="Train on the passive prismatic cutter-suspension plant.",
)
p.add_argument("--floating_cutter_stiffness", type=float, default=None)
p.add_argument("--floating_cutter_damping", type=float, default=None)
p.add_argument("--floating_cutter_effort_limit", type=float, default=None)
p.add_argument("--adaptive_trajectories", action="store_true")
p.add_argument("--adaptive_episode_length", type=float, default=40.0)
p.add_argument("--adaptive_amplitude", type=float, default=8.0)
p.add_argument("--adaptive_frequency", type=float, default=1.6)
p.add_argument("--adaptive_frequency_jitter", type=float, default=0.20)
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
p.add_argument("--cylinder_target_max_lead_per_axis_m", default="", help="Per-cylinder anti-windup lead caps in metres, comma separated, one per cylinder (e.g. '0.05,0.05,0.05,0.018'). Overrides --cylinder_target_max_lead_m per axis. Measured: relaxing the cap halves lateral error on the sustained-offset path families, but the SAME relaxation on c4 (head pitch) collapses head alignment.")
p.add_argument(
    "--tracking_amplitude",
    type=float,
    default=None,
    help=(
        "Override terrain_tracking_amplitude_scale. Stage 6 sets 2.2 as the "
        "point of the stage; ramping 1.4 -> 1.8 -> 2.2 exists because a wider "
        "path demands more slew and slew tilts the head -- jumping straight to "
        "2.2 measured head alignment 0.5911 on the stage-5 checkpoint."
    ),
)
p.add_argument(
    "--head_frame_observation",
    default="",
    choices=("", "true", "false"),
    help=(
        "Expose the JOINT-FRAME head tilt and the arm slew to the policy "
        "(2 extra observation columns). The policy otherwise sees only an "
        "unsigned alignment cosine and a WORLD-frame axis, neither of which is "
        "the error the head joint can act on."
    ),
)
p.add_argument(
    "--head_tilt_arm_frame",
    default="",
    choices=("", "true", "false"),
    help=(
        "Measure head tilt about the joint's own axis instead of the world "
        "Y-Z plane. Measured: the world projection under-reads the correctable "
        "tilt by 54 percent exactly where alignment collapses."
    ),
)
p.add_argument(
    "--height_schedule",
    default="",
    help=(
        "Comma-separated training relief scales, e.g. '0.50,0.70'. Empty keeps "
        "the wide default (0.35..1.15). 0.0 is allowed and gives a perfectly "
        "flat environment. The held-out gate tuple is never changed by this."
    ),
)
p.add_argument(
    "--mow_strip_width_m",
    type=float,
    default=None,
    help=(
        "Width of the green verge the cutter tracks. The path bound is "
        "0.5*width - 0.16 m of half-swing, so this is what decides how far the "
        "arm must extend laterally. MUST be given to the evaluator too: it "
        "changes the terrain being scored, and a wide-track score is NOT "
        "comparable with a 1.8 m one."
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
    "--train_context_only",
    action="store_true",
    help=(
        "Train ONLY the actor's context columns: freeze every actor parameter "
        "and mask the input layer's gradient so just the context slice moves. "
        "The point is that this CANNOT walk away from the distilled behaviour. "
        "A nominal plant normalises to the zero context vector, so with the "
        "base's zeroed context weights the policy is exactly the frozen one at "
        "iteration 0 and stays exactly the frozen one wherever the context is "
        "nominal; all it can learn is how to modulate itself by the payload. "
        "Measured 2026-09-06: an unanchored full-policy fine-tune on the wide "
        "payload range dropped ground contact from 0.788 to 0.457 (adaptive) "
        "and 0.023 (blind), i.e. PPO bought lateral error by flying the cutter "
        "off the ground. This is the fix for that, and it is ~256 parameters "
        "rather than the whole actor."
    ),
)
p.add_argument(
    "--freeze_actor_iterations",
    type=int,
    default=0,
    help=(
        "Keep the resumed actor fixed while the critic adapts to a new speed "
        "distribution. The actor is unfrozen after this many PPO iterations."
    ),
)
p.add_argument(
    "--obs_normalizer",
    choices=("freeze", "adapt"),
    default="freeze",
    help="Freeze resumed observation statistics; adapt is an explicit ablation.",
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
    choices=("rom", "idfd", "id_aam_fd"),
    help=(
        "Which actuator converts the cylinder command into motion. "
        "rom = PPO3 command ROM; idfd = PPO4 ID->FD; id_aam_fd = causal "
        "AAM -> ID adapter -> FD. Everything else about training is identical, "
        "which is the point."
    ),
)
p.add_argument(
    "--online_aam_residual_scale",
    "--online-aam-residual-scale",
    dest="online_aam_residual_scale",
    type=float,
    default=None,
    help=(
        "Override the PPO-ID(AAM)-FD latent residual scale for a controlled "
        "curriculum. The AAM observation remains enabled and its value is "
        "recorded in the run manifest."
    ),
)
p.add_argument(
    "--online_aam_latent_mode",
    choices=("estimated", "true", "zero", "mean", "shuffled"),
    default=None,
    help=(
        "Latent supplied to the policy and ID adapter. 'true' is the "
        "privileged frozen-SD warm-start phase; production deployment uses "
        "'estimated'."
    ),
)
p.add_argument(
    "--factor_domain",
    default="",
    choices=("", "nominal", "hydraulic", "mechanical", "combined_train", "combined_test"),
    help=(
        "Randomize the plant per EPISODE RESET from the audited deployment "
        "factor manifest. Without this the plant is nominal for every episode, "
        "the AAM latent z_hat is therefore constant, and both consumers of it "
        "learn to ignore it: measured 2026-08-14, a policy trained on nominal "
        "ended with first-layer weights of 0.023/0.016 on the two z_hat "
        "channels against a median of 1.36, and the conditioned residual made "
        "every gate WORSE because it perturbs a command the frozen ID already "
        "gets right. Under combined_train the sign flips -- residual 0.25 vs "
        "0.0 at 3 km/h gives headroom 0.00231 vs 0.00018 and lateral 0.3757 "
        "vs 0.4378. Train ID(AAM)-FD with this set."
    ),
)
p.add_argument(
    "--factor_key",
    default="",
    help="Explicit factor key/run id, e.g. flow_min or combined_test:102.",
)
p.add_argument("--factor_manifest", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

stages = tuple(int(part) for part in a.stages.split(",") if part.strip())
if not stages or any(stage not in (1, 2, 3, 4, 5, 6) for stage in stages):
    p.error("--stages must be a comma-separated subset of 1,2,3,4,5,6")
if a.stage_iterations <= 0 or a.save_interval <= 0:
    p.error("stage iterations and save interval must be positive")
if a.freeze_actor_iterations < 0:
    p.error("--freeze_actor_iterations must be non-negative")
if a.learning_rate is not None and a.learning_rate <= 0.0:
    p.error("--learning_rate must be positive")
if a.gamma is not None and not 0.0 < a.gamma <= 1.0:
    p.error("--gamma must be in (0, 1]")
if a.lam is not None and not 0.0 < a.lam <= 1.0:
    p.error("--lam must be in (0, 1]")
if min(a.demo_lambda, a.demo_kl_beta, a.demo_max_kl) < 0.0:
    p.error("demo regularization weights and KL cap must be non-negative")
if min(
    a.latent_teacher_lambda,
    a.latent_teacher_kl_beta,
    a.latent_teacher_max_kl,
) < 0.0:
    p.error("latent-teacher weights and KL cap must be non-negative")
if a.demo_regularized and a.latent_teacher_regularized:
    p.error("DLS demo and latent-teacher regularization are mutually exclusive")
if a.latent_teacher_regularized and (a.actuator != "id_aam_fd" or not a.resume):
    p.error("--latent_teacher_regularized requires ID-AAM-FD and --resume")
if a.latent_teacher_regularized and a.online_aam_latent_mode not in (None, "estimated"):
    p.error("latent-teacher student must execute with estimated z_hat")
if a.demo_lateral_threshold < 0.0:
    p.error("--demo_lateral_threshold must be non-negative")
if a.speed_segment_duration_s < 0.0:
    p.error("--speed_segment_duration_s must be non-negative")
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
if (a.vehicle_speed_min_kmh is None) != (a.vehicle_speed_max_kmh is None):
    p.error(
        "--vehicle_speed_min_kmh and --vehicle_speed_max_kmh must be given together"
    )
if a.vehicle_speed_min_kmh is not None and not (
    0.0 < a.vehicle_speed_min_kmh <= a.vehicle_speed_max_kmh
):
    p.error("truck speed range must satisfy 0 < min <= max")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import verge_task  # noqa: F401,E402  (registers the gym task)
from verge_task.distill import assert_teacher_disabled  # noqa: E402
from verge_task.vehicle_speed import configure_path_horizon  # noqa: E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    TerrainSolePolicyPPORunnerCfg,  # noqa: F401  (default actuator)
    apply_terrain_curriculum_stage,
    configure_floating_cutter,
    configure_terrain_gpu_physx_memory,
)
from verge_task.actuators import resolve_actuator  # noqa: E402
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)
from verge_task.terrain import (  # noqa: E402
    TRACKING_PATH_FAMILIES,
    enable_adaptive_trajectory_curriculum,
)

# One task, two actuators. Resolved once here so every downstream use --
# env construction, runner config, checkpoint tagging -- cannot disagree.
TASK_ID, RUNNER_CFG_CLS, ACTUATOR_LABEL = resolve_actuator(a.actuator)
print(f"[PPO-TERRAIN] actuator={a.actuator} ({ACTUATOR_LABEL}) task={TASK_ID}", flush=True)
from verge_task.terrain_convergence import summarize_convergence  # noqa: E402


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
    # Must be re-asserted here rather than once at construction: the stage
    # curriculum assigns `lateral_tracking_std_m` itself (stage 6 -> 0.30), so
    # an override applied before the curriculum is silently reverted on every
    # stage transition -- the exact failure this function exists to prevent.
    if a.head_penalty_weight is not None:
        override_cfg(
            cfg, "terrain_head_penalty_weight", float(a.head_penalty_weight)
        )
    if a.rate_penalty_weight is not None:
        override_cfg(cfg, "sole_rate_penalty_weight", float(a.rate_penalty_weight))
    if a.action_magnitude_weight is not None:
        override_cfg(
            cfg, "terrain_action_magnitude_weight", float(a.action_magnitude_weight)
        )
    if a.lateral_tracking_std_m is not None:
        override_cfg(cfg, "lateral_tracking_std_m", float(a.lateral_tracking_std_m))
    if a.lateral_tracking_weight is not None:
        override_cfg(cfg, "lateral_tracking_weight", float(a.lateral_tracking_weight))
    if a.dense_lateral_error_weight is not None:
        override_cfg(
            cfg, "dense_lateral_error_weight", float(a.dense_lateral_error_weight)
        )
    for flag, field in (
        ("contact_tail_fraction", "terrain_contact_tail_fraction"),
        ("contact_tail_multiplier", "terrain_contact_tail_multiplier"),
        ("lateral_tail_fraction", "terrain_lateral_tail_fraction"),
        ("lateral_tail_multiplier", "terrain_lateral_tail_multiplier"),
    ):
        value = getattr(a, flag)
        if value is not None:
            override_cfg(cfg, field, float(value))
    if a.cylinder_target_max_lead_m is not None:
        override_cfg(
            cfg, "cylinder_target_max_lead_m", float(a.cylinder_target_max_lead_m)
        )
    if a.cylinder_target_leak is not None:
        override_cfg(cfg, "cylinder_target_leak", float(a.cylinder_target_leak))
    if a.cylinder_target_max_lead_per_axis_m:
        override_cfg(
            cfg,
            "cylinder_target_max_lead_per_axis_m",
            tuple(
                float(part)
                for part in a.cylinder_target_max_lead_per_axis_m.split(",")
                if part.strip()
            ),
        )


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
        from verge_task.actuators import assert_checkpoint_actuator
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
        print("[PPO3] --head_action_to_c4_gain is deprecated and ignored; q5 is direct", flush=True)
    env_cfg.sole_head_action_to_c4_gain = 0.0
    if a.sole_ik_hint:
        raise ValueError("--sole_ik_hint is not part of the terrain observation contract")
    env_cfg.sole_ik_hint = False
    # --- contact-safe overrides, applied BEFORE the observation width is
    # derived so a widened observation and the network built for it cannot
    # disagree. ---
    if a.contact_observation:
        env_cfg.terrain_contact_observation = a.contact_observation == "true"
    # WITH the other observation flags, and BEFORE observation_space_for_cfg
    # below. Applied later -- in the curriculum block, where it reads naturally
    # next to --tracking_amplitude -- the width was already frozen and training
    # built a 96-wide env for a 98-wide checkpoint. Both actuators died on
    # "checkpoint observation width 98 does not match the frozen terrain
    # contract 96" before a single iteration ran.
    if a.head_frame_observation:
        env_cfg.terrain_head_frame_observation = a.head_frame_observation == "true"
    if a.head_tilt_arm_frame:
        env_cfg.terrain_head_tilt_arm_frame = a.head_tilt_arm_frame == "true"
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
    if a.contact_latch:
        env_cfg.terrain_contact_loss_latches_failure = a.contact_latch == "true"
    if a.contact_latch_release_steps is not None:
        env_cfg.terrain_contact_loss_latch_release_steps = int(
            a.contact_latch_release_steps
        )
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
    if a.online_aam_residual_scale is not None:
        if a.actuator != "id_aam_fd":
            p.error("--online_aam_residual_scale requires --actuator id_aam_fd")
        if not 0.0 <= a.online_aam_residual_scale <= 1.0:
            p.error("--online_aam_residual_scale must be in [0, 1]")
        env_cfg.online_aam_residual_scale = float(a.online_aam_residual_scale)
    if a.online_aam_latent_mode is not None:
        if a.actuator != "id_aam_fd":
            p.error("--online_aam_latent_mode requires --actuator id_aam_fd")
        env_cfg.online_aam_latent_mode = str(a.online_aam_latent_mode)
        env_cfg.online_aam_policy_latent_mode = str(a.online_aam_latent_mode)
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
    if a.vehicle_speed_min_kmh is not None:
        env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_min_kmh)
        env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_max_kmh)
    else:
        env_cfg.vehicle_speed_min_kmh = float(a.vehicle_speed_kmh)
        env_cfg.vehicle_speed_max_kmh = float(a.vehicle_speed_kmh)
    env_cfg.vehicle_speed_per_episode = True
    env_cfg.vehicle_speed_sampling = str(a.vehicle_speed_sampling)
    # Plant randomization is latched once per episode reset, matching how the
    # AAM's own dataset was collected (one constant factor row per run) and how
    # truck speed is already sampled. Switching factors mid-episode would put
    # the AAM out of its training distribution and make estimator lag
    # indistinguishable from estimator error; that belongs in a mid-episode
    # switch TEST, not in the training regime.
    if a.factor_domain or a.factor_key:
        env_cfg.deployment_factor_enabled = True
        env_cfg.deployment_factor_domain = a.factor_domain or "nominal"
        env_cfg.deployment_factor_key = str(a.factor_key)
        if a.factor_manifest:
            env_cfg.deployment_factor_manifest = str(Path(a.factor_manifest).resolve())
        env_cfg.deployment_factor_seed = int(a.seed)
    env_cfg.vehicle_training_segment_duration_s = float(
        a.speed_segment_duration_s
    )
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
    if a.gamma is not None:
        runner_cfg.algorithm.gamma = float(a.gamma)
    if a.lam is not None:
        runner_cfg.algorithm.lam = float(a.lam)
    if a.init_std is not None:
        runner_cfg.actor.distribution_cfg.init_std = float(a.init_std)

    apply_terrain_curriculum_stage(env_cfg, stages[0])
    apply_plant_overrides(env_cfg)
    configure_path_horizon(env_cfg)
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
    if a.lateral_focus:
        # The direct-q5 held-out failure is localized to lane_change_hold and
        # offset_return. Preserve the canonical centre lead-in: phase-shifting
        # these families made a centre-deployed arm begin inside a late
        # manoeuvre and measured 35-63% clearance termination while head
        # alignment collapsed from 0.992 to 0.640 in twenty updates.
        # The first lateral-only attempt repeated each failure family six
        # times, abruptly making 61% of the batch a two-family task. That
        # regressed the deterministic checkpoint in 15 updates. Two extra
        # copies each give the failures 33% exposure (5/15 env groups) while
        # preserving the mixed distribution the checkpoint learned.
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
            *("offset_return",) * 2,
            *("lane_change_hold",) * 2,
        )
        env_cfg.tracking_path_start_phase_schedule = ()
    if a.boom_focus:
        # The MEASURED worst region, found by the V1 strict gate matrix and not
        # by the V1 diagnostic. `boom_sweep` at relief 1.0 produced 251
        # consecutive airborne steps and a 1.18 m gap -- while passing all 17
        # historical gates, because its contact duty cycle was 0.9012.
        #
        # The V1 diagnostic used `s_curve` because that is where the lift is
        # visible, and treated `boom_sweep` as a control family because it is
        # the low-tail control for HEAD ALIGNMENT. For contact it is the
        # opposite, so `--contact_focus` was aimed at the second-worst region.
        #
        # Half the environments are boom_sweep at late phases; the remaining
        # half keeps one copy of every other family so the ten the policy
        # already handles are not forgotten.
        late = (0.55, 0.65, 0.75, 0.85, 0.90, 0.95)
        others = (
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
            # s_curve twice: it is the second-worst measured family and the one
            # every V1 artifact is indexed on.
            "s_curve",
            "boom_sweep",
        )
        boom = ("boom_sweep",) * len(late) * 2
        env_cfg.tracking_path_family_schedule = (*others, *boom)
        env_cfg.tracking_path_start_phase_schedule = (
            *(-1.0 for _ in others),
            *late,
            *late,
        )
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
    if a.speed_segment_duration_s > 0.0 and not (
        a.tail_focus or a.lateral_focus or a.boom_focus or a.contact_focus
    ):
        phases = (0.0, 0.25, 0.50, 0.75)
        env_cfg.tracking_path_family_schedule = tuple(
            family for phase in phases for family in TRACKING_PATH_FAMILIES
        )
        env_cfg.tracking_path_start_phase_schedule = tuple(
            phase for phase in phases for _ in TRACKING_PATH_FAMILIES
        )
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
    # Stage 6 sets its own amplitude (2.2) as the POINT of the stage -- a wide
    # tracking line over flat ground with discrete chunks. Overwriting it here
    # would silently turn stage 6 back into a narrow-path task and the whole
    # stage would measure something else.
    if 6 not in stages:
        env_cfg.terrain_tracking_amplitude_scale = (
            1.0 if (a.tail_focus or a.contact_focus or a.boom_focus) else 1.8
        )
    if a.tracking_amplitude is not None:
        if a.tracking_amplitude <= 0.0:
            raise ValueError("--tracking_amplitude must be positive")
        env_cfg.terrain_tracking_amplitude_scale = float(a.tracking_amplitude)
    if a.mow_strip_width_m is not None:
        # The tanh bound is 0.5*width - 0.16, so anything at or below 0.32 m
        # leaves no half-swing at all and every family collapses onto the
        # centre line -- a silently trivial task rather than an error.
        if a.mow_strip_width_m <= 0.34:
            raise ValueError(
                "--mow_strip_width_m must exceed 0.34 m to leave any lateral "
                "swing after the 0.16 m green margin"
            )
        env_cfg.mow_strip_width_m = float(a.mow_strip_width_m)
    if a.height_schedule:
        scales = tuple(
            float(part) for part in a.height_schedule.split(",") if part.strip()
        )
        # 0.0 is permitted and means a perfectly flat environment; it is how a
        # pure tracking case with no terrain disturbance is mixed in alongside
        # the chunked ones. Negatives would invert the relief.
        if not scales or any(value < 0.0 for value in scales):
            raise ValueError(
                "--height_schedule must be non-negative comma-separated scales"
            )
        env_cfg.terrain_height_scales = scales

    if a.floating_cutter:
        env_cfg.terrain_floating_cutter = True
    if a.floating_cutter_stiffness is not None:
        env_cfg.terrain_floating_cutter_stiffness_npm = float(
            a.floating_cutter_stiffness
        )
    if a.floating_cutter_damping is not None:
        env_cfg.terrain_floating_cutter_damping_nspm = float(
            a.floating_cutter_damping
        )
    if a.floating_cutter_effort_limit is not None:
        env_cfg.terrain_floating_cutter_effort_limit_n = float(
            a.floating_cutter_effort_limit
        )
    if a.adaptive_trajectories:
        enable_adaptive_trajectory_curriculum(
            env_cfg,
            episode_length_s=float(a.adaptive_episode_length),
            amplitude_scale=float(a.adaptive_amplitude),
            frequency_scale=float(a.adaptive_frequency),
            frequency_jitter=float(a.adaptive_frequency_jitter),
        )
    configure_floating_cutter(env_cfg)

    # The inherited trajectory exists only for reset deployment. Scored terrain
    # episodes always use the sensed verge reference and a fixed deployment.
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    env_cfg.randomize_trajectory_shape = False
    env_cfg.randomize_start_phase = False
    # The teacher may provide labels, but never executes during PPO. Distillation
    # hands over a checkpoint; --demo_regularized keeps a filtered label anchor
    # on policy-visited recovery states.
    #
    # Pin the deployment trajectory BEFORE construction, the same way
    # distill_terrain.py and evaluate_gates.py do. `apply_terrain_curriculum_stage`
    # leaves `randomize_trajectory_shape = True`, and a randomized library entry
    # with `truck_motion_mode = "parked"` sets `vehicle_speed_min_kmh =
    # vehicle_speed_max_kmh = 0.0` inside the base env's `_select_trajectory`.
    # That overwrites --vehicle_speed_kmh silently, and pinning the shape after
    # construction (which is where these lines used to live) never restores it.
    # See PPO_ROM/scripts/train.py for the measured 600-iteration run this cost.
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
        assert_teacher_disabled(env_cfg, context="PPO3 PPO training")
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
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
            "deployment trajectory's truck_motion_mode overwrote it; training "
            "on a parked truck while acceptance drives is the defect this "
            "check exists to stop."
        )
    print(
        f"[PPO-TERRAIN-DRIVE] truck speed verified at {built_speed_kmh[0]:.2f}-"
        f"{built_speed_kmh[1]:.2f} km/h after construction "
        f"(trajectory={raw.cfg.trajectory_shape})",
        flush=True,
    )
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
    if os.environ.get("MULAG_ARMA_PHI"):
        # A-RMA Phase 3: RL fine-tuning with the FROZEN Phase 2 estimator in the
        # loop, so the policy is optimised on the z_hat it will actually receive
        # rather than on the true context it saw in Phase 1.
        from verge_task.arma_phase3 import Phase3LatentAdapter, load_phi

        _phi_path = os.environ["MULAG_ARMA_PHI"]
        _phi, _payload = load_phi(_phi_path, a.device)
        _ctx = int(_payload["context_dim"])
        wrapped = Phase3LatentAdapter(
            wrapped, _phi,
            base_dim=int(raw.cfg.observation_space) - _ctx,
            context_dim=_ctx,
            history=int(_payload["history"]),
            target_mean=_payload["target_mean"],
            target_std=_payload["target_std"],
            num_envs=int(raw.num_envs),
            action_dim=int(raw.cfg.action_space),
            device=a.device,
        )
        print(
            f"[PPO3-ARMA] Phase 3: frozen phi from {_phi_path} "
            f"(history={_payload['history']}, context_dim={_ctx}) substituted into "
            "the observation; PPO now optimises on z_hat",
            flush=True,
        )
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
        "training_start_iteration": int(
            resume_payload.get("iter", 0) if resume_payload is not None else 0
        ),
        "teacher_mode": str(raw.cfg.sole_teacher_mode),
        "teacher_execution_during_ppo": False,
        "demo_regularized": bool(a.demo_regularized),
        "demo_lambda": float(a.demo_lambda) if a.demo_regularized else None,
        "demo_kl_beta": float(a.demo_kl_beta) if a.demo_regularized else None,
        "demo_max_kl": float(a.demo_max_kl) if a.demo_regularized else None,
        "demo_lateral_threshold": (
            float(a.demo_lateral_threshold) if a.demo_regularized else None
        ),
        "latent_teacher_regularized": bool(a.latent_teacher_regularized),
        "latent_teacher_lambda": (
            float(a.latent_teacher_lambda)
            if a.latent_teacher_regularized
            else None
        ),
        "latent_teacher_kl_beta": (
            float(a.latent_teacher_kl_beta)
            if a.latent_teacher_regularized
            else None
        ),
        "latent_teacher_max_kl": (
            float(a.latent_teacher_max_kl)
            if a.latent_teacher_regularized
            else None
        ),
        "direct_target_demo": bool(raw.cfg.enable_contact_tracking_demo),
        "num_envs": int(raw.num_envs),
        "observation_width": int(raw.cfg.observation_space),
        "action_width": int(raw.cfg.action_space),
        "head_joint_action": bool(raw.cfg.sole_head_joint_action),
        "head_control_mode": (
            "direct_joint_position" if raw.cfg.sole_head_joint_action else "disabled"
        ),
        "head_action_to_c4_gain": float(raw.cfg.sole_head_action_to_c4_gain),
        "automatic_head_level_gain": float(raw.cfg.terrain_messerkopf_level_gain),
        "path_families": sorted(set(raw._tracking_path_family_names)),
        "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
        "terrain_base_seed": int(raw._mow_strip_cfg().seed),
        "terrain_height_scales": sorted(
            set(float(value) for value in raw._terrain_height_scales)
        ),
        "vehicle_speed_min_kmh": float(raw.cfg.vehicle_speed_min_kmh),
        "vehicle_speed_max_kmh": float(raw.cfg.vehicle_speed_max_kmh),
        "vehicle_speed_per_episode": bool(
            getattr(raw.cfg, "vehicle_speed_per_episode", False)
        ),
        "vehicle_speed_sampling": str(
            getattr(raw.cfg, "vehicle_speed_sampling", "uniform")
        ),
        "speed_segment_duration_s": float(
            getattr(raw.cfg, "vehicle_training_segment_duration_s", 0.0)
        ),
        # A run trained on nominal cannot support any AAM claim; record the
        # plant randomization so a checkpoint's own manifest says which it was.
        "deployment_factor_enabled": bool(
            getattr(raw.cfg, "deployment_factor_enabled", False)
        ),
        "deployment_factor_domain": str(
            getattr(raw.cfg, "deployment_factor_domain", "nominal")
        ),
        "path_length_m": float(raw.cfg.terrain_tracking_active_length_m),
        "episode_length_s": float(raw.cfg.episode_length_s),
        "tail_focus": bool(a.tail_focus),
        "lateral_focus": bool(a.lateral_focus),
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
        "gamma": float(runner_cfg.algorithm.gamma),
        "lam": float(runner_cfg.algorithm.lam),
        "obs_normalizer": str(a.obs_normalizer),
        "freeze_actor_iterations": int(a.freeze_actor_iterations),
        "seed": int(a.seed),
        "stages": list(stages),
        "online_aam_residual_scale": (
            float(a.online_aam_residual_scale)
            if a.online_aam_residual_scale is not None
            else None
        ),
        "online_aam_latent_mode": str(
            getattr(raw.cfg, "online_aam_latent_mode", "estimated")
        ),
        "online_aam_policy_latent_mode": str(
            getattr(raw.cfg, "online_aam_policy_latent_mode", "estimated")
        ),
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
        if a.train_context_only:
            actor = runner.alg.actor
            linears = [m for m in actor.mlp.modules()
                       if isinstance(m, torch.nn.Linear)]
            if not linears:
                raise RuntimeError("--train_context_only found no Linear layer")
            first = linears[0]
            width = int(first.weight.shape[1])
            n_ctx = int(raw.cfg.observation_space) - 66
            if n_ctx <= 0 or width != int(raw.cfg.observation_space):
                raise RuntimeError(
                    "--train_context_only needs a context slot: input width "
                    f"{width}, observation_space {raw.cfg.observation_space}"
                )
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            first.weight.requires_grad_(True)
            # requires_grad alone would train the whole first layer, so the
            # task columns are masked out of the gradient every backward pass.
            _mask = torch.zeros_like(first.weight)
            _mask[:, width - n_ctx:] = 1.0
            first.weight.register_hook(lambda g: g * _mask)
            print(
                f"[PPO3-SOLE] train_context_only: {n_ctx} context columns of "
                f"{width}, {first.weight.shape[0] * n_ctx} trainable actor "
                f"parameters, everything else frozen",
                flush=True,
            )
        if os.environ.get("MULAG_RMA_ENCODER"):
            # RMA Phase 1 proper: mu is trained JOINTLY with the policy under
            # PPO. Installed AFTER runner.load so the warm start is in place
            # first; the zero-initialised decoder means this must not change
            # the policy's output at all.
            from verge_task.rma_encoder import wrap_actor_critic

            _latent = int(os.environ.get("MULAG_RMA_LATENT", "2"))
            _ctx = 6 if bool(getattr(raw.cfg, "online_aam_policy_privileged_e", False)) else 2
            _info = wrap_actor_critic(
                runner,
                base_dim=int(raw.cfg.observation_space) - _ctx,
                context_dim=_ctx,
                latent_dim=_latent,
            )
            print(f"[PPO3-RMA] jointly-trained encoder installed: {_info}", flush=True)
            if _info["init_output_drift"] > 1e-6:
                raise RuntimeError(
                    "RMA encoder changed the warm-started policy at init "
                    f"(drift {_info['init_output_drift']}); comparison would be confounded"
                )
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

    # RSL-RL's interval saves call ``runner.save(path)`` without ``infos``.
    # Those files used to lose the terrain schema and became impossible to
    # evaluate after an interrupted block. Wrap every save path once so a
    # periodic checkpoint is just as recoverable as ``--checkpoint_out``.
    original_runner_save = runner.save

    def save_with_terrain_provenance(path: str, infos=None) -> None:
        checkpoint_infos = {
            "source": "PPO3 terrain PPO",
            "schema": "terrain_shared_v2",
            "actuator": str(a.actuator),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iteration": int(runner.current_learning_iteration),
            "num_envs": int(raw.num_envs),
            "observation_width": int(raw.cfg.observation_space),
            "action_width": int(raw.cfg.action_space),
            "head_joint_action": bool(raw.cfg.sole_head_joint_action),
            "head_action_to_c4_gain": float(raw.cfg.sole_head_action_to_c4_gain),
            "initialization_source": resume_source,
            "teacher_enabled_during_ppo": bool(a.demo_regularized),
            "teacher_labels_enabled": bool(a.demo_regularized),
            "teacher_execution_during_ppo": False,
            "demo_regularized": bool(a.demo_regularized),
            "demo_lambda": float(a.demo_lambda) if a.demo_regularized else None,
            "demo_kl_beta": float(a.demo_kl_beta) if a.demo_regularized else None,
            "demo_max_kl": float(a.demo_max_kl) if a.demo_regularized else None,
            "latent_teacher_regularized": bool(a.latent_teacher_regularized),
            "latent_teacher_lambda": (
                float(a.latent_teacher_lambda)
                if a.latent_teacher_regularized
                else None
            ),
            "latent_teacher_kl_beta": (
                float(a.latent_teacher_kl_beta)
                if a.latent_teacher_regularized
                else None
            ),
            "latent_teacher_max_kl": (
                float(a.latent_teacher_max_kl)
                if a.latent_teacher_regularized
                else None
            ),
            "terrain_curriculum_version": str(raw.cfg.terrain_curriculum_version),
            "terrain_base_seed": int(raw._mow_strip_cfg().seed),
            "path_families": sorted(set(raw._tracking_path_family_names)),
            "terrain_height_scales": sorted(
                set(float(value) for value in raw._terrain_height_scales)
            ),
            "optimizer_reset_on_resume": bool(a.reset_optimizer_on_resume),
            "resume_q5_std": a.resume_q5_std,
            "tail_focus": bool(a.tail_focus),
            "lateral_focus": bool(a.lateral_focus),
        }
        if infos:
            checkpoint_infos.update(infos)
        original_runner_save(path, infos=checkpoint_infos)

    runner.save = save_with_terrain_provenance

    def observation_normalizers() -> list:
        """Return empirical input normalizers owned by actor or critic."""

        return [
            module
            for owner in (runner.alg.actor, runner.alg.critic)
            for module in owner.modules()
            if type(module).__name__ == "EmpiricalNormalization"
        ]

    _frozen_normalizers = observation_normalizers() if a.obs_normalizer == "freeze" else []
    if _frozen_normalizers:
        from verge_task.trajectory_regularized import (
            assert_normalizers_unchanged,
            freeze_normalizers_across_train_mode,
            reseed_frozen_speed_normalizer,
        )

        # Repair BEFORE the snapshot, or the drift guard preserves the damage
        # for the whole run. A frozen normalizer resumed from a narrower speed
        # range does not normalize the speed channel, it amplifies it: the
        # PPO-ID(AAM)-FD warm start carried mean 0.6283 m/s (2.3 km/h) and std
        # 0.1874, putting 7 km/h at +6.7 sigma while every other channel is
        # order 1. The same defect cost PPO-ROM eleven hours of training that
        # could not have worked. The AAM observation is 68-D (66 plus z_hat(2),
        # appended), so the speed channel index is unchanged.
        _normalizer_repairs = reseed_frozen_speed_normalizer(
            runner.alg,
            float(raw.cfg.vehicle_speed_min_kmh),
            float(raw.cfg.vehicle_speed_max_kmh),
        )
        for _repair in _normalizer_repairs:
            print(f"[VERGE-NORM] reseeded {_repair}", flush=True)
        if not _normalizer_repairs:
            print(
                "[VERGE-NORM] resumed normalizer already covers the sampled "
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
            return result

        runner.alg.update = update_with_frozen_normalizer_guard
        print(
            f"[PPO3-NORM] froze {len(_frozen_normalizers)} observation "
            "normalizer(s) for the resumed policy",
            flush=True,
        )

    demo_prior_actor = None
    demo_teacher_steps: list[torch.Tensor] = []
    demo_weight_steps: list[torch.Tensor] = []
    if a.demo_regularized:
        from verge_task.demo_regularized import demo_loss, gaussian_kl, state_filter

        demo_prior_actor = copy.deepcopy(runner.alg.actor).to(a.device)
        demo_prior_actor.eval()
        for parameter in demo_prior_actor.parameters():
            parameter.requires_grad_(False)

        process_env_step_without_demo = runner.alg.process_env_step

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
            return process_env_step_without_demo(obs, rewards, dones, extras)

        runner.alg.process_env_step = process_env_step_with_demo
        update_without_demo = runner.alg.update

        def update_with_demo_regularization(*args, **kwargs):
            storage = runner.alg.storage
            rollout_observations = storage.observations.clone()
            teacher_steps = list(demo_teacher_steps)
            weight_steps = list(demo_weight_steps)
            demo_teacher_steps.clear()
            demo_weight_steps.clear()

            result = update_without_demo(*args, **kwargs)
            if (
                not teacher_steps
                or len(teacher_steps) != int(storage.num_transitions_per_env)
                or not any(
                    parameter.requires_grad
                    for parameter in runner.alg.actor.parameters()
                )
            ):
                return result

            observations = rollout_observations.flatten(0, 1)
            teacher_actions = torch.stack(teacher_steps, dim=0).flatten(0, 1)
            weights = torch.stack(weight_steps, dim=0).flatten(0, 1)
            runner.alg.actor(observations, stochastic_output=True)
            policy_mean = runner.alg.actor.distribution.mean
            policy_std = runner.alg.actor.distribution.std
            with torch.no_grad():
                demo_prior_actor(observations, stochastic_output=True)
                prior_mean = demo_prior_actor.distribution.mean
                prior_std = demo_prior_actor.distribution.std
            filtered_demo = demo_loss(
                policy_mean=policy_mean,
                teacher_action=teacher_actions,
                weights=weights,
            )
            prior_kl = gaussian_kl(
                mean_q=policy_mean,
                std_q=policy_std,
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

            prior_kl_before_projection = float(prior_kl.detach())
            projection_alpha = 0.0
            max_kl = float(a.demo_max_kl)
            if max_kl > 0.0 and prior_kl_before_projection > max_kl:
                current_parameters = {
                    name: parameter.detach().clone()
                    for name, parameter in runner.alg.actor.named_parameters()
                }
                prior_parameters = dict(demo_prior_actor.named_parameters())

                def set_prior_blend(alpha: float) -> None:
                    with torch.no_grad():
                        for name, parameter in runner.alg.actor.named_parameters():
                            parameter.copy_(
                                (1.0 - alpha) * current_parameters[name]
                                + alpha * prior_parameters[name]
                            )

                def measure_prior_kl() -> torch.Tensor:
                    with torch.no_grad():
                        runner.alg.actor(observations, stochastic_output=True)
                        current_mean = runner.alg.actor.distribution.mean
                        current_std = runner.alg.actor.distribution.std
                        demo_prior_actor(observations, stochastic_output=True)
                        return gaussian_kl(
                            mean_q=current_mean,
                            std_q=current_std,
                            mean_p=demo_prior_actor.distribution.mean,
                            std_p=demo_prior_actor.distribution.std,
                        ).detach()

                low, high = 0.0, 1.0
                for _ in range(12):
                    middle = 0.5 * (low + high)
                    set_prior_blend(middle)
                    if float(measure_prior_kl()) > max_kl:
                        low = middle
                    else:
                        high = middle
                set_prior_blend(high)
                projection_alpha = high

            result = dict(result)
            result.update(
                {
                    "demo": float(filtered_demo.detach()),
                    "demo_kl": float(measure_prior_kl())
                    if projection_alpha > 0.0
                    else prior_kl_before_projection,
                    "demo_kl_pre_projection": prior_kl_before_projection,
                    "demo_projection_alpha": projection_alpha,
                    "demo_weight_fraction": float(weights.mean()),
                }
            )
            return result

        runner.alg.update = update_with_demo_regularization
        print(
            "[PPO3-DEMO] filtered policy-state DLS labels active: "
            f"lambda={a.demo_lambda:g} beta={a.demo_kl_beta:g} "
            f"max_kl={a.demo_max_kl:g}; teacher execution disabled",
            flush=True,
        )

    if a.latent_teacher_regularized:
        from verge_task.demo_regularized import gaussian_kl

        if int(raw.cfg.observation_space) < 2:
            raise RuntimeError("latent-teacher training requires two latent columns")
        if str(getattr(raw.cfg, "online_aam_latent_mode", "")) != "estimated":
            raise RuntimeError("latent-teacher controller must remain in estimated mode")
        if str(getattr(raw.cfg, "online_aam_policy_latent_mode", "")) != "estimated":
            raise RuntimeError("latent-teacher student must observe estimated z_hat")
        if not hasattr(raw, "_aam_privileged_z"):
            raise RuntimeError("environment has no privileged latent label buffer")

        # One frozen actor serves both roles: with true z it is the privileged
        # action teacher; with estimated z it is the trust-region prior.  It is
        # copied before any student update and never executes in the plant.
        latent_teacher_actor = copy.deepcopy(runner.alg.actor).to(a.device)
        latent_teacher_actor.eval().requires_grad_(False)
        latent_teacher_z_steps: list[torch.Tensor] = []

        act_without_latent_teacher = runner.alg.act

        def act_with_latent_teacher_labels(obs):
            label = raw._aam_privileged_z.detach().clone()
            if label.shape != (int(raw.num_envs), 2) or not bool(
                torch.isfinite(label).all()
            ):
                raise RuntimeError(
                    f"invalid privileged latent label shape/value: {tuple(label.shape)}"
                )
            latent_teacher_z_steps.append(label)
            return act_without_latent_teacher(obs)

        runner.alg.act = act_with_latent_teacher_labels
        update_without_latent_teacher = runner.alg.update

        def update_with_latent_teacher(*args, **kwargs):
            storage = runner.alg.storage
            rollout_observations = storage.observations.clone()
            true_z_steps = list(latent_teacher_z_steps)
            latent_teacher_z_steps.clear()

            result = update_without_latent_teacher(*args, **kwargs)
            if (
                len(true_z_steps) != int(storage.num_transitions_per_env)
                or not any(
                    parameter.requires_grad
                    for parameter in runner.alg.actor.parameters()
                )
            ):
                return result

            observations = rollout_observations.flatten(0, 1)
            privileged_z = torch.stack(true_z_steps, dim=0).flatten(0, 1)
            student_policy_obs = observations["policy"]
            if student_policy_obs.shape[-1] != int(raw.cfg.observation_space):
                raise RuntimeError("latent-teacher observation width changed")
            teacher_observations = observations.clone()
            teacher_observations["policy"] = torch.cat(
                (student_policy_obs[..., :-2], privileged_z), dim=-1
            )

            with torch.no_grad():
                latent_teacher_actor(
                    teacher_observations, stochastic_output=True
                )
                teacher_mean = latent_teacher_actor.distribution.mean.detach().clone()
                latent_teacher_actor(observations, stochastic_output=True)
                prior_mean = latent_teacher_actor.distribution.mean.detach().clone()
                prior_std = latent_teacher_actor.distribution.std.detach().clone()

            runner.alg.actor(observations, stochastic_output=True)
            student_mean = runner.alg.actor.distribution.mean
            student_std = runner.alg.actor.distribution.std
            imitation = ((student_mean - teacher_mean) ** 2).mean()
            prior_kl = gaussian_kl(
                mean_q=student_mean,
                std_q=student_std,
                mean_p=prior_mean,
                std_p=prior_std,
            )
            regularizer = (
                float(a.latent_teacher_lambda) * imitation
                + float(a.latent_teacher_kl_beta) * prior_kl
            )
            runner.alg.optimizer.zero_grad()
            regularizer.backward()
            torch.nn.utils.clip_grad_norm_(
                runner.alg.actor.parameters(), runner.alg.max_grad_norm
            )
            runner.alg.optimizer.step()

            pre_projection_kl = float(prior_kl.detach())
            projection_alpha = 0.0
            max_kl = float(a.latent_teacher_max_kl)
            current_parameters = {
                name: parameter.detach().clone()
                for name, parameter in runner.alg.actor.named_parameters()
            }
            prior_parameters = dict(latent_teacher_actor.named_parameters())

            def set_prior_blend(alpha: float) -> None:
                with torch.no_grad():
                    for name, parameter in runner.alg.actor.named_parameters():
                        parameter.copy_(
                            (1.0 - alpha) * current_parameters[name]
                            + alpha * prior_parameters[name]
                        )

            def measure_prior_kl() -> torch.Tensor:
                with torch.no_grad():
                    runner.alg.actor(observations, stochastic_output=True)
                    current_mean = runner.alg.actor.distribution.mean
                    current_std = runner.alg.actor.distribution.std
                    latent_teacher_actor(observations, stochastic_output=True)
                    return gaussian_kl(
                        mean_q=current_mean,
                        std_q=current_std,
                        mean_p=latent_teacher_actor.distribution.mean,
                        std_p=latent_teacher_actor.distribution.std,
                    ).detach()

            post_step_kl = float(measure_prior_kl())
            if max_kl > 0.0 and post_step_kl > max_kl:
                low, high = 0.0, 1.0
                for _ in range(12):
                    middle = 0.5 * (low + high)
                    set_prior_blend(middle)
                    if float(measure_prior_kl()) > max_kl:
                        low = middle
                    else:
                        high = middle
                set_prior_blend(high)
                projection_alpha = high

            result = dict(result)
            result.update(
                {
                    "latent_teacher_imitation": float(imitation.detach()),
                    "latent_teacher_prior_kl": (
                        float(measure_prior_kl())
                        if projection_alpha > 0.0
                        else post_step_kl
                    ),
                    "latent_teacher_prior_kl_pre_regularizer": pre_projection_kl,
                    "latent_teacher_projection_alpha": projection_alpha,
                    "latent_teacher_z_mae": float(
                        (student_policy_obs[..., -2:] - privileged_z)
                        .abs()
                        .mean()
                    ),
                    "latent_teacher_action_delta": float(
                        (prior_mean - teacher_mean).abs().mean()
                    ),
                }
            )
            return result

        runner.alg.update = update_with_latent_teacher
        print(
            "[PPO-ID-AAM-FD-LATENT-TEACHER] frozen true-z action labels active: "
            f"lambda={a.latent_teacher_lambda:g} "
            f"beta={a.latent_teacher_kl_beta:g} "
            f"max_kl={a.latent_teacher_max_kl:g}; "
            "student/controller remain estimated, teacher never executes",
            flush=True,
        )

    def critic_warmup(freeze: int) -> None:
        """Adapt the new-speed critic before allowing actor updates."""

        schedule = runner.alg.schedule
        base_lr = float(runner.alg.learning_rate)
        runner.alg.schedule = "fixed"
        runner.alg.actor.requires_grad_(False)
        for module in observation_normalizers():
            module.eval()
        print(
            f"[PPO3-WARMUP] actor frozen for {freeze} critic-only iterations",
            flush=True,
        )
        runner.learn(num_learning_iterations=freeze, init_at_random_ep_len=False)
        warmup_iteration = int(runner.current_learning_iteration)
        warmup_checkpoint = log_dir / "warmup_checkpoint.pt"
        resume_iteration = warmup_iteration + 1
        runner.current_learning_iteration = resume_iteration
        try:
            runner.save(
                str(warmup_checkpoint),
                infos={
                    "checkpoint_kind": "critic_warmup_complete",
                    "warmup_iterations": int(freeze),
                    "warmup_actor_frozen": True,
                    "warmup_iteration": int(resume_iteration),
                    "vehicle_speed_min_kmh": float(raw.cfg.vehicle_speed_min_kmh),
                    "vehicle_speed_max_kmh": float(raw.cfg.vehicle_speed_max_kmh),
                },
            )
        finally:
            runner.current_learning_iteration = warmup_iteration
        (log_dir / "warmup_checkpoint.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(warmup_checkpoint),
                    "checkpoint_kind": "critic_warmup_complete",
                    "warmup_iterations": int(freeze),
                    "warmup_actor_frozen": True,
                    "warmup_iteration": int(resume_iteration),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            )
            + "\n"
        )
        runner.alg.actor.requires_grad_(True)
        if a.obs_normalizer == "adapt":
            for module in observation_normalizers():
                module.train()
        runner.alg.schedule = schedule
        runner.alg.learning_rate = base_lr
        for param_group in runner.alg.optimizer.param_groups:
            param_group["lr"] = base_lr
        print(
            f"[PPO3-WARMUP] saved {warmup_checkpoint}; actor unfrozen and "
            f"learning rate reset to {base_lr:.6g}",
            flush=True,
        )

    final_stage = stages[-1]
    last_convergence_result = None
    for stage in stages:
        apply_terrain_curriculum_stage(raw.cfg, stage)
        apply_plant_overrides(raw.cfg)
        configure_path_horizon(raw.cfg)
        raw.cfg.trajectory_shape = str(raw.cfg.terrain_deployment_trajectory_shape)
        raw.cfg.randomize_trajectory_shape = False
        raw.cfg.randomize_start_phase = False
        print(
            f"[PPO3-VERGE] === stage {stage} === "
            f"truck={raw.cfg.vehicle_speed_min_kmh:.2f}-"
            f"{raw.cfg.vehicle_speed_max_kmh:.2f} km/h per episode "
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
        warmed_up = 0
        if stage == stages[0] and a.freeze_actor_iterations > 0:
            warmed_up = int(a.freeze_actor_iterations)
            if warmed_up > int(a.stage_iterations):
                raise RuntimeError(
                    "--freeze_actor_iterations cannot exceed --stage_iterations"
                )
            critic_warmup(warmed_up)
        if not (a.until_converged and stage == final_stage):
            remaining = int(a.stage_iterations) - warmed_up
            if remaining > 0:
                runner.learn(
                    num_learning_iterations=remaining,
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
            "teacher_enabled_during_ppo": bool(a.demo_regularized),
            "teacher_labels_enabled": bool(a.demo_regularized),
            "teacher_execution_during_ppo": False,
            "demo_regularized": bool(a.demo_regularized),
            "demo_lambda": float(a.demo_lambda) if a.demo_regularized else None,
            "demo_kl_beta": float(a.demo_kl_beta) if a.demo_regularized else None,
            "demo_max_kl": float(a.demo_max_kl) if a.demo_regularized else None,
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
