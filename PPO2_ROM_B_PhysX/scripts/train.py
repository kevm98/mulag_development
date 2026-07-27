#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Train robust PPO2 on independently randomized ROM-B factors."""

from __future__ import annotations

import argparse
import copy
import json
import signal
import sys
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser(description=__doc__)
# Every ROM-B artifact was produced at 128 environments. See the contact-buffer
# guard below for what happens above it.
VALIDATED_MAX_NUM_ENVS = 128
p.add_argument("--num_envs", type=int, default=VALIDATED_MAX_NUM_ENVS)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--balanced_shape_rounds", type=int, default=50)
p.add_argument("--iterations_per_shape", type=int, default=2)
p.add_argument(
    "--start_curriculum_stage",
    type=int,
    choices=(1, 2, 3, 4),
    default=4,
    help="Initial ROM-B stage; Stages 1-3 are prevalidated by full audits.",
)
p.add_argument("--trajectory_shapes", default="line,arc,sine,circle,square")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--residual_scale_m", type=float, default=0.002)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.00025)
p.add_argument("--save_interval", type=int, default=10)
p.add_argument("--learning_rate", type=float, default=5.0e-6)
p.add_argument(
    "--max_stage_rounds",
    type=int,
    default=12,
    help="Stop a run that cannot advance from one curriculum stage.",
)
p.add_argument(
    "--failed_shape_iteration_multiplier",
    type=int,
    default=3,
    help="Extra training allocation for shapes that fail held-out promotion.",
)
p.add_argument(
    "--curriculum_eval_steps",
    type=int,
    default=120,
    help="Scored deterministic steps per shape and curriculum round.",
)
p.add_argument(
    "--curriculum_eval_seed_groups",
    type=int,
    default=1,
    help=(
        "Held-out seed groups evaluated per shape every round. All groups are "
        "fixed for the whole run, so widening this broadens validation without "
        "allowing seeds to be shopped. One group reproduces the legacy "
        "single-seed-per-shape behaviour, which is too narrow to declare "
        "robustness on its own."
    ),
)
p.add_argument(
    "--curriculum_eval_seed_offset",
    type=int,
    default=100000,
    help=(
        "Base offset for curriculum seeds. Must stay clear of the final-test "
        "families used by evaluate.py (300000 Monte Carlo, 400000 h sweep)."
    ),
)
p.add_argument(
    "--remediation",
    action="store_true",
    help=(
        "Start from --resume_checkpoint weights at --start_curriculum_stage "
        "without inheriting that run's curriculum counters. Required to "
        "continue from an already-'complete' run: restoring its counters would "
        "let the first passing round declare completion again."
    ),
)
p.add_argument(
    "--minimum_full_scale_rounds",
    type=int,
    default=0,
    help=(
        "Override the full-scale rounds required before completion. 0 keeps "
        "the ROMBCurriculumThresholds default."
    ),
)
p.add_argument(
    "--ppo2_a_checkpoint",
    default=str(
        ROOT / "logs/rsl_rl/ppo2_rom_residual_4cyl_joint"
        / "2026-07-21_02-09-15/model_607.pt"
    ),
)
p.add_argument("--resume_checkpoint", default="")
p.add_argument(
    "--resume_run_dir",
    default="",
    help="Interrupted run containing the curriculum history to restore.",
)
p.add_argument(
    "--command_rom_checkpoint",
    default=str(
        ROOT / "ROM-cylinders_pipeline/checkpoints"
        / "plant_reachable_v1_finetune_blend25/best_rom.pt"
    ),
)
p.add_argument(
    "--trajectory_library_path",
    default=str(
        ROOT / "PPO2_ROM_PhysX/models"
        / "plant_reachable_trajectory_library_v3_circle_c7_retime.json"
    ),
)
p.add_argument(
    "--train_safety_terminations",
    action="store_true",
    help=(
        "Collect PPO experience with safety violations terminating the episode "
        "per replica, matching how the evaluator scores. The default defers "
        "them, so a clearance/head/limit crossing costs only a bounded per-step "
        "penalty during training while ending the episode during evaluation. "
        "Resets stay unsynchronized, which is what previously made safety "
        "terminations stall a run."
    ),
)
p.add_argument(
    "--plant_observer",
    action="store_true",
    help=(
        "Append the plant-RESPONSE observer to the observation (42 -> 50). It is "
        "a temporal summary of commanded-vs-achieved motion, not a factor "
        "estimator: cylinder goal error also reflects trajectory, load, controller "
        "state and the policy's own action. Without some temporal signal a "
        "single-frame MLP cannot adapt to latent dynamics at all and "
        "randomization has nothing to teach. No ROM-B factor is exposed. A "
        "narrower checkpoint is zero-padded on load, so transfer is exact."
    ),
)
p.add_argument(
    "--exclude_contaminated_clearance_reward",
    action="store_true",
    help=(
        "EXPERIMENT FLAG. Removes clearance from the ROM-B reward entirely: "
        "residual_ground_penalty_weight=0, rom_b_clearance_margin_weight=0, and "
        "clearance excluded from the -20 hard-safety event. Head, cylinder-limit, "
        "truck and acceleration terms are unaffected. Motivation: the clearance "
        "measurement carries a ~0.164 m frame offset "
        "(reports/rom_b_frame_decomposition.md). This is NOT a fix -- evaluation "
        "still terminates on the contaminated signal, curriculum promotion still "
        "counts those failures, and final acceptance remains contaminated. Use it "
        "only for a clearly labelled pilot, never for final remediation."
    ),
)
p.add_argument(
    "--clearance_frame_consistent",
    action="store_true",
    help=(
        "Use the frame-consistent clearance signal: orientation-aware cutter "
        "(_compute_cutter_pos_w) against a reference derived from the live truck "
        "pose, so the truck's ride height cancels instead of leaking into the "
        "measurement. Fixes the ~0.164 m offset rather than hiding it, and unlike "
        "--exclude_contaminated_clearance_reward it also corrects termination, "
        "curriculum promotion and acceptance. Changes every clearance number, so "
        "results are NOT comparable with earlier runs."
    ),
)
p.add_argument(
    "--truck_reaction_shaping",
    type=float,
    default=0.0,
    metavar="WEIGHT",
    help=(
        "Enable the bounded truck-reaction margin penalty with this weight "
        "(negative; -2.0 is the analogue of the existing head penalty). Measured "
        "2026-07-25, truck_raw_reaction is the dominant remaining termination "
        "cause (0.0143 vs 0.0050 head alignment) and rises with plant stiffness. "
        "It is the only gate that terminates without an approach gradient, so "
        "the policy is punished for crossing a line it is never pushed away "
        "from. This adds the shoulder only: the 0.05 m termination threshold is "
        "unchanged. Default 0.0 leaves the reward bit-exact."
    ),
)
p.add_argument(
    "--privileged_oracle",
    action="store_true",
    help=(
        "DIAGNOSTIC ONLY, NEVER CERTIFIABLE. Put the true hidden factors in the "
        "actor and critic observation (42 -> 47). Deliberately breaks the "
        "factor-blind contract. Purpose: an online estimator such as AAM can "
        "only be worse than the truth, so this policy's margin over factor-blind "
        "ROM-B is a hard ceiling on AAM's margin over ROM-B. If the oracle does "
        "not beat ROM-B, no adaptation scheme can, and the AAM comparison is not "
        "worth building. Writes to its own experiment name; cannot overwrite "
        "model_53 or any ROM-B artifact."
    ),
)
p.add_argument("--transfer_only", action="store_true")
p.add_argument(
    "--curriculum_eval_only",
    action="store_true",
    help="Run one deterministic curriculum evaluation and exit without PPO updates.",
)
p.add_argument(
    "--curriculum_eval_force_zero",
    action="store_true",
    help="Evaluation-only audit of the deterministic zero-residual baseline.",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

shapes = tuple(part.strip() for part in a.trajectory_shapes.split(",") if part.strip())
if not shapes:
    p.error("at least one trajectory shape is required")
if a.balanced_shape_rounds <= 0 or a.iterations_per_shape <= 0:
    p.error("balanced rounds and iterations per shape must be positive")
if a.num_envs <= 0 or a.path_speed_mps <= 0.0 or a.learning_rate <= 0.0:
    p.error("num_envs, path speed, and learning rate must be positive")
if a.num_envs > VALIDATED_MAX_NUM_ENVS:
    # Verified 2026-07-25: --num_envs 4096 overflows the PhysX GPU contact buffer
    # from t=15 s, before a single PPO update. PhysX then *silently drops
    # contacts* -- it does not stop -- so the dynamics quietly degrade: the first
    # curriculum probes reported acc95 48-54 m/s^2 against 31-40 in the 128-env
    # audit, safety rate 0.13-0.15 against 0.01-0.03, and completion 0.85-0.87.
    # The run finally died with NaN in cylinder_q. Corrupt physics that still
    # produces plausible-looking numbers is the dangerous failure mode here, so
    # this is a hard stop rather than a warning.
    p.error(
        f"--num_envs {a.num_envs} exceeds the validated {VALIDATED_MAX_NUM_ENVS}. "
        "Larger scenes overflow the PhysX contact buffer, which silently drops "
        "contacts and corrupts the dynamics before failing with NaN. Every ROM-B "
        "artifact to date was produced at 128. Raise this ceiling only after "
        "confirming the contact buffer is sized for the new scene."
    )
if a.curriculum_eval_steps < 60:
    p.error("curriculum evaluation must contain at least 60 scored steps")
if a.max_stage_rounds < 3:
    p.error("max stage rounds must be at least 3")
if a.failed_shape_iteration_multiplier < 1:
    p.error("failed-shape iteration multiplier must be positive")
if not 0.0 < a.residual_scale_m <= 0.004:
    p.error("residual scale must be in (0, 0.004] m")
if not 0.0 <= a.cylinder4_residual_scale_m <= 0.0005:
    p.error("cylinder-4 residual scale must be in [0, 0.0005] m")
if a.transfer_only and a.curriculum_eval_only:
    p.error("--transfer_only and --curriculum_eval_only are mutually exclusive")
if a.curriculum_eval_force_zero and not a.curriculum_eval_only:
    p.error("--curriculum_eval_force_zero requires --curriculum_eval_only")
if a.resume_run_dir and not a.resume_checkpoint:
    p.error("--resume_run_dir requires --resume_checkpoint")
if a.curriculum_eval_seed_groups < 1:
    p.error("curriculum evaluation needs at least one seed group")
if a.minimum_full_scale_rounds < 0:
    p.error("minimum full-scale rounds must be non-negative")
if a.remediation and not a.resume_checkpoint:
    p.error("--remediation requires --resume_checkpoint")
if a.remediation and a.resume_run_dir:
    p.error("--remediation deliberately ignores curriculum history; drop --resume_run_dir")
if a.curriculum_eval_seed_offset in (300000, 400000):
    p.error(
        "curriculum seeds must not reuse the final-test seed families "
        "(300000 Monte Carlo, 400000 h sweep)"
    )
if a.exclude_contaminated_clearance_reward and a.clearance_frame_consistent:
    p.error(
        "--exclude_contaminated_clearance_reward hides the contaminated signal; "
        "--clearance_frame_consistent corrects it. Use one or the other."
    )
if a.truck_reaction_shaping > 0.0:
    p.error("--truck_reaction_shaping is a penalty weight and must be <= 0")
if a.privileged_oracle and a.plant_observer:
    p.error(
        "--privileged_oracle already supplies the true factors; the observer's "
        "inferred features would confound the upper bound. Use one or the other."
    )
if a.train_safety_terminations:
    # Verified 2026-07-25: this crashes partway through the first PPO round with
    # "Residual-policy resets are synchronized; physical preposition received
    # 1 of N environments" (PPO2_ROM_PhysX/policy/env.py:684). The parent's
    # `_run_reset_preposition` runs the validated cosine deployment for the whole
    # batch and rejects a partial reset, so per-replica safety terminations are
    # not supported by the environment as it stands. Synchronizing resets instead
    # would make one replica's violation reset all of them, which is the stall the
    # deferral was introduced to avoid. Closing the train/evaluate cost-structure
    # gap therefore needs the parent reset path to support partial physical
    # prepositioning - a shared-environment change, out of ROM-B scope.
    p.error(
        "--train_safety_terminations is not supported by the current environment: "
        "the parent requires synchronized resets for physical prepositioning "
        "(policy/env.py:684). Fix the parent reset path before enabling this."
    )

_STOP_REQUESTED = False


def _request_graceful_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(
        f"[PPO2-ROM-B] received signal={signum}; "
        "will stop at the next safe boundary",
        flush=True,
    )


for _signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
    if hasattr(signal, _signal_name):
        signal.signal(getattr(signal, _signal_name), _request_graceful_stop)

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import PPO2_ROM_B_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_B_PhysX.env import PRIVILEGED_OBS_DIM
from PPO2_ROM_B_PhysX.env_cfg import PPO2ROMBPPORunnerCfg
from PPO2_ROM_B_PhysX.runtime import (
    PLANT_LAG_OBS_DIM,
    SUPPORTED_FACTOR_NAMES,
    pad_state_dict_for_wider_observation,
    ROMBCurriculumController,
    ROMBCurriculumThresholds,
    held_out_evaluation_seed,
    run_deterministic_probe,
)
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage
from PPO2_ROM_PhysX.policy_4cyl.checkpoint_transfer import configure_joint_finetune


def _round_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        raise ValueError("curriculum round contains no shape metrics")
    return {
        "max_shape_mean_error_m": max(float(row["mean_error_m"]) for row in rows),
        "max_shape_p95_error_m": max(float(row["p95_error_m"]) for row in rows),
        "max_shape_p95_acceleration_mps2": max(
            float(row["p95_acceleration_mps2"]) for row in rows
        ),
        "min_head_alignment": min(
            float(row["min_head_alignment"]) for row in rows
        ),
        "max_clearance_error_m": max(
            float(row["max_clearance_error_m"]) for row in rows
        ),
        "max_post_constraint_truck_drift_m": max(
            float(row["max_post_constraint_truck_drift_m"]) for row in rows
        ),
        "max_termination_rate": max(
            float(row["max_termination_rate"]) for row in rows
        ),
    }


def _gate_score(metrics: dict[str, float], thresholds) -> tuple[int, float, float]:
    violations = (
        max(0.0, metrics["max_shape_mean_error_m"] / thresholds.max_shape_mean_error_m - 1.0),
        max(0.0, metrics["max_shape_p95_error_m"] / thresholds.max_shape_p95_error_m - 1.0),
        max(0.0, metrics["max_shape_p95_acceleration_mps2"] / thresholds.max_shape_p95_acceleration_mps2 - 1.0),
        max(0.0, thresholds.min_head_alignment - metrics["min_head_alignment"]),
        max(0.0, metrics["max_clearance_error_m"] / thresholds.max_clearance_error_m - 1.0),
        max(0.0, metrics["max_post_constraint_truck_drift_m"] / thresholds.max_post_constraint_truck_drift_m - 1.0),
        max(0.0, metrics["max_termination_rate"] / thresholds.max_termination_rate - 1.0),
    )
    return (
        sum(value > 0.0 for value in violations),
        float(sum(violations)),
        float(metrics["max_shape_mean_error_m"]),
    )


def _failed_shapes(rows: list[dict[str, object]], thresholds) -> set[str]:
    failed: set[str] = set()
    for row in rows:
        metrics = _round_metrics([row])
        if _gate_score(metrics, thresholds)[0] > 0:
            failed.add(str(row["shape"]))
    return failed


def _restore_algorithm_state(runner, state: dict[str, object]) -> None:
    # RSL-RL updates normalization buffers while collecting under
    # torch.inference_mode(). Restore those inference tensors in the same mode.
    with torch.inference_mode():
        runner.alg.actor.load_state_dict(state["actor_state_dict"])
        runner.alg.critic.load_state_dict(state["critic_state_dict"])
    runner.alg.optimizer.load_state_dict(state["optimizer_state_dict"])


def _candidate_improves_or_preserves_acceptance(
    candidate: tuple[int, float, float],
    accepted: tuple[int, float, float],
    *,
    tracking_tolerance_m: float = 0.002,
) -> bool:
    """Accept better gates or a passing candidate within tracking tolerance."""

    if candidate[:2] < accepted[:2]:
        return True
    if candidate[0] == 0 and accepted[0] == 0:
        return candidate[2] <= accepted[2] + float(tracking_tolerance_m)
    return candidate < accepted


def _evaluate_curriculum(raw, env, runner, *, round_index: int) -> list[dict[str, object]]:
    deterministic_policy = runner.get_inference_policy(device=raw.device)
    rows: list[dict[str, object]] = []
    for group in range(int(a.curriculum_eval_seed_groups)):
        for shape_index, shape in enumerate(shapes):
            # Groups are offset by whole shape blocks, so every (group, shape)
            # pair gets a distinct seed that is fixed for the entire run.
            evaluation_seed = held_out_evaluation_seed(
                int(a.seed),
                group * len(shapes) + shape_index,
                offset=int(a.curriculum_eval_seed_offset),
            )
            row = run_deterministic_probe(
                raw,
                env,
                deterministic_policy,
                shape=shape,
                seed=evaluation_seed,
                scored_steps=int(a.curriculum_eval_steps),
                force_zero_policy=bool(a.curriculum_eval_force_zero),
            )
            row["seed_group"] = group + 1
            rows.append(row)
            print(
                f"[PPO2-ROM-B EVAL] round={round_index + 1} "
                f"group={group + 1}/{a.curriculum_eval_seed_groups} "
                f"shape={shape} seed={evaluation_seed} "
                f"mean={float(row['mean_error_m']):.5f}m "
                f"p95={float(row['p95_error_m']):.5f}m "
                f"acc95={float(row['p95_acceleration_mps2']):.2f}m/s2 "
                f"head={float(row['min_head_alignment']):.5f} "
                f"clearance={float(row['max_clearance_error_m']):.5f}m "
                f"safety_rate={float(row['max_termination_rate']):.5f} "
                f"completion={float(row['completion_fraction']):.3f}",
                flush=True,
            )
    return rows


def main() -> None:
    task = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    # The plant-lag observer appends 8 inferred dimensions. Set the flag before
    # the observation width so both agree at environment construction.
    cfg.rom_b_plant_observer = bool(a.plant_observer)
    cfg.clearance_frame_consistent = bool(a.clearance_frame_consistent)
    if a.clearance_frame_consistent:
        print(
            "[PPO2-ROM-B] frame-consistent clearance ENABLED: reward, termination "
            "and acceptance all use the corrected signal. Clearance numbers are "
            "not comparable with runs before 2026-07-25.",
            flush=True,
        )
    if a.exclude_contaminated_clearance_reward:
        cfg.residual_ground_penalty_weight = 0.0
        cfg.rom_b_clearance_margin_weight = 0.0
        cfg.rom_b_clearance_in_hard_safety_event = False
        print(
            "[PPO2-ROM-B] PILOT: clearance excluded from the ROM-B reward "
            "(ground penalty 0, margin 0, out of the hard event). Evaluation, "
            "curriculum promotion and acceptance still use the contaminated "
            "signal -- results are not final-grade.",
            flush=True,
        )
    cfg.rom_b_truck_reaction_margin_weight = float(a.truck_reaction_shaping)
    if a.truck_reaction_shaping != 0.0:
        print(
            "[PPO2-ROM-B] truck-reaction shaping ENABLED: weight "
            f"{float(a.truck_reaction_shaping):.2f}, soft "
            f"{float(cfg.rom_b_truck_reaction_soft_m):.3f} m -> hard "
            f"{float(cfg.rom_b_truck_reaction_hard_m):.3f} m. The 0.05 m "
            "termination threshold is UNCHANGED; only the approach gradient is "
            "new.",
            flush=True,
        )
    cfg.rom_b_privileged_factors = bool(a.privileged_oracle)
    cfg.observation_space = (
        42 + (8 if a.plant_observer else 0) + (5 if a.privileged_oracle else 0)
    )
    cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).resolve())
    cfg.trajectory_library_path = str(Path(a.trajectory_library_path).resolve())
    cfg.trajectory_shapes = shapes
    cfg.trajectory_shape = shapes[0]
    cfg.randomize_trajectory_shape = False
    cfg.procedural_trajectory_probability = 0.0
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.residual_scale_m = float(a.residual_scale_m)
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.command_rom_direction_guard = True
    # The deterministic probe overrides both of these for scoring and restores
    # them afterwards, so this setting only governs PPO experience collection.
    cfg.defer_safety_terminations = not bool(a.train_safety_terminations)
    cfg.synchronize_resets = not bool(a.train_safety_terminations)
    cfg.rom_b_mode = "random"
    thresholds = ROMBCurriculumThresholds()
    if int(a.minimum_full_scale_rounds) > 0:
        thresholds = replace(
            thresholds, minimum_full_scale_rounds=int(a.minimum_full_scale_rounds)
        )
    curriculum = ROMBCurriculumController(
        stage=int(a.start_curriculum_stage), thresholds=thresholds
    )
    cfg.rom_b_randomization_scale = curriculum.scale
    resume_history: list[dict[str, object]] = []
    resume_run_dir: Path | None = None
    if a.remediation:
        # Weights only. Counters start at zero so the run must earn completion
        # again against the (broader) curriculum seed groups.
        print(
            "[PPO2-ROM-B] remediation: loading weights from "
            f"{Path(a.resume_checkpoint).resolve()} at stage "
            f"{curriculum.stage}; curriculum counters reset to zero",
            flush=True,
        )
    elif a.resume_checkpoint:
        resume_run_dir = (
            Path(a.resume_run_dir).expanduser().resolve()
            if a.resume_run_dir
            else Path(a.resume_checkpoint).expanduser().resolve().parent
        )
        history_path = resume_run_dir / "rom_b_curriculum_history.json"
        if not history_path.is_file():
            raise FileNotFoundError(
                f"resume curriculum history is missing: {history_path}"
            )
        payload = json.loads(history_path.read_text())
        if not isinstance(payload, list) or not payload:
            raise ValueError("resume curriculum history must be a non-empty list")
        resume_history = payload
        last_decision = resume_history[-1]["decision"]
        if int(last_decision["stage"]) != curriculum.stage:
            raise ValueError(
                "resume stage mismatch: history has "
                f"{last_decision['stage']}, CLI selected {curriculum.stage}"
            )
        curriculum.restore_progress(
            stage_rounds=int(last_decision["stage_rounds"]),
            consecutive_passes=int(last_decision["consecutive_passes"]),
        )

    total_iterations = int(a.balanced_shape_rounds) * int(a.iterations_per_shape) * (
        len(shapes) - 1 + int(a.failed_shape_iteration_multiplier)
    )
    agent_cfg = PPO2ROMBPPORunnerCfg()
    agent_cfg.seed = int(a.seed)
    agent_cfg.device = a.device
    agent_cfg.max_iterations = total_iterations
    agent_cfg.save_interval = int(a.save_interval)
    agent_cfg.algorithm.learning_rate = float(a.learning_rate)
    if a.privileged_oracle:
        # Physically separate tree: an oracle checkpoint must never be reachable
        # by anything that expects a factor-blind ROM-B policy.
        agent_cfg.experiment_name = "ppo2_rom_b_privileged_oracle"
    log_dir = (
        ROOT / "logs/rsl_rl" / agent_cfg.experiment_name
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(log_dir)
    source_checkpoint = a.resume_checkpoint or a.ppo2_a_checkpoint
    run_contract = {
        "variant": "PPO2-ROM_B",
        "base_policy": str(Path(a.ppo2_a_checkpoint).resolve()),
        "resume_policy": (
            str(Path(a.resume_checkpoint).resolve()) if a.resume_checkpoint else None
        ),
        "resume_run_dir": str(resume_run_dir) if resume_run_dir else None,
        "restored_curriculum_rounds": len(resume_history),
        "restored_stage_rounds": curriculum.stage_rounds,
        "restored_consecutive_passes": curriculum.consecutive_passes,
        "command_rom": cfg.command_rom_checkpoint,
        "observations": int(cfg.observation_space),
        "plant_observer": bool(a.plant_observer),
        "actions": 4,
        # Hardcoding these True was wrong once --privileged_oracle existed: the
        # oracle run recorded itself as factor-blind while feeding the true
        # factors to both networks.
        "actor_factor_blind": not bool(a.privileged_oracle),
        "critic_factor_blind": not bool(a.privileged_oracle),
        "factor_sampling": "independent_uniform_per_environment_at_reset",
        "curriculum": curriculum.contract(),
        "initial_curriculum_stage": curriculum.stage,
        "initial_randomization_scale": curriculum.scale,
        "nominal_stage_prevalidation": {
            "status": "passed_three_consecutive_full_audits",
            "checkpoint": str(Path(a.ppo2_a_checkpoint).resolve()),
            "scored_steps_per_shape": 120,
            "environments_per_shape": 128,
            "artifacts": [
                "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-02-18/rom_b_curriculum_preflight.json",
                "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-06-42/rom_b_curriculum_preflight.json",
                "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-10-50/rom_b_curriculum_preflight.json",
            ],
        },
        "prevalidated_randomization_stages": {
            "stage_2_scale_0.4": "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-16-15/rom_b_curriculum_preflight.json",
            "stage_3_scale_0.7": "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-20-29/rom_b_curriculum_preflight.json",
        },
        "first_unaccepted_stage": {
            "stage": 4,
            "scale": 1.0,
            "artifact": "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_11-24-44/rom_b_curriculum_preflight.json",
            "failed_shape": "line",
            "termination_rate": 0.0625,
        },
        "curriculum_evaluation": {
            "policy": "deterministic_actor_mean",
            "source": "dedicated_post_round_rollout",
            "scored_steps_per_shape": int(a.curriculum_eval_steps),
            "environments_per_shape": int(a.num_envs),
            "seed_offset": int(a.curriculum_eval_seed_offset),
            "seed_groups_per_shape": int(a.curriculum_eval_seed_groups),
            "rows_per_round": int(a.curriculum_eval_seed_groups) * len(shapes),
            "seed_policy": "fixed_held_out_per_shape_and_group_across_all_rounds",
            "safety_resets": "captured_without_automatic_environment_reset",
            "final_test_seed_families_excluded": [300000, 400000],
        },
        "remediation": bool(a.remediation),
        "exclude_contaminated_clearance_reward": bool(
            a.exclude_contaminated_clearance_reward
        ),
        "clearance_frame_consistent": bool(a.clearance_frame_consistent),
        "privileged_oracle": bool(a.privileged_oracle),
        "factor_blind": not bool(a.privileged_oracle),
        "certifiable": not bool(a.privileged_oracle),
        "truck_reaction_margin_weight": float(a.truck_reaction_shaping),
        "truck_reaction_soft_m": float(cfg.rom_b_truck_reaction_soft_m),
        "truck_reaction_hard_m": float(cfg.rom_b_truck_reaction_hard_m),
        "clearance_reward_terms": {
            "residual_ground_penalty_weight": float(cfg.residual_ground_penalty_weight),
            "rom_b_clearance_margin_weight": float(cfg.rom_b_clearance_margin_weight),
            "rom_b_clearance_in_hard_safety_event": bool(
                cfg.rom_b_clearance_in_hard_safety_event
            ),
        },
        "supported_factor_set": list(SUPPORTED_FACTOR_NAMES),
        "collection_safety_terminations": bool(a.train_safety_terminations),
        "collection_defer_safety_terminations": bool(cfg.defer_safety_terminations),
        "collection_synchronize_resets": bool(cfg.synchronize_resets),
        "curriculum_thresholds": asdict(curriculum.thresholds),
        "max_stage_rounds": int(a.max_stage_rounds),
        "failed_shape_iteration_multiplier": int(
            a.failed_shape_iteration_multiplier
        ),
        "checkpoint_selection": "restore_best_gate_score_after_regression",
        "factor_order": list(SUPPORTED_FACTOR_NAMES),
        "factor_ranges": {
            "e_comp": [-0.125, 0.020],
            "d_seal_j": [0.0, 1.0],
        },
        "nominal_factors": [0.0] * len(SUPPORTED_FACTOR_NAMES),
        "health_scale": {
            "range": [0.4, 1.2],
            "role": "correlated_evaluation_coordinate_only",
            "nominal": 1.0,
        },
        "excluded_effects": [
            "pressure_limit", "valve_time_constant", "mechanical_clearance",
        ],
        "shapes": list(shapes),
        "iterations": total_iterations,
        "learning_rate": float(a.learning_rate),
        "residual_scale_m": float(a.residual_scale_m),
        "cylinder4_residual_scale_m": float(a.cylinder4_residual_scale_m),
        "seed": int(a.seed),
    }
    (log_dir / "ppo2_rom_b_run_contract.json").write_text(
        json.dumps(run_contract, indent=2) + "\n"
    )
    print(
        f"[PPO2-ROM-B] iterations={total_iterations} envs={a.num_envs} "
        f"obs={cfg.observation_space} actions=4 shapes={','.join(shapes)} "
        f"log={log_dir}",
        flush=True,
    )

    gym_env = gym.make(task, cfg=cfg)
    try:
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
        load_path = Path(source_checkpoint).resolve()
        if a.privileged_oracle or a.plant_observer:
            # Both widen the observation, so the source checkpoint is zero-padded
            # into the wider model. The padded columns start at zero *and* their
            # inputs start at zero, so the widened policy is numerically
            # identical to the source at iteration 0; training then decides
            # whether the new inputs earn their place.
            #
            # For the oracle this matters for fairness, not convenience.
            # `model_53` was itself a fine-tune of PPO2_A, so an oracle trained
            # from scratch would differ from ROM-B in initialization *and*
            # budget as well as in observation content, and any gap would be
            # uninterpretable. Same start, same budget, one variable.
            added = PRIVILEGED_OBS_DIM if a.privileged_oracle else PLANT_LAG_OBS_DIM
            payload = torch.load(load_path, map_location="cpu", weights_only=False)
            targets = {
                "actor_state_dict": runner.alg.actor.state_dict(),
                "critic_state_dict": runner.alg.critic.state_dict(),
            }
            adaptation = {}
            for key, target in targets.items():
                source = payload.get(key)
                if source is None:
                    continue
                payload[key], adaptation[key] = pad_state_dict_for_wider_observation(
                    source, target, added_dims=added
                )
            if a.resume_checkpoint:
                # Optimizer moments are indexed by parameter shape; a widened
                # first layer invalidates them.
                payload.pop("optimizer_state_dict", None)
            load_path = log_dir / "widened_source_checkpoint.pt"
            torch.save(payload, load_path)
            run_contract["widened_observation_transfer"] = adaptation
            label = (
                "privileged oracle" if a.privileged_oracle else "plant observer"
            )
            print(
                f"[PPO2-ROM-B] {label} ON: obs 42 -> {42 + added}; source "
                f"checkpoint zero-padded and written to {load_path}",
                flush=True,
            )
        runner.load(
            str(load_path),
            load_cfg={
                "actor": True,
                "critic": True,
                # Remediation restarts optimization: fresh moments and a
                # fresh iteration counter, weights only.
                "optimizer": (
                    bool(a.resume_checkpoint)
                    and not a.remediation
                    and not a.plant_observer
                    and not a.privileged_oracle
                ),
                "iteration": bool(a.resume_checkpoint) and not a.remediation,
                "rnd": False,
            },
        )
        optimization_contract = configure_joint_finetune(runner)
        run_contract["optimization_contract"] = optimization_contract
        (log_dir / "ppo2_rom_b_run_contract.json").write_text(
            json.dumps(run_contract, indent=2) + "\n"
        )
        print(
            "[PPO2-ROM-B] initialized="
            + str(Path(source_checkpoint).resolve())
            + (" (zero-padded 42 -> 47 for the oracle)" if a.privileged_oracle else "")
            + f" optimizer={'restored' if (a.resume_checkpoint and not a.remediation and not a.plant_observer and not a.privileged_oracle) else 'fresh'} "
            f"iteration={runner.current_learning_iteration} "
            "actor=joint_trainable exploration_std=frozen",
            flush=True,
        )
        if a.transfer_only:
            print("[PPO2-ROM-B] transfer-only complete; PPO was not started", flush=True)
            return
        if a.curriculum_eval_only:
            rows = _evaluate_curriculum(
                gym_env.unwrapped, env, runner, round_index=0
            )
            preflight = {
                "policy": str(Path(source_checkpoint).resolve()),
                "stage": curriculum.stage,
                "scale": curriculum.scale,
                "metrics": rows,
                "aggregate": _round_metrics(rows),
            }
            (log_dir / "rom_b_curriculum_preflight.json").write_text(
                json.dumps(preflight, indent=2) + "\n"
            )
            print(
                "[PPO2-ROM-B] deterministic curriculum preflight complete; "
                "PPO was not started",
                flush=True,
            )
            return

        completed = len(resume_history) * len(shapes)
        curriculum_history: list[dict[str, object]] = copy.deepcopy(resume_history)
        curriculum_complete = False
        curriculum_stalled = False
        curriculum_interrupted = False
        accepted_rows = _evaluate_curriculum(
            gym_env.unwrapped, env, runner, round_index=-1
        )
        accepted_aggregate = _round_metrics(accepted_rows)
        accepted_score = _gate_score(accepted_aggregate, curriculum.thresholds)
        accepted_state = copy.deepcopy(runner.alg.save())
        focus_shapes = _failed_shapes(accepted_rows, curriculum.thresholds)
        initial_evaluation = {
            "stage": curriculum.stage,
            "scale": curriculum.scale,
            "metrics": accepted_rows,
            "aggregate": accepted_aggregate,
            "gate_score": list(accepted_score),
            "focus_shapes": sorted(focus_shapes),
        }
        (log_dir / "rom_b_initial_evaluation.json").write_text(
            json.dumps(initial_evaluation, indent=2) + "\n"
        )
        print(
            f"[PPO2-ROM-B] initial gate score={accepted_score} "
            f"focus_shapes={sorted(focus_shapes)}",
            flush=True,
        )
        for round_index in range(
            len(resume_history), int(a.balanced_shape_rounds)
        ):
            ordered = shapes if round_index % 2 == 0 else tuple(reversed(shapes))
            stage_before = curriculum.stage
            scale_before = curriculum.scale
            for shape in ordered:
                if _STOP_REQUESTED:
                    break
                shape_iterations = int(a.iterations_per_shape) * (
                    int(a.failed_shape_iteration_multiplier)
                    if shape in focus_shapes else 1
                )
                gym_env.unwrapped.set_trajectory_shape(shape)
                with torch.inference_mode():
                    gym_env.unwrapped.reset(
                        seed=int(a.seed) + round_index * len(shapes) + completed
                    )
                if completed:
                    runner.current_learning_iteration += 1
                print(
                    f"[PPO2-ROM-B] round={round_index + 1}/"
                    f"{a.balanced_shape_rounds} shape={shape} "
                    f"iterations={shape_iterations} focused={shape in focus_shapes}",
                    flush=True,
                )
                runner.learn(
                    num_learning_iterations=shape_iterations,
                    init_at_random_ep_len=False,
                )
                completed += 1
            if _STOP_REQUESTED:
                _restore_algorithm_state(runner, accepted_state)
                curriculum_interrupted = True
                break
            candidate_rows = _evaluate_curriculum(
                gym_env.unwrapped, env, runner, round_index=round_index
            )
            candidate_aggregate = _round_metrics(candidate_rows)
            candidate_score = _gate_score(
                candidate_aggregate, curriculum.thresholds
            )
            accepted_candidate = _candidate_improves_or_preserves_acceptance(
                candidate_score, accepted_score
            )
            if accepted_candidate:
                accepted_rows = candidate_rows
                accepted_aggregate = candidate_aggregate
                accepted_score = candidate_score
                accepted_state = copy.deepcopy(runner.alg.save())
            else:
                _restore_algorithm_state(runner, accepted_state)
            focus_shapes = _failed_shapes(accepted_rows, curriculum.thresholds)
            decision = curriculum.update(accepted_aggregate)
            record: dict[str, object] = {
                "round": int(round_index + 1),
                "checkpoint_iteration": int(runner.current_learning_iteration),
                "training_stage": int(stage_before),
                "training_scale": float(scale_before),
                "metric_source": "deterministic_actor_mean_evaluation",
                "candidate_metrics": candidate_rows,
                "candidate_aggregate": candidate_aggregate,
                "candidate_gate_score": list(candidate_score),
                "accepted_candidate": accepted_candidate,
                "rolled_back": not accepted_candidate,
                "evaluation_metrics": accepted_rows,
                "aggregate": accepted_aggregate,
                "accepted_gate_score": list(accepted_score),
                "next_focus_shapes": sorted(focus_shapes),
                "decision": {
                    "stage": decision.stage,
                    "scale": decision.scale,
                    "passed": decision.passed,
                    "advanced": decision.advanced,
                    "complete": decision.complete,
                    "failed_checks": list(decision.failed_checks),
                    "stage_rounds": decision.stage_rounds,
                    "consecutive_passes": decision.consecutive_passes,
                },
            }
            curriculum_history.append(record)
            (log_dir / "rom_b_curriculum_history.json").write_text(
                json.dumps(curriculum_history, indent=2) + "\n"
            )
            print(
                f"[PPO2-ROM-B] curriculum round={round_index + 1} "
                f"trained_stage={stage_before} scale={scale_before:.2f} "
                f"passed={decision.passed} failed={list(decision.failed_checks)} "
                f"next_stage={decision.stage} next_scale={decision.scale:.2f} "
                f"consecutive={decision.consecutive_passes}",
                flush=True,
            )
            if accepted_candidate:
                accepted_path = log_dir / (
                    f"model_{runner.current_learning_iteration}_accepted.pt"
                )
                runner.save(
                    str(accepted_path),
                    infos={"curriculum_record": record},
                )
            if decision.advanced:
                gym_env.unwrapped.set_rom_b_randomization_scale(decision.scale)
                accepted_rows = _evaluate_curriculum(
                    gym_env.unwrapped, env, runner, round_index=round_index
                )
                accepted_aggregate = _round_metrics(accepted_rows)
                accepted_score = _gate_score(
                    accepted_aggregate, curriculum.thresholds
                )
                accepted_state = copy.deepcopy(runner.alg.save())
                focus_shapes = _failed_shapes(
                    accepted_rows, curriculum.thresholds
                )
            if decision.complete:
                curriculum_complete = True
                print(
                    "[PPO2-ROM-B] full-scale acceptance gates passed; "
                    "stopping before the maximum iteration budget",
                    flush=True,
                )
                break
            if _STOP_REQUESTED:
                curriculum_interrupted = True
                break
            if (
                not decision.advanced
                and decision.stage_rounds >= int(a.max_stage_rounds)
            ):
                curriculum_stalled = True
                print(
                    f"[PPO2-ROM-B] stage={decision.stage} failed to advance "
                    f"within {a.max_stage_rounds} rounds; stopping for audit",
                    flush=True,
                )
                break
        run_contract["training_outcome"] = {
            "status": (
                "full_scale_gates_passed"
                if curriculum_complete else "maximum_round_budget_reached"
            ),
            "final_stage": curriculum.stage,
            "final_scale": curriculum.scale,
            "rounds_completed": len(curriculum_history),
            "shape_phases_completed": completed,
        }
        if curriculum_stalled:
            run_contract["training_outcome"]["status"] = "stage_stalled"
        if curriculum_interrupted:
            run_contract["training_outcome"]["status"] = "interrupted"
        (log_dir / "ppo2_rom_b_run_contract.json").write_text(
            json.dumps(run_contract, indent=2) + "\n"
        )
        final_path = log_dir / (
            f"model_{runner.current_learning_iteration}_ppo2_rom_b_final.pt"
        )
        runner.save(str(final_path), infos={"run_contract": run_contract})
        print(f"[PPO2-ROM-B] complete checkpoint={final_path}", flush=True)
    finally:
        gym_env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
