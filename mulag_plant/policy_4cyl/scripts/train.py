#!/usr/bin/env python3
"""Train the constrained four-cylinder Policy from model_249."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--warmup_iterations", type=int, default=0)
p.add_argument("--low_authority_iterations", type=int, default=0)
p.add_argument("--full_exact_iterations", type=int, default=0)
p.add_argument("--full_mixed_iterations", type=int, default=0)
p.add_argument("--balanced_shape_rounds", type=int, default=3)
p.add_argument("--iterations_per_shape", type=int, default=4)
p.add_argument("--cylinder4_low_scale_m", type=float, default=0.00025)
p.add_argument("--cylinder4_full_scale_m", type=float, default=0.00050)
p.add_argument("--procedural_probability", type=float, default=0.15)
p.add_argument("--save_interval", type=int, default=10)
p.add_argument("--trajectory_shapes", default="line,arc,sine,circle,square")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--trajectory_library_path", default="")
p.add_argument(
    "--trajectory_2d",
    action="store_true",
    help="Use the standalone line/sine_chain/circle/rounded_square task.",
)
p.add_argument(
    "--fresh_start",
    action="store_true",
    help="Start a new policy when the observation contract is wider than the legacy checkpoint.",
)
p.add_argument("--command_rom_checkpoint", default="")
p.add_argument(
    "--command_rom_direction_guard",
    action=argparse.BooleanOptionalAction,
    default=None,
)
p.add_argument("--cylinder123_std", type=float, default=0.001)
p.add_argument("--cylinder4_std", type=float, default=0.08)
p.add_argument("--refinement_cylinder4_std", type=float, default=0.03)
p.add_argument("--learning_rate", type=float, default=None)
p.add_argument(
    "--joint_finetune",
    action="store_true",
    help="Train the accepted three-action teacher and pitch branch together.",
)
p.add_argument("--early_stop_probe_interval_rounds", type=int, default=2)
p.add_argument("--early_stop_probe_steps", type=int, default=120)
p.add_argument("--early_stop_min_mean_improvement", type=float, default=0.03)
p.add_argument("--early_stop_min_p95_improvement", type=float, default=0.03)
p.add_argument("--early_stop_max_acceleration_ratio", type=float, default=1.05)
p.add_argument("--early_stop_max_shape_acceleration_ratio", type=float, default=1.10)
p.add_argument("--early_stop_max_shape_regression_m", type=float, default=0.002)
p.add_argument("--early_stop_consecutive_passes", type=int, default=2)
p.add_argument("--early_stop_min_completion_fraction", type=float, default=0.95)
p.add_argument("--disable_early_stop", action="store_true")
p.add_argument(
    "--defer_safety_resets",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Keep finite safety violations as metrics/reward signals and reset at "
        "the normal episode timeout; prevents one vectorized outlier from "
        "forcing an expensive synchronized physical reset."
    ),
)
p.add_argument(
    "--checkpoint3",
    default=str(
        ROOT
        / "logs/rsl_rl/ppo2_rom_residual/2026-07-19_18-55-23/model_249.pt"
    ),
)
p.add_argument(
    "--checkpoint4",
    default=None,
    help="Optional 42/4 checkpoint to continue with a fresh optimizer.",
)
p.add_argument("--skip_contract_check", action="store_true")
p.add_argument("--preflight_only", action="store_true")
p.add_argument(
    "--transfer_only",
    action="store_true",
    help="Build the 42/4 runner, transfer model_249, and exit without PPO updates.",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

phase_iterations = (
    int(a.warmup_iterations),
    int(a.low_authority_iterations),
    int(a.full_exact_iterations),
    int(a.full_mixed_iterations),
)
trajectory_shapes = tuple(
    part.strip() for part in a.trajectory_shapes.split(",") if part.strip()
)
balanced_iterations = (
    int(a.balanced_shape_rounds)
    * int(a.iterations_per_shape)
    * len(trajectory_shapes)
)
if any(value < 0 for value in phase_iterations):
    p.error("curriculum iteration counts must be non-negative")
if a.balanced_shape_rounds < 0 or a.iterations_per_shape <= 0:
    p.error("balanced rounds must be non-negative and iterations per shape positive")
if not trajectory_shapes:
    p.error("at least one trajectory shape is required")
if balanced_iterations <= 0 and sum(phase_iterations) <= 0:
    p.error("balanced training or the legacy curriculum must have positive iterations")
if not 0.0 < a.cylinder4_low_scale_m <= a.cylinder4_full_scale_m <= 0.0005:
    p.error("cylinder-4 authority must satisfy 0 < low <= full <= 0.0005 m")
if not 0.0 <= a.procedural_probability <= 1.0:
    p.error("--procedural_probability must be in [0, 1]")
if a.path_speed_mps <= 0.0 or a.save_interval <= 0:
    p.error("path speed and save interval must be positive")
if (
    a.cylinder123_std <= 0.0
    or a.cylinder4_std <= 0.0
    or a.refinement_cylinder4_std <= 0.0
):
    p.error("policy standard deviations must be positive")
if a.learning_rate is not None and a.learning_rate <= 0.0:
    p.error("--learning_rate must be positive")
if a.early_stop_probe_interval_rounds <= 0 or a.early_stop_probe_steps < 20:
    p.error("early-stop probe interval must be positive and steps at least 20")
if a.early_stop_consecutive_passes <= 0:
    p.error("--early_stop_consecutive_passes must be positive")
if not 0.0 <= a.early_stop_min_mean_improvement < 1.0:
    p.error("mean improvement must be in [0, 1)")
if not 0.0 <= a.early_stop_min_p95_improvement < 1.0:
    p.error("p95 improvement must be in [0, 1)")
if a.early_stop_max_acceleration_ratio <= 0.0 or a.early_stop_max_shape_acceleration_ratio <= 0.0:
    p.error("acceleration ratios must be positive")
if a.early_stop_max_shape_regression_m < 0.0:
    p.error("shape regression tolerance must be non-negative")
if not 0.0 < a.early_stop_min_completion_fraction <= 1.0:
    p.error("completion fraction must be in (0, 1]")
if a.joint_finetune and a.checkpoint4:
    p.error("joint fine-tuning must restart from --checkpoint3, not a prior four-action run")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from rrlab_compat.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import mulag_plant.tasks  # noqa: F401,E402
from mulag_plant.policy.env_cfg import apply_residual_curriculum_stage
from mulag_plant.policy_4cyl.checkpoint_transfer import (
    assert_frozen_teacher_unchanged,
    configure_joint_finetune,
    freeze_teacher_and_train_cylinder4_head,
    snapshot_frozen_teacher,
    transfer_checkpoint_into_runner,
)
from mulag_plant.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg,
    FourCylinderTrajectory2DPPORunnerCfg,
)
from mulag_plant.policy_4cyl.promotion_probe import (
    paired_promotion_probe,
)
from isaaclab_tasks.utils import parse_env_cfg


@torch.no_grad()
def _zero_action_preflight(raw) -> None:
    """Confirm that the 42/4 interface preserves the deterministic teacher."""
    raw.cfg.force_zero_policy = True
    saved = (
        float(raw.cfg.cutter_clearance_hard_band_m),
        float(raw.cfg.truck_drift_termination_m),
        float(raw.cfg.truck_raw_reaction_termination_m),
    )
    raw.cfg.cutter_clearance_hard_band_m = 10.0
    raw.cfg.truck_drift_termination_m = 10.0
    raw.cfg.truck_raw_reaction_termination_m = 10.0
    raw.reset(seed=int(a.seed))
    zero = torch.zeros((raw.num_envs, 4), device=raw.device)
    max_residual = 0.0
    max_clearance_error_by_env = torch.zeros(
        raw.num_envs, dtype=torch.float32, device=raw.device
    )
    for _ in range(100):
        _, _, terminated, _, extras = raw.step(zero)
        if terminated.any():
            raise RuntimeError(
                "four-cylinder zero-action preflight terminated: "
                f"{extras.get('log', {}) if isinstance(extras, dict) else extras}"
            )
        max_residual = max(max_residual, float(raw._residual_filtered.abs().max()))
        clearance_error = raw._compute_cutter_world_z() - raw._original_reference_world_z
        max_clearance_error_by_env = torch.maximum(
            max_clearance_error_by_env, clearance_error.abs()
        )
    raw.cfg.cutter_clearance_hard_band_m = saved[0]
    raw.cfg.truck_drift_termination_m = saved[1]
    raw.cfg.truck_raw_reaction_termination_m = saved[2]
    raw_drift = float(raw._max_truck_drift.max())
    retained_drift = float(raw._max_truck_post_constraint_drift.max())
    clearance_pass_fraction = float(
        (max_clearance_error_by_env <= saved[0]).float().mean()
    )
    max_clearance_error = float(max_clearance_error_by_env.max())
    p95_clearance_error = float(
        torch.quantile(max_clearance_error_by_env, 0.95)
    )
    min_pass_fraction = float(a.early_stop_min_completion_fraction)
    catastrophic_clearance = max(1.5 * saved[0], saved[0] + 0.05)
    if (
        max_residual > 1.0e-7
        or clearance_pass_fraction < min_pass_fraction
        or max_clearance_error > catastrophic_clearance
        or raw_drift > saved[2]
        or retained_drift > saved[1]
    ):
        worst = torch.topk(max_clearance_error_by_env, k=min(8, raw.num_envs))
        measured_x, _ = raw._refresh_cylinder_state()
        print(
            "[4CYL PREFLIGHT] worst_clearance_envs="
            f"{worst.indices.detach().cpu().tolist()} values="
            f"{worst.values.detach().cpu().tolist()} "
            f"measured_cylinder={measured_x[worst.indices].detach().cpu().tolist()} "
            f"goal_cylinder={raw._baseline_cylinder_goal[worst.indices].detach().cpu().tolist()} "
            f"tool={raw._compute_ee_pos_local()[worst.indices].detach().cpu().tolist()} "
            f"reference={raw._original_reference_tool[worst.indices].detach().cpu().tolist()}",
            flush=True,
        )
        raise RuntimeError(
            "four-cylinder zero-action contract failed: "
            f"residual={max_residual:.3e} "
            f"clearance_p95={p95_clearance_error:.5f}m "
            f"clearance_max={max_clearance_error:.5f}m "
            f"clearance_pass={clearance_pass_fraction:.3f}/"
            f"{min_pass_fraction:.3f} "
            f"raw_reaction={raw_drift:.5f}m retained_drift={retained_drift:.6f}m"
        )
    print(
        "[4CYL PREFLIGHT] PASS "
        f"residual={max_residual:.3e} "
        f"clearance_p95={p95_clearance_error:.5f}m "
        f"clearance_max={max_clearance_error:.5f}m "
        f"clearance_pass={clearance_pass_fraction:.3f} "
        f"raw_reaction={raw_drift:.5f}m retained_drift={retained_drift:.6f}m",
        flush=True,
    )
    raw.cfg.force_zero_policy = False
    if not bool(a.preflight_only):
        raw.reset(seed=int(a.seed))


@torch.inference_mode()
def _probe_policy(raw, env, policy, *, shape: str, seed: int) -> dict[str, float | bool]:
    """Run a short deterministic vectorized safety/tracking probe."""
    raw.set_trajectory_shape(shape)
    raw.reset(seed=int(seed))
    obs = env.get_observations()
    errors: list[torch.Tensor] = []
    accelerations: list[torch.Tensor] = []
    alignments: list[torch.Tensor] = []
    drifts: list[torch.Tensor] = []
    scored = 0
    terminated = False
    for _ in range(int(a.early_stop_probe_steps) + 4000):
        obs, _, dones, _ = env.step(policy(obs))
        if bool(dones.any()):
            terminated = True
            break
        active = (raw._preposition_counter <= 0) & (raw._settle_counter <= 0)
        if not bool(active.any()):
            continue
        errors.append(torch.linalg.norm(
            raw._original_reference_tool[active]
            - raw._compute_ee_pos_local()[active],
            dim=-1,
        ).detach().cpu())
        accelerations.append(torch.linalg.norm(
            raw._measured_tool_acceleration_truck[active], dim=-1
        ).detach().cpu())
        alignments.append(
            raw._measure_head_down_alignment()[active].detach().cpu()
        )
        drifts.append(raw._truck_post_constraint_drift_xy[active].detach().cpu())
        scored += 1
        if scored >= int(a.early_stop_probe_steps):
            break
    if not errors:
        return {"completed": False}
    error = torch.cat(errors)
    acceleration = torch.cat(accelerations)
    alignment = torch.cat(alignments)
    drift = torch.cat(drifts)
    return {
        "completed": bool(not terminated and scored >= int(a.early_stop_probe_steps)),
        "mean_error_m": float(torch.mean(error)),
        "p95_error_m": float(torch.quantile(error, 0.95)),
        "p95_acceleration_mps2": float(torch.quantile(acceleration, 0.95)),
        "min_head_alignment": float(torch.min(alignment)),
        "max_truck_drift_m": float(torch.max(drift)),
    }


@torch.inference_mode()
def _promotion_probe(raw, env, baseline_policy, candidate_policy, *, round_index: int):
    """Compare candidate and accepted policy on every exact shape."""
    per_shape: dict[str, dict[str, object]] = {}
    for shape_index, shape in enumerate(trajectory_shapes):
        seed = int(a.seed) + 10000 + round_index * len(trajectory_shapes) + shape_index
        baseline = _probe_policy(
            raw, env, baseline_policy, shape=shape, seed=seed
        )
        candidate = _probe_policy(
            raw, env, candidate_policy, shape=shape, seed=seed
        )
        per_shape[shape] = {"baseline": baseline, "candidate": candidate}

    def _average(policy_name: str, key: str) -> float:
        return sum(
            float(values[policy_name].get(key, float("inf")))
            for values in per_shape.values()
        ) / len(per_shape)

    baseline_mean = _average("baseline", "mean_error_m")
    candidate_mean = _average("candidate", "mean_error_m")
    baseline_p95 = _average("baseline", "p95_error_m")
    candidate_p95 = _average("candidate", "p95_error_m")
    baseline_acceleration = _average("baseline", "p95_acceleration_mps2")
    candidate_acceleration = _average("candidate", "p95_acceleration_mps2")
    checks = {
        "completed": all(
            bool(values[name].get("completed"))
            for values in per_shape.values()
            for name in ("baseline", "candidate")
        ),
        "mean_error_improved": candidate_mean <= baseline_mean * (
            1.0 - float(a.early_stop_min_mean_improvement)
        ),
        "p95_error_improved": candidate_p95 <= baseline_p95 * (
            1.0 - float(a.early_stop_min_p95_improvement)
        ),
        "aggregate_acceleration_safe": candidate_acceleration
        <= baseline_acceleration * float(a.early_stop_max_acceleration_ratio),
        "per_shape_error_safe": all(
            float(values["candidate"].get("mean_error_m", float("inf")))
            <= float(values["baseline"].get("mean_error_m", float("inf")))
            + float(a.early_stop_max_shape_regression_m)
            for values in per_shape.values()
        ),
        "per_shape_acceleration_safe": all(
            float(values["candidate"].get("p95_acceleration_mps2", float("inf")))
            <= float(values["baseline"].get("p95_acceleration_mps2", 0.0))
            * float(a.early_stop_max_shape_acceleration_ratio)
            for values in per_shape.values()
        ),
        "alignment_safe": min(
            float(values["candidate"].get("min_head_alignment", -1.0))
            for values in per_shape.values()
        ) >= 0.98,
        "truck_drift_safe": max(
            float(values["candidate"].get("max_truck_drift_m", float("inf")))
            for values in per_shape.values()
        ) <= 0.001,
    }
    return {
        "round": int(round_index + 1),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "aggregate": {
            "baseline_mean_error_m": baseline_mean,
            "candidate_mean_error_m": candidate_mean,
            "baseline_p95_error_m": baseline_p95,
            "candidate_p95_error_m": candidate_p95,
            "baseline_p95_acceleration_mps2": baseline_acceleration,
            "candidate_p95_acceleration_mps2": candidate_acceleration,
        },
        "per_shape": per_shape,
    }


def _promotion_probe_per_environment(
    raw, env, baseline_policy, candidate_policy, *, round_index: int
):
    """Run the repaired matched per-environment promotion gate."""
    result = paired_promotion_probe(
        raw,
        env,
        baseline_policy,
        candidate_policy,
        shapes=trajectory_shapes,
        seed=int(a.seed) + 10000 + round_index * len(trajectory_shapes),
        scored_steps=int(a.early_stop_probe_steps),
        min_completion_fraction=float(a.early_stop_min_completion_fraction),
        min_mean_improvement=float(a.early_stop_min_mean_improvement),
        min_p95_improvement=float(a.early_stop_min_p95_improvement),
        max_acceleration_ratio=float(a.early_stop_max_acceleration_ratio),
        max_shape_acceleration_ratio=float(
            a.early_stop_max_shape_acceleration_ratio
        ),
        max_shape_regression_m=float(a.early_stop_max_shape_regression_m),
    )
    result["round"] = int(round_index + 1)
    return result


def main() -> None:
    task = (
        "RRLAB-Mulag-PPO2-A-Trajectory2D-v0"
        if a.trajectory_2d
        else "RRLAB-Mulag-PPO2-A-Policy-V3-v0"
    )
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 45 if a.trajectory_2d else 42
    cfg.trajectory_geometry_observation = bool(a.trajectory_2d)
    if a.trajectory_2d:
        if not a.fresh_start:
            p.error("--trajectory_2d requires --fresh_start; legacy 42-D weights are not silently migrated")
        cfg.trajectory_library_path = str(
            ROOT / "mulag_plant/models/trajectory_2d_library_v2.json"
        )
    if a.trajectory_library_path:
        cfg.trajectory_library_path = str(
            Path(a.trajectory_library_path).expanduser().resolve()
        )
    if a.command_rom_checkpoint:
        cfg.command_rom_checkpoint = str(
            Path(a.command_rom_checkpoint).expanduser().resolve()
        )
    if a.command_rom_direction_guard is not None:
        cfg.command_rom_direction_guard = bool(a.command_rom_direction_guard)
    cfg.trajectory_shapes = trajectory_shapes
    balanced_mode = int(a.balanced_shape_rounds) > 0
    cfg.randomize_trajectory_shape = not balanced_mode
    if balanced_mode:
        cfg.trajectory_shape = trajectory_shapes[0]
    if a.trajectory_2d:
        # The standalone task follows the deployment station protocol: each
        # shape starts at a verified endpoint after the truck has parked. The
        # balanced trainer already covers both travel directions; arbitrary
        # mid-path resets are not part of the approved route and can exercise
        # unverified preposition poses during the zero-action gate.
        cfg.randomize_start_phase = False
    cfg.procedural_trajectory_probability = 0.0
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.cylinder4_residual_scale_m = 0.0
    # Reapply the stricter four-cylinder safety values after the shared stage helper.
    cfg.residual_head_penalty_weight = -8.0
    cfg.residual_ground_penalty_weight = -4.0
    cfg.residual_head_threshold = 0.997
    cfg.head_down_termination_alignment = 0.98
    cfg.cutter_clearance_hard_band_m = 0.20
    cfg.clearance_relative_to_reference = True
    cfg.defer_safety_terminations = bool(a.defer_safety_resets)

    agent_cfg = (
        FourCylinderTrajectory2DPPORunnerCfg()
        if a.trajectory_2d
        else FourCylinderResidualPolicyPPORunnerCfg()
    )
    if a.joint_finetune and not a.trajectory_2d:
        agent_cfg.experiment_name = "ppo2_a_policy_v3"
    agent_cfg.seed = int(a.seed)
    agent_cfg.device = a.device
    total_iterations = balanced_iterations if balanced_mode else sum(phase_iterations)
    agent_cfg.max_iterations = total_iterations
    agent_cfg.save_interval = int(a.save_interval)
    if a.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = float(a.learning_rate)

    log_dir = (
        ROOT / "logs/rsl_rl" / agent_cfg.experiment_name
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(log_dir)
    print(
        f"[PPO2-ROM-4CYL] iterations={total_iterations} "
        f"balanced_shapes={balanced_mode} "
        f"defer_safety_resets={cfg.defer_safety_terminations} "
        f"c4_authority=0->{a.cylinder4_low_scale_m:.6f}->"
        f"{a.cylinder4_full_scale_m:.6f}m log={log_dir}",
        flush=True,
    )

    gym_env = gym.make(task, cfg=cfg)
    try:
        if not a.skip_contract_check:
            _zero_action_preflight(gym_env.unwrapped)
        if a.preflight_only:
            print("[PPO2-ROM-4CYL] preflight-only complete; PPO was not started", flush=True)
            return
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
        if a.trajectory_2d:
            print("[PPO2-ROM-4CYL] fresh 2-D actor; historical line checkpoint untouched", flush=True)
        elif a.checkpoint4:
            checkpoint4 = str(Path(a.checkpoint4).expanduser().resolve())
            runner.load(
                checkpoint4,
                load_cfg={
                    "actor": True,
                    "critic": True,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
            with torch.no_grad():
                runner.alg.get_policy().distribution.std_param[:] = torch.tensor(
                    [
                        float(a.cylinder123_std),
                        float(a.cylinder123_std),
                        float(a.cylinder123_std),
                        float(a.cylinder4_std),
                    ],
                    device=runner.alg.get_policy().distribution.std_param.device,
                )
            print(
                f"[PPO2-ROM-4CYL] resumed4={checkpoint4} optimizer=fresh",
                flush=True,
            )
        else:
            transfer = transfer_checkpoint_into_runner(
                runner,
                a.checkpoint3,
                cylinder123_std=float(a.cylinder123_std),
                cylinder4_std=float(a.cylinder4_std),
            )
            print(f"[PPO2-ROM-4CYL] transfer={transfer}", flush=True)
            if (
                transfer["fourth_output_weight_max"] != 0.0
                or transfer["fourth_output_bias"] != 0.0
            ):
                raise RuntimeError("transferred cylinder-4 actor output is not exactly zero")
        if a.transfer_only:
            print("[PPO2-ROM-4CYL] transfer-only complete; PPO was not started", flush=True)
            return
        baseline_policy = None
        teacher_snapshot = None
        if a.trajectory_2d:
            # The geometry channels change the policy contract, so there is no
            # valid 42-D teacher to freeze or transfer. Train the new 45-D
            # actor jointly from scratch; the old line actor remains available
            # for the comparison gate.
            a.joint_finetune = True
            a.disable_early_stop = True
            joint = configure_joint_finetune(runner)
            print(f"[PPO2-ROM-4CYL] trajectory_2d_joint_start={joint}", flush=True)
        elif a.joint_finetune:
            baseline_policy = copy.deepcopy(runner.alg.get_policy()).to(
                gym_env.unwrapped.device
            ).eval()
            joint = configure_joint_finetune(runner)
            print(f"[PPO2-ROM-4CYL] joint_finetune={joint}", flush=True)
        else:
            freeze = freeze_teacher_and_train_cylinder4_head(runner)
            teacher_snapshot = snapshot_frozen_teacher(runner.alg.get_policy())
            print(f"[PPO2-ROM-4CYL] frozen_teacher={freeze}", flush=True)

        if balanced_mode:
            phases = []
            for round_index in range(int(a.balanced_shape_rounds)):
                ordered_shapes = (
                    trajectory_shapes
                    if round_index % 2 == 0
                    else tuple(reversed(trajectory_shapes))
                )
                std = (
                    float(a.cylinder4_std)
                    if round_index == 0
                    else float(a.refinement_cylinder4_std)
                )
                for shape in ordered_shapes:
                    phases.append((
                        f"balanced_r{round_index + 1}_{shape}",
                        int(a.iterations_per_shape),
                        float(a.cylinder4_low_scale_m),
                        0.0,
                        shape,
                        std,
                    ))
        else:
            phases = [
                ("teacher_lock", phase_iterations[0], 0.0, 0.0, None, float(a.cylinder4_std)),
                ("c4_0p25mm", phase_iterations[1], float(a.cylinder4_low_scale_m), 0.0, None, float(a.cylinder4_std)),
                ("c4_0p50mm_exact", phase_iterations[2], float(a.cylinder4_full_scale_m), 0.0, None, float(a.refinement_cylinder4_std)),
                (
                    "c4_0p50mm_mixed",
                    phase_iterations[3],
                    float(a.cylinder4_full_scale_m),
                    float(a.procedural_probability),
                    None,
                    float(a.refinement_cylinder4_std),
                ),
            ]
        completed_phases = 0
        consecutive_probe_passes = 0
        probe_history: list[dict[str, object]] = []
        early_stopped = False
        for phase_index, (
            name,
            iterations,
            authority,
            procedural_probability,
            shape,
            cylinder4_std,
        ) in enumerate(phases):
            if iterations <= 0:
                continue
            gym_env.unwrapped.cfg.cylinder4_residual_scale_m = authority
            gym_env.unwrapped.cfg.procedural_trajectory_probability = procedural_probability
            with torch.no_grad():
                runner.alg.get_policy().distribution.std_param[:] = torch.tensor(
                    [
                        float(a.cylinder123_std),
                        float(a.cylinder123_std),
                        float(a.cylinder123_std),
                        float(cylinder4_std),
                    ],
                    device=runner.alg.get_policy().distribution.std_param.device,
                )
            if shape is not None:
                gym_env.unwrapped.set_trajectory_shape(shape)
                # RSL-RL rollout collection creates Isaac state buffers under
                # inference mode. Subsequent synchronized resets must use the
                # same mode or PyTorch rejects in-place state updates.
                with torch.inference_mode():
                    gym_env.unwrapped.reset(seed=int(a.seed) + phase_index)
            if completed_phases > 0:
                runner.current_learning_iteration += 1
            print(
                f"[PPO2-ROM-4CYL] phase={name} iterations={iterations} "
                f"c4_authority={authority:.6f}m "
                f"c4_std={cylinder4_std:.4f} "
                f"procedural_probability={procedural_probability:.2f}",
                flush=True,
            )
            runner.learn(
                num_learning_iterations=int(iterations), init_at_random_ep_len=False
            )
            if teacher_snapshot is not None:
                assert_frozen_teacher_unchanged(
                    runner.alg.get_policy(), teacher_snapshot
                )
                print(
                    f"[PPO2-ROM-4CYL] frozen teacher invariant PASS after {name}",
                    flush=True,
                )
            completed_phases += 1
            completed_round = (
                balanced_mode
                and completed_phases % len(trajectory_shapes) == 0
            )
            if (
                a.joint_finetune
                and completed_round
                and not a.disable_early_stop
            ):
                round_index = completed_phases // len(trajectory_shapes) - 1
                if (
                    (round_index + 1)
                    % int(a.early_stop_probe_interval_rounds)
                    == 0
                ):
                    probe = _promotion_probe_per_environment(
                        gym_env.unwrapped,
                        env,
                        baseline_policy,
                        runner.alg.get_policy(),
                        round_index=round_index,
                    )
                    probe["iteration"] = int(runner.current_learning_iteration)
                    probe_history.append(probe)
                    (log_dir / "promotion_probe_history.json").write_text(
                        json.dumps(probe_history, indent=2) + "\n"
                    )
                    aggregate = probe["aggregate"]
                    print(
                        f"[4CYL PROMOTION PROBE] round={round_index + 1} "
                        f"passed={probe['passed']} "
                        f"mean={aggregate['baseline_mean_error_m']:.5f}->"
                        f"{aggregate['candidate_mean_error_m']:.5f}m "
                        f"p95={aggregate['baseline_p95_error_m']:.5f}->"
                        f"{aggregate['candidate_p95_error_m']:.5f}m "
                        f"acc95={aggregate['baseline_p95_acceleration_mps2']:.2f}->"
                        f"{aggregate['candidate_p95_acceleration_mps2']:.2f} "
                        f"checks={probe['checks']}",
                        flush=True,
                    )
                    consecutive_probe_passes = (
                        consecutive_probe_passes + 1
                        if bool(probe["passed"])
                        else 0
                    )
                    if consecutive_probe_passes >= int(
                        a.early_stop_consecutive_passes
                    ):
                        early_path = log_dir / (
                            f"model_{runner.current_learning_iteration}_promotion.pt"
                        )
                        runner.save(
                            str(early_path),
                            infos={"promotion_probe": probe},
                        )
                        print(
                            f"[4CYL EARLY STOP] promotion target passed "
                            f"{consecutive_probe_passes} consecutive probes; "
                            f"saved {early_path}",
                            flush=True,
                        )
                        early_stopped = True
                        break
        print(
            f"[PPO2-ROM-4CYL] complete final_iteration="
            f"{runner.current_learning_iteration} early_stopped={early_stopped} "
            f"log={log_dir}",
            flush=True,
        )
    finally:
        gym_env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
