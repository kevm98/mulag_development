#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Balanced, smoothness-aware V3 training for PPO1 -> ID -> FD -> PhysX."""

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
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--balanced_shape_rounds", type=int, default=40)
p.add_argument("--iterations_per_shape", type=int, default=2)
p.add_argument(
    "--trajectory_shapes",
    default="line,arc,sine,circle,square,sine,circle,square",
)
p.add_argument(
    "--trajectory_library_path",
    default=str(
        ROOT / "PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json"
    ),
)
p.add_argument("--checkpoint", required=True)
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--residual_scale_m", type=float, default=0.002)
p.add_argument("--ee_correction_gain", type=float, default=0.25)
p.add_argument("--idfd_target_filter_beta", type=float, default=0.0)
p.add_argument("--policy_std", type=float, default=0.002)
p.add_argument("--cylinder4_std", type=float, default=0.03)
p.add_argument("--refinement_cylinder4_std", type=float, default=0.015)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.00025)
p.add_argument("--learning_rate", type=float, default=5.0e-6)
p.add_argument("--save_interval", type=int, default=10)
p.add_argument("--skip_contract_check", action="store_true")
p.add_argument("--preflight_only", action="store_true")
p.add_argument(
    "--defer_safety_resets",
    action=argparse.BooleanOptionalAction,
    default=True,
)
p.add_argument("--early_stop_probe_interval_rounds", type=int, default=2)
p.add_argument("--early_stop_probe_steps", type=int, default=120)
p.add_argument("--preflight_min_pass_fraction", type=float, default=0.94)
p.add_argument("--early_stop_min_completion_fraction", type=float, default=0.95)
p.add_argument("--early_stop_min_mean_improvement", type=float, default=0.02)
p.add_argument("--early_stop_min_p95_improvement", type=float, default=0.02)
p.add_argument("--early_stop_max_acceleration_ratio", type=float, default=1.05)
p.add_argument("--early_stop_max_shape_acceleration_ratio", type=float, default=1.08)
p.add_argument("--early_stop_max_shape_regression_m", type=float, default=0.001)
p.add_argument("--early_stop_consecutive_passes", type=int, default=2)
p.add_argument("--disable_early_stop", action="store_true")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

shapes = tuple(part.strip() for part in a.trajectory_shapes.split(",") if part.strip())
valid_shapes = {"line", "arc", "sine", "circle", "square"}
if not shapes or set(shapes) - valid_shapes:
    p.error(f"--trajectory_shapes must contain only {sorted(valid_shapes)}")
if a.num_envs < 1 or a.balanced_shape_rounds < 1 or a.iterations_per_shape < 1:
    p.error("environment, round, and per-shape iteration counts must be positive")
if a.path_speed_mps <= 0.0 or a.residual_scale_m <= 0.0:
    p.error("path speed and residual scale must be positive")
if a.policy_std <= 0.0 or a.learning_rate <= 0.0 or a.save_interval < 1:
    p.error("policy std, learning rate, and save interval must be positive")
if not 0.0 < a.early_stop_min_completion_fraction <= 1.0:
    p.error("completion fraction must be in (0, 1]")
if not 0.0 < a.preflight_min_pass_fraction <= 1.0:
    p.error("preflight pass fraction must be in (0, 1]")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import PPO1_ID_FD_PhysX.tasks  # noqa: F401,E402
from PPO1_ID_FD_PhysX.policy.env_cfg import (
    apply_residual_curriculum_stage,
)
from PPO1_ID_FD_PhysX.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg,
)
from PPO2_ROM_PhysX.policy_4cyl.checkpoint_transfer import (
    configure_joint_finetune,
    transfer_checkpoint_into_runner,
)
from PPO2_ROM_PhysX.policy_4cyl.promotion_probe import paired_promotion_probe
from isaaclab_tasks.utils import parse_env_cfg


@torch.no_grad()
def _zero_action_preflight(raw) -> None:
    """Check the zero-residual ID-FD baseline on every exact V3 shape."""
    raw.cfg.force_zero_policy = True
    saved = (
        float(raw.cfg.cutter_clearance_hard_band_m),
        float(raw.cfg.truck_drift_termination_m),
        float(raw.cfg.truck_raw_reaction_termination_m),
    )
    raw.cfg.cutter_clearance_hard_band_m = 10.0
    raw.cfg.truck_drift_termination_m = 10.0
    raw.cfg.truck_raw_reaction_termination_m = 10.0
    zero = torch.zeros((raw.num_envs, 4), device=raw.device)
    catastrophic = max(1.5 * saved[0], saved[0] + 0.05)
    try:
        for shape_index, shape in enumerate(tuple(dict.fromkeys(shapes))):
            raw.set_trajectory_shape(shape)
            raw.reset(seed=int(a.seed) + shape_index)
            max_clearance = torch.zeros(raw.num_envs, device=raw.device)
            max_residual = 0.0
            for _ in range(100):
                _, _, terminated, _, extras = raw.step(zero)
                if bool(terminated.any()):
                    raise RuntimeError(
                        f"ID-FD V3 preflight terminated on {shape}: "
                        f"{extras.get('log', {}) if isinstance(extras, dict) else extras}"
                    )
                max_residual = max(
                    max_residual, float(raw._residual_filtered.abs().max())
                )
                clearance = (
                    raw._compute_cutter_world_z() - raw._original_reference_world_z
                )
                max_clearance = torch.maximum(max_clearance, clearance.abs())
            pass_fraction = float((max_clearance <= saved[0]).float().mean())
            p95 = float(torch.quantile(max_clearance, 0.95))
            maximum = float(max_clearance.max())
            raw_reaction = float(raw._max_truck_drift.max())
            retained_drift = float(raw._max_truck_post_constraint_drift.max())
            if (
                max_residual > 1.0e-7
                or pass_fraction < float(a.preflight_min_pass_fraction)
                or maximum > catastrophic
                or raw_reaction > saved[2]
                or retained_drift > saved[1]
            ):
                raise RuntimeError(
                    f"ID-FD V3 zero-action contract failed on {shape}: "
                    f"residual={max_residual:.3e} clearance_p95={p95:.5f}m "
                    f"clearance_max={maximum:.5f}m pass={pass_fraction:.3f} "
                    f"raw_reaction={raw_reaction:.5f}m retained={retained_drift:.6f}m"
                )
            print(
                f"[IDFD V3 PREFLIGHT] {shape} PASS "
                f"clearance_p95={p95:.5f}m clearance_max={maximum:.5f}m "
                f"pass={pass_fraction:.3f} raw_reaction={raw_reaction:.5f}m "
                f"retained={retained_drift:.6f}m",
                flush=True,
            )
    finally:
        raw.cfg.cutter_clearance_hard_band_m = saved[0]
        raw.cfg.truck_drift_termination_m = saved[1]
        raw.cfg.truck_raw_reaction_termination_m = saved[2]
    raw.cfg.force_zero_policy = False
    if not a.preflight_only:
        raw.reset(seed=int(a.seed))


def _run_probe(raw, env, baseline_policy, candidate_policy, round_index: int):
    result = paired_promotion_probe(
        raw,
        env,
        baseline_policy,
        candidate_policy,
        shapes=tuple(dict.fromkeys(shapes)),
        seed=int(a.seed) + 10000 + round_index * len(shapes),
        scored_steps=int(a.early_stop_probe_steps),
        min_completion_fraction=float(a.early_stop_min_completion_fraction),
        min_mean_improvement=float(a.early_stop_min_mean_improvement),
        min_p95_improvement=float(a.early_stop_min_p95_improvement),
        max_acceleration_ratio=float(a.early_stop_max_acceleration_ratio),
        max_shape_acceleration_ratio=float(a.early_stop_max_shape_acceleration_ratio),
        max_shape_regression_m=float(a.early_stop_max_shape_regression_m),
    )
    result["round"] = round_index + 1
    return result


def main() -> None:
    task = "RRLAB-Mulag-PPO1-A-Policy-V3-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.trajectory_library_path = str(Path(a.trajectory_library_path).expanduser().resolve())
    cfg.trajectory_shapes = shapes
    cfg.trajectory_shape = shapes[0]
    cfg.randomize_trajectory_shape = False
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.residual_scale_m = float(a.residual_scale_m)
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.ee_correction_gain = float(a.ee_correction_gain)
    cfg.idfd_target_filter_beta = float(a.idfd_target_filter_beta)
    cfg.defer_safety_terminations = bool(a.defer_safety_resets)
    # Same anti-oscillation objective and safety envelope as ROM V3.
    cfg.residual_tracking_std_m = 0.04
    cfg.residual_tracking_guard_weight = -2.0
    cfg.residual_tracking_guard_scale_m = 0.04
    cfg.residual_action_rate_weight = -0.20
    cfg.residual_tool_acceleration_weight = -0.50
    cfg.residual_tool_acceleration_scale_mps2 = 30.0
    cfg.residual_head_penalty_weight = -8.0
    cfg.residual_ground_penalty_weight = -4.0
    cfg.residual_head_threshold = 0.997
    cfg.head_down_termination_alignment = 0.98
    cfg.cutter_clearance_hard_band_m = 0.20
    cfg.truck_raw_reaction_termination_m = 0.08
    cfg.clearance_relative_to_reference = True

    total_iterations = int(a.balanced_shape_rounds) * int(a.iterations_per_shape) * len(shapes)
    agent_cfg = FourCylinderResidualPolicyPPORunnerCfg()
    agent_cfg.experiment_name = "ppo1_a_policy_v3"
    agent_cfg.seed = int(a.seed)
    agent_cfg.device = a.device
    agent_cfg.max_iterations = total_iterations
    agent_cfg.save_interval = int(a.save_interval)
    agent_cfg.algorithm.learning_rate = float(a.learning_rate)
    agent_cfg.algorithm.schedule = "fixed"
    agent_cfg.algorithm.entropy_coef = 0.0

    log_dir = (
        ROOT / "logs/rsl_rl/id_fd_residual_v3"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(log_dir)
    print(
        f"[IDFD V3] iterations={total_iterations} envs={a.num_envs} "
        f"defer_safety_resets={cfg.defer_safety_terminations} "
        f"scale={cfg.residual_scale_m:.4f}m speed={cfg.path_speed_mps:.3f}m/s "
        f"log={log_dir}",
        flush=True,
    )

    gym_env = gym.make(task, cfg=cfg)
    try:
        if not a.skip_contract_check:
            _zero_action_preflight(gym_env.unwrapped)
        if a.preflight_only:
            print("[IDFD V3] preflight-only complete; PPO was not started", flush=True)
            return
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
        checkpoint = str(Path(a.checkpoint).expanduser().resolve())
        transfer = transfer_checkpoint_into_runner(
            runner,
            checkpoint,
            cylinder123_std=float(a.policy_std),
            cylinder4_std=float(a.cylinder4_std),
        )
        joint = configure_joint_finetune(runner)
        baseline_policy = copy.deepcopy(runner.alg.get_policy()).to(
            gym_env.unwrapped.device
        ).eval()
        print(
            f"[IDFD V3] transfer={transfer} joint_finetune={joint} "
            f"optimizer=fresh c123_std={a.policy_std:.4f} "
            f"c4_std={a.cylinder4_std:.4f}",
            flush=True,
        )

        completed_phases = 0
        consecutive_passes = 0
        history: list[dict[str, object]] = []
        early_stopped = False
        for round_index in range(int(a.balanced_shape_rounds)):
            ordered = shapes if round_index % 2 == 0 else tuple(reversed(shapes))
            c4_std = (
                float(a.cylinder4_std)
                if round_index == 0
                else float(a.refinement_cylinder4_std)
            )
            for shape in ordered:
                raw = gym_env.unwrapped
                with torch.no_grad():
                    runner.alg.get_policy().distribution.std_param[:] = torch.tensor(
                        [a.policy_std, a.policy_std, a.policy_std, c4_std],
                        device=runner.alg.get_policy().distribution.std_param.device,
                    )
                raw.set_trajectory_shape(shape)
                with torch.inference_mode():
                    raw.reset(seed=int(a.seed) + completed_phases)
                if completed_phases > 0:
                    runner.current_learning_iteration += 1
                print(
                    f"[IDFD V3] phase=balanced_r{round_index + 1}_{shape} "
                    f"iterations={a.iterations_per_shape} residual_scale="
                    f"{cfg.residual_scale_m:.4f}m c4_authority="
                    f"{cfg.cylinder4_residual_scale_m:.6f}m c4_std={c4_std:.4f}",
                    flush=True,
                )
                runner.learn(
                    num_learning_iterations=int(a.iterations_per_shape),
                    init_at_random_ep_len=False,
                )
                completed_phases += 1

            if (
                not a.disable_early_stop
                and (round_index + 1) % int(a.early_stop_probe_interval_rounds) == 0
            ):
                probe = _run_probe(
                    gym_env.unwrapped,
                    env,
                    baseline_policy,
                    runner.alg.get_policy(),
                    round_index,
                )
                probe["iteration"] = int(runner.current_learning_iteration)
                history.append(probe)
                (log_dir / "promotion_probe_history.json").write_text(
                    json.dumps(history, indent=2) + "\n"
                )
                aggregate = probe["aggregate"]
                print(
                    f"[IDFD V3 PROBE] round={round_index + 1} "
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
                consecutive_passes = consecutive_passes + 1 if probe["passed"] else 0
                if consecutive_passes >= int(a.early_stop_consecutive_passes):
                    path = log_dir / f"model_{runner.current_learning_iteration}_promotion.pt"
                    runner.save(str(path), infos={"promotion_probe": probe})
                    print(f"[IDFD V3] early-stop promotion={path}", flush=True)
                    early_stopped = True
                    break

        final = log_dir / f"model_{runner.current_learning_iteration}_final.pt"
        runner.save(str(final), infos={"early_stopped": early_stopped})
        print(
            f"[IDFD V3] complete final_iteration={runner.current_learning_iteration} "
            f"early_stopped={early_stopped} checkpoint={final}",
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
