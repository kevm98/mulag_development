#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Evaluate a standalone 45-D PPO trajectory-2D checkpoint in PhysX."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True)
p.add_argument("--shapes", default="line,sine_chain,circle,rounded_square")
p.add_argument("--seed", type=int, default=17)
p.add_argument("--num_seeds", type=int, default=1)
p.add_argument(
    "--one_way",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run each shape until one endpoint is reached instead of a fixed screen.",
)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument(
    "--residual_scale_m",
    type=float,
    default=None,
    help="Optional scale for the trained cylinders-1..3 residual (m).",
)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.0005)
p.add_argument(
    "--zero_policy",
    action="store_true",
    help="Ignore learned residual actions and evaluate the deterministic teacher.",
)
p.add_argument(
    "--out",
    default="reports/trajectory_2d/evaluation_final.json",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

shapes = tuple(part.strip() for part in a.shapes.split(",") if part.strip())
valid_shapes = {"line", "sine_chain", "circle", "rounded_square"}
if not shapes or not set(shapes).issubset(valid_shapes):
    p.error("--shapes must contain line, sine_chain, circle, and/or rounded_square")
if a.num_seeds < 1 or a.steps < 60:
    p.error("--num_seeds must be positive and --steps must be at least 60")
if not 0.0 <= a.cylinder4_residual_scale_m <= 0.0005:
    p.error("cylinder-4 authority must be in [0, 0.0005] m")
if a.residual_scale_m is not None and not 0.0 <= a.residual_scale_m <= 0.004:
    p.error("residual scale must be in [0, 0.004] m")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import mulag_plant.tasks  # noqa: F401,E402
from mulag_plant.policy.env_cfg import apply_residual_curriculum_stage
from mulag_plant.policy_4cyl.env_cfg import FourCylinderTrajectory2DPPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg


def _run_shape(raw, env, policy, *, shape: str, seed: int) -> dict[str, object]:
    raw.set_trajectory_shape(shape)
    # RSL-RL rollout collection uses inference tensors; synchronized physical
    # reset must remain inside the same inference context.
    with torch.inference_mode():
        raw.reset(seed=int(seed))
    obs = env.get_observations()
    errors: list[float] = []
    nearest_errors: list[float] = []
    normal_errors: list[float] = []
    accelerations: list[float] = []
    alignments: list[float] = []
    clearance_errors: list[float] = []
    drifts: list[float] = []
    residuals: list[float] = []
    saturations: list[float] = []
    path_s: list[float] = []
    termination_log: dict[str, float] = {}
    terminated = False
    total_steps = 0
    target_steps = int(a.steps)
    one_way_time = float(raw._one_way_s)

    while total_steps < target_steps:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        total_steps += 1
        active = bool(
            raw._preposition_counter[0] <= 0 and raw._settle_counter[0] <= 0
        )
        if active:
            tool = raw._compute_ee_pos_local()
            tool_error = torch.linalg.norm(
                raw._original_reference_tool - tool, dim=-1
            )[0]
            nearest_delta, normal_error = raw._nearest_trajectory_geometry(tool)
            acceleration = torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)[0]
            target_clip = (
                raw._action_target_delta_x_pre_gain_clip_mask
                | raw._action_target_delta_x_post_gain_clip_mask
                | raw._action_target_delta_xdot_clip_mask
            )[0].any()
            valve_clip = (
                raw._action_hydraulic[0].abs()
                >= 0.999 * float(getattr(raw.cfg, "max_abs_valve", 1.0))
            ).any()
            errors.append(float(tool_error))
            nearest_errors.append(float(torch.linalg.norm(nearest_delta[0])))
            normal_errors.append(float(torch.abs(normal_error[0])))
            accelerations.append(float(acceleration))
            alignments.append(float(raw._measure_head_down_alignment()[0]))
            clearance_errors.append(
                float(
                    torch.abs(
                        raw._compute_cutter_world_z()[0]
                        - raw._original_reference_world_z[0]
                    )
                )
            )
            drifts.append(float(raw._truck_post_constraint_drift_xy[0]))
            residuals.append(
                float(torch.linalg.norm(raw._residual_filtered[0] * raw._residual_scale_vector()))
            )
            saturations.append(float(bool(target_clip or valve_clip)))
            path_s.append(float(raw._reference_path_s[0]))

        if bool(dones.any()):
            raw_log = raw.extras.get("log", {}) if hasattr(raw, "extras") else {}
            for key, value in raw_log.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    termination_log[str(key)] = float(value.detach().cpu())
                elif isinstance(value, (float, int)):
                    termination_log[str(key)] = float(value)
            terminated = True
            break
        if a.one_way and active and float(raw._scored_elapsed[0]) >= one_way_time:
            break

    if len(errors) < 20:
        return {
            "completed": False,
            "terminated": terminated,
            "active_steps": len(errors),
            "total_steps": total_steps,
            "trajectory_shape": shape,
            "truck_motion_mode": raw._trajectory_truck_motion_mode,
            "truck_speed_mps": float(raw._trajectory_truck_speed_mps),
        }

    err = np.asarray(errors)
    # A practical settling measure for a moving path: the longest contiguous
    # portion whose tool error remains inside the 15 cm working band.
    settled = err <= 0.15
    longest_settled = current_settled = 0
    for value in settled:
        current_settled = current_settled + 1 if value else 0
        longest_settled = max(longest_settled, current_settled)
    return {
        "completed": bool(not terminated and (not a.one_way or float(raw._scored_elapsed[0]) >= one_way_time)),
        "terminated": terminated,
        "active_steps": len(errors),
        "total_steps": total_steps,
        "trajectory_shape": shape,
        "truck_motion_mode": raw._trajectory_truck_motion_mode,
        "truck_speed_mps": float(raw._trajectory_truck_speed_mps),
        "effective_arm_speed_mps": float(raw._effective_path_speed_mps()),
        "one_way_time_s": one_way_time,
        "elapsed_s": float(raw._scored_elapsed[0]),
        "mean_tool_error_m": float(np.mean(err)),
        "p95_tool_error_m": float(np.percentile(err, 95)),
        "max_tool_error_m": float(np.max(err)),
        "overshoot_beyond_0p20_m": float(max(0.0, np.max(err) - 0.20)),
        "mean_nearest_path_error_m": float(np.mean(nearest_errors)),
        "p95_nearest_path_error_m": float(np.percentile(nearest_errors, 95)),
        "p95_abs_normal_error_m": float(np.percentile(normal_errors, 95)),
        "p95_clearance_error_m": float(np.percentile(clearance_errors, 95)),
        "p95_tool_acceleration_mps2": float(np.percentile(accelerations, 95)),
        "min_head_alignment": float(np.min(alignments)),
        "max_clearance_error_m": float(np.max(clearance_errors)),
        "max_truck_post_constraint_drift_m": float(np.max(drifts)),
        "mean_residual_norm_m": float(np.mean(residuals)),
        "max_residual_norm_m": float(np.max(residuals)),
        "saturation_fraction": float(np.mean(saturations)),
        "max_path_s_m": float(np.max(path_s)),
        "settled_error_band_m": 0.15,
        "longest_settled_steps": int(longest_settled),
        "longest_settled_time_s": float(longest_settled * raw.step_dt),
        "termination_log": termination_log,
    }


def main() -> None:
    task = "RRLAB-Mulag-PPO2-A-Trajectory2D-PLAY-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=1)
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 45
    cfg.trajectory_geometry_observation = True
    cfg.trajectory_library_path = str(ROOT / "mulag_plant/models/trajectory_2d_library_v2.json")
    cfg.trajectory_shapes = shapes
    cfg.randomize_trajectory_shape = False
    cfg.randomize_start_phase = False
    cfg.path_speed_mps = float(a.path_speed_mps)
    if a.residual_scale_m is not None:
        cfg.residual_scale_m = float(a.residual_scale_m)
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.force_zero_policy = bool(a.zero_policy)
    cfg.clearance_relative_to_reference = True
    cfg.clearance_frame_consistent = True
    cfg.cutter_clearance_hard_band_m = 0.20
    cfg.head_down_termination_alignment = 0.98
    cfg.episode_length_s = 300.0

    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    agent_cfg = FourCylinderTrajectory2DPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=a.device)
    checkpoint = str(Path(a.checkpoint).expanduser().resolve())
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=raw.device)

    runs: list[dict[str, object]] = []
    try:
        for shape in shapes:
            for seed in range(int(a.seed), int(a.seed) + int(a.num_seeds)):
                result = _run_shape(raw, env, policy, shape=shape, seed=seed)
                result["seed"] = seed
                runs.append(result)
                print(
                    f"[4CYL 2D EVAL] {shape} seed={seed} "
                    f"completed={result.get('completed')} "
                    f"mean={result.get('mean_tool_error_m', float('inf')):.5f}m "
                    f"p95={result.get('p95_tool_error_m', float('inf')):.5f}m "
                    f"normal95={result.get('p95_abs_normal_error_m', float('inf')):.5f}m "
                    f"sat={result.get('saturation_fraction', float('inf')):.4f}",
                    flush=True,
                )
    finally:
        env.close()

    completed = all(bool(run.get("completed")) for run in runs)
    result = {
        "passed": bool(
            completed
            and all(float(run.get("p95_clearance_error_m", 999.0)) <= 0.20 for run in runs)
            and all(float(run.get("p95_tool_error_m", 999.0)) <= 0.20 for run in runs)
            and all(float(run.get("p95_abs_normal_error_m", 999.0)) <= 0.15 for run in runs)
            and all(float(run.get("min_head_alignment", -1.0)) >= 0.98 for run in runs)
            and all(float(run.get("max_truck_post_constraint_drift_m", 999.0)) <= 0.02 for run in runs)
        ),
        "mode": "one_way" if a.one_way else "fixed_screen",
        "checkpoint": checkpoint,
        "trajectory_library": str(ROOT / "mulag_plant/models/trajectory_2d_library_v2.json"),
        "shapes": list(shapes),
        "seeds": list(range(int(a.seed), int(a.seed) + int(a.num_seeds))),
        "runs": runs,
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[4CYL 2D EVAL] {'PASS' if result['passed'] else 'FAIL'} wrote {out}", flush=True)
    if not result["passed"]:
        raise SystemExit(2)


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
