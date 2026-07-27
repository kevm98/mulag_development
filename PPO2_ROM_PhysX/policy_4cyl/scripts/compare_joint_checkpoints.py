#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Compare joint four-cylinder checkpoints with matched per-environment probes."""

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
p.add_argument("--checkpoints", required=True)
p.add_argument("--labels", default="model_79,model_99,model_119")
p.add_argument("--shapes", default="line,arc,sine,circle,square")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--scored_steps", type=int, default=120)
p.add_argument("--seed", type=int, default=17017)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.00025)
p.add_argument("--min_completion_fraction", type=float, default=0.95)
p.add_argument("--min_mean_improvement", type=float, default=0.03)
p.add_argument("--min_p95_improvement", type=float, default=0.03)
p.add_argument("--max_acceleration_ratio", type=float, default=1.05)
p.add_argument("--max_shape_acceleration_ratio", type=float, default=1.10)
p.add_argument("--max_shape_regression_m", type=float, default=0.002)
p.add_argument(
    "--baseline_checkpoint",
    default=str(
        ROOT / "logs/rsl_rl/ppo2_rom_residual/2026-07-19_18-55-23/model_249.pt"
    ),
)
p.add_argument(
    "--out",
    default="outputs/ppo2_rom_residual_4cyl_joint/checkpoint_comparison.json",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

checkpoints = tuple(part.strip() for part in a.checkpoints.split(",") if part.strip())
labels = tuple(part.strip() for part in a.labels.split(",") if part.strip())
shapes = tuple(part.strip() for part in a.shapes.split(",") if part.strip())
if not checkpoints or len(checkpoints) != len(labels):
    p.error("--checkpoints and --labels must have equal nonzero lengths")
if a.num_envs < 1 or a.scored_steps < 20:
    p.error("num envs must be positive and scored steps at least 20")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import PPO2_ROM_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage
from PPO2_ROM_PhysX.policy_4cyl.checkpoint_transfer import (
    transfer_checkpoint_into_runner,
)
from PPO2_ROM_PhysX.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg,
)
from PPO2_ROM_PhysX.policy_4cyl.promotion_probe import (
    compare_probe_results,
    run_policy_probe,
)
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = "RRLAB-Mulag-PPO2-A-Policy-V3-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 42
    cfg.trajectory_shape = shapes[0]
    cfg.trajectory_shapes = shapes
    cfg.randomize_trajectory_shape = False
    # Per-environment phase diversity is required for meaningful completion
    # fractions. With a fixed phase all 64 replicas complete or fail together.
    cfg.randomize_start_phase = True
    cfg.path_speed_mps = 0.18
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.clearance_relative_to_reference = True
    cfg.cutter_clearance_hard_band_m = 0.20
    cfg.head_down_termination_alignment = 0.98
    cfg.episode_length_s = 300.0

    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    agent_cfg = FourCylinderResidualPolicyPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    baseline_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=a.device
    )
    transfer_checkpoint_into_runner(
        baseline_runner,
        a.baseline_checkpoint,
        cylinder123_std=0.005,
        cylinder4_std=0.02,
    )
    baseline_policy = baseline_runner.get_inference_policy(device=raw.device)
    candidate_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=a.device
    )

    try:
        baseline_runs = {
            shape: run_policy_probe(
                raw,
                env,
                baseline_policy,
                shape=shape,
                seed=int(a.seed) + index,
                scored_steps=int(a.scored_steps),
            )
            for index, shape in enumerate(shapes)
        }
        results = {}
        for label, checkpoint in zip(labels, checkpoints):
            candidate_runner.load(
                str(Path(checkpoint).expanduser().resolve()),
                load_cfg={
                    "actor": True,
                    "critic": False,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
            candidate_policy = candidate_runner.get_inference_policy(device=raw.device)
            candidate_runs = {
                shape: run_policy_probe(
                    raw,
                    env,
                    candidate_policy,
                    shape=shape,
                    seed=int(a.seed) + index,
                    scored_steps=int(a.scored_steps),
                )
                for index, shape in enumerate(shapes)
            }
            result = compare_probe_results(
                baseline_runs,
                candidate_runs,
                min_completion_fraction=float(a.min_completion_fraction),
                min_mean_improvement=float(a.min_mean_improvement),
                min_p95_improvement=float(a.min_p95_improvement),
                max_acceleration_ratio=float(a.max_acceleration_ratio),
                max_shape_acceleration_ratio=float(a.max_shape_acceleration_ratio),
                max_shape_regression_m=float(a.max_shape_regression_m),
            )
            result["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
            results[label] = result
            aggregate = result["aggregate"]
            print(
                f"[JOINT COMPARE] {label} passed={result['passed']} "
                f"mean={aggregate['baseline_mean_error_m']:.5f}->"
                f"{aggregate['candidate_mean_error_m']:.5f}m "
                f"p95={aggregate['baseline_p95_error_m']:.5f}->"
                f"{aggregate['candidate_p95_error_m']:.5f}m "
                f"acc95={aggregate['baseline_p95_acceleration_mps2']:.2f}->"
                f"{aggregate['candidate_p95_acceleration_mps2']:.2f}",
                flush=True,
            )
        payload = {
            "baseline_checkpoint": str(Path(a.baseline_checkpoint).resolve()),
            "num_envs": int(a.num_envs),
            "scored_steps": int(a.scored_steps),
            "results": results,
        }
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[JOINT COMPARE] wrote {out}", flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
