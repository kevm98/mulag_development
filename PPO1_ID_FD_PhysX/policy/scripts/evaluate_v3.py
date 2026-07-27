#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Batched paired evaluation for a PPO1_A V3 Policy checkpoint."""

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
p.add_argument("--shapes", default="line,arc,sine,circle,square")
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--scored_steps", type=int, default=120)
p.add_argument("--seed", type=int, default=10192)
p.add_argument("--min_completion_fraction", type=float, default=0.95)
p.add_argument("--max_acceleration_ratio", type=float, default=1.05)
p.add_argument("--max_shape_acceleration_ratio", type=float, default=1.08)
p.add_argument("--max_shape_regression_m", type=float, default=0.001)
p.add_argument(
    "--trajectory_library_path",
    default=str(
        ROOT / "PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json"
    ),
)
p.add_argument(
    "--out",
    default="outputs/id_fd_policy_v3/checkpoint_evaluation.json",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

shapes = tuple(part.strip() for part in a.shapes.split(",") if part.strip())
if not shapes:
    p.error("--shapes must not be empty")
if a.num_envs < 1 or a.scored_steps < 20:
    p.error("num_envs must be positive and scored_steps must be at least 20")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import PPO1_ID_FD_PhysX.tasks  # noqa: F401,E402
from PPO1_ID_FD_PhysX.policy.env_cfg import apply_residual_curriculum_stage
from PPO1_ID_FD_PhysX.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg,
)
from PPO2_ROM_PhysX.policy_4cyl.promotion_probe import paired_promotion_probe
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = "RRLAB-Mulag-PPO1-A-Policy-V3-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 42
    cfg.trajectory_library_path = str(
        Path(a.trajectory_library_path).expanduser().resolve()
    )
    cfg.trajectory_shape = shapes[0]
    cfg.trajectory_shapes = shapes
    cfg.randomize_trajectory_shape = False
    cfg.randomize_start_phase = True
    cfg.path_speed_mps = 0.18
    cfg.cylinder4_residual_scale_m = 0.00025
    cfg.clearance_relative_to_reference = True
    cfg.cutter_clearance_hard_band_m = 0.20
    cfg.head_down_termination_alignment = 0.98
    cfg.episode_length_s = 300.0

    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    agent_cfg = FourCylinderResidualPolicyPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=a.device)
    runner.load(
        str(Path(a.checkpoint).expanduser().resolve()),
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
    )
    candidate_policy = runner.get_inference_policy(device=raw.device)

    def zero_policy(obs):
        return torch.zeros((obs.shape[0], 4), device=obs.device)

    try:
        result = paired_promotion_probe(
            raw,
            env,
            zero_policy,
            candidate_policy,
            shapes=shapes,
            seed=int(a.seed),
            scored_steps=int(a.scored_steps),
            min_completion_fraction=float(a.min_completion_fraction),
            min_mean_improvement=0.0,
            min_p95_improvement=0.0,
            max_acceleration_ratio=float(a.max_acceleration_ratio),
            max_shape_acceleration_ratio=float(a.max_shape_acceleration_ratio),
            max_shape_regression_m=float(a.max_shape_regression_m),
        )
        result["checkpoint"] = str(Path(a.checkpoint).expanduser().resolve())
        result["num_envs"] = int(a.num_envs)
        result["scored_steps"] = int(a.scored_steps)
        result["seed"] = int(a.seed)
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        aggregate = result["aggregate"]
        print(
            f"[PPO1_A EVAL] passed={result['passed']} "
            f"mean={aggregate['baseline_mean_error_m']:.5f}->"
            f"{aggregate['candidate_mean_error_m']:.5f}m "
            f"p95={aggregate['baseline_p95_error_m']:.5f}->"
            f"{aggregate['candidate_p95_error_m']:.5f}m "
            f"acc95={aggregate['baseline_p95_acceleration_mps2']:.2f}->"
            f"{aggregate['candidate_p95_acceleration_mps2']:.2f}",
            flush=True,
        )
        print(f"[PPO1_A EVAL] wrote {out}", flush=True)
        if not result["passed"]:
            raise SystemExit(2)
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
