#!/usr/bin/env python3
"""Playback/evaluate horizon-IDFD PPO checkpoint."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--fd_horizon_gain_per_cylinder", type=str, default="2.0,3.0,3.0,3.0")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL checkpoint to load")
parser.add_argument("--episodes", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""): args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np, torch
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

fd_gains = [float(x.strip()) for x in args_cli.fd_horizon_gain_per_cylinder.split(",")]
env_cfg = MulagUpperLevelEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
if getattr(args_cli, "device", None): env_cfg.device = args_cli.device; env_cfg.sim.device = args_cli.device

base_env = HorizonIDFDEnv(
    base_cfg=env_cfg,
    horizon_steps=args_cli.horizon_steps,
    action_scale_m=0.01,
    fd_gains=fd_gains,
    device=args_cli.device,
)
env = RslRlVecEnvWrapper(base_env)

# Build runner and load checkpoint
runner = OnPolicyRunner(env, {}, log_dir="/tmp/horizon_play", device=args_cli.device)
runner.load(args_cli.checkpoint)
print(f"Loaded checkpoint: {args_cli.checkpoint}", flush=True)

obs, _ = env.reset()
for ep in range(args_cli.episodes):
    total_rew = 0.0
    for step in range(100):  # max 100 horizon steps per episode
        with torch.no_grad():
            action = runner.get_inference_policy()(obs)
        obs, rew, terminated, truncated, _ = env.step(action)
        total_rew += float(rew.mean().cpu())
        if terminated.any() or truncated.any():
            break
    print(f"Episode {ep+1}: total_reward={total_rew:.2f}", flush=True)

env.close()
simulation_app.close()
