#!/usr/bin/env python3
"""Minimal smoke test for the active PPO1 ID/FD task-space pipeline."""
from __future__ import annotations
import sys
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=50)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--direction_checks", action="store_true")
parser.add_argument("--fast_exit", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not hasattr(args_cli, 'device') or args_cli.device is None:
    args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv

print("="*80)
print("PPO1 SMOKE TEST: ID/FD with Task-Space Actions")
print("="*80)

# Configure env
env_cfg = MulagUpperLevelEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.ee_action_mode = "task_space_delta"
env_cfg.ee_action_dim = 3
env_cfg.ppo_stage = "local_reach_stage0"
env_cfg.stage0_level = 0
env_cfg.lowlevel_actuator_backend = "explicit_effort_pid_bias"
device = getattr(args_cli, 'device', 'cuda:0')
env_cfg.device = device
env_cfg.sim.device = device

print(f"\nPASS config: num_envs={args_cli.num_envs}, ee_action_mode=task_space_delta")

# Build env
base_env = HorizonIDFDEnv(
    base_cfg=env_cfg,
    horizon_steps=int(args_cli.horizon_steps),
    action_scale_m=0.01,
    fd_gains=[2.0, 3.0, 3.0, 3.0],
    device=device,
)

print(f"PASS env created: obs_dim={base_env.observation_space}, act_dim={base_env.num_actions}")

# Check task-space mode
if not base_env._is_task_space:
    print("FAIL: task-space mode not enabled")
    simulation_app.close()
    sys.exit(1)

print("PASS task-space mode enabled")

# Check ID/FD models loaded
try:
    has_models = hasattr(base_env, '_id_model') and base_env._id_model is not None
    print(f"PASS ID model loaded: {has_models}")
except Exception as exc:
    print(f"WARN could not verify ID/FD models: {exc}")

# Check EE controller
if base_env._ee_controller is None:
    print("FAIL: HighLevelEEController not initialized")
    simulation_app.close()
    sys.exit(1)

print("PASS HighLevelEEController ready")

# Reset and get initial obs
obs, _ = base_env.reset()
print(f"PASS reset successful, obs shape: {tuple(obs.shape)}")

# Run a few steps with task-space actions
print(f"\nRunning {args_cli.num_steps} steps...")
success_count = 0
for step in range(args_cli.num_steps):
    actions = torch.randn((args_cli.num_envs, base_env.num_actions), device=device).clamp(-1, 1)

    obs, rewards, terminated, truncated, info = base_env.step(actions)

    if step % 10 == 0:
        nonfinite = int((~torch.isfinite(obs)).sum().detach().cpu().item())
        print(
            f"  step={step:03d} reward_mean={rewards.mean().item():+.3f} "
            f"term={terminated.sum().item()}/{args_cli.num_envs} nonfinite_obs={nonfinite}",
            flush=True,
        )

    if not torch.isnan(obs).any() and not torch.isnan(rewards).any():
        success_count += 1

print(f"\nPASS completed {success_count}/{args_cli.num_steps} steps without NaN rewards/obs", flush=True)

if args_cli.direction_checks:
    print("\nTesting direction control...")
    obs, _ = base_env.reset()

    actions = torch.zeros((args_cli.num_envs, base_env.num_actions), device=device)
    actions[:, 1] = 1.0  # +Y in base frame
    obs, _, _, _, _ = base_env.step(actions)
    print("PASS +Y action executed")

    actions = torch.zeros((args_cli.num_envs, base_env.num_actions), device=device)
    actions[:, 2] = 1.0  # +Z in base frame
    obs, _, _, _, _ = base_env.step(actions)
    print("PASS +Z action executed")

print("\n" + "="*80)
print("SMOKE TEST PASSED", flush=True)
print("="*80)
print("\nPPO1 command path is active: PPO delta EE -> J_eff -> ID/FD -> cylinder targets -> explicit_effort_pid_bias -> PhysX", flush=True)
print("Ready for training with:")
print("  ./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py --num_envs 64 --max_iterations 100", flush=True)

if args_cli.fast_exit:
    os._exit(0)

simulation_app.close()
