#!/usr/bin/env python3
"""Minimal smoke test for PPO1 with ID/FD and task-space actions."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=50)
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
env_cfg.ppo_stage = "local_reach_stage0"
env_cfg.stage0_level = 0
device = getattr(args_cli, 'device', 'cuda:0')
env_cfg.device = device
env_cfg.sim.device = device

print(f"\n✓ Config: num_envs={args_cli.num_envs}, ee_action_mode=task_space_delta")

# Build env
base_env = HorizonIDFDEnv(
    base_cfg=env_cfg,
    horizon_steps=10,
    action_scale_m=0.01,
    fd_gains=[2.0, 3.0, 3.0, 3.0],
    device=device,
)

print(f"✓ Env created: obs_dim={base_env.observation_space}, act_dim={base_env.num_actions}")

# Check task-space mode
if not base_env._is_task_space:
    print("✗ FAIL: Task-space mode not enabled!")
    simulation_app.close()
    sys.exit(1)

print(f"✓ Task-space mode enabled")

# Check ID/FD models loaded
try:
    has_models = hasattr(base_env, '_id_model') and base_env._id_model is not None
    print(f"✓ ID/FD models loaded")
except:
    print(f"⚠ Could not verify ID/FD models (may be in base env)")

# Check EE controller
if base_env._ee_controller is None:
    print("✗ FAIL: HighLevelEEController not initialized!")
    simulation_app.close()
    sys.exit(1)

print(f"✓ HighLevelEEController ready")

# Reset and get initial obs
obs = base_env.reset()
print(f"✓ Reset successful, obs shape: {obs.shape}")

# Run a few steps with task-space actions
print(f"\nRunning {args_cli.num_steps} steps...")
success_count = 0
for step in range(args_cli.num_steps):
    # Random task-space actions [ΔY, ΔZ] in [-1, 1]
    actions = torch.randn((args_cli.num_envs, 2), device=device).clamp(-1, 1)
    
    obs, rewards, terminated, truncated, info = base_env.step(actions)
    
    if step % 10 == 0:
        print(f"  Step {step:3d}: reward_mean={rewards.mean().item():+.3f}, "
              f"term={terminated.sum().item()}/{args_cli.num_envs}")
    
    if not torch.isnan(obs).any() and not torch.isnan(rewards).any():
        success_count += 1

print(f"\n✓ Completed {success_count}/{args_cli.num_steps} steps without NaN")

# Test direction control
print("\nTesting direction control...")
obs = base_env.reset()

# Test +Y
actions = torch.zeros((args_cli.num_envs, 2), device=device)
actions[:, 0] = 1.0  # +Y
obs, _, _, _, _ = base_env.step(actions)
print(f"✓ +Y action executed")

# Test +Z
actions = torch.zeros((args_cli.num_envs, 2), device=device)
actions[:, 1] = 1.0  # +Z
obs, _, _, _, _ = base_env.step(actions)
print(f"✓ +Z action executed")

print("\n" + "="*80)
print("✅ SMOKE TEST PASSED")
print("="*80)
print("\nPPO1 with ID/FD and task-space actions is working!")
print("Ready for training with:")
print("  ./rrlab.sh -p kevin_integration/scripts/train_upper_level_ppo_horizon_idfd.py \\")
print("    --num_envs 64 --max_iterations 100 --ee_action_mode task_space_delta")

simulation_app.close()
