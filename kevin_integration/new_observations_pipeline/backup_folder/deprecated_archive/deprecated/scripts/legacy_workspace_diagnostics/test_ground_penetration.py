#!/usr/bin/env python3
"""Test ground penetration fix - verify end effector stays above ground."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=100)
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
print("GROUND PENETRATION TEST")
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

print(f"\n✓ Config: num_envs={args_cli.num_envs}")
print(f"  PhysX solver iterations: min_pos={env_cfg.sim.physx.min_position_iteration_count}, "
      f"min_vel={env_cfg.sim.physx.min_velocity_iteration_count}")

# Build env
base_env = HorizonIDFDEnv(
    base_cfg=env_cfg,
    horizon_steps=10,
    action_scale_m=0.01,
    fd_gains=[2.0, 3.0, 3.0, 3.0],
    device=device,
)

print(f"✓ Env created")

# Reset
obs = base_env.reset()
print(f"✓ Reset complete")

# Run steps and monitor end effector height
print(f"\nRunning {args_cli.num_steps} steps, monitoring EE height...")
min_ee_z = torch.full((args_cli.num_envs,), float('inf'), device=device)
penetration_count = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)

for step in range(args_cli.num_steps):
    # Random actions pushing down (to stress test ground contact)
    actions = torch.randn((args_cli.num_envs, 2), device=device)
    actions[:, 1] = -1.0  # Push -Z (down)

    obs, rewards, terminated, truncated, info = base_env.step(actions)

    # Get EE position from base env
    if hasattr(base_env._env, '_ee_pos_w'):
        ee_pos_w = base_env._env._ee_pos_w
        ee_z = ee_pos_w[:, 2]

        # Track minimum height
        min_ee_z = torch.minimum(min_ee_z, ee_z)

        # Count penetrations (below 0cm - ground is at -1cm)
        penetrations = ee_z < 0.0
        penetration_count += penetrations.long()

        if step % 20 == 0:
            print(f"  Step {step:3d}: "
                  f"EE_Z min={ee_z.min().item()*1000:+.1f}mm, "
                  f"mean={ee_z.mean().item()*1000:+.1f}mm, "
                  f"penetrations={penetrations.sum().item()}/{args_cli.num_envs}")

# Final report
print(f"\n{'='*80}")
print("RESULTS:")
print(f"{'='*80}")
print(f"Minimum EE height recorded: {min_ee_z.min().item()*1000:+.2f}mm")
print(f"Penetration frequency:")
for env_id in range(args_cli.num_envs):
    pct = 100.0 * penetration_count[env_id].item() / args_cli.num_steps
    print(f"  Env {env_id}: {penetration_count[env_id].item()}/{args_cli.num_steps} steps ({pct:.1f}%)")

total_penetrations = penetration_count.sum().item()
total_steps = args_cli.num_steps * args_cli.num_envs
pct = 100.0 * total_penetrations / total_steps
print(f"  Total: {total_penetrations}/{total_steps} ({pct:.1f}%)")

print(f"\n{'='*80}")
if pct < 5.0:
    print("✅ PASS: Ground penetration < 5% (acceptable with aggressive downward actions)")
elif pct < 20.0:
    print("⚠️  WARNING: Ground penetration 5-20% (marginal, may need further tuning)")
else:
    print("❌ FAIL: Ground penetration > 20% (contact solver needs improvement)")
print(f"{'='*80}")

print("\nNote: Ground is at z=-10mm. Some penetration is acceptable with")
print("aggressive downward actions. The key is it doesn't sink continuously.")

simulation_app.close()
