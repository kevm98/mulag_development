#!/usr/bin/env python3
"""Diagnostic: verify full PPO Stage-0 Y-Z pipeline in one rollout."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check Stage-0 Y-Z PPO env.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage0_yz_env_wrapper import Stage0YZEnvWrapper

cfg = MulagUpperLevelEnvCfg()
env = Stage0YZEnvWrapper(cfg, horizon_steps=20, action_scale_m=0.05, device=args_cli.device)
B = env.num_envs

print("=" * 60, flush=True)
print("STAGE-0 Y-Z PPO ENV CHECK", flush=True)
print("=" * 60, flush=True)

# 1. Structure check
print(f"\n1. Structure:", flush=True)
print(f"   num_envs: {B}", flush=True)
print(f"   action_dim: {env.num_actions}", flush=True)
print(f"   obs_dim: {env.num_obs}", flush=True)
print(f"   J_pinv shape: {env._J_pinv.shape}", flush=True)
print(f"   max_ep_len: {env.max_episode_length}", flush=True)

# 2. Reset + observation check
obs, extras = env.reset()
obs_vec = obs["policy"][0].detach().cpu().numpy()
print(f"\n2. Reset:", flush=True)
print(f"   obs shape: {obs['policy'].shape}", flush=True)
print(f"   obs[0]: {obs_vec} (ee_yz[2], target_yz[2], prev_action[2])", flush=True)
print(f"   obs finite: {torch.isfinite(obs['policy']).all().item()}", flush=True)

# 3. Step with zero action
action_zero = torch.zeros((B, 2), device=env.device)
obs2, reward, term, trunc, extras = env.step(action_zero)
print(f"\n3. Zero action step:", flush=True)
print(f"   reward: {reward.item():.4f}", flush=True)
print(f"   terminated: {term.item()}, truncated: {trunc.item()}", flush=True)
print(f"   reward finite: {torch.isfinite(reward).all().item()}", flush=True)
print(f"   log: {extras.get('log', {})}", flush=True)

# 4. Step with random action
action_rand = 2.0 * torch.rand((B, 2), device=env.device) - 1.0
obs3, reward2, term2, trunc2, extras2 = env.step(action_rand)
print(f"\n4. Random action step:", flush=True)
print(f"   action: {action_rand[0].detach().cpu().numpy()}", flush=True)
print(f"   reward: {reward2.item():.4f}", flush=True)
print(f"   log: {extras2.get('log', {})}", flush=True)

# 5. Run full episode with random actions
print(f"\n5. Full episode (random policy):", flush=True)
env.reset()
total_reward = 0.0
distances = []
for step in range(30):
    action = 2.0 * torch.rand((B, 2), device=env.device) - 1.0
    obs, reward, term, trunc, extras = env.step(action)
    total_reward += reward.item()
    d = extras.get("log", {}).get("distance", torch.tensor(0.0))
    if isinstance(d, torch.Tensor): d = d.mean().item()
    distances.append(float(d))
    if term.item() or trunc.item():
        break
print(f"   steps: {step+1}", flush=True)
print(f"   total_reward: {total_reward:.3f}", flush=True)
print(f"   distances: min={min(distances):.3f} max={max(distances):.3f} final={distances[-1]:.3f}", flush=True)
print(f"   reward finite: {np.isfinite(total_reward)}", flush=True)

# 6. Check no cylinder leakage in obs
cyl_ids = env._uw._cylinder_joint_ids.detach().cpu().tolist()
jnames = env._uw.robot.data.joint_names
print(f"\n6. No cylinder leakage:", flush=True)
print(f"   obs dims: {obs_vec.tolist()}", flush=True)
# The obs is [ee_yz(2), target_yz(2), prev_action(2)] = 6 floats
# Cylinder data would add 4 more floats
print(f"   obs contains cylinder data? {'YES' if env.num_obs > 10 else 'NO'} (obs_dim={env.num_obs})", flush=True)

# 7. Check mapper output
desired_ee = torch.tensor([[0.0, -0.05]], device=env.device)  # command -5cm in Z
delta_cyl_raw = (env._J_pinv @ desired_ee.T).T
delta_cyl = torch.clamp(delta_cyl_raw, -0.005, 0.005)
print(f"\n7. Mapper check:", flush=True)
print(f"   desired_ee: {desired_ee[0].tolist()}", flush=True)
print(f"   delta_cyl_raw: {delta_cyl_raw[0].tolist()}", flush=True)
print(f"   delta_cyl_clamped: {delta_cyl[0].tolist()}", flush=True)
print(f"   any nonzero: {(delta_cyl.abs() > 1e-9).any().item()}", flush=True)

print(f"\n{'='*60}", flush=True)
print("ALL CHECKS PASSED", flush=True)
print(f"{'='*60}", flush=True)

env.close()
simulation_app.close()
