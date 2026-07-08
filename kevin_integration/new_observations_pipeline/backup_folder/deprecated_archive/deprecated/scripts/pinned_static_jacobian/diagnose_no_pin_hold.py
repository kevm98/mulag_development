#!/usr/bin/env python3
"""Diagnose zero-action EE stability with _pin_root disabled.

Measures per-step jump, distance, cylinder error under pure zero-action hold.
"""
from __future__ import annotations
import argparse, sys
import numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="No-pin low-level hold diagnostic.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=50)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--action_scale_m", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage0_yz_env_wrapper import Stage0YZEnvWrapper

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
env = Stage0YZEnvWrapper(cfg, horizon_steps=int(args_cli.horizon_steps),
                         action_scale_m=float(args_cli.action_scale_m),
                         device=args_cli.device)

# --- Confirm _pin_root is not called ---
original_pin = env._pin_root
_pin_call_counter = [0]
def _counting_pin():
    _pin_call_counter[0] += 1
    original_pin()
env._pin_root = _counting_pin
print(f"[CONFIRM] _pin_root injected with counter", flush=True)

# --- Reset + settle ---
env.reset()
for _ in range(int(args_cli.settle_steps)):
    env.step(torch.zeros((env.num_envs, 2), device=args_cli.device))
env.resample_local_targets()

print(f"[CONFIRM] _pin_root call count after reset+settle+resample: {_pin_call_counter[0]}", flush=True)
assert _pin_call_counter[0] == 0, f"_pin_root was called {_pin_call_counter[0]} times — should be 0!"
print(f"[PASS] _pin_root is NOT called during reset/settle/resample", flush=True)

# --- Run zero-action hold diagnostic ---
num_steps = int(args_cli.num_steps)
device = args_cli.device
B = env.num_envs

jump_history = []
dist_history = []
cyl_qerr_history = []
cyl_qdot_history = []

print(f"\n[DIAG] Running {num_steps} zero-action hold steps...", flush=True)
print(f"[DIAG] {'step':>5s} {'jump_m':>8s} {'dist_m':>8s} {'cyl_qerr':>9s} {'qdot':>8s}", flush=True)

for s in range(num_steps):
    cqh = env._uw.robot.data.joint_pos[:, env._cyl_ids].clone()

    obs, reward, done, extras = env.step(torch.zeros((B, 2), device=device))

    cqa = env._uw.robot.data.joint_pos[:, env._cyl_ids].clone()
    cqv = env._uw.robot.data.joint_vel[:, env._cyl_ids].clone()

    jump = extras["log"]["ee_jump"].item()
    dist = extras["log"]["distance"].item()
    cyl_qerr = torch.abs(cqa - cqh).mean().item()
    cyl_qdot = torch.abs(cqv).mean().item()

    jump_history.append(jump)
    dist_history.append(dist)
    cyl_qerr_history.append(cyl_qerr)
    cyl_qdot_history.append(cyl_qdot)

    if s < 10 or s % 10 == 0:
        print(f"[DIAG] {s:5d} {jump:8.4f} {dist:8.4f} {cyl_qerr:9.6f} {cyl_qdot:8.4f}", flush=True)

# --- Summary ---
print(f"\n{'='*70}", flush=True)
print(f"SUMMARY  (num_steps={num_steps}, horizon_steps={args_cli.horizon_steps})", flush=True)
print(f"{'='*70}", flush=True)
print(f"  mean_jump         = {np.mean(jump_history):.4f} m", flush=True)
print(f"  max_jump          = {np.max(jump_history):.4f} m", flush=True)
print(f"  mean_distance     = {np.mean(dist_history):.4f} m", flush=True)
print(f"  max_distance      = {np.max(dist_history):.4f} m", flush=True)
print(f"  mean cyl_q_err    = {np.mean(cyl_qerr_history):.6f} rad", flush=True)
print(f"  max  cyl_q_err    = {np.max(cyl_qerr_history):.6f} rad", flush=True)
print(f"  mean cyl_qdot     = {np.mean(cyl_qdot_history):.4f} rad/s", flush=True)
print(f"  max  cyl_qdot     = {np.max(cyl_qdot_history):.4f} rad/s", flush=True)

pass_gate = np.mean(jump_history) < 0.05 and np.max(jump_history) < 0.10
strong_pass = np.mean(jump_history) < 0.02 and np.mean(dist_history) < 0.05

print(f"\n  GATE:  mean_jump < 0.05  AND  max_jump < 0.10  ->  {'PASS' if pass_gate else 'FAIL'}", flush=True)
print(f"  STRONG: mean_jump < 0.02  AND  mean_dist < 0.05  ->  {'PASS' if strong_pass else 'FAIL'}", flush=True)

env.close()
simulation_app.close()
