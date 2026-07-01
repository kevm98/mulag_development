#!/usr/bin/env python3
"""Collect Persistently Exciting FD dataset from PhysX/adapter.

Strategy for persistent excitation:
1. Long episodes (1000+ steps) to let cylinders move through full range.
2. Random-walk valve commands with slow drift for wide extension coverage.
3. Superimposed chirp/sweep for dynamic excitation at each operating point.
4. One-cylinder and multi-cylinder phases.
5. Forced low/mid/high extension exploration.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=12)
parser.add_argument("--steps_per_ep", type=int, default=800)
parser.add_argument("--output_dir", type=str, default="outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""): args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

outdir = Path(args_cli.output_dir); outdir.mkdir(parents=True, exist_ok=True)
cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.low_level_servo_mode = "none"
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True
cfg.log_target_delta_xdot_clipping = False
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env.unwrapped if hasattr(env, "unwrapped") else env
dt = float(uw.step_dt); dev = uw.device
za = torch.zeros((uw.num_envs, uw.cfg.action_space), device=dev)

X_batches, Y_batches = [], []
np.random.seed(42)

# Valve command levels for rich excitation
VALVE_LEVELS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70,
                -0.02, -0.05, -0.08, -0.12, -0.18, -0.25, -0.35, -0.50, -0.70]

def _excitation_profile(step, ep, ext_cur):
    """Generate persistently exciting valve commands.
    
    Phases per episode:
    0-100: Zero drift measurement
    100-300: One-cylinder slow ramps + chirp (wide extension exploration)
    300-500: Multi-cylinder random-walk with superimposed chirp
    500-700: Targeted sweeps at current operating point
    700-800: Chirp burst for dynamic identification
    """
    t = step * dt
    vc = np.zeros((uw.num_envs, 4), dtype=np.float32)
    
    if step < 100:
        # Zero drift: measure natural cylinder dynamics
        return vc
    
    elif step < 300:
        # One-cylinder slow ramps: drive each cylinder through its range
        phase_step = step - 100
        cyl_idx = (phase_step // 50) % 4
        sub_step = phase_step % 50
        # Slow triangle wave: 0 -> max -> -max -> 0 over 50 steps
        if sub_step < 13:
            vc[:, cyl_idx] = VALVE_LEVELS[sub_step + 4]  # ramping up positive
        elif sub_step < 25:
            vc[:, cyl_idx] = VALVE_LEVELS[25 - sub_step + 3]  # ramping down
        elif sub_step < 38:
            vc[:, cyl_idx] = VALVE_LEVELS[min(sub_step - 25 + 12, len(VALVE_LEVELS)-1)]  # ramping down negative
        else:
            vc[:, cyl_idx] = VALVE_LEVELS[min(38 - sub_step + 11, len(VALVE_LEVELS)-1)]  # ramping back up
        # Add small chirp on other cylinders for coupling data
        for ci in range(4):
            if ci != cyl_idx:
                vc[:, ci] = 0.02 * np.sin(2 * np.pi * (3 + ci) * t)
    
    elif step < 500:
        # Multi-cylinder random walk with drift bias for coverage
        rng = np.random.RandomState(step + ep * 10000)
        for ci in range(4):
            # Bias toward unexplored extension regions
            e = ext_cur[0, ci] if ext_cur.shape[0] > 0 else 0.5
            # Push away from edges: if extension is low (<0.35), bias positive
            if e < 0.35:
                bias = 0.15
            elif e > 0.60:
                bias = -0.15
            else:
                bias = 0.0
            vc[:, ci] = bias + rng.uniform(-0.25, 0.25)
    
    elif step < 700:
        # Targeted sweeps at current operating point
        rng = np.random.RandomState(step + ep * 10000)
        for ci in range(4):
            # Sweep through 5 valve levels, hold each for 40 steps
            level_idx = ((step - 500) // 40) % 5
            levels = [0.0, 0.1, -0.1, 0.25, -0.25]
            vc[:, ci] = levels[level_idx]
            # Add small random perturbation for richness
            vc[:, ci] += rng.uniform(-0.03, 0.03)
    
    else:
        # Chirp burst: rich dynamic excitation
        for ci in range(4):
            amp = 0.15 + 0.05 * ci
            freq = 4.0 + ci * 3.0 + ep * 0.5
            vc[:, ci] = amp * np.sin(2 * np.pi * freq * t)
    
    return np.clip(vc, -0.9, 0.9)

for ep in range(args_cli.episodes):
    obs, _ = env.reset()
    uw.set_debug_valve_cmd(torch.zeros((uw.num_envs, 4), device=dev))
    ext_current = np.zeros((uw.num_envs, 4), dtype=np.float32)
    
    for step in range(args_cli.steps_per_ep):
        vc_np = _excitation_profile(step, ep, ext_current)
        vc = torch.from_numpy(vc_np).to(device=dev)
        uw.set_debug_valve_cmd(vc)
        
        cq_b = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
        cqd_b = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids].detach().clone()
        x_b, xd_b = uw._cylinder_adapter.current_from_cylinder_joints(cq_b, cqd_b)
        ext_current = x_b.detach().cpu().numpy()
        
        obs, _, _, _, _ = env.step(za)
        
        cq_a = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
        cqd_a = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids].detach().clone()
        x_a, xd_a = uw._cylinder_adapter.current_from_cylinder_joints(cq_a, cqd_a)
        delta_x = (x_a - x_b).detach().cpu().numpy()
        
        X = np.concatenate([vc_np, x_b.detach().cpu().numpy(), xd_b.detach().cpu().numpy()], axis=1)
        X_batches.append(X)
        Y_batches.append(delta_x)
    
    ns = len(X_batches)
    # Incremental save every episode
    try:
        X_ep = np.concatenate(X_batches, axis=0).astype(np.float32)
        Y_ep = np.concatenate(Y_batches, axis=0).astype(np.float32)
        np.savez(outdir / f"fd_dataset_ep{ep+1:02d}.npz", X=X_ep, Y=Y_ep)
    except Exception:
        pass
    print(f"FD collect ep {ep+1}/{args_cli.episodes} ({ns} samples, ext_range=[{ext_current.min():.4f},{ext_current.max():.4f}])", flush=True)

env.close()
simulation_app.close()
print("Collection complete. Data saved per episode.", flush=True)
