#!/usr/bin/env python3
"""Collect Persistently Exciting ID teacher labels using valve-space PID.

Strategy:
- Long episodes (800+ steps) for wide extension coverage.
- One-cylinder velocity sweeps at low/mid/high extension.
- Multi-cylinder coordinated velocity commands.
- Valve-space PID with high gains produces teacher labels.
- Only "good" segments (sign-correct tracking) are saved for ID training.
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
parser.add_argument("--kp", type=float, default=40.0)
parser.add_argument("--ki", type=float, default=15.0)
parser.add_argument("--kd", type=float, default=2.0)
parser.add_argument("--integral_limit", type=float, default=0.5)
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

VELS = [-0.04, -0.03, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.03, 0.04]
N_V = len(VELS)

e_int = torch.zeros((uw.num_envs, 4), device=dev)
e_prev = torch.zeros((uw.num_envs, 4), device=dev)

X_id_batches, Y_id_batches = [], []
good_flags, sign_per_cyl = [], []

def _target_velocity(step, ep, ext_cur):
    """Persistently exciting velocity targets.
    
    Phases:
    0-100: One-cylinder full-range sweeps
    100-250: Multi-cylinder coordinated moves
    250-400: Random velocity exploration with anti-drift bias
    400-600: Targeted sweeps at current operating point
    600-800: Chirp velocity profiles
    """
    v_des = np.zeros((uw.num_envs, 4), dtype=np.float32)
    t = step * dt
    
    if step < 100:
        # One-cylinder full sweeps through all velocities
        cyl_idx = (step // 10) % 4
        vel_idx = (step % 10)
        if vel_idx < N_V:
            v_des[:, cyl_idx] = VELS[vel_idx]
    
    elif step < 250:
        hold_step = (step - 100) % 15
        if hold_step == 0:
            rng = np.random.RandomState(step + ep * 10000)
            for ci in range(4):
                if rng.random() < 0.7:
                    v_des[:, ci] = rng.choice(VELS)
            _target_velocity._last = v_des.copy()
        elif hasattr(_target_velocity, '_last'):
            v_des = _target_velocity._last.copy()
    
    elif step < 400:
        # Random velocity with anti-drift: push cylinders away from edges
        rng = np.random.RandomState(step + ep * 10000)
        for ci in range(4):
            e = ext_cur[0, ci] if len(ext_cur) > 0 else 0.5
            if e < 0.35:
                bias = 0.02  # Push positive to expand range
            elif e > 0.60:
                bias = -0.02  # Push negative to expand range
            else:
                bias = 0.0
            v_des[:, ci] = bias + rng.choice(VELS)
    
    elif step < 600:
        # Targeted sweeps: hold each velocity for 20 steps
        phase = ((step - 400) // 20) % (N_V + 2)
        for ci in range(4):
            if phase < N_V:
                v_des[:, ci] = VELS[phase]
            elif phase == N_V:
                v_des[:, ci] = 0.0  # Zero-velocity hold
            else:
                # Chirp burst on this cylinder
                v_des[:, ci] = 0.03 * np.sin(2 * np.pi * (5 + ci) * t)
    
    else:
        # Chirp velocity profile for dynamic excitation
        for ci in range(4):
            amp = 0.02 + 0.01 * ci
            freq = 3.0 + ci * 2.0 + ep * 0.3
            v_des[:, ci] = amp * np.sin(2 * np.pi * freq * t)
    
    return np.clip(v_des, -0.05, 0.05)

for ep in range(args_cli.episodes):
    obs, _ = env.reset()
    uw.set_debug_valve_cmd(torch.zeros((uw.num_envs, 4), device=dev))
    e_int.zero_(); e_prev.zero_()
    ext_current = np.zeros((uw.num_envs, 4), dtype=np.float32)
    
    for step in range(args_cli.steps_per_ep):
        v_des_np = _target_velocity(step, ep, ext_current)
        v_des = torch.from_numpy(v_des_np).to(device=dev)
        
        cq = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
        cqd = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]
        x_cur, xd_cur = uw._cylinder_adapter.current_from_cylinder_joints(cq, cqd)
        ext_current = x_cur.detach().cpu().numpy()
        
        e_v = v_des - xd_cur
        e_int = e_int + e_v * dt
        e_int = torch.clamp(e_int, -args_cli.integral_limit, args_cli.integral_limit)
        de_v = (e_v - e_prev) / max(dt, 1e-9)
        e_prev = e_v.detach().clone()
        
        valve_teacher = args_cli.kp * e_v + args_cli.ki * e_int + args_cli.kd * de_v
        valve_teacher = torch.clamp(valve_teacher, -0.95, 0.95)
        # Limit boom cylinder valve to avoid violent motion
        valve_teacher[:, 1] = torch.clamp(valve_teacher[:, 1], -0.4, 0.4)
        
        uw.set_debug_valve_cmd(valve_teacher)
        x_b = x_cur.detach().clone()
        
        obs, _, _, _, _ = env.step(za)
        
        cq_a = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
        cqd_a = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]
        x_a, xd_a = uw._cylinder_adapter.current_from_cylinder_joints(cq_a, cqd_a)
        delta_x_meas = x_a - x_b
        desired_delta = v_des * dt
        
        # ID input: [desired_delta(4), desired_vel(4), ext(4), extvel(4)]
        X_id = np.concatenate([
            desired_delta.detach().cpu().numpy(),
            v_des_np,
            x_cur.detach().cpu().numpy(),
            xd_cur.detach().cpu().numpy(),
        ], axis=1)
        Y_id = valve_teacher.detach().cpu().numpy()
        
        # Track sign correctness per cylinder
        signs_correct = (np.sign(delta_x_meas.detach().cpu().numpy()) == np.sign(v_des_np)) | (np.abs(v_des_np) < 0.002)
        all_good = signs_correct.all(axis=1)[0]
        sign_per_cyl.append(signs_correct[0])
        
        # Only save good samples for ID training
        if all_good:
            X_id_batches.append(X_id)
            Y_id_batches.append(Y_id)
        good_flags.append(all_good)
    
    good_ns = len(X_id_batches)
    # Incremental save
    if good_ns > 0:
        try:
            X_ep = np.concatenate(X_id_batches, axis=0).astype(np.float32)
            Y_ep = np.concatenate(Y_id_batches, axis=0).astype(np.float32)
            np.savez(outdir / f'id_dataset_ep{ep+1:02d}.npz', X=X_ep, Y=Y_ep)
        except Exception:
            pass
    frac = np.mean(good_flags[-args_cli.steps_per_ep:]) if good_flags else 0
    spc = np.mean(sign_per_cyl[-args_cli.steps_per_ep:], axis=0) if sign_per_cyl else [0]*4
    print(f"ID collect ep {ep+1}/{args_cli.episodes} (good={good_ns} samples, frac={frac:.2f}, sign_per_cyl={[f'{s:.2f}' for s in spc]})", flush=True)

env.close()
simulation_app.close()
print("ID collection complete.", flush=True)
