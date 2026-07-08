#!/usr/bin/env python3
"""Test Y-Z task-space mapper with measured J_eff and D-only backend."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Y-Z EE mapper.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--lambda_val", type=float, default=1.0)
parser.add_argument("--max_delta_m", type=float, default=0.005)
parser.add_argument("--target_mag_m", type=float, default=0.03)
parser.add_argument("--horizon_steps", type=int, default=20)
parser.add_argument("--kd_scale", type=float, default=0.2)
parser.add_argument("--kp_scale", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from isaaclab.utils.math import quat_apply_inverse

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
cfg.allow_debug_manual_cylinder_targets = True
cfg.lowlevel_actuator_backend = "explicit_effort_pd"
cfg.explicit_pd_gain_scale = args_cli.kp_scale
cfg.explicit_pd_damping_scale = args_cli.kd_scale
cfg.explicit_pd_effort_scale = 1.0
cfg.explicit_force_rate_limit_fraction = 0.01
cfg.explicit_pd_velocity_filter_beta = 0.8
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg); uw = env; B = env.num_envs; dev = uw.device
cyl_ids = uw._cylinder_joint_ids; tool_id = uw._tool_joint_id; ee_id = uw._ee_body_id
dt = float(uw.step_dt); H = int(args_cli.horizon_steps); jnames = uw.robot.data.joint_names

# Fixed measured J_eff (Y-Z rows, 4 cylinder columns)
J_yz = torch.tensor([
    [-1.74,  29.81,  20.45,  20.29],
    [-1.51, -24.73, -30.88, -31.66],
], device=dev, dtype=torch.float64)

lam = float(args_cli.lambda_val)
max_delta = float(args_cli.max_delta_m)

# Precompute damped least-squares pinv
# pinv = J^T @ inv(J@J^T + λ²I)
JT = J_yz.T  # [4, 2]
JJT = J_yz @ JT  # [2, 2]
I2 = torch.eye(2, device=dev, dtype=torch.float64)
JJT_damped = JJT + lam * lam * I2
JJT_inv = torch.linalg.inv(JJT_damped)
J_pinv = JT @ JJT_inv  # [4, 2]

def tcp_b():
    ee = uw.robot.data.body_link_state_w[:, ee_id, :3]; bp = uw.robot.data.root_pos_w
    bq = uw.robot.data.root_quat_w; bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return quat_apply_inverse(bq, ee - bp)

def run_trial(target_yz, label=""):
    """Run one trial: settle, command target, measure EE."""
    env.reset()
    root_pose = uw.robot.data.root_state_w[:, :7].clone()
    root_vel = torch.zeros((B, 6), device=dev)
    cq_hold = uw.robot.data.joint_pos[:, cyl_ids].clone()
    tool_target = uw.robot.data.joint_pos[:, tool_id].unsqueeze(-1).clone()

    # Settle
    for _ in range(60):
        uw.robot.write_root_pose_to_sim(root_pose)
        uw.robot.write_root_velocity_to_sim(root_vel)
        uw.set_debug_manual_cylinder_targets(cq_hold, tool_position_target=tool_target)
        env.step(torch.zeros((B, 5), device=dev))
    cq_hold = uw.robot.data.joint_pos[:, cyl_ids].clone()

    ee_start = tcp_b().detach().clone()

    # Compute cylinder delta from mapper
    if target_yz is not None:
        dee = torch.tensor(target_yz, device=dev, dtype=torch.float64).unsqueeze(0)
        delta_cyl_raw = (J_pinv @ dee.T).T  # [1, 4]
        delta_cyl = torch.clamp(delta_cyl_raw, -max_delta, max_delta)
        clipped = (delta_cyl - delta_cyl_raw).abs().gt(1e-8).float().mean().item()
        delta_cyl = delta_cyl.float()

        cq_target = cq_hold + delta_cyl
        # Clamp to joint limits
        limits = uw.robot.data.soft_joint_pos_limits[:, cyl_ids]
        cq_target = torch.clamp(cq_target, limits[..., 0], limits[..., 1])
    else:
        cq_target = cq_hold
        delta_cyl = torch.zeros((1, 4), device=dev)
        clipped = 0.0

    # Execute horizon
    for _ in range(H):
        uw.robot.write_root_pose_to_sim(root_pose)
        uw.robot.write_root_velocity_to_sim(root_vel)
        uw.set_debug_manual_cylinder_targets(cq_target, tool_position_target=tool_target)
        env.step(torch.zeros((B, 5), device=dev))

    ee_end = tcp_b().detach().clone()
    ee_delta = (ee_end - ee_start)[0].detach().cpu().numpy()
    dq = (uw.robot.data.joint_pos[:, cyl_ids] - cq_hold)[0].detach().cpu().numpy()
    cq_applied = (cq_target - cq_hold)[0].detach().cpu().numpy()

    return ee_start[0].cpu().numpy(), ee_end[0].cpu().numpy(), ee_delta, dq, cq_applied, delta_cyl[0].cpu().numpy(), clipped

# Directions to test: +Y, -Y, +Z, -Z, +Y+Z, +Y-Z, -Y+Z, -Y-Z
mag = float(args_cli.target_mag_m)
directions = {
    "+Y":   (0.0, mag),
    "-Y":   (0.0, -mag),
    "+Z":   (mag, 0.0),
    "-Z":   (-mag, 0.0),
    "+Y+Z": (0.7*mag, 0.7*mag),
    "+Y-Z": (0.7*mag, -0.7*mag),
    "-Y+Z": (-0.7*mag, 0.7*mag),
    "-Y-Z": (-0.7*mag, -0.7*mag),
}

print("=" * 70, flush=True)
print(f"Y-Z MAPPER TEST (λ={lam}, max_δ={max_delta}m, target={mag}m, H={H})", flush=True)
print("=" * 70, flush=True)

print(f"J_yz:\n{J_yz.cpu().numpy()}", flush=True)
print(f"J_pinv [4x2]:\n{J_pinv.cpu().numpy()}", flush=True)

results = []
zero_trials = []

# Run zero baseline 3 times
for _ in range(3):
    ee0, ee1, dee, dq, cq_app, dc, clip = run_trial(None, "zero")
    zero_trials.append({"ee_delta": dee, "dq": dq})

zero_drift_mean = np.mean([t["ee_delta"] for t in zero_trials], axis=0)
zero_drift_norm = float(np.linalg.norm(zero_drift_mean))
print(f"\nZero drift: {zero_drift_mean} norm={zero_drift_norm:.4f}m", flush=True)

for name, (dz, dy) in directions.items():
    target = (dz, dy)  # (Z, Y) to match J rows
    ee0, ee1, dee_raw, dq, cq_app, dc, clip = run_trial(target, name)

    # Drift subtraction
    dee_sub = dee_raw - zero_drift_mean
    actual_dy = dee_sub[1]  # Y component
    actual_dz = dee_sub[0]  # Z component

    # Direction cosine
    target_vec = np.array([dz, dy])
    actual_vec = np.array([actual_dz, actual_dy])
    dot_prod = np.dot(actual_vec, target_vec)
    norms = np.linalg.norm(actual_vec) * np.linalg.norm(target_vec)
    dir_cos = dot_prod / max(norms, 1e-9)

    # Error metrics
    ee_start_yz = ee0[[1, 0]]  # reorder to (Z, Y)
    target_yz = ee_start_yz + target_vec
    initial_err = float(np.linalg.norm(target_vec))
    final_err = float(np.linalg.norm(target_yz - ee1[[1, 0]]))
    improvement = initial_err - final_err

    s2d = float(np.linalg.norm(dee_sub)) / max(zero_drift_norm, 1e-9)

    print(f"\n  {name}: init_err={initial_err:.4f} final_err={final_err:.4f} "
          f"imp={improvement:+.4f}m dir_cos={dir_cos:.2f} "
          f"clip={clip:.2f} s2d={s2d:.1f}",
          flush=True)
    print(f"    dee_raw=[{dee_raw[0]:.4f},{dee_raw[1]:.4f},{dee_raw[2]:.4f}] "
          f"dee_sub=[{dee_sub[0]:.4f},{dee_sub[1]:.4f},{dee_sub[2]:.4f}]",
          flush=True)
    print(f"    cyl_delta=[{dc[0]*1000:+.1f},{dc[1]*1000:+.1f},{dc[2]*1000:+.1f},{dc[3]*1000:+.1f}]mm "
          f"dq=[{dq[0]*1000:+.1f},{dq[1]*1000:+.1f},{dq[2]*1000:+.1f},{dq[3]*1000:+.1f}]mm",
          flush=True)

    results.append({
        "dir": name, "initial_err": initial_err, "final_err": final_err,
        "improvement": improvement, "dir_cos": dir_cos,
        "s2d": s2d, "clip": clip,
        "dee_sub_y": actual_dy, "dee_sub_z": actual_dz,
    })

# Summary
n = len(results)
mean_improve = np.mean([r["improvement"] for r in results])
median_dir_cos = np.median([r["dir_cos"] for r in results])
improved_count = sum(1 for r in results if r["improvement"] > 0)
mean_final = np.mean([r["final_err"] for r in results])
mean_initial = np.mean([r["initial_err"] for r in results])

print(f"\n{'='*70}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Mean initial error: {mean_initial:.4f} m", flush=True)
print(f"  Mean final error:   {mean_final:.4f} m", flush=True)
print(f"  Mean improvement:   {mean_improve:+.4f} m", flush=True)
print(f"  Improved: {improved_count}/{n}", flush=True)
print(f"  Median dir cosine:  {median_dir_cos:.2f}", flush=True)

pass_mapper = (mean_final < mean_initial and improved_count >= 6
               and median_dir_cos > 0.5)
print(f"\n  MAPPER PASS: {pass_mapper}", flush=True)

env.close()
simulation_app.close()
