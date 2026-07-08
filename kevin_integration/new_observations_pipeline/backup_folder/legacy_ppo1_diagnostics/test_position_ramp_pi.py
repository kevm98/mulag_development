#!/usr/bin/env python3
"""Position-ramp PI for cylinder velocity tracking — bypasses ID/FD pipeline."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cylinders", type=str, default="0")
parser.add_argument("--velocities", type=str, default="-0.02,-0.01,0.01,0.02")
parser.add_argument("--kp_pos", type=float, default=0.5)
parser.add_argument("--ki_pos", type=float, default=0.2)
parser.add_argument("--delta_clip", type=float, default=0.005)
parser.add_argument("--integral_limit", type=float, default=0.01)
parser.add_argument("--duration_s", type=float, default=2.0)
parser.add_argument("--settle_s", type=float, default=0.5)
parser.add_argument("--output_dir", type=str, default="outputs/cylinder_diagnostics/paper_matched_pipeline/pos_ramp_pi_test")
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
kp, ki = args_cli.kp_pos, args_cli.ki_pos
dclip, ilim = args_cli.delta_clip, args_cli.integral_limit
x_cmd = torch.zeros((uw.num_envs, 4), device=dev)
e_int = torch.zeros((uw.num_envs, 4), device=dev)
e_prev = torch.zeros((uw.num_envs, 4), device=dev)
vels = [float(x.strip()) for x in args_cli.velocities.split(",") if x.strip()]
cids = [int(x.strip()) for x in args_cli.cylinders.split(",") if x.strip()]
steps = max(int(round(args_cli.duration_s / dt)), 1)
settle = max(int(round(args_cli.settle_s / dt)), 0)
results = []

for ci in cids:
    for vv in vels:
        print(f"[pos-ramp-pi] cyl={ci} v={vv:+.4f}", flush=True)
        obs, _ = env.reset()
        za = torch.zeros((uw.num_envs, uw.cfg.action_space), device=dev)
        cq = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
        x_cur, _ = uw._cylinder_adapter.current_from_cylinder_joints(cq, torch.zeros_like(cq))
        x_cmd = x_cur.detach().clone(); e_int.zero_(); e_prev.zero_()
        v_des = torch.zeros((uw.num_envs, 4), device=dev)
        v_des[:, ci] = float(vv)
        samples = []
        for s_idx in range(steps + settle):
            cq = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
            cqd = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]
            xm, xdm = uw._cylinder_adapter.current_from_cylinder_joints(cq, cqd)
            xm = torch.nan_to_num(xm, nan=0.0, posinf=0.0, neginf=0.0)
            xdm = torch.nan_to_num(xdm, nan=0.0, posinf=0.0, neginf=0.0)
            x_cmd = x_cmd + v_des * dt
            ep = x_cmd - xm
            e_int = e_int + ep * dt
            e_int = torch.clamp(e_int, -ilim, ilim)
            dep = (ep - e_prev) / max(dt, 1e-9)
            e_prev = ep.detach().clone()
            dx = kp * ep + ki * e_int
            dx = torch.clamp(dx, -dclip, dclip)
            di = uw._cylinder_adapter.simscape_delta_to_isaac_delta(dx)
            pt = cq + di
            lo = uw.robot.data.soft_joint_pos_limits[:, uw._cylinder_joint_ids, 0]
            hi = uw.robot.data.soft_joint_pos_limits[:, uw._cylinder_joint_ids, 1]
            pt = torch.clamp(pt, lo, hi)
            uw.robot.set_joint_position_target(pt, joint_ids=uw._cylinder_joint_ids)
            obs, _, _, _, _ = env.step(za)
            if s_idx < settle: continue
            samples.append({"m": float(xdm[:, ci].mean().cpu()), "ep": float(ep[:, ci].mean().cpu())})
        if not samples: results.append({"cyl": ci, "v": vv, "err": "no samples"}); continue
        ma = np.array([s["m"] for s in samples])
        sa = float(np.mean(np.sign(ma) == np.sign(vv))) if abs(vv) > 0 else 1.0
        rm = float(np.sqrt(np.mean((ma - vv)**2)))
        at = float(np.abs(ma).mean() / max(abs(vv), 1e-9))
        print(f"  rmse={rm:.5f} sign={sa:.2f} atten={at:.2f} mean={float(ma.mean()):+.5f}", flush=True)
        results.append({"cyl": ci, "v_des": vv, "v_meas_mean": float(ma.mean()), "rmse": rm, "sign_acc": sa, "atten": at})

env.close(); simulation_app.close()
csv_path = outdir / "pos_ramp_pi_results.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cyl","v_des","v_meas_mean","rmse","sign_acc","atten"])
    w.writeheader(); w.writerows(results)
print(f"\nDone. Results: {csv_path}")
