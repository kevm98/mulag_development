#!/usr/bin/env python3
"""Sweep fd_gain, tracking_gain across servo_modes."""
from __future__ import annotations
import argparse, csv, sys, traceback, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--servo_modes", type=str, default="id_pid_fd_tracking_servo")
parser.add_argument("--fd_gains", type=str, default="1,5")
parser.add_argument("--tracking_gains", type=str, default="0,1.0")
parser.add_argument("--delta_clips", type=str, default="0.005")
parser.add_argument("--cylinders", type=str, default="0")
parser.add_argument("--velocities", type=str, default="-0.01,0.01")
parser.add_argument("--duration_s", type=float, default=1.0)
parser.add_argument("--settle_s", type=float, default=0.2)
parser.add_argument("--output_dir", type=str, default="outputs/cylinder_diagnostics/paper_matched_pipeline")
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
servo_modes = [s.strip() for s in args_cli.servo_modes.split(",") if s.strip()]
fd_gains = [float(x.strip()) for x in args_cli.fd_gains.split(",")]
tr_gains = [float(x.strip()) for x in args_cli.tracking_gains.split(",")]
d_clips = [float(x.strip()) for x in args_cli.delta_clips.split(",")]
cids = [int(x.strip()) for x in args_cli.cylinders.split(",")]
vels = [float(x.strip()) for x in args_cli.velocities.split(",")]

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
# Use absolute path for ID checkpoint
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True; cfg.log_target_delta_xdot_clipping = False
cfg.tracking_kp = 40.0; cfg.tracking_ki = 15.0; cfg.tracking_kd = 2.0
cfg.tracking_integral_limit = 0.5; cfg.cyl_vel_cmd_limit = 0.05; cfg.valve_cmd_limit = 1.0
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env.unwrapped; dt = float(uw.step_dt); dev = uw.device
steps = max(int(round(args_cli.duration_s / dt)), 1)
settle = max(int(round(args_cli.settle_s / dt)), 0)
total_trials = len(servo_modes)*len(fd_gains)*len(tr_gains)*len(d_clips)*len(cids)*len(vels)
print(f"ENV_READY {total_trials}", flush=True)

results = []; trial_n = 0

for servo_mode in servo_modes:
    cfg.low_level_servo_mode = servo_mode
    for fg in fd_gains:
        for tg in tr_gains:
            for dc in d_clips:
                cfg.fd_gain = fg; cfg.tracking_gain = tg; cfg.tracking_delta_clip_m = dc
                row = {"servo_mode": servo_mode, "fd_gain": fg, "tracking_gain": tg, "delta_clip": dc}
                cyl_sas, cyl_ats, cyl_rms = [], [], []
                for ci in cids:
                    for vv in vels:
                        trial_n += 1
                        try:
                            print(f"T{trial_n}", end="", flush=True)
                            obs, _ = env.reset()
                            print("R", end="", flush=True)
                            za = torch.zeros((uw.num_envs, uw.cfg.action_space), device=dev)
                            v_des = torch.zeros((uw.num_envs, 4), device=dev); v_des[:, ci] = float(vv)
                            uw.set_debug_cylinder_velocity_target(v_des)
                            uw._servo_fd_gain.fill_(fg)
                            uw._tracking_e_int.zero_(); uw._tracking_e_prev.zero_()
                            samples = []
                            for si in range(steps + settle):
                                obs, _, _, _, _ = env.step(za)
                                if si < settle: continue
                                samples.append(float(uw._cyl_vel_measured[:, ci].mean().cpu()))
                            if not samples: continue
                            ma = np.array(samples)
                            sa = float(np.mean(np.sign(ma) == np.sign(vv))) if abs(vv) > 0 else 1.0
                            rm = float(np.sqrt(np.mean((ma - vv)**2)))
                            at = float(np.abs(ma).mean() / max(abs(vv), 1e-9))
                            cyl_sas.append(sa); cyl_ats.append(at); cyl_rms.append(rm)
                            print(f".{sa:.2f}", flush=True)
                        except Exception as e:
                            print(f"!{e}", flush=True)
                if cyl_sas:
                    row["mean_sign_accuracy"] = float(np.mean(cyl_sas))
                    row["mean_attenuation"] = float(np.mean(cyl_ats))
                    row["mean_rmse"] = float(np.mean(cyl_rms))
                    results.append(row)
                    print(f" CFG {servo_mode} fg={fg:.0f} tg={tg:.1f}: sign={row['mean_sign_accuracy']:.3f}", flush=True)
                torch.cuda.empty_cache()

print(f"\nDONE {len(results)} configs", flush=True)

if results:
    csv_path = outdir / "deployment_actuator_tracking.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["servo_mode","fd_gain","tracking_gain","delta_clip","mean_sign_accuracy","mean_attenuation","mean_rmse"])
        w.writeheader(); w.writerows(results)
    best = max(results, key=lambda r: r["mean_sign_accuracy"])
    report = [f"# Deployment Actuator Tracking",
              f"Best: {best['servo_mode']} fg={best['fd_gain']} tg={best['tracking_gain']} sign={best['mean_sign_accuracy']:.3f}"]
    (outdir / "deployment_actuator_tracking_report.md").write_text("\n".join(report))
    print(f"BEST {best['servo_mode']} fg={best['fd_gain']} tg={best['tracking_gain']} sign={best['mean_sign_accuracy']:.3f}", flush=True)
env.close(); simulation_app.close()
