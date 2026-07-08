#!/usr/bin/env python3
"""Calibrate fd_gain for current_horizon_idfd_ff controller only."""
from __future__ import annotations
import argparse, csv, sys, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--fd_gains", type=str, default="1.0,1.5,2.0,2.5,3.0")
parser.add_argument("--delta_cmds", type=str, default="-0.02,-0.01,-0.005,0.005,0.01,0.02")
parser.add_argument("--horizon_steps", type=str, default="10,15,20")
parser.add_argument("--cylinders", type=str, default="0,1,2,3")
parser.add_argument("--settle_s", type=float, default=0.2)
parser.add_argument("--output_dir", type=str,
                    default="outputs/cylinder_diagnostics/paper_matched_pipeline/controller_comparison")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""): args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from finetune_id_physx_dynamic import VelocityToValveMLP

outdir = Path(args_cli.output_dir); outdir.mkdir(parents=True, exist_ok=True)
fd_gains = [float(x.strip()) for x in args_cli.fd_gains.split(",")]
deltas = [float(x.strip()) for x in args_cli.delta_cmds.split(",")]
horizons = [int(x.strip()) for x in args_cli.horizon_steps.split(",")]
cids = [int(x.strip()) for x in args_cli.cylinders.split(",")]

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
cfg.low_level_servo_mode = "none"
cfg.enable_scripted_vehicle_motion = False; cfg.enable_startup_summary = True
cfg.log_target_delta_xdot_clipping = False
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env.unwrapped; dt = float(uw.step_dt); dev = uw.device
settle = max(int(round(args_cli.settle_s / dt)), 0)
model_dt = float(getattr(uw, "_sequence_model_dt", 0.05))

id_ckpt = torch.load(str(id_path), map_location=dev, weights_only=False)
id_model = VelocityToValveMLP().to(dev)
id_model.load_state_dict(id_ckpt["model_state_dict"]); id_model.eval()
X_mean_t = torch.from_numpy(id_ckpt["X_mean"]).to(dev)
X_std_t = torch.from_numpy(id_ckpt["X_std"]).to(dev)

za = torch.zeros((uw.num_envs, uw.cfg.action_space), device=dev)
results = []; trial_n = 0
total = len(fd_gains) * len(horizons) * len(cids) * len(deltas)
print(f"CAL_START {total}", flush=True)

for fg in fd_gains:
    for h in horizons:
        prev_meas = {ci: 0.0 for ci in cids}
        for ci in cids:
            for dx_des in deltas:
                trial_n += 1
                obs, _ = env.reset()
                for _ in range(settle): obs, _, _, _, _ = env.step(za)
                cq0 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
                x0, xd0 = uw._cylinder_adapter.current_from_cylinder_joints(cq0, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
                ee0 = uw.robot.data.body_state_w[:, uw._ee_body_id, 0:3].detach().clone()
                
                # ID/FD rollout (cloned history)
                id_hist = uw._sequence_id._step_history.detach().clone()
                fd_hist = uw._sequence_fd._step_history.detach().clone()
                x_r = x0.detach().clone(); xd_r = xd0.detach().clone()
                fd_sum = 0.0
                for _ in range(h):
                    dx_step = dx_des / h; v_step = dx_step / model_dt
                    dd = torch.zeros((1, 4), device=dev); dd[:, ci] = dx_step
                    vd = torch.zeros((1, 4), device=dev); vd[:, ci] = v_step
                    with torch.no_grad():
                        valve = id_model(torch.cat([dd, vd, x_r, xd_r], -1).sub(X_mean_t).div(X_std_t))
                    uw._sequence_fd.push_current_state(valve, x_r, xd_r)
                    fd_d = float(uw._sequence_fd.predict_delta_extension().delta_extension_raw[:, ci].mean().cpu())
                    fd_sum += fd_d; x_r[:, ci] += fd_d; xd_r[:, ci] = v_step
                uw._sequence_id._step_history = id_hist
                uw._sequence_fd._step_history = fd_hist
                
                # Controller A: delta_cmd_H = fg * fd_sum
                e_prev = dx_des - prev_meas[ci] if prev_meas[ci] != 0 else 0.0
                delta_cmd_H = fg * fd_sum
                delta_cmd_H = np.clip(delta_cmd_H, -0.02, 0.02)
                
                # Smooth ramp
                dx_learned = torch.zeros((1, 4), device=dev); dx_learned[:, ci] = float(delta_cmd_H)
                dx_isaac = uw._cylinder_adapter.simscape_delta_to_isaac_delta(dx_learned)
                for i in range(1, h + 1):
                    uw._debug_manual_cylinder_position_target = (cq0 + (i/h) * dx_isaac).detach().clone()
                    obs, _, _, _, _ = env.step(za)
                uw._debug_manual_cylinder_position_target = None
                
                cq1 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
                x1, _ = uw._cylinder_adapter.current_from_cylinder_joints(cq1, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
                dx_meas = float((x1 - x0)[:, ci].mean().cpu())
                ee1 = uw.robot.data.body_state_w[:, uw._ee_body_id, 0:3].detach().clone()
                prev_meas[ci] = dx_meas
                
                sign_ok = 1 if np.sign(dx_meas) == np.sign(dx_des) else 0
                disp_r = abs(dx_meas) / max(abs(dx_des), 1e-9)
                fd_c = abs(fg * fd_sum) / max(abs(delta_cmd_H), 1e-9)
                
                results.append(dict(fg=fg, h=h, cyl=ci, dx_des=dx_des, dx_meas=dx_meas,
                                   fd_sum=fd_sum, delta_cmd=delta_cmd_H,
                                   sign_ok=sign_ok, disp_r=disp_r, fd_c=fd_c,
                                   ee_d=float((ee1-ee0).norm(dim=-1).mean().cpu())))
                
                if trial_n % 30 == 0:
                    g = sum(1 for r in results if r["sign_ok"])
                    print(f"  [{trial_n}/{total}] fg={fg:.1f} h={h} cyl={ci} dx={dx_des:+.3f}: "
                          f"meas={dx_meas:+.5f} sign={'OK' if sign_ok else '??'} "
                          f"ratio={disp_r:.2f} good={g}/{len(results)}", flush=True)

# Save
_csv = outdir / "current_horizon_idfd_ff_gain_calibration.csv"
with _csv.open("w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=list(results[0].keys()))
    _w.writeheader(); _w.writerows(results)
print(f"CSV: {_csv}", flush=True)

# Report
lines = ["# FD Gain Calibration: current_horizon_idfd_ff", f"- trials: {len(results)}"]
best_fg = {}
for fg in fd_gains:
    fr = [r for r in results if r["fg"] == fg]
    if not fr: continue
    sa = np.mean([r["sign_ok"] for r in fr])
    dr = np.mean([r["disp_r"] for r in fr])
    grav = sum(1 for r in fr if r["cyl"] in (1,2) and r["dx_des"] > 0 and not r["sign_ok"])
    lines.append(f"fg={fg:.1f}: sign={sa:.3f} ratio={dr:.3f} grav_fails={grav} n={len(fr)}")

for ci in cids:
    best_fg[ci] = {"fg": 1.0, "ratio": 0, "sign": 0}
    lines.append(f"\ncyl {ci}:")
    for fg in fd_gains:
        cr = [r for r in results if r["fg"] == fg and r["cyl"] == ci]
        if not cr: continue
        sa = np.mean([r["sign_ok"] for r in cr])
        dr = np.mean([r["disp_r"] for r in cr])
        lines.append(f"  fg={fg:.1f}: sign={sa:.3f} ratio={dr:.3f}")
        if dr > best_fg[ci]["ratio"] and sa >= 0.93:
            best_fg[ci] = {"fg": fg, "ratio": dr, "sign": sa}

overall_best = sorted([(ci, v["fg"], v["ratio"]) for ci, v in best_fg.items()], key=lambda x: x[2], reverse=True)
lines.append(f"\nBest per cylinder: {[(ci, f'{fg:.1f}', f'{r:.3f}') for ci, fg, r in overall_best]}")
rec_fg = max(fd_gains, key=lambda fg: np.mean([r["disp_r"] for r in results if r["fg"] == fg and r["sign_ok"]]))
lines.append(f"\nRecommended fd_gain: {rec_fg}")

(outdir / "current_horizon_idfd_ff_gain_calibration_report.md").write_text("\n".join(lines))
print(f"Recommended fd_gain={rec_fg}", flush=True)

env.close(); simulation_app.close()
