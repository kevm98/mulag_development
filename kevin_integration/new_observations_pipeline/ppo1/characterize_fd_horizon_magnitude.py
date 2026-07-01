#!/usr/bin/env python3
"""Characterize FD horizon magnitude vs actual PhysX displacement per cylinder.

For each cylinder and horizon displacement:
1. Apply diagnostic PhysX position ramp (bypass ID/FD) → measure delta_x_meas_H
2. Run ID/FD closed-loop rollout → compute delta_x_fd_H  
3. calibration_ratio = delta_x_meas_H / delta_x_fd_H
"""
from __future__ import annotations
import argparse, csv, sys, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--delta_cmds", type=str, default="-0.02,-0.01,-0.005,0.005,0.01,0.02")
parser.add_argument("--horizon_steps", type=str, default="5,10,15,20")
parser.add_argument("--cylinders", type=str, default="0,1,2,3")
parser.add_argument("--settle_s", type=float, default=0.3)
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
deltas = [float(x.strip()) for x in args_cli.delta_cmds.split(",") if x.strip()]
horizons = [int(x.strip()) for x in args_cli.horizon_steps.split(",") if x.strip()]
cids = [int(x.strip()) for x in args_cli.cylinders.split(",") if x.strip()]

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
cfg.low_level_servo_mode = "none"  # We'll bypass the servo entirely
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True; cfg.log_target_delta_xdot_clipping = False
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env.unwrapped; dt = float(uw.step_dt); dev = uw.device
settle = max(int(round(args_cli.settle_s / dt)), 0)

# Load fine-tuned ID for rollout
from finetune_id_physx_dynamic import VelocityToValveMLP
id_ckpt = torch.load(str(id_path), map_location=dev, weights_only=False)
id_model = VelocityToValveMLP().to(dev)
id_model.load_state_dict(id_ckpt["model_state_dict"]); id_model.eval()
X_mean_t = torch.from_numpy(id_ckpt["X_mean"]).to(dev)
X_std_t = torch.from_numpy(id_ckpt["X_std"]).to(dev)

za = torch.zeros((uw.num_envs, uw.cfg.action_space), device=dev)
model_dt = float(getattr(uw, "_sequence_model_dt", 0.05))
results = []; trial_n = 0
print("CHAR_START", flush=True)

for h in horizons:
    for ci in cids:
        for dx_des in deltas:
            trial_n += 1
            
            # --- Phase 1: Diagnostic PhysX position ramp ---
            obs, _ = env.reset()
            for _ in range(settle):
                obs, _, _, _, _ = env.step(za)
                uw._debug_manual_cylinder_position_target = None  # clear override after step
            
            cq0 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
            x0, xd0 = uw._cylinder_adapter.current_from_cylinder_joints(cq0, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
            
            # Compute target in Isaac space: same delta as desired in learned space
            dx_learned = torch.zeros((1, 4), device=dev)
            dx_learned[:, ci] = float(dx_des)
            dx_isaac = uw._cylinder_adapter.simscape_delta_to_isaac_delta(dx_learned)
            cq_target = cq0 + dx_isaac
            
            # Write position ramp directly to PhysX (bypasses everything)
            for i in range(1, h + 1):
                frac = i / h
                interp_target = cq0 + frac * dx_isaac
                uw._debug_manual_cylinder_position_target = interp_target.detach().clone()
                obs, _, _, _, _ = env.step(za)
                uw._debug_manual_cylinder_position_target = None  # clear override after step
            
            cq1 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
            x1, xd1 = uw._cylinder_adapter.current_from_cylinder_joints(cq1, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
            dx_meas = float((x1 - x0)[:, ci].mean().cpu())
            
            # --- Phase 2: ID/FD closed-loop rollout ---
            # Reset env to same state
            obs, _ = env.reset()
            for _ in range(settle):
                obs, _, _, _, _ = env.step(za)
                uw._debug_manual_cylinder_position_target = None  # clear override after step
            
            cq_roll = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
            x_roll, xd_roll = uw._cylinder_adapter.current_from_cylinder_joints(cq_roll, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
            
            # Clone GRU histories
            id_hist_saved = uw._sequence_id._step_history.detach().clone() if hasattr(uw, '_sequence_id') and uw._sequence_id is not None else None
            fd_hist_saved = uw._sequence_fd._step_history.detach().clone() if hasattr(uw, '_sequence_fd') and uw._sequence_fd is not None else None
            
            valve_cmd = torch.zeros((uw.num_envs, 4), device=dev)
            fd_delta_sum = 0.0
            
            for step_h in range(h):
                # Per-step ID input correctly scaled
                remaining = h - step_h
                dx_step = dx_des / remaining
                v_step = dx_step / model_dt
                
                dd_step = torch.zeros((1, 4), device=dev); dd_step[:, ci] = dx_step
                vd_step = torch.zeros((1, 4), device=dev); vd_step[:, ci] = v_step
                
                # Fine-tuned ID: [desired_delta(4), desired_vel(4), ext(4), extvel(4)]
                id_input = torch.cat([dd_step, vd_step, x_roll, xd_roll], dim=-1)
                id_input_n = (id_input - X_mean_t) / X_std_t
                with torch.no_grad():
                    valve_step = id_model(id_input_n)
                valve_cmd = valve_step
                
                # Push current state + valve into FD history and predict delta
                if uw._sequence_fd is not None:
                    uw._sequence_fd.push_current_state(valve_cmd, x_roll, xd_roll)
                    fd_out = uw._sequence_fd.predict_delta_extension()
                    fd_delta = float(fd_out.delta_extension_raw[:, ci].mean().cpu())
                else:
                    fd_delta = 0.0
                
                fd_delta_sum += fd_delta
                
                # Update rollout state
                x_roll[:, ci] += fd_delta
                xd_roll[:, ci] = v_step  # approximate
            
            # Restore GRU histories
            if id_hist_saved is not None:
                uw._sequence_id._step_history = id_hist_saved
            if fd_hist_saved is not None:
                uw._sequence_fd._step_history = fd_hist_saved
            
            # Compute calibration
            cal_ratio = dx_meas / fd_delta_sum if abs(fd_delta_sum) > 1e-9 else 0.0
            
            results.append(dict(h=h, cyl=ci, dx_des=dx_des, dx_meas=dx_meas,
                               dx_fd=fd_delta_sum, cal_ratio=cal_ratio))
            
            print(f"  T{trial_n} h={h} cyl={ci} dx={dx_des:+.3f}: "
                  f"meas={dx_meas:+.6f} fd={fd_delta_sum:+.6f} ratio={cal_ratio:.2f}", flush=True)

env.close(); simulation_app.close()

if results:
    csv_path = outdir / "fd_horizon_magnitude_characterization.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    
    lines = ["# FD Horizon Magnitude Characterization"]
    for ci in cids:
        cr = [r for r in results if r["cyl"] == ci and abs(r["cal_ratio"]) > 0.01]
        if cr:
            ratios = [r["cal_ratio"] for r in cr]
            med = float(np.median(ratios))
            p25 = float(np.percentile(ratios, 25))
            p75 = float(np.percentile(ratios, 75))
            lines.append(f"- cyl {ci}: median={med:.2f} p25={p25:.2f} p75={p75:.2f} n={len(cr)}")
    lines.append(f"\n## Recommendations")
    for ci in cids:
        cr = [r for r in results if r["cyl"] == ci and abs(r["cal_ratio"]) > 0.01]
        if cr:
            med = float(np.median([r["cal_ratio"] for r in cr]))
            if med > 0:
                lines.append(f"- cyl {ci}: fd_horizon_gain ~ {1.0/med if med > 0.01 else 100:.0f}")
    
    (outdir / "fd_horizon_magnitude_characterization_report.md").write_text("\n".join(lines))
    print(f"\nCSV: {csv_path}", flush=True)
else:
    print("NO RESULTS", flush=True)
