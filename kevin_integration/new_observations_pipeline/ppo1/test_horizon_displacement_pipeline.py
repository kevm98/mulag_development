#!/usr/bin/env python3
"""Horizon displacement pipeline with closed-loop ID/FD rollout.

Critical fixes:
1. Per-step ID input (delta_x_des_H / H, not horizon-level)
2. Closed-loop ID/FD rollout over H steps
3. Cloned GRU history for rollout (doesn't corrupt live state)
4. Tracking error uses PREVIOUS horizon error
5. Smooth PhysX position ramp execution
6. ID/FD contribution logged separately from tracking correction
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
parser.add_argument("--fd_horizon_gains", type=str, default="1,3,10")
parser.add_argument("--tracking_gains", type=str, default="0,0.25,0.5,1.0")
parser.add_argument("--cylinders", type=str, default="0,1,2,3")
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
deltas = [float(x.strip()) for x in args_cli.delta_cmds.split(",") if x.strip()]
horizons = [int(x.strip()) for x in args_cli.horizon_steps.split(",") if x.strip()]
fd_gains = [float(x.strip()) for x in args_cli.fd_horizon_gains.split(",") if x.strip()]
tr_gains = [float(x.strip()) for x in args_cli.tracking_gains.split(",") if x.strip()]
cids = [int(x.strip()) for x in args_cli.cylinders.split(",") if x.strip()]

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
cfg.low_level_servo_mode = "none"
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True; cfg.log_target_delta_xdot_clipping = False
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env.unwrapped; dt = float(uw.step_dt); dev = uw.device
settle = max(int(round(args_cli.settle_s / dt)), 0)
model_dt = float(getattr(uw, "_sequence_model_dt", 0.05))

# Load fine-tuned ID
from finetune_id_physx_dynamic import VelocityToValveMLP
id_ckpt = torch.load(str(id_path), map_location=dev, weights_only=False)
id_model = VelocityToValveMLP().to(dev)
id_model.load_state_dict(id_ckpt["model_state_dict"]); id_model.eval()
X_mean_t = torch.from_numpy(id_ckpt["X_mean"]).to(dev)
X_std_t = torch.from_numpy(id_ckpt["X_std"]).to(dev)

za = torch.zeros((uw.num_envs, uw.cfg.action_space), device=dev)
prev_tracking_error = {ci: 0.0 for ci in cids}
results = []; trial_n = 0
print("H_START", flush=True)

for h in horizons:
    for fg in fd_gains:
        for tg in tr_gains:
            for ci in cids:
                # Per-cylinder rollout state
                prev_meas = 0.0
                trigger_key = f"{h}_{fg}_{tg}_{ci}_prev"
                if trigger_key in prev_tracking_error:
                    prev_meas = prev_tracking_error[trigger_key]
                
                for dx_des in deltas:
                    trial_n += 1
                    try:
                        obs, _ = env.reset()
                        for _ in range(settle): obs, _, _, _, _ = env.step(za)
                        
                        cq0 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
                        ee0 = uw.robot.data.body_state_w[:, uw._ee_body_id, 0:3].detach().clone()  # EE position before
                        x0, xd0 = uw._cylinder_adapter.current_from_cylinder_joints(
                            cq0, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
                        
                        # --- Phase 1: Closed-loop ID/FD rollout (cloned history) ---
                        id_hist_saved = uw._sequence_id._step_history.detach().clone()
                        fd_hist_saved = uw._sequence_fd._step_history.detach().clone()
                        
                        x_roll = x0.detach().clone()
                        xd_roll = xd0.detach().clone()
                        valve_roll = torch.zeros((uw.num_envs, 4), device=dev)
                        fd_delta_sum = 0.0
                        vff_abs_sum = 0.0
                        
                        for step_h in range(h):
                            # FIX 1: Per-step ID input
                            remaining = h - step_h
                            dx_step = dx_des / h  # total / H, not remaining-based
                            v_step = dx_step / model_dt
                            
                            dd_step = torch.zeros((1, 4), device=dev); dd_step[:, ci] = dx_step
                            vd_step = torch.zeros((1, 4), device=dev); vd_step[:, ci] = v_step
                            
                            # Fine-tuned ID
                            id_input = torch.cat([dd_step, vd_step, x_roll, xd_roll], dim=-1)
                            id_input_n = (id_input - X_mean_t) / X_std_t
                            with torch.no_grad():
                                valve_step = id_model(id_input_n)
                            valve_roll = valve_step
                            vff_abs_sum += float(valve_step[:, ci].abs().mean().cpu())
                            
                            # FIX 3: Use cloned FD history
                            uw._sequence_fd.push_current_state(valve_roll, x_roll, xd_roll)
                            fd_out = uw._sequence_fd.predict_delta_extension()
                            fd_delta = float(fd_out.delta_extension_raw[:, ci].mean().cpu())
                            fd_delta_sum += fd_delta
                            
                            # Update rollout state
                            x_roll[:, ci] += fd_delta
                            xd_roll[:, ci] = v_step
                        
                        # Restore live histories
                        uw._sequence_id._step_history = id_hist_saved
                        uw._sequence_fd._step_history = fd_hist_saved
                        
                        # --- Phase 2: Deployment execution ---
                        # FIX 4: Tracking error from PREVIOUS horizon
                        tracking_err = dx_des - prev_meas if prev_meas != 0 else 0.0
                        
                        # FIX 6: Deployment delta
                        delta_cmd_H = fg * fd_delta_sum + tg * tracking_err
                        delta_cmd_H = np.clip(delta_cmd_H, -0.02, 0.02)
                        
                        idfd_contrib = abs(fg * fd_delta_sum) / max(abs(delta_cmd_H), 1e-9) if abs(delta_cmd_H) > 1e-9 else 0
                        
                        # Smooth position ramp execution
                        dx_learned = torch.zeros((1, 4), device=dev)
                        dx_learned[:, ci] = float(delta_cmd_H)
                        dx_isaac = uw._cylinder_adapter.simscape_delta_to_isaac_delta(dx_learned)
                        
                        for i in range(1, h + 1):
                            frac = i / h
                            interp_target = cq0 + frac * dx_isaac
                            uw._debug_manual_cylinder_position_target = interp_target.detach().clone()
                            obs, _, _, _, _ = env.step(za)
                        
                        uw._debug_manual_cylinder_position_target = None
                        
                        # Measure result
                        cq1 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
                        x1, _ = uw._cylinder_adapter.current_from_cylinder_joints(
                            cq1, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
                        dx_meas = float((x1 - x0)[:, ci].mean().cpu())
                        ee1 = uw.robot.data.body_state_w[:, uw._ee_body_id, 0:3].detach().clone()  # EE position after
                        
                        # Store tracking error for next horizon
                        prev_tracking_error[trigger_key] = dx_meas
                        
                        # Metrics
                        sign_ok = 1 if (np.sign(dx_meas) == np.sign(dx_des) and abs(dx_des) > 1e-9) else 0
                        disp_r = abs(dx_meas) / max(abs(dx_des), 1e-9)
                        
                        results.append(dict(h=h, fg=fg, tg=tg, cyl=ci, dx_des=dx_des,
                                           dx_meas=dx_meas, fd_sum=fd_delta_sum,
                                           delta_cmd=delta_cmd_H, sign_ok=sign_ok,
                                           disp_ratio=disp_r, idfd_contrib=idfd_contrib,
                                           vff_avg=vff_abs_sum/h))
                        
                        if trial_n % 15 == 0:
                            good = sum(1 for r in results if r["sign_ok"])
                            print(f"  T{trial_n} h={h} fg={fg:.0f} tg={tg:.1f} cyl={ci} dx={dx_des:+.3f}: "
                                  f"meas={dx_meas:+.5f} sign={'OK' if sign_ok else '??'} "
                                  f"ratio={disp_r:.2f} idfd={idfd_contrib:.2f} | good={good}/{len(results)}", flush=True)
                    
                    except Exception as e:
                        print(f"  E{trial_n}: {e}", flush=True)

env.close(); simulation_app.close()

if results:
    csv_path = outdir / "horizon_displacement_pipeline.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    
    sa = np.mean([r["sign_ok"] for r in results])
    dr = np.mean([r["disp_ratio"] for r in results])
    idfd = np.mean([r["idfd_contrib"] for r in results])
    
    sign_ok_rs = [r for r in results if r["sign_ok"]]
    if sign_ok_rs:
        best = min(sign_ok_rs, key=lambda r: abs(1.0 - r["disp_ratio"]))
    else:
        best = max(results, key=lambda r: r["sign_ok"] * 10)
    
    lines = [f"# Horizon Displacement Pipeline Report",
             f"- trials: {len(results)}, sign_acc: {sa:.3f}, disp_ratio: {dr:.3f}, idfd_contrib: {idfd:.3f}",
             f"- best: h={best['h']} fg={best['fg']} tg={best['tg']} cyl={best['cyl']} dx_des={best['dx_des']}",
             f"  meas={best['dx_meas']:.6f} ratio={best['disp_ratio']:.3f} sign_ok={best['sign_ok']} idfd={best['idfd_contrib']:.3f}"]
    (outdir / "horizon_displacement_pipeline_report.md").write_text("\n".join(lines))
    print(f"\nDONE sign={sa:.3f} ratio={dr:.3f} idfd={idfd:.3f} csv={csv_path}", flush=True)
else:
    print("NO_RESULTS", flush=True)
