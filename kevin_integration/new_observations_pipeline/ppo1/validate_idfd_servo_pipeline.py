#!/usr/bin/env python3
"""Validate ID/FD servo pipeline end-to-end."""
from __future__ import annotations
import argparse, sys, csv
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--velocities", type=str, default="-0.03,-0.02,-0.01,-0.005,0.005,0.01,0.02,0.03")
parser.add_argument("--cylinders", type=str, default="0,1,2,3")
parser.add_argument("--duration_s", type=float, default=2.0)
parser.add_argument("--settle_s", type=float, default=0.5)
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
cfg.low_level_servo_mode = "none"  # We'll inject ID output as valve override
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True
cfg.log_target_delta_xdot_clipping = False
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env.unwrapped; dt = float(uw.step_dt); dev = uw.device

# Load fine-tuned ID
id_ckpt = torch.load(outdir / "id_physx_finetuned_best.pt", map_location="cpu", weights_only=False)
from finetune_id_physx_dynamic import VelocityToValveMLP
id_model = VelocityToValveMLP().to(dev)
id_model.load_state_dict(id_ckpt["model_state_dict"])
id_model.eval()
X_mean = torch.from_numpy(id_ckpt["X_mean"]).to(dev)
X_std = torch.from_numpy(id_ckpt["X_std"]).to(dev)

vels = [float(x.strip()) for x in args_cli.velocities.split(",") if x.strip()]
cids = [int(x.strip()) for x in args_cli.cylinders.split(",") if x.strip()]
steps = max(int(round(args_cli.duration_s / dt)), 1)
settle = max(int(round(args_cli.settle_s / dt)), 0)

results = []

for ci in cids:
    for vv in vels:
        print(f"[validate] cyl={ci} v={vv:+.4f}", flush=True)
        obs, _ = env.reset()
        za = torch.zeros((uw.num_envs, uw.cfg.action_space), device=dev)
        v_des = torch.zeros((uw.num_envs, 4), device=dev)
        v_des[:, ci] = float(vv)
        
        samples = []
        for si in range(steps + settle):
            cq = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
            cqd = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]
            x_cur, xd_cur = uw._cylinder_adapter.current_from_cylinder_joints(cq, cqd)
            
            # ID prediction: [desired_delta, desired_vel, ext, extvel] -> valve_cmd
            dd = v_des * dt
            id_input = torch.cat([dd, v_des, x_cur, xd_cur], dim=-1)
            id_input_n = (id_input - X_mean) / X_std
            with torch.no_grad():
                valve_id = id_model(id_input_n)
            valve_id = torch.clamp(valve_id, -0.9, 0.9)
            
            # Inject valve through debug override -> original FD -> adapter -> PhysX
            uw.set_debug_valve_cmd(valve_id)
            x_b = x_cur.detach().clone()
            obs, _, _, _, _ = env.step(za)
            
            cq_a = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
            cqd_a = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]
            x_a, xd_a = uw._cylinder_adapter.current_from_cylinder_joints(cq_a, cqd_a)
            
            if si < settle: continue
            samples.append({
                "v_des": vv, "v_meas": float(xd_a[:, ci].mean().cpu()),
                "valve_id": float(valve_id[:, ci].mean().cpu()),
                "delta_meas": float((x_a - x_b)[:, ci].mean().cpu()),
            })
        
        if not samples: results.append({"cyl": ci, "v_des": vv, "err": "no_samples"}); continue
        ma = np.array([s["v_meas"] for s in samples])
        va = np.array([s["valve_id"] for s in samples])
        da = np.array([s["delta_meas"] for s in samples])
        sa = float(np.mean(np.sign(ma) == np.sign(vv))) if abs(vv) > 0 else 1.0
        rm = float(np.sqrt(np.mean((ma - vv)**2)))
        at = float(np.abs(ma).mean() / max(abs(vv), 1e-9))
        print(f"  rmse={rm:.5f} sign={sa:.2f} atten={at:.2f} valve_mean={va.mean():+.3f} delta_mean={da.mean():+.6f}", flush=True)
        results.append({"cylinder": ci, "v_desired": vv, "v_measured_mean": float(ma.mean()),
                        "rmse": rm, "sign_accuracy": sa, "attenuation": at,
                        "valve_mean": float(va.mean()), "delta_mean": float(da.mean())})

env.close(); simulation_app.close()

csv_path = outdir / "idfd_servo_validation.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cylinder","v_desired","v_measured_mean","rmse","sign_accuracy","attenuation","valve_mean","delta_mean"])
    w.writeheader(); w.writerows(results)

report = [f"# ID/FD Servo Validation Report",
          f"- fine-tuned ID: id_physx_finetuned_best.pt",
          f"- FD: original (sequence FD checkpoint)",
          f"- trials: {len(results)}"]
for ci in cids:
    ci_res = [r for r in results if r.get("cylinder") == ci]
    if ci_res:
        sas = [r["sign_accuracy"] for r in ci_res]
        report.append(f"- cyl {ci}: sign_acc={np.mean(sas):.3f}, best={max(sas):.3f}")
(outdir / "idfd_servo_validation_report.md").write_text("\n".join(report))
print(f"Results: {csv_path}")
