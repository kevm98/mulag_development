#!/usr/bin/env python3
"""Single-cylinder step test with explicit effort PD."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-cylinder step test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--gain_scale", type=float, default=1.0)
parser.add_argument("--damping_scale", type=float, default=1.0)
parser.add_argument("--effort_scale", type=float, default=1.0)
parser.add_argument("--step_m", type=float, default=0.001)
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
cfg.explicit_pd_gain_scale = float(args_cli.gain_scale)
cfg.explicit_pd_damping_scale = float(args_cli.damping_scale)
cfg.explicit_pd_effort_scale = float(args_cli.effort_scale)
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg); uw = env; B = env.num_envs; dev = uw.device
cyl_ids = uw._cylinder_joint_ids; tool_id = uw._tool_joint_id; ee_id = uw._ee_body_id
dt = float(uw.step_dt); pv = uw.robot.root_physx_view
jnames = uw.robot.data.joint_names
step_m = float(args_cli.step_m)

def tcp_b():
    ee = uw.robot.data.body_link_state_w[:, ee_id, :3]; bp = uw.robot.data.root_pos_w
    bq = uw.robot.data.root_quat_w; bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return quat_apply_inverse(bq, ee - bp)

print("=" * 70, flush=True)
print(f"SINGLE-CYLINDER STEP TEST (explicit PD, step={step_m}m, gain={args_cli.gain_scale})", flush=True)
print("=" * 70, flush=True)

results = []
for c in range(4):
    for sign in [+1, -1]:
        env.reset()
        root_pose = uw.robot.data.root_state_w[:, :7].clone()
        root_vel = torch.zeros((B, 6), device=dev)
        cq_hold = uw.robot.data.joint_pos[:, cyl_ids].clone()
        tool_target = uw.robot.data.joint_pos[:, tool_id].unsqueeze(-1).clone()
        ee0 = tcp_b()[0].detach().cpu().numpy()
        q0 = cq_hold[0].detach().cpu().numpy()

        # Set target: perturb cylinder c
        cq_target = cq_hold.clone()
        cq_target[0, c] += sign * step_m

        # Run 1 second with explicit PD
        q_final = None; ee_final = None; effort_final = None
        for s in range(int(1.0 / dt)):
            uw.robot.write_root_pose_to_sim(root_pose); uw.robot.write_root_velocity_to_sim(root_vel)
            uw.set_debug_manual_cylinder_targets(cq_target, tool_position_target=tool_target)
            env.step(torch.zeros((B, 5), device=dev))
            if s == int(1.0 / dt) - 1:
                q_final = uw.robot.data.joint_pos[0, cyl_ids].detach().cpu().numpy()
                ee_final = tcp_b()[0].detach().cpu().numpy()
                effort = pv.get_dof_actuation_forces()
                effort_final = effort[0].detach().cpu().numpy() if isinstance(effort, torch.Tensor) else np.zeros(21)

        cyl_delta = q_final - q0
        ee_delta = ee_final - ee0
        effort_vals = np.array([effort_final[jid] for jid in cyl_ids.tolist()])

        # Check: did cylinder c move in the commanded direction?
        cyl_c_moved = cyl_delta[c]
        sign_correct = (cyl_c_moved * sign) > 0

        print(f"\n  cyl{c} {'+' if sign>0 else '-'}{step_m*1000:.0f}mm: "
              f"dq_c={cyl_c_moved*1000:+.3f}mm "
              f"sign_ok={sign_correct} "
              f"ee_delta={[f'{x:.4f}' for x in ee_delta]} "
              f"effort_c={effort_vals[c]:.0f}N",
              flush=True)
        print(f"    all dq={[f'{x*1000:+.3f}' for x in cyl_delta]}mm  "
              f"all effort={[f'{x:.0f}' for x in effort_vals]}",
              flush=True)

        results.append({
            "cylinder": c, "sign": sign,
            "cyl_delta_c_mm": cyl_c_moved * 1000,
            "sign_correct": sign_correct,
            "ee_delta_norm": float(np.linalg.norm(ee_delta)),
            "effort_c": effort_vals[c],
        })

# Summary
n = len(results)
sign_acc = sum(1 for r in results if r["sign_correct"]) / n
nonzero_effort = sum(1 for r in results if abs(r["effort_c"]) > 1) / n
cyl_movements = [r["cyl_delta_c_mm"] for r in results]
ee_movements = [r["ee_delta_norm"] for r in results]

print(f"\n{'='*70}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Sign accuracy: {sign_acc:.2f} ({sum(1 for r in results if r['sign_correct'])}/{n})", flush=True)
print(f"  Nonzero effort: {nonzero_effort:.2f}", flush=True)
print(f"  Cyl delta range: [{min(cyl_movements):+.3f}, {max(cyl_movements):+.3f}] mm", flush=True)
print(f"  EE delta range:  [{min(ee_movements):.4f}, {max(ee_movements):.4f}] m", flush=True)
print(f"  ALL PASS: {sign_acc >= 6/8 and nonzero_effort >= 0.75}", flush=True)

env.close(); simulation_app.close()
