#!/usr/bin/env python3
"""Structured hold test: calibration → verify → runtime → metrics.

Timeline:
  1. reset + pin root
  2. set q_target = current q
  3. run bias calibration (PID with ki_calib)
  4. verify calibration (bias_window_count > 0, force_bias nonzero)
  5. retarget q_des_runtime = current q
  6. reset metric buffers
  7. run runtime hold (bias + P + D + optional tiny I)
  8. compute metrics from runtime phase only

Phase 1 term isolation:
  --pid_mode bias_only | bias_d | bias_d_p | bias_d_p_i
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Structured hold test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=5.0)
parser.add_argument("--calibration_time_s", type=float, default=3.0)
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--pid_mode", type=str, default="bias_d_p",
                    choices=["bias_only","bias_d","bias_d_p","bias_d_p_i"])
parser.add_argument("--kp_scale", type=float, default=0.2)
parser.add_argument("--kd_scale", type=float, default=0.6)
parser.add_argument("--ki_scale", type=float, default=0.0)
parser.add_argument("--force_rate_limit", type=float, default=0.01)
parser.add_argument("--vel_filter_beta", type=float, default=0.8)
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
cfg.lowlevel_actuator_backend = args_cli.backend
cfg.bias_calibration_requires_static = False
cfg.bias_calibration_time_s = float(args_cli.calibration_time_s)
cfg.bias_calibration_ki_scale = 3.0
cfg.bias_runtime_ki_scale = float(args_cli.ki_scale) if args_cli.pid_mode == "bias_d_p_i" else 0.0

# Set term scales based on mode
mode = args_cli.pid_mode
kp_s = float(args_cli.kp_scale) if mode in ("bias_d_p", "bias_d_p_i") else 0.0
kd_s = float(args_cli.kd_scale) if mode in ("bias_d", "bias_d_p", "bias_d_p_i") else 0.0
cfg.explicit_pid_kp_scale = kp_s
cfg.explicit_pid_kd_scale = kd_s
cfg.explicit_pd_gain_scale = kp_s
cfg.explicit_pd_damping_scale = kd_s
cfg.explicit_pd_effort_scale = 1.0
cfg.explicit_pid_effort_scale = 1.0
cfg.explicit_force_rate_limit_fraction = float(args_cli.force_rate_limit)
cfg.explicit_pd_velocity_filter_beta = float(args_cli.vel_filter_beta)

if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg); uw = env; B = env.num_envs; dev = uw.device
cyl_ids = uw._cylinder_joint_ids; tool_id = uw._tool_joint_id; ee_id = uw._ee_body_id
dt = float(uw.step_dt); pv = uw.robot.root_physx_view
jnames = uw.robot.data.joint_names

def tcp_b():
    ee = uw.robot.data.body_link_state_w[:, ee_id, :3]; bp = uw.robot.data.root_pos_w
    bq = uw.robot.data.root_quat_w; bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return quat_apply_inverse(bq, ee - bp)

# ====================================================================
# STEP 1-2: Reset + pin + set target
# ====================================================================
env.reset()
root_pose = uw.robot.data.root_state_w[:, :7].clone(); root_vel = torch.zeros((B, 6), device=dev)
q_initial = uw.robot.data.joint_pos[:, cyl_ids].clone()
tool_target = uw.robot.data.joint_pos[:, tool_id].unsqueeze(-1).clone()

# ====================================================================
# STEP 3: Calibration phase
# ====================================================================
calib_s = float(args_cli.calibration_time_s)
calib_steps = int(calib_s / (dt / float(uw.cfg.decimation)))  # account for decimation
print(f"CALIBRATION: {calib_s}s ({calib_steps} steps) mode={mode}", flush=True)
for s in range(calib_steps):
    uw.robot.write_root_pose_to_sim(root_pose); uw.robot.write_root_velocity_to_sim(root_vel)
    uw.set_debug_manual_cylinder_targets(q_initial, tool_position_target=tool_target)
    env.step(torch.zeros((B, 5), device=dev))

# ====================================================================
# STEP 4: Verify calibration
# ====================================================================
calib_ok = True
if hasattr(uw, '_explicit_bias_calibrated'):
    calib_ok = bool(uw._explicit_bias_calibrated[0].item())
    bias_val = uw._explicit_force_bias[0].detach().cpu().numpy() if hasattr(uw, '_explicit_force_bias') else np.zeros(4)
    bias_count = uw._explicit_bias_window_count[0].item() if hasattr(uw, '_explicit_bias_window_count') else 0
    print(f"CALIBRATION: bias_calibrated={calib_ok} window_count={bias_count}", flush=True)
    for i, jid in enumerate(cyl_ids.tolist()):
        print(f"  {jnames[jid]}: bias={bias_val[i]:.0f}N", flush=True)
    if not calib_ok or bias_count == 0:
        print("CALIBRATION FAILED — cannot proceed to hold metrics", flush=True)
        env.close(); simulation_app.close(); exit(1)

# ====================================================================
# STEP 5: Retarget to calibrated equilibrium
# ====================================================================
q_runtime = uw.robot.data.joint_pos[:, cyl_ids].clone()
ee_runtime_start = tcp_b()[0].detach().cpu().numpy()
print(f"RUNTIME START: EE={ee_runtime_start}", flush=True)

# ====================================================================
# STEP 6-7: Runtime hold phase
# ====================================================================
ks = 10  # control steps per horizon sample
horizons_per_sec = int(1.0 / (dt * ks))
total_horizons = int(float(args_cli.duration_s) * horizons_per_sec)

ee_samples = []
eff_samples = []
fc_samples = []

for h in range(total_horizons):
    for _ in range(ks):
        uw.robot.write_root_pose_to_sim(root_pose); uw.robot.write_root_velocity_to_sim(root_vel)
        uw.set_debug_manual_cylinder_targets(q_runtime, tool_position_target=tool_target)
        env.step(torch.zeros((B, 5), device=dev))
    ee_samples.append(tcp_b()[0].detach().cpu().numpy())
    effort = pv.get_dof_actuation_forces()
    eff_arr = effort[0].detach().cpu().numpy() if isinstance(effort, torch.Tensor) else np.zeros(21)
    eff_samples.append(np.array([eff_arr[jid] for jid in cyl_ids.tolist()]))
    if hasattr(uw, '_last_force_cmd'):
        fc_samples.append(uw._last_force_cmd[0].detach().cpu().numpy())

ee_samples = np.array(ee_samples)

# ====================================================================
# STEP 8: Runtime-only metrics
# ====================================================================
if len(ee_samples) > 1:
    per_h = np.linalg.norm(ee_samples[1:] - ee_samples[:-1], axis=1)
    mean_h = float(np.mean(per_h))
    max_h = float(np.max(per_h))
    rms_h = float(np.sqrt(np.mean(per_h**2)))
    total = float(np.linalg.norm(ee_samples[-1] - ee_samples[0]))
    osc = float(np.sum(per_h) / total) if total > 1e-9 else 0
else:
    mean_h = max_h = rms_h = total = osc = 0

# Saturation
eff_lims = np.array([100e6, 10e6, 1e6, 100e3])
sat = float(np.mean(np.abs(eff_samples[-10:]) / (eff_lims + 1e-9))) if len(eff_samples) > 10 else 0

# Cylinder errors at end
cq_end = uw.robot.data.joint_pos[0, cyl_ids].detach().cpu().numpy()
cyl_errs = q_runtime[0].detach().cpu().numpy() - cq_end

print(f"\n{'='*60}", flush=True)
print(f"RUNTIME HOLD METRICS (mode={mode})", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Total drift:   {total:.4f} m", flush=True)
print(f"  Mean/horizon:  {mean_h:.6f} m  (target <0.005)", flush=True)
print(f"  Max/horizon:   {max_h:.4f} m  (target <0.02)", flush=True)
print(f"  RMS/horizon:   {rms_h:.6f} m  (target <0.01)", flush=True)
print(f"  Osc index:     {osc:.2f}  (target <2.0)", flush=True)
print(f"  Saturation:    {sat:.3f}  (target <0.3)", flush=True)
print(f"  Runtime start: {ee_samples[0]}", flush=True)
print(f"  Runtime end:   {ee_samples[-1]}", flush=True)

print(f"\n  Per-cylinder:", flush=True)
for i, jid in enumerate(cyl_ids.tolist()):
    fc = fc_samples[-1][i] if fc_samples else 0
    ea = eff_samples[-1][i] if len(eff_samples) > 0 else 0
    fb = bias_val[i] if 'bias_val' in dir() else 0
    print(f"    {jnames[jid]:30s} err={cyl_errs[i]*1000:+.2f}mm fc={fc:.0f}N bias={fb:.0f}N effort={ea:.0f}N", flush=True)

pass_hold = (mean_h < 0.005 and max_h < 0.02 and rms_h < 0.01
             and osc < 2.0 and total < 0.03 and sat < 0.3)
print(f"\n  PASS: {pass_hold}", flush=True)

env.close(); simulation_app.close()
