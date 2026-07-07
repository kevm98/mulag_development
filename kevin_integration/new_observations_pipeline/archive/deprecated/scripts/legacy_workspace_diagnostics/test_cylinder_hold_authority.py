#!/usr/bin/env python3
"""Test whether cylinder PD actuators can hold the arm under gravity.

Isolated test using HorizonIDFDEnv's debug_manual_cylinder_targets mechanism
to ensure targets persist through _apply_action.
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cylinder hold authority test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=5.0)
parser.add_argument("--hold_gain_scale", type=float, default=1.0)
parser.add_argument("--force_limit_scale", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from isaaclab.actuators import ImplicitActuatorCfg

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
cfg.horizon_pin_root_pose = True
cfg.horizon_reset_settle_steps = 0
cfg.ee_action_mode = "task_space_delta"
if getattr(args_cli, "device", None):
    cfg.device = args_cli.device
    cfg.sim.device = args_cli.device

# Apply gain/force scaling to actuator config
gs = float(args_cli.hold_gain_scale)
fs = float(args_cli.force_limit_scale)
base_acts = cfg.robot_cfg.actuators
scaled = {}
for name, act_cfg in base_acts.items():
    if hasattr(act_cfg, 'stiffness') and 'cylinder' in name.lower():
        scaled[name] = ImplicitActuatorCfg(
            joint_names_expr=act_cfg.joint_names_expr,
            stiffness=act_cfg.stiffness * gs,
            damping=act_cfg.damping * gs,
            effort_limit_sim=act_cfg.effort_limit_sim * fs,
        )
    else:
        scaled[name] = act_cfg
cfg.robot_cfg.actuators = scaled

id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

# Use HorizonIDFDEnv for proper target mechanism
env = HorizonIDFDEnv(base_cfg=cfg, horizon_steps=10, action_scale_m=0.01,
                     fd_gains=[2.0,3.0,3.0,3.0], device=args_cli.device)
uw = env._uw
B = env.num_envs
cyl_ids = uw._cylinder_joint_ids
dt = float(uw.step_dt)

print("=" * 60, flush=True)
print(f"CYLINDER HOLD AUTHORITY TEST (gain={gs}, force={fs})", flush=True)
print("=" * 60, flush=True)

# Reset
env.reset()

# Get hold target (set by the wrapper's reset)
cq0 = uw.robot.data.joint_pos[:, cyl_ids].detach().clone()
hold_target = env._cylinder_hold_target.detach().clone()
tool_target = env._tool_hold_target.detach().clone()

print(f"  Hold target: {hold_target[0].detach().cpu().numpy()}", flush=True)
print(f"  Current cyl: {cq0[0].detach().cpu().numpy()}", flush=True)

# Stabilize: write hold targets via debug mechanism for 60 steps
print("Stabilizing for 1s...", flush=True)
for _ in range(60):
    uw.set_debug_manual_cylinder_targets(hold_target, tool_position_target=tool_target)
    env._pin_root_pose_if_enabled()
    uw.step(torch.zeros((B, 5), device=uw.device))
    env._pin_root_pose_if_enabled()

# Refresh hold target to current settled position
cq_stable = uw.robot.data.joint_pos[:, cyl_ids].detach().clone()
hold_target = cq_stable.clone()
ee_stable = env._tcp_pos_base_frame()[0].detach().cpu().numpy()

print(f"  Settled cyl:  {cq_stable[0].detach().cpu().numpy()}", flush=True)
print(f"  Settled EE:   {ee_stable}", flush=True)

# ---- Main hold test ----
duration_s = float(args_cli.duration_s)
total_steps = int(duration_s / dt)
horizon_steps = 10

print(f"\nHolding for {duration_s}s ({total_steps} steps)...", flush=True)

ee_samples = []
cyl_samples = []
step = 0
horizon_ee_start = ee_stable.copy()
horizon_count = 0
ee_drift_per_horizon = []

while step < total_steps:
    step += 1
    uw.set_debug_manual_cylinder_targets(hold_target, tool_position_target=tool_target)
    env._pin_root_pose_if_enabled()
    uw.step(torch.zeros((B, 5), device=uw.device))
    env._pin_root_pose_if_enabled()

    if step % 10 == 0:
        ee = env._tcp_pos_base_frame()[0].detach().cpu().numpy()
        cq = uw.robot.data.joint_pos[0, cyl_ids].detach().cpu().numpy()
        ee_samples.append(ee)
        cyl_samples.append(cq)

        horizon_count += 1
        if horizon_count >= horizon_steps:
            drift = float(np.linalg.norm(ee - horizon_ee_start))
            ee_drift_per_horizon.append(drift)
            horizon_ee_start = ee.copy()
            horizon_count = 0

ee_final = env._tcp_pos_base_frame()[0].detach().cpu().numpy()
cq_final = uw.robot.data.joint_pos[0, cyl_ids].detach().cpu().numpy()
target_err = (hold_target[0] - torch.from_numpy(cq_final).to(uw.device)).detach().cpu().numpy()

# Results
ee_total_drift = float(np.linalg.norm(ee_final - ee_stable))
ee_max_h = max(ee_drift_per_horizon) if ee_drift_per_horizon else 0
ee_mean_h = np.mean(ee_drift_per_horizon) if ee_drift_per_horizon else 0
cyl_max_drift = float(np.max(np.abs(cq_final - cq_stable[0].detach().cpu().numpy())))

# Read applied efforts
try:
    efforts = uw.robot.data.applied_joint_effort[0, cyl_ids].detach().cpu().numpy()
    has_effort = True
except:
    efforts = None
    has_effort = False

print(f"\nResults:", flush=True)
print(f"  EE drift total:     {ee_total_drift:.4f} m", flush=True)
print(f"  EE drift max/horizon: {ee_max_h:.4f} m", flush=True)
print(f"  EE drift mean/horizon: {ee_mean_h:.4f} m", flush=True)
print(f"  Cyl max drift:       {cyl_max_drift:.6f} m", flush=True)
print(f"  Target error:        {target_err}", flush=True)
print(f"  EE start: {ee_stable}", flush=True)
print(f"  EE end:   {ee_final}", flush=True)
if has_effort and efforts is not None:
    print(f"  Applied efforts:     {efforts}", flush=True)

# Effort limits from config
eff_limits = {}
for name, act in base_acts.items():
    if hasattr(act, 'stiffness') and 'cylinder' in name.lower():
        eff_limits[name] = act.effort_limit_sim * fs
print(f"  Effort limits (scaled): {eff_limits}", flush=True)

pass_hold = ee_mean_h < 0.005 and ee_total_drift < 0.03
print(f"\nPASS: {pass_hold}", flush=True)
if not pass_hold:
    if gs < 10:
        print(f"  Try increasing hold_gain_scale (current={gs}).", flush=True)
    else:
        print(f"  Gain=10 still fails. Possible force limit saturation or unstable pose.", flush=True)

env.close()
simulation_app.close()
