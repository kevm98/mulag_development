#!/usr/bin/env python3
"""Audit actual PhysX DOF drive properties — read back from simulation.

Answers: Did the cfg stiffness/force actually reach PhysX?
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Audit PhysX DOF drives.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--stiffness_scale", type=float, default=200.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from isaaclab.actuators import ImplicitActuatorCfg

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
cfg.allow_debug_manual_cylinder_targets = True
if getattr(args_cli, "device", None):
    cfg.device = args_cli.device; cfg.sim.device = args_cli.device

# Set extreme stiffness to verify it reaches PhysX
ss = float(args_cli.stiffness_scale)
cfg.robot_cfg.actuators = {
    'drehzapfen_cylinder': ImplicitActuatorCfg(joint_names_expr=['Drehzapfen_joint'], stiffness=15_000_000*ss, damping=1_000_000*ss, effort_limit_sim=100_000_000*ss),
    'ausleger_i_cylinder': ImplicitActuatorCfg(joint_names_expr=['Ausleger_I_joint'], stiffness=6_000_000*ss, damping=600_000*ss, effort_limit_sim=10_000_000*ss),
    'ausleger_ii_cylinder': ImplicitActuatorCfg(joint_names_expr=['Ausleger_II_joint'], stiffness=1_000_000*ss, damping=500_000*ss, effort_limit_sim=1_000_000*ss),
    'messerkopf_schwenk_cylinder': ImplicitActuatorCfg(joint_names_expr=['Messerkopf_Schwenk_joint'], stiffness=1_000_000*ss, damping=500_000*ss, effort_limit_sim=100_000*ss),
    'messerkopf_passive_coupled': ImplicitActuatorCfg(joint_names_expr=['Messerkopf_joint'], stiffness=1_000_000*ss, damping=500_000*ss, effort_limit_sim=100_000*ss),
    'mulag_wheels': ImplicitActuatorCfg(joint_names_expr=['Wheel_Front_Left_joint','Wheel_Front_Right_joint','Wheel_Rear_Left_joint','Wheel_Rear_Right_joint','Wheel_Front_Left_Steering_joint','Wheel_Front_Right_Steering_joint'], stiffness=10000, damping=1000, effort_limit_sim=100000),
}

id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env
print("=" * 70, flush=True)
print(f"PHYSX DOF DRIVE AUDIT (cfg_stiffness_scale={ss})", flush=True)
print("=" * 70, flush=True)

# Get articulation view
art = uw.robot
pv = art.root_physx_view
num_dofs = pv.max_dof_count if hasattr(pv, 'max_dof_count') else 21
num_links = pv.max_links if hasattr(pv, 'max_links') else 0

print(f"\nArticulation: dof_count={num_dofs} link_count={num_links}", flush=True)
print(f"DOF names: {art.data.joint_names}", flush=True)
print(f"Body names: {art.body_names}", flush=True)

# Read PhysX drive properties
env.reset()
print(f"\n{'idx':>3s} {'name':<35s} {'limit_lower':>12s} {'limit_upper':>12s} {'position':>12s} {'stiffness':>14s} {'damping':>14s} {'max_force':>14s} {'actuator':<25s}", flush=True)
print("-" * 140, flush=True)

# Collect per-DOF data
positions = art.data.joint_pos[0].detach().cpu().numpy()
velocities = art.data.joint_vel[0].detach().cpu().numpy()
limits = art.data.soft_joint_pos_limits[0].detach().cpu().numpy()
defaults = art.data.default_joint_pos[0].detach().cpu().numpy()

# Read stiffness/damping from PhysX view
try:
    dof_stiffness = pv.get_dof_stiffness()  # Returns list of tensors
    dof_damping = pv.get_dof_damping()
    dof_max_forces = pv.get_dof_max_forces()
    dof_armatures = pv.get_dof_armatures()
except Exception as e:
    print(f"  Cannot read drive properties: {e}", flush=True)
    dof_stiffness = None

# Build actuator name lookup
actuator_map = {}
if hasattr(cfg.robot_cfg, 'actuators'):
    for act_name, act_cfg in cfg.robot_cfg.actuators.items():
        jn_expr = getattr(act_cfg, 'joint_names_expr', '')
        if isinstance(jn_expr, (list, tuple)):
            for jn in jn_expr:
                actuator_map[jn] = act_name
        elif isinstance(jn_expr, str):
            actuator_map[jn_expr] = act_name

for idx in range(min(num_dofs, len(positions))):
    name = art.data.joint_names[idx] if idx < len(art.data.joint_names) else f"dof_{idx}"
    pos = positions[idx]
    vel = velocities[idx]
    lo = limits[idx, 0] if idx < limits.shape[0] else 0
    hi = limits[idx, 1] if idx < limits.shape[0] else 0
    default = defaults[idx] if idx < len(defaults) else 0

    # Read drive props
    stiff = "N/A"
    damp = "N/A"
    max_f = "N/A"
    if dof_stiffness is not None:
        try:
            stiff_val = dof_stiffness[0][idx].item() if dof_stiffness[0].ndim > 0 else dof_stiffness[0].item()
            damp_val = dof_damping[0][idx].item() if dof_damping[0].ndim > 0 else dof_damping[0].item()
            maxf_val = dof_max_forces[0][idx].item() if dof_max_forces[0].ndim > 0 else dof_max_forces[0].item()
            stiff = f"{stiff_val:.1f}"
            damp = f"{damp_val:.1f}"
            max_f = f"{maxf_val:.1f}"
        except (IndexError, AttributeError):
            pass

    act_name = actuator_map.get(name, "(none)")

    print(f"{idx:3d} {name:<35s} {lo:12.6f} {hi:12.6f} {pos:12.6f} {stiff:>14s} {damp:>14s} {max_f:>14s} {act_name:<25s}", flush=True)

# Also print which joints are the cylinder IDs
print(f"\n--- Active cylinder joint IDs ---", flush=True)
for cid_name in ['_cylinder_joint_ids', '_arm_joint_ids', '_tool_joint_id', '_ee_body_id']:
    val = getattr(uw, cid_name, None)
    if val is not None:
        if isinstance(val, torch.Tensor):
            ids = val.detach().cpu().tolist() if val.ndim == 1 else val.flatten().detach().cpu().tolist()
            names = [art.data.joint_names[i] if i < len(art.data.joint_names) else f"?{i}" for i in ids]
            print(f"  {cid_name}: {ids} -> {names}", flush=True)
        else:
            print(f"  {cid_name}: {val}", flush=True)

# Cylinder joint positions
cyl_ids = uw._cylinder_joint_ids
print(f"\n--- Cylinder joint state ---", flush=True)
print(f"  pos:  {art.data.joint_pos[0, cyl_ids].detach().cpu().tolist()}", flush=True)
print(f"  vel:  {art.data.joint_vel[0, cyl_ids].detach().cpu().tolist()}", flush=True)
print(f"  limits: {art.data.soft_joint_pos_limits[0, cyl_ids].detach().cpu().tolist()}", flush=True)

# Verify stiffness values match cfg
print(f"\n--- Stiffness verification ---", flush=True)
if dof_stiffness is not None:
    for cid_name, ids in [('cylinder_joints', cyl_ids.detach().cpu().tolist())]:
        for idx, jid in enumerate(ids):
            cfg_stiff = list(cfg.robot_cfg.actuators.values())[idx].stiffness if idx < len(cfg.robot_cfg.actuators) else '?'
            try:
                physx_stiff = dof_stiffness[0][jid].item() if dof_stiffness[0].ndim > 0 else dof_stiffness[0].item()
                match = "MATCH" if abs(physx_stiff - float(cfg_stiff)) < 0.01 * abs(physx_stiff) else "MISMATCH"
                print(f"  {art.data.joint_names[jid]}: cfg={cfg_stiff:.0f} physx={physx_stiff:.0f} {match}", flush=True)
            except:
                pass

env.close()
simulation_app.close()
