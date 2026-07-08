#!/usr/bin/env python3
"""Inspect USD joint/constraint prims for the Mulag robot.

Lists every joint/constraint, its type, bodies, axis, drive API, limits.
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Audit USD closed-loop cylinders.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from pxr import Usd, UsdPhysics, PhysxSchema

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device

id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env
sim = uw.sim

# Access the USD stage
stage = sim.stage
if stage is None:
    print("ERROR: No USD stage accessible", flush=True)
    env.close(); simulation_app.close(); exit(1)

print("=" * 70, flush=True)
print("USD JOINT / CONSTRAINT INVENTORY", flush=True)
print("=" * 70, flush=True)

# Get the robot prim path
robot_prim_path = "/World/envs/env_0/Robot"
robot_prim = stage.GetPrimAtPath(robot_prim_path)
if not robot_prim:
    print(f"ERROR: Robot prim not found at {robot_prim_path}", flush=True)
    env.close(); simulation_app.close(); exit(1)

# Collect all joint-like prims under the robot
joint_prims = []
cylinder_prims = []

# Keywords for cylinder-related prims
cyl_keywords = ['zylinder', 'cylinder', 'kolben', 'rohr', 'piston', 'rod', 'hydraulic', 'hebe', 'knick', 'schwenk']

def traverse(prim, depth=0):
    """Recursively find joint/constraint prims."""
    if not prim:
        return
    path = prim.GetPath().pathString
    name = prim.GetName()
    ptype = prim.GetTypeName()

    # Check if it's a joint or constraint
    is_joint = (
        ptype in ('PhysicsRevoluteJoint', 'PhysicsPrismaticJoint', 'PhysicsFixedJoint',
                   'PhysicsDistanceJoint', 'PhysicsJoint', 'PhysicsSphericalJoint',
                   'PhysicsDriveAPI', 'PhysxJointAPI', 'PhysxArticulationAPI')
        or 'Joint' in ptype or 'Drive' in ptype or 'Constraint' in ptype
    )

    # Check for cylinder keywords in name or path
    is_cylinder = any(kw in name.lower() or kw.lower() in path.lower() for kw in cyl_keywords)

    if is_joint or is_cylinder:
        # Get detailed info
        info = {"path": path, "name": name, "type": ptype, "is_joint": is_joint, "is_cylinder": is_cylinder}

        # Try to get body0/body1
        try:
            jp = UsdPhysics.Joint(prim) if prim else None
        except:
            jp = None
        if jp:
            try: info["body0"] = str(jp.GetBody0Rel().GetTargets()[0]) if jp.GetBody0Rel().GetTargets() else "?"
            except: info["body0"] = "?"
            try: info["body1"] = str(jp.GetBody1Rel().GetTargets()[0]) if jp.GetBody1Rel().GetTargets() else "?"
            except: info["body1"] = "?"

        # Check DriveAPI
        drive_api = UsdPhysics.DriveAPI.Apply(prim, "linear" if "Prismatic" in ptype else "angular") if prim else None
        has_drive = False
        if prim:
            for api_name in ['linear', 'angular']:
                try:
                    da = UsdPhysics.DriveAPI.Get(stage, prim.GetPath().AppendProperty(f"drive:{api_name}:physics:targetPosition"))
                    if da:
                        has_drive = True
                        break
                except:
                    pass
        info["has_drive"] = has_drive

        if is_joint:
            joint_prims.append(info)
        if is_cylinder and not is_joint:
            cylinder_prims.append(info)

    # Traverse children
    for child in prim.GetChildren():
        traverse(child, depth + 1)

print("Traversing robot prim tree...", flush=True)
traverse(robot_prim)

# ---- Report Joints ----
print(f"\n{'='*70}")
print(f"JOINT/CONSTRAINT PRIMS ({len(joint_prims)} found)")
print(f"{'='*70}")
for j in joint_prims:
    print(f"  path: {j['path']}")
    print(f"    type={j['type']}  body0={j.get('body0','?')}  body1={j.get('body1','?')}  drive={j['has_drive']}")

# ---- Report Cylinder Bodies ----
print(f"\n{'='*70}")
print(f"CYLINDER-RELATED NON-JOINT PRIMS ({len(cylinder_prims)} found)")
print(f"{'='*70}")
for c in cylinder_prims:
    print(f"  path: {c['path']}  type={c['type']}")

# ---- IsaacLab articulation joints vs all joints ----
print(f"\n{'='*70}")
print("ISAACLAB ARTICULATION JOINTS")
print(f"{'='*70}")
jnames = uw.robot.data.joint_names
cyl_jids = uw._cylinder_joint_ids.detach().cpu().tolist()
for i, name in enumerate(jnames):
    is_cyl = "*** CYLINDER_TARGET ***" if i in cyl_jids else ""
    limits = uw.robot.data.soft_joint_pos_limits[0, i].detach().cpu().numpy()
    jtype = 'REVOLUTE' if abs(limits[1]-limits[0]) < 10 else 'PRISMATIC' if abs(limits[1]-limits[0]) > 100 else 'OTHER'
    act = ''
    for an, ac in uw.cfg.robot_cfg.actuators.items():
        jne = getattr(ac, 'joint_names_expr', '')
        if isinstance(jne, str) and jne == name: act = an; break
        if isinstance(jne, (list,tuple)) and name in jne: act = an; break
    print(f"  [{i:2d}] {name:40s} {jtype:10s} limits=[{limits[0]:.4f},{limits[1]:.4f}] act={act} {is_cyl}")

# ---- Check USD drive properties for cylinder constraint joints ----
print(f"\n{'='*70}")
print("CONSTRAINT DRIVE INSPECTION (cylinder-related joints)")
print(f"{'='*70}")
for j in joint_prims:
    if any(kw in j['name'].lower() for kw in cyl_keywords) or 'constraint' in j['name'].lower():
        prim = stage.GetPrimAtPath(j['path'])
        if prim:
            # Try to read drive properties
            for drive_type in ['linear', 'angular']:
                try:
                    tp_attr = prim.GetAttribute(f"drive:{drive_type}:physics:targetPosition")
                    if tp_attr and tp_attr.IsValid():
                        val = tp_attr.Get()
                        print(f"  {j['name']}: drive:{drive_type}:targetPosition = {val}")
                except:
                    pass
                try:
                    stiff_attr = prim.GetAttribute(f"drive:{drive_type}:physics:stiffness")
                    if stiff_attr and stiff_attr.IsValid():
                        val = stiff_attr.Get()
                        print(f"  {j['name']}: drive:{drive_type}:stiffness = {val}")
                except:
                    pass

env.close()
simulation_app.close()
