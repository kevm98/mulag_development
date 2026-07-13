#!/usr/bin/env python3
"""Test 1: Runtime articulation inspection."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import torch
from kevin_integration.new_observations_pipeline.extensive_debug.scripts.debug_utils import (
    build_mulag_env, inspect_articulation, get_tcp_pose,
    get_cylinder_state, get_all_joint_state, save_json, save_log,
    ARM_JOINT_NAMES, CYLINDER_JOINT_NAMES, EE_BODY_NAME, _close_sim_app,
)

def main():
    p = argparse.ArgumentParser(description="Test 1: Articulation inspection")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", action="store_true", default=False)
    args = p.parse_args()
    headless = not args.viewer
    device = args.device
    print("=" * 72)
    print("Test 1: Runtime Articulation Inspection")
    print("=" * 72)
    log_lines = []
    env = None
    try:
        print("Building MULAG env...")
        env, cfg = build_mulag_env(num_envs=1, device=device, headless=headless)
        robot = env.robot
        log_lines.append(f"Device: {device}")
        log_lines.append(f"Num bodies: {robot.num_bodies}, joints: {robot.num_joints}")

        info = inspect_articulation(env)
        info["test_name"] = "articulation_inspection"
        info["device"] = device

        print(f"\nBodies ({robot.num_bodies}):")
        for i, name in enumerate(robot.body_names):
            print(f"  [{i:3d}] {name}")
            log_lines.append(f"  [{i:3d}] {name}")

        print(f"\nJoints ({robot.num_joints}):")
        for i, name in enumerate(robot.joint_names):
            plim = robot.data.soft_joint_pos_limits[0, i].detach().cpu().tolist()
            def_pos = robot.data.default_joint_pos[0, i].detach().cpu().item()
            cur_pos = robot.data.joint_pos[0, i].detach().cpu().item()
            line = (f"  [{i:3d}] {name:40s} pos_lim=[{plim[0]:+.4f}, {plim[1]:+.4f}] "
                    f"default={def_pos:+.6f} current={cur_pos:+.6f}")
            print(line)
            log_lines.append(line)

        # Resolve arm/cylinder joint IDs
        arm_joint_ids, cylinder_joint_ids = [], []
        for name in ARM_JOINT_NAMES:
            found = False
            for i, jn in enumerate(robot.joint_names):
                if jn == name:
                    arm_joint_ids.append(i)
                    if name in CYLINDER_JOINT_NAMES:
                        cylinder_joint_ids.append(i)
                    found = True
                    break
            status = f"joint_id={arm_joint_ids[-1]}" if found else "NOT FOUND!"
            print(f"  {name:40s} -> {status}")
            log_lines.append(f"  {name:40s} -> {status}")
        info["arm_joint_ids"] = arm_joint_ids
        info["cylinder_joint_ids"] = cylinder_joint_ids

        active_prismatic, passive_coupled = [], []
        for name in robot.joint_names:
            if name in CYLINDER_JOINT_NAMES:
                active_prismatic.append(name)
            elif name in ARM_JOINT_NAMES and name not in CYLINDER_JOINT_NAMES:
                passive_coupled.append(name)
        info["active_prismatic_cylinders"] = active_prismatic
        info["passive_coupled_candidates"] = passive_coupled
        print(f"\nActive prismatic (cylinder) joints: {active_prismatic}")
        print(f"Passive/coupled candidates:          {passive_coupled}")

        print(f"\nActuator groups:")
        for gn, gd in info.get("actuator_groups", {}).items():
            print(f"  {gn}: joints={gd.get('joint_names_expr')} stiff={gd.get('stiffness')} damp={gd.get('damping')}")
            log_lines.append(f"  {gn}: {json.dumps(gd)}")

        tcp = get_tcp_pose(env, env_id=0)
        info["tcp_pose"] = tcp
        print(f"\nTCP ({EE_BODY_NAME}): world={tcp['world_pos']} base={tcp['base_pos']}")

        cyl_state = get_cylinder_state(env, env_id=0)
        info["cylinder_state"] = cyl_state
        for name, s in cyl_state.items():
            print(f"  {name:40s} pos={s['pos']:+.6f} m  vel={s['vel']:+.6f} m/s")

        arm_state = get_all_joint_state(env, env_id=0)
        info["all_arm_joint_state"] = arm_state

        print(f"\nArm body poses (world frame):")
        for name in ARM_JOINT_NAMES:
            if name in info.get("body_poses", {}):
                bp = info["body_poses"][name]
                print(f"  {name:40s} pos={bp['pos']}")

        save_json(info, "articulation_inspection.json")
        save_log("\n".join(log_lines), "articulation_inspection.log")
        print(f"\nTest 1 complete.")
    except Exception as exc:
        import traceback
        err = f"ERROR: {exc}\n{traceback.format_exc()}"
        print(err); log_lines.append(err)
        save_log("\n".join(log_lines), "articulation_inspection.log")
        save_json({"status": "FAILED", "error": str(exc)}, "articulation_inspection.json")
        sys.exit(1)
    finally:
        if env is not None:
            env.close()
        _close_sim_app()

if __name__ == "__main__":
    main()
