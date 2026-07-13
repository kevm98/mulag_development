#!/usr/bin/env python3
"""Test 4: Single-cylinder perturbation test.

Perturbs one cylinder/prismatic coordinate at a time by +0.002 m.
Measures which joints moved, cylinder deltas, TCP deltas.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import torch
from kevin_integration.new_observations_pipeline.extensive_debug.scripts.debug_utils import (
    _close_sim_app,
    build_mulag_env, get_tcp_pose, get_cylinder_state, get_all_joint_state,
    save_json, save_csv, save_log, ARM_JOINT_NAMES, CYLINDER_JOINT_NAMES,
)

EPSILON = 0.002  # m
SETTLE_STEPS = 30
MEASURE_STEPS = 10

def run_cyl_perturbation(env, cyl_name, cyl_joint_id, device):
    robot = env.robot
    env.reset()
    zero = torch.zeros((1, 5), device=device)
    for _ in range(SETTLE_STEPS):
        env.step(zero)
    initial_q = robot.data.joint_pos[0].detach().cpu().clone()
    initial_tcp = get_tcp_pose(env, env_id=0)
    initial_cyl = get_cylinder_state(env, env_id=0)
    all_ids = list(range(robot.num_joints))
    target = initial_q.clone().unsqueeze(0).to(device)
    target[0, cyl_joint_id] += EPSILON
    limits = robot.data.soft_joint_pos_limits[0].to(device)
    target = torch.clamp(target, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
    for _ in range(MEASURE_STEPS):
        robot.set_joint_position_target(target, joint_ids=all_ids)
        env.step(zero)
    final_q = robot.data.joint_pos[0].detach().cpu().clone()
    final_tcp = get_tcp_pose(env, env_id=0)
    final_cyl = get_cylinder_state(env, env_id=0)
    delta_q = (final_q - initial_q).tolist()
    delta_tcp_w = [final_tcp["world_pos"][i] - initial_tcp["world_pos"][i] for i in range(3)]
    delta_tcp_b = [final_tcp["base_pos"][i] - initial_tcp["base_pos"][i] for i in range(3)]
    cyl_deltas = {}
    for name in CYLINDER_JOINT_NAMES:
        if name in initial_cyl and name in final_cyl:
            cyl_deltas[name] = final_cyl[name]["pos"] - initial_cyl[name]["pos"]
    return {
        "cylinder_perturbed": cyl_name, "cylinder_joint_id": cyl_joint_id,
        "epsilon_m": EPSILON, "delta_q": delta_q,
        "delta_tcp_world": delta_tcp_w, "delta_tcp_base": delta_tcp_b,
        "cylinder_deltas": cyl_deltas,
    }

def main():
    p = argparse.ArgumentParser(description="Test 4: Single-cylinder perturbation")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", action="store_true", default=False)
    args = p.parse_args()
    device = args.device
    headless = not args.viewer
    print("=" * 72)
    print(f"Test 4: Single-Cylinder Perturbation (eps={EPSILON} m)")
    print("=" * 72)
    log_lines = []
    env = None
    try:
        env, cfg = build_mulag_env(num_envs=1, device=device, headless=headless)
        robot = env.robot
        cyl_map = {}
        for name in CYLINDER_JOINT_NAMES:
            for i, jn in enumerate(robot.joint_names):
                if jn == name:
                    cyl_map[name] = i
                    break
        results = []
        for cname in CYLINDER_JOINT_NAMES:
            if cname not in cyl_map:
                print(f"  SKIP {cname}: not found")
                log_lines.append(f"  SKIP {cname}: not found")
                continue
            cid = cyl_map[cname]
            print(f"\nPerturbing {cname} (id={cid}) by +{EPSILON} m...")
            r = run_cyl_perturbation(env, cname, cid, device)
            results.append(r)
            print(f"  Delta TCP world: {r['delta_tcp_world']}")
            print(f"  Delta TCP base:  {r['delta_tcp_base']}")
            print(f"  Cylinder deltas: {r['cylinder_deltas']}")
            log_lines.append(f"{cname}: TCP_world={r['delta_tcp_world']} TCP_base={r['delta_tcp_base']}")
        csv_rows = []
        joint_map = {}
        for name in ARM_JOINT_NAMES:
            for i, jn in enumerate(robot.joint_names):
                if jn == name:
                    joint_map[name] = i
                    break
        for r in results:
            row = {"cylinder_perturbed": r["cylinder_perturbed"], "joint_id": r["cylinder_joint_id"],
                   "eps_m": r["epsilon_m"],
                   "dtcp_wx": r["delta_tcp_world"][0], "dtcp_wy": r["delta_tcp_world"][1], "dtcp_wz": r["delta_tcp_world"][2],
                   "dtcp_bx": r["delta_tcp_base"][0], "dtcp_by": r["delta_tcp_base"][1], "dtcp_bz": r["delta_tcp_base"][2]}
            for nm in CYLINDER_JOINT_NAMES:
                row[f"dcyl_{nm}"] = r["cylinder_deltas"].get(nm, 0.0)
            for nm in ARM_JOINT_NAMES:
                if nm in joint_map:
                    row[f"dq_{nm}"] = r["delta_q"][joint_map[nm]]
            csv_rows.append(row)
        save_json(results, "single_cylinder_perturbation.json")
        save_csv(csv_rows, "single_cylinder_perturbation.csv")
        save_log("\n".join(log_lines), "single_cylinder_perturbation.log")
        print(f"\nTest 4 complete. {len(results)} cylinders perturbed.")
    except Exception as exc:
        import traceback
        err = f"ERROR: {exc}\n{traceback.format_exc()}"
        print(err); log_lines.append(err)
        save_log("\n".join(log_lines), "single_cylinder_perturbation.log")
        save_json({"status": "FAILED", "error": str(exc)}, "single_cylinder_perturbation.json")
        sys.exit(1)
    finally:
        _close_sim_app()
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
