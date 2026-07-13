#!/usr/bin/env python3
"""Run tests 4-9 within a single SimulationApp to avoid GPU OOM."""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import isaacsim  # noqa
from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})

import torch, numpy as np
from kevin_integration.new_observations_pipeline.extensive_debug.scripts.debug_utils import (
    build_mulag_env, get_tcp_pose, get_cylinder_state, get_all_joint_state,
    save_json, save_csv, save_report, save_log,
    ARM_JOINT_NAMES, CYLINDER_JOINT_NAMES,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

EPSILON = 0.002
SETTLE = 30
MEASURE = 8
DT = 1.0/120.0

def cyl_perturbation(env, device):
    """Test 4: Perturb each cylinder by +eps m."""
    robot = env.robot
    cyl_map = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jn in enumerate(robot.joint_names):
            if jn == name:
                cyl_map[name] = i
                break
    zero = torch.zeros((1,5), device=device)
    all_ids = list(range(robot.num_joints))
    results = []
    for cname in CYLINDER_JOINT_NAMES:
        if cname not in cyl_map:
            continue
        cid = cyl_map[cname]
        env.reset()
        for _ in range(SETTLE):
            env.step(zero)
        tcp0 = get_tcp_pose(env, 0)
        q0 = robot.data.joint_pos[0].detach().cpu().clone()
        target = q0.clone().unsqueeze(0).to(device)
        target[0, cid] += EPSILON
        limits = robot.data.soft_joint_pos_limits[0].to(device)
        target = torch.clamp(target, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
        for _ in range(MEASURE):
            robot.set_joint_position_target(target, joint_ids=all_ids)
            env.step(zero)
        tcp1 = get_tcp_pose(env, 0)
        cyl1 = get_cylinder_state(env, 0)
        dtcp_w = [tcp1["world_pos"][i]-tcp0["world_pos"][i] for i in range(3)]
        dtcp_b = [tcp1["base_pos"][i]-tcp0["base_pos"][i] for i in range(3)]
        dcyl = {n: cyl1[n]["pos"]-cyl1.get(n,{"pos":0})["pos"] if n in cyl1 else 0 for n in CYLINDER_JOINT_NAMES}
        results.append({"cylinder": cname, "joint_id": cid, "eps_m": EPSILON,
                        "dtcp_world": dtcp_w, "dtcp_base": dtcp_b, "cylinder_deltas": dcyl})
        print(f"  {cname}: dTCP_world={dtcp_w} dTCP_base={dtcp_b}")
    return results

def numerical_jacobian(env, device):
    """Test 6: Numerical J_cyl via central differences."""
    robot = env.robot
    cyl_map = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jn in enumerate(robot.joint_names):
            if jn == name:
                cyl_map[name] = i
                break
    zero = torch.zeros((1,5), device=device)
    all_ids = list(range(robot.num_joints))
    J_w = np.zeros((3,4))
    J_b = np.zeros((3,4))
    eps = EPSILON
    for c, cname in enumerate(CYLINDER_JOINT_NAMES):
        if cname not in cyl_map:
            continue
        cid = cyl_map[cname]
        env.reset()
        for _ in range(SETTLE):
            env.step(zero)
        q0 = robot.data.joint_pos[0].detach().cpu().clone()
        target_p = q0.clone().unsqueeze(0).to(device)
        target_p[0, cid] += eps
        limits = robot.data.soft_joint_pos_limits[0].to(device)
        target_p = torch.clamp(target_p, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
        for _ in range(MEASURE):
            robot.set_joint_position_target(target_p, joint_ids=all_ids)
            env.step(zero)
        tcp_p = get_tcp_pose(env, 0)
        env.reset()
        for _ in range(SETTLE):
            env.step(zero)
        q0n = robot.data.joint_pos[0].detach().cpu().clone()
        target_n = q0n.clone().unsqueeze(0).to(device)
        target_n[0, cid] -= eps
        target_n = torch.clamp(target_n, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
        for _ in range(MEASURE):
            robot.set_joint_position_target(target_n, joint_ids=all_ids)
            env.step(zero)
        tcp_n = get_tcp_pose(env, 0)
        for ax in range(3):
            J_w[ax,c] = (tcp_p["world_pos"][ax] - tcp_n["world_pos"][ax]) / (2*eps)
            J_b[ax,c] = (tcp_p["base_pos"][ax] - tcp_n["base_pos"][ax]) / (2*eps)
        print(f"  J_cyl col {c} ({cname}): world={J_w[:,c]} base={J_b[:,c]}")

    # Validate with damped pseudoinverse
    lam = 0.05
    JT = J_w.T
    pinv = JT @ np.linalg.inv(J_w @ JT + lam * np.eye(3))
    desired = np.array([0.005, 0.003, -0.002])
    ds = pinv @ desired
    env.reset()
    for _ in range(SETTLE):
        env.step(zero)
    tcp0 = get_tcp_pose(env, 0)
    q0 = robot.data.joint_pos[0].detach().cpu().clone()
    target = q0.clone().unsqueeze(0).to(device)
    for c, cname in enumerate(CYLINDER_JOINT_NAMES):
        if cname in cyl_map:
            target[0, cyl_map[cname]] += ds[c]
    target = torch.clamp(target, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
    for _ in range(MEASURE):
        robot.set_joint_position_target(target, joint_ids=all_ids)
        env.step(zero)
    tcp1 = get_tcp_pose(env, 0)
    actual = np.array([tcp1["world_pos"][i]-tcp0["world_pos"][i] for i in range(3)])
    pred = J_w @ ds
    S_w = np.linalg.svd(J_w, compute_uv=False)
    S_b = np.linalg.svd(J_b, compute_uv=False)
    cond_w = S_w[0]/S_w[-1] if S_w[-1]>1e-9 else float('inf')
    cond_b = S_b[0]/S_b[-1] if S_b[-1]>1e-9 else float('inf')
    err = float(np.linalg.norm(actual-desired))
    print(f"  J_world:\n{J_w}")
    print(f"  J_base:\n{J_b}")
    print(f"  SVs world: {S_w}, cond={cond_w:.2f}")
    print(f"  SVs base: {S_b}, cond={cond_b:.2f}")
    print(f"  Validation: desired={desired} actual={actual} err={err:.6f}")
    return J_w, J_b, S_w, S_b, cond_w, cond_b, desired, actual, ds, err

def q4_q5_coupling(env, device):
    """Test 7: Check Messerkopf_joint coupling to Messerkopf_Schwenk_joint."""
    robot = env.robot
    jmap = {}
    for name in ARM_JOINT_NAMES:
        for i, jn in enumerate(robot.joint_names):
            if jn == name:
                jmap[name] = i
                break
    j4 = jmap["Messerkopf_Schwenk_joint"]
    j5 = jmap["Messerkopf_joint"]
    zero = torch.zeros((1,5), device=device)
    all_ids = list(range(robot.num_joints))
    eps = 0.005
    # Test A: cmd q4, observe q5
    env.reset()
    for _ in range(SETTLE):
        env.step(zero)
    q0a = robot.data.joint_pos[0].detach().cpu().clone()
    ta = q0a.clone().unsqueeze(0).to(device)
    ta[0,j4] += eps
    limits = robot.data.soft_joint_pos_limits[0].to(device)
    ta = torch.clamp(ta, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
    for _ in range(MEASURE):
        robot.set_joint_position_target(ta, joint_ids=all_ids)
        env.step(zero)
    q1a = robot.data.joint_pos[0].detach().cpu().clone()
    dq4a = float(q1a[j4]-q0a[j4])
    dq5a = float(q1a[j5]-q0a[j5])
    # Test B: cmd q5, observe q4
    env.reset()
    for _ in range(SETTLE):
        env.step(zero)
    q0b = robot.data.joint_pos[0].detach().cpu().clone()
    tb = q0b.clone().unsqueeze(0).to(device)
    tb[0,j5] += eps
    tb = torch.clamp(tb, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
    for _ in range(MEASURE):
        robot.set_joint_position_target(tb, joint_ids=all_ids)
        env.step(zero)
    q1b = robot.data.joint_pos[0].detach().cpu().clone()
    dq4b = float(q1b[j4]-q0b[j4])
    dq5b = float(q1b[j5]-q0b[j5])
    # Test C: coupling ratio
    env.reset()
    for _ in range(SETTLE):
        env.step(zero)
    q0c = robot.data.joint_pos[0].detach().cpu().clone()
    tc = q0c.clone().unsqueeze(0).to(device)
    tc[0,j4] += eps*3
    tc = torch.clamp(tc, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
    for _ in range(MEASURE*2):
        robot.set_joint_position_target(tc, joint_ids=all_ids)
        env.step(zero)
    q1c = robot.data.joint_pos[0].detach().cpu().clone()
    dq4c = float(q1c[j4]-q0c[j4])
    dq5c = float(q1c[j5]-q0c[j5])
    ratio = dq5c/dq4c if abs(dq4c)>1e-9 else float('inf')
    print(f"  Test A (cmd q4): dq4={dq4a:+.6f} dq5={dq5a:+.6f}")
    print(f"  Test B (cmd q5): dq4={dq4b:+.6f} dq5={dq5b:+.6f}")
    print(f"  Test C (large q4): dq4={dq4c:+.6f} dq5={dq5c:+.6f} ratio={ratio:+.4f}")
    return {"test_a": {"dq4": dq4a, "dq5": dq5a}, "test_b": {"dq4": dq4b, "dq5": dq5b},
            "test_c": {"dq4": dq4c, "dq5": dq5c, "ratio": ratio},
            "conclusion": "q5 IS coupled to q4" if abs(ratio)>0.01 else "q5 appears independent"}

def ab_comparison(env, device, J_w):
    """Test 9: A/B/C/D convention comparison."""
    robot = env.robot
    cyl_map = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jn in enumerate(robot.joint_names):
            if jn == name:
                cyl_map[name] = i
                break
    zero = torch.zeros((1,5), device=device)
    all_ids = list(range(robot.num_joints))
    results = []
    for conv, label in [("A", "Current (sign=-1)"), ("B", "Flipped signs"), ("C", "Flipped + rate limit"), ("D", "J_cyl TCP")]:
        env.reset()
        for _ in range(SETTLE):
            env.step(zero)
        tcp0 = get_tcp_pose(env, 0)
        if conv == "D":
            desired = np.array([0.005, 0.003, -0.002])
            JT = J_w.T
            pinv = JT @ np.linalg.inv(J_w @ JT + 0.05*np.eye(3))
            delta = pinv @ desired
        else:
            delta = np.array([0.002, 0.002, 0.001, 0.001])
        if conv in ("B", "C"):
            delta = -delta
        if conv == "C":
            delta = np.clip(delta, -0.002, 0.002)
        q0 = robot.data.joint_pos[0].detach().cpu().clone()
        target = q0.clone().unsqueeze(0).to(device)
        limits = robot.data.soft_joint_pos_limits[0].to(device)
        for c, cn in enumerate(CYLINDER_JOINT_NAMES):
            if cn in cyl_map:
                target[0, cyl_map[cn]] += delta[c]
        target = torch.clamp(target, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
        max_vels = np.zeros(4)
        for _ in range(MEASURE):
            robot.set_joint_position_target(target, joint_ids=all_ids)
            env.step(zero)
            for c, cn in enumerate(CYLINDER_JOINT_NAMES):
                if cn in cyl_map:
                    v = abs(float(robot.data.joint_vel[0, cyl_map[cn]].detach().cpu().item()))
                    if v > max_vels[c]:
                        max_vels[c] = v
        tcp1 = get_tcp_pose(env, 0)
        dtcp = np.array([tcp1["world_pos"][i]-tcp0["world_pos"][i] for i in range(3)])
        tcp_mag = float(np.linalg.norm(dtcp))
        results.append({"convention": conv, "label": label, "cmd_delta": delta.tolist(),
                        "dtcp_world": dtcp.tolist(), "tcp_mag": tcp_mag,
                        "max_cyl_vel": max_vels.tolist(), "stable": tcp_mag < 0.5})
        print(f"  {conv}: TCP_mag={tcp_mag:.6f} stable={tcp_mag<0.5}")
    return results

def main():
    device = "cuda:0"
    print("="*72)
    print("Running Tests 4,6,7,9 within single SimulationApp")
    print("="*72)
    env = None
    try:
        env, cfg = build_mulag_env(num_envs=1, device=device, headless=True)

        # Test 4
        print("\n--- Test 4: Single-Cylinder Perturbation ---")
        cyl_results = cyl_perturbation(env, device)
        save_json(cyl_results, "single_cylinder_perturbation.json")
        csv_rows = []
        for r in cyl_results:
            row = {"cylinder": r["cylinder"], "dtcp_wx": r["dtcp_world"][0],
                   "dtcp_wy": r["dtcp_world"][1], "dtcp_wz": r["dtcp_world"][2],
                   "dtcp_bx": r["dtcp_base"][0], "dtcp_by": r["dtcp_base"][1], "dtcp_bz": r["dtcp_base"][2]}
            for n in CYLINDER_JOINT_NAMES:
                row[f"dcyl_{n}"] = r["cylinder_deltas"].get(n, 0)
            csv_rows.append(row)
        save_csv(csv_rows, "single_cylinder_perturbation.csv")
        save_log(f"Test 4 results: {len(cyl_results)} cylinders", "single_cylinder_perturbation.log")

        # Test 6
        print("\n--- Test 6: Numerical Cylinder Jacobian ---")
        J_w, J_b, S_w, S_b, cond_w, cond_b, desired, actual, ds, err = numerical_jacobian(env, device)
        jac_data = {"J_world": J_w.tolist(), "J_base": J_b.tolist(),
                    "sv_world": S_w.tolist(), "sv_base": S_b.tolist(),
                    "cond_world": float(cond_w), "cond_base": float(cond_b),
                    "validation": {"desired": desired.tolist(), "actual": actual.tolist(),
                                   "delta_s": ds.tolist(), "error_norm": err}}
        save_json(jac_data, "numerical_cylinder_jacobian.json")
        jac_csv = []
        for c in range(4):
            jac_csv.append({"cyl": CYLINDER_JOINT_NAMES[c],
                            "Jw_x": J_w[0,c], "Jw_y": J_w[1,c], "Jw_z": J_w[2,c],
                            "Jb_x": J_b[0,c], "Jb_y": J_b[1,c], "Jb_z": J_b[2,c]})
        save_csv(jac_csv, "numerical_cylinder_jacobian.csv")
        save_csv([{"desired_x": desired[0], "desired_y": desired[1], "desired_z": desired[2],
                   "actual_x": actual[0], "actual_y": actual[1], "actual_z": actual[2],
                   "error_norm": err}], "numerical_jacobian_validation.csv")
        save_log(f"J_world:\n{J_w}\nJ_base:\n{J_b}\nSVs world: {S_w} cond={cond_w:.2f}\nErr: {err:.6f}",
                 "numerical_cylinder_jacobian.log")
        md = []
        md.append("# Numerical Jacobian Validation\n\n")
        md.append(f"## J_cyl_world (3x4)\n```\n{J_w}\n```\n\n")
        md.append(f"## J_cyl_base (3x4)\n```\n{J_b}\n```\n\n")
        md.append(f"SVs world: {S_w}, cond={cond_w:.2f}\n")
        md.append(f"SVs base: {S_b}, cond={cond_b:.2f}\n\n")
        md.append(f"## Validation\n- Desired: {desired}\n- Actual: {actual}\n- Error: {err:.6f} m\n")
        save_report("".join(md), "numerical_jacobian_validation.md")

        # Test 7
        print("\n--- Test 7: q4/q5 Coupling ---")
        coupling = q4_q5_coupling(env, device)
        save_json(coupling, "q4_q5_coupling.json")
        save_csv([{"test": "A_cmd_q4", "dq4": coupling["test_a"]["dq4"], "dq5": coupling["test_a"]["dq5"]},
                  {"test": "B_cmd_q5", "dq4": coupling["test_b"]["dq4"], "dq5": coupling["test_b"]["dq5"]},
                  {"test": "C_coupling", "dq4": coupling["test_c"]["dq4"], "dq5": coupling["test_c"]["dq5"],
                   "ratio": coupling["test_c"]["ratio"]}], "q4_q5_coupling.csv")
        md7 = []
        md7.append("# q4/q5 Coupling\n\n")
        md7.append(f"A (cmd q4): dq4={coupling['test_a']['dq4']:+.6f} dq5={coupling['test_a']['dq5']:+.6f}\n")
        md7.append(f"B (cmd q5): dq4={coupling['test_b']['dq4']:+.6f} dq5={coupling['test_b']['dq5']:+.6f}\n")
        md7.append(f"C (large q4): ratio q5/q4 = {coupling['test_c']['ratio']:+.4f}\n\n")
        md7.append(f"**Conclusion**: {coupling['conclusion']}\n")
        ratio = coupling['test_c']['ratio']
        if abs(ratio) > 0.01:
            md7.append("A 5-DOF Jacobian is invalid. An effective 4-input Jacobian is needed.\n")
        save_report("".join(md7), "q4_q5_coupling.md")
        save_log(f"Coupling ratio: {ratio}", "q4_q5_coupling.log")

        # Test 9
        print("\n--- Test 9: A/B Comparison ---")
        ab = ab_comparison(env, device, J_w)
        save_json(ab, "ab_comparison.json")
        ab_csv = []
        for r in ab:
            row = {"conv": r["convention"], "label": r["label"], "tcp_mag": r["tcp_mag"],
                   "dtcp_x": r["dtcp_world"][0], "dtcp_y": r["dtcp_world"][1], "dtcp_z": r["dtcp_world"][2],
                   "stable": r["stable"]}
            ab_csv.append(row)
        save_csv(ab_csv, "ab_comparison.csv")
        md9 = ["# A/B Comparison\n\n"]
        md9.append("| Conv | TCP Mag (m) | Stable |\n|------|------------|--------|\n")
        for r in ab:
            md9.append(f"| {r['convention']} | {r['tcp_mag']:.6f} | {r['stable']} |\n")
        save_report("".join(md9), "ab_comparison.md")
        save_log(f"AB results: {ab}", "ab_comparison.log")

        print("\n" + "="*72)
        print("All remaining tests complete!")
        print("="*72)
    except Exception as exc:
        import traceback
        print(f"ERROR: {exc}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        if env is not None:
            env.close()
        _app.close()

if __name__ == "__main__":
    main()
