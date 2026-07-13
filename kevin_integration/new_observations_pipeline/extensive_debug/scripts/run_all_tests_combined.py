#!/usr/bin/env python3
"""Run ALL tests within a single SimulationApp. Saves results incrementally."""
from __future__ import annotations
import sys, json, math, os, time
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import isaacsim  # noqa
from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})

import torch, numpy as np

# Must import debug_utils AFTER SimulationApp is created
from kevin_integration.new_observations_pipeline.extensive_debug.scripts.debug_utils import (
    build_mulag_env, get_tcp_pose, get_cylinder_state, get_all_joint_state,
    save_json, save_csv, save_report, save_log, inspect_articulation,
    ARM_JOINT_NAMES, CYLINDER_JOINT_NAMES, EE_BODY_NAME,
    RESULTS_DIR, LOGS_DIR, REPORTS_DIR,
)

EPSILON = 0.002
EPSILON_RAD = 0.005
SETTLE = 30
MEASURE = 10

def main():
    device = "cuda:0"
    print("="*72)
    print("Extensive Debug — All Tests (Combined)")
    print("="*72)
    env = None
    try:
        env, cfg = build_mulag_env(num_envs=1, device=device, headless=True)
        robot = env.robot
        zero = torch.zeros((1,5), device=device)
        all_ids = list(range(robot.num_joints))
        limits = robot.data.soft_joint_pos_limits[0].to(device)

        cyl_map = {}
        for name in CYLINDER_JOINT_NAMES:
            for i, jn in enumerate(robot.joint_names):
                if jn == name:
                    cyl_map[name] = i
                    break
        joint_map = {}
        for name in ARM_JOINT_NAMES:
            for i, jn in enumerate(robot.joint_names):
                if jn == name:
                    joint_map[name] = i
                    break

        def reset_and_settle():
            env.reset()
            for _ in range(SETTLE):
                env.step(zero)

        def apply_target_and_measure(target_tensor):
            for _ in range(MEASURE):
                robot.set_joint_position_target(target_tensor, joint_ids=all_ids)
                env.step(zero)

        # ── Test 1: Articulation Inspection ──────────────────────────
        print("\n--- Test 1: Articulation Inspection ---")
        info = inspect_articulation(env)
        info["test_name"] = "articulation_inspection"
        info["device"] = device
        info["arm_joint_ids"] = [joint_map.get(n) for n in ARM_JOINT_NAMES]
        info["cylinder_joint_ids"] = [cyl_map.get(n) for n in CYLINDER_JOINT_NAMES]
        active_prismatic = [n for n in CYLINDER_JOINT_NAMES if n in cyl_map]
        passive_coupled = [n for n in ARM_JOINT_NAMES if n not in CYLINDER_JOINT_NAMES and n in joint_map]
        info["active_prismatic_cylinders"] = active_prismatic
        info["passive_coupled_candidates"] = passive_coupled
        tcp = get_tcp_pose(env, 0)
        info["tcp_pose"] = tcp
        info["cylinder_state"] = get_cylinder_state(env, 0)
        info["all_arm_joint_state"] = get_all_joint_state(env, 0)
        save_json(info, "articulation_inspection.json")
        log_lines = [f"Bodies: {robot.num_bodies}, Joints: {robot.num_joints}"]
        for i, n in enumerate(robot.joint_names):
            plim = robot.data.soft_joint_pos_limits[0,i].detach().cpu().tolist()
            def_p = robot.data.default_joint_pos[0,i].detach().cpu().item()
            cur_p = robot.data.joint_pos[0,i].detach().cpu().item()
            log_lines.append(f"[{i:3d}] {n:40s} lim=[{plim[0]:+.4f},{plim[1]:+.4f}] def={def_p:+.6f} cur={cur_p:+.6f}")
        log_lines.append(f"Active cylinders: {active_prismatic}")
        log_lines.append(f"Passive/coupled: {passive_coupled}")
        log_lines.append(f"TCP world: {tcp['world_pos']}")
        log_lines.append(f"TCP base:  {tcp['base_pos']}")
        save_log("\n".join(log_lines), "articulation_inspection.log")
        print(f"  Bodies: {robot.num_bodies}, Joints: {robot.num_joints}")
        print(f"  Active cylinders: {active_prismatic}")
        print(f"  Passive/coupled: {passive_coupled}")

        # ── Test 3: Single-Joint Perturbation ────────────────────────
        print("\n--- Test 3: Single-Joint Perturbation ---")
        jt_results = []
        for jname in ARM_JOINT_NAMES:
            if jname not in joint_map:
                continue
            jid = joint_map[jname]
            reset_and_settle()
            tcp0 = get_tcp_pose(env, 0)
            q0 = robot.data.joint_pos[0].detach().cpu().clone()
            target = q0.clone().unsqueeze(0).to(device)
            target[0, jid] += EPSILON_RAD
            target = torch.clamp(target, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
            apply_target_and_measure(target)
            tcp1 = get_tcp_pose(env, 0)
            q1 = robot.data.joint_pos[0].detach().cpu().clone()
            cyl1 = get_cylinder_state(env, 0)
            dtcp_w = [tcp1["world_pos"][i]-tcp0["world_pos"][i] for i in range(3)]
            dtcp_b = [tcp1["base_pos"][i]-tcp0["base_pos"][i] for i in range(3)]
            dcyl = {n: cyl1[n]["pos"]-cyl1.get(n,{"pos":0})["pos"] if n in cyl1 else 0 for n in CYLINDER_JOINT_NAMES}
            dq = (q1-q0).tolist()
            jt_results.append({"joint": jname, "joint_id": jid, "eps_rad": EPSILON_RAD,
                               "dtcp_world": dtcp_w, "dtcp_base": dtcp_b, "cylinder_deltas": dcyl})
            print(f"  {jname}: dTCP_w={[f'{v:+.4f}' for v in dtcp_w]} dTCP_b={[f'{v:+.4f}' for v in dtcp_b]}")
        save_json(jt_results, "single_joint_perturbation.json")
        jt_csv = []
        for r in jt_results:
            row = {"joint": r["joint"], "dtcp_wx": r["dtcp_world"][0], "dtcp_wy": r["dtcp_world"][1],
                   "dtcp_wz": r["dtcp_world"][2], "dtcp_bx": r["dtcp_base"][0], "dtcp_by": r["dtcp_base"][1],
                   "dtcp_bz": r["dtcp_base"][2]}
            for n in CYLINDER_JOINT_NAMES:
                row[f"dcyl_{n}"] = r["cylinder_deltas"].get(n,0)
            jt_csv.append(row)
        save_csv(jt_csv, "single_joint_perturbation.csv")
        save_log(f"Joint perturbation: {len(jt_results)} joints", "single_joint_perturbation.log")

        # ── Test 4: Single-Cylinder Perturbation ─────────────────────
        print("\n--- Test 4: Single-Cylinder Perturbation ---")
        cyl_results = []
        for cname in CYLINDER_JOINT_NAMES:
            if cname not in cyl_map:
                continue
            cid = cyl_map[cname]
            reset_and_settle()
            tcp0 = get_tcp_pose(env, 0)
            cyl0 = get_cylinder_state(env, 0)
            q0 = robot.data.joint_pos[0].detach().cpu().clone()
            target = q0.clone().unsqueeze(0).to(device)
            target[0, cid] += EPSILON
            target = torch.clamp(target, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
            apply_target_and_measure(target)
            tcp1 = get_tcp_pose(env, 0)
            q1 = robot.data.joint_pos[0].detach().cpu().clone()
            cyl1 = get_cylinder_state(env, 0)
            dtcp_w = [tcp1["world_pos"][i]-tcp0["world_pos"][i] for i in range(3)]
            dtcp_b = [tcp1["base_pos"][i]-tcp0["base_pos"][i] for i in range(3)]
            dcyl = {}
            for n in CYLINDER_JOINT_NAMES:
                if n in cyl0 and n in cyl1:
                    dcyl[n] = cyl1[n]["pos"] - cyl0[n]["pos"]
            dq = {}
            for n in ARM_JOINT_NAMES:
                if n in joint_map:
                    dq[n] = float(q1[joint_map[n]] - q0[joint_map[n]])
            cyl_results.append({"cylinder": cname, "joint_id": cid, "eps_m": EPSILON,
                               "dtcp_world": dtcp_w, "dtcp_base": dtcp_b,
                               "cylinder_deltas": dcyl, "joint_deltas": dq})
            print(f"  {cname}: dTCP_w={[f'{v:+.4f}' for v in dtcp_w]} dTCP_b={[f'{v:+.4f}' for v in dtcp_b]}")
        save_json(cyl_results, "single_cylinder_perturbation.json")
        cyl_csv = []
        for r in cyl_results:
            row = {"cylinder": r["cylinder"], "dtcp_wx": r["dtcp_world"][0], "dtcp_wy": r["dtcp_world"][1],
                   "dtcp_wz": r["dtcp_world"][2], "dtcp_bx": r["dtcp_base"][0], "dtcp_by": r["dtcp_base"][1],
                   "dtcp_bz": r["dtcp_base"][2]}
            for n in CYLINDER_JOINT_NAMES:
                row[f"dcyl_{n}"] = r["cylinder_deltas"].get(n,0)
            for n in ARM_JOINT_NAMES:
                row[f"dq_{n}"] = r["joint_deltas"].get(n,0)
            cyl_csv.append(row)
        save_csv(cyl_csv, "single_cylinder_perturbation.csv")
        save_log(f"Cylinder perturbation: {len(cyl_results)} cylinders", "single_cylinder_perturbation.log")

        # ── Test 6: Numerical Cylinder Jacobian ──────────────────────
        print("\n--- Test 6: Numerical Cylinder Jacobian ---")
        J_w = np.zeros((3,4))
        J_b = np.zeros((3,4))
        for c, cname in enumerate(CYLINDER_JOINT_NAMES):
            if cname not in cyl_map:
                continue
            cid = cyl_map[cname]
            reset_and_settle()
            q0 = robot.data.joint_pos[0].detach().cpu().clone()
            tp = q0.clone().unsqueeze(0).to(device)
            tp[0, cid] += EPSILON
            tp = torch.clamp(tp, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
            apply_target_and_measure(tp)
            tcp_p = get_tcp_pose(env, 0)
            reset_and_settle()
            q0n = robot.data.joint_pos[0].detach().cpu().clone()
            tn = q0n.clone().unsqueeze(0).to(device)
            tn[0, cid] -= EPSILON
            tn = torch.clamp(tn, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
            apply_target_and_measure(tn)
            tcp_n = get_tcp_pose(env, 0)
            for ax in range(3):
                J_w[ax,c] = (tcp_p["world_pos"][ax] - tcp_n["world_pos"][ax]) / (2*EPSILON)
                J_b[ax,c] = (tcp_p["base_pos"][ax] - tcp_n["base_pos"][ax]) / (2*EPSILON)
            print(f"  J_cyl col {c} ({cname}): world={J_w[:,c]}")
        lam = 0.05
        JT = J_w.T
        pinv = JT @ np.linalg.inv(J_w @ JT + lam * np.eye(3))
        desired = np.array([0.005, 0.003, -0.002])
        ds = pinv @ desired
        reset_and_settle()
        tcp0 = get_tcp_pose(env, 0)
        q0 = robot.data.joint_pos[0].detach().cpu().clone()
        target = q0.clone().unsqueeze(0).to(device)
        for c, cn in enumerate(CYLINDER_JOINT_NAMES):
            if cn in cyl_map:
                target[0, cyl_map[cn]] += ds[c]
        target = torch.clamp(target, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
        apply_target_and_measure(target)
        tcp1 = get_tcp_pose(env, 0)
        actual = np.array([tcp1["world_pos"][i]-tcp0["world_pos"][i] for i in range(3)])
        pred = J_w @ ds
        S_w = np.linalg.svd(J_w, compute_uv=False)
        S_b = np.linalg.svd(J_b, compute_uv=False)
        cond_w = S_w[0]/S_w[-1] if S_w[-1]>1e-9 else float('inf')
        cond_b = S_b[0]/S_b[-1] if S_b[-1]>1e-9 else float('inf')
        err = float(np.linalg.norm(actual-desired))
        jac_data = {"J_world": J_w.tolist(), "J_base": J_b.tolist(),
                    "sv_world": S_w.tolist(), "sv_base": S_b.tolist(),
                    "cond_world": float(cond_w), "cond_base": float(cond_b),
                    "validation": {"desired": desired.tolist(), "actual": actual.tolist(),
                                   "delta_s": ds.tolist(), "error_norm": err}}
        save_json(jac_data, "numerical_cylinder_jacobian.json")
        jac_csv = [{"cyl": CYLINDER_JOINT_NAMES[c], "Jw_x": J_w[0,c], "Jw_y": J_w[1,c], "Jw_z": J_w[2,c],
                     "Jb_x": J_b[0,c], "Jb_y": J_b[1,c], "Jb_z": J_b[2,c]} for c in range(4)]
        save_csv(jac_csv, "numerical_cylinder_jacobian.csv")
        save_csv([{"desired_x": desired[0], "desired_y": desired[1], "desired_z": desired[2],
                   "actual_x": actual[0], "actual_y": actual[1], "actual_z": actual[2],
                   "error_norm": err}], "numerical_jacobian_validation.csv")
        print(f"  SVs: {S_w}, cond={cond_w:.2f}, err={err:.6f}")

        # ── Test 7: q4/q5 Coupling ───────────────────────────────────
        print("\n--- Test 7: q4/q5 Coupling ---")
        j4 = joint_map["Messerkopf_Schwenk_joint"]
        j5 = joint_map["Messerkopf_joint"]
        eps_r = 0.005
        # Test A
        reset_and_settle()
        q0a = robot.data.joint_pos[0].detach().cpu().clone()
        ta = q0a.clone().unsqueeze(0).to(device)
        ta[0,j4] += eps_r
        ta = torch.clamp(ta, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
        apply_target_and_measure(ta)
        q1a = robot.data.joint_pos[0].detach().cpu().clone()
        dq4a, dq5a = float(q1a[j4]-q0a[j4]), float(q1a[j5]-q0a[j5])
        # Test B
        reset_and_settle()
        q0b = robot.data.joint_pos[0].detach().cpu().clone()
        tb = q0b.clone().unsqueeze(0).to(device)
        tb[0,j5] += eps_r
        tb = torch.clamp(tb, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
        apply_target_and_measure(tb)
        q1b = robot.data.joint_pos[0].detach().cpu().clone()
        dq4b, dq5b = float(q1b[j4]-q0b[j4]), float(q1b[j5]-q0b[j5])
        # Test C
        reset_and_settle()
        q0c = robot.data.joint_pos[0].detach().cpu().clone()
        tc = q0c.clone().unsqueeze(0).to(device)
        tc[0,j4] += eps_r*3
        tc = torch.clamp(tc, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
        for _ in range(MEASURE*2):
            robot.set_joint_position_target(tc, joint_ids=all_ids)
            env.step(zero)
        q1c = robot.data.joint_pos[0].detach().cpu().clone()
        dq4c, dq5c = float(q1c[j4]-q0c[j4]), float(q1c[j5]-q0c[j5])
        ratio = dq5c/dq4c if abs(dq4c)>1e-9 else float('inf')
        coupling = {"test_a": {"dq4": dq4a, "dq5": dq5a}, "test_b": {"dq4": dq4b, "dq5": dq5b},
                    "test_c": {"dq4": dq4c, "dq5": dq5c, "ratio": ratio},
                    "conclusion": "q5 IS coupled to q4" if abs(ratio)>0.01 else "q5 appears independent"}
        save_json(coupling, "q4_q5_coupling.json")
        save_csv([{"test": "A_cmd_q4", "dq4": dq4a, "dq5": dq5a},
                  {"test": "B_cmd_q5", "dq4": dq4b, "dq5": dq5b},
                  {"test": "C_coupling", "dq4": dq4c, "dq5": dq5c, "ratio": ratio}], "q4_q5_coupling.csv")
        md7 = f"# q4/q5 Coupling\n\nA: dq4={dq4a:+.6f} dq5={dq5a:+.6f}\nB: dq4={dq4b:+.6f} dq5={dq5b:+.6f}\nC: ratio={ratio:+.4f}\n\n**Conclusion**: {coupling['conclusion']}\n"
        save_report(md7, "q4_q5_coupling.md")
        print(f"  A: dq4={dq4a:+.6f} dq5={dq5a:+.6f}  B: dq4={dq4b:+.6f} dq5={dq5b:+.6f}  ratio={ratio:+.4f}")

        # ── Test 9: A/B Comparison ───────────────────────────────────
        print("\n--- Test 9: A/B Comparison ---")
        ab_results = []
        for conv, label in [("A", "Current sign=-1"), ("B", "Flipped signs"), ("C", "Flipped+rate"), ("D", "J_cyl")]:
            reset_and_settle()
            tcp0 = get_tcp_pose(env, 0)
            if conv == "D":
                d = np.array([0.005, 0.003, -0.002])
                JT2 = J_w.T
                delta = JT2 @ np.linalg.inv(J_w @ JT2 + 0.05*np.eye(3)) @ d
            else:
                delta = np.array([0.002, 0.002, 0.001, 0.001])
            if conv in ("B", "C"):
                delta = -delta
            if conv == "C":
                delta = np.clip(delta, -0.002, 0.002)
            q0 = robot.data.joint_pos[0].detach().cpu().clone()
            target = q0.clone().unsqueeze(0).to(device)
            for c, cn in enumerate(CYLINDER_JOINT_NAMES):
                if cn in cyl_map:
                    target[0, cyl_map[cn]] += delta[c]
            target = torch.clamp(target, limits[:,0].unsqueeze(0), limits[:,1].unsqueeze(0))
            for _ in range(MEASURE):
                robot.set_joint_position_target(target, joint_ids=all_ids)
                env.step(zero)
            tcp1 = get_tcp_pose(env, 0)
            dtcp = np.array([tcp1["world_pos"][i]-tcp0["world_pos"][i] for i in range(3)])
            tcp_mag = float(np.linalg.norm(dtcp))
            ab_results.append({"conv": conv, "label": label, "tcp_mag": tcp_mag, "dtcp": dtcp.tolist()})
            print(f"  {conv}: TCP_mag={tcp_mag:.6f}m")
        save_json(ab_results, "ab_comparison.json")
        ab_csv = [{"conv": r["conv"], "label": r["label"], "tcp_mag": r["tcp_mag"],
                   "dtcp_x": r["dtcp"][0], "dtcp_y": r["dtcp"][1], "dtcp_z": r["dtcp"][2]} for r in ab_results]
        save_csv(ab_csv, "ab_comparison.csv")
        md9 = "# A/B Comparison\n\n| Conv | TCP Mag |\n|------|--------|\n"
        for r in ab_results:
            md9 += f"| {r['conv']} | {r['tcp_mag']:.6f} |\n"
        save_report(md9, "ab_comparison.md")

        print("\n" + "="*72)
        print("ALL TESTS COMPLETE")
        print("="*72)
    except Exception as exc:
        import traceback
        print(f"ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Use os._exit to avoid segfault on SimulationApp cleanup
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
if __name__ == "__main__":
    main()
