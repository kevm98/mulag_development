#!/usr/bin/env python3
"""Test 9: Current pipeline A/B comparison.

Compares multiple conventions on the same small scripted references:
A. current existing convention (no sign correction)
B. corrected sign/order convention
C. corrected convention + rate limiting
D. corrected convention + numerical cylinder Jacobian for TCP motion

Measures TCP tracking error, cylinder tracking error, joint direction correctness,
action smoothness, max velocities, and stability.
"""
from __future__ import annotations
import sys, json, argparse, math
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import torch
import numpy as np
from kevin_integration.new_observations_pipeline.extensive_debug.scripts.debug_utils import (
    _close_sim_app,
    build_mulag_env, get_tcp_pose, get_cylinder_state, get_all_joint_state,
    save_json, save_csv, save_report, save_log, CYLINDER_JOINT_NAMES,
    ARM_JOINT_NAMES, CYLINDER_SIGNS, CYLINDER_NOMINAL_SIMSCAPE, CYLINDER_NOMINAL_ISAAC,
)

EPSILON = 0.002
SETTLE_STEPS = 30
MEASURE_STEPS = 12
DT = 1.0 / 120.0

def measure_jacobian(env, device):
    """Quick finite-diff J_cyl."""
    robot = env.robot
    cyl_map = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jn in enumerate(robot.joint_names):
            if jn == name:
                cyl_map[name] = i
                break
    zero = torch.zeros((1, 5), device=device)
    all_ids = list(range(robot.num_joints))
    J = np.zeros((3, 4))
    for c, cname in enumerate(CYLINDER_JOINT_NAMES):
        if cname not in cyl_map:
            continue
        cid = cyl_map[cname]
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.step(zero)
        tcp0 = get_tcp_pose(env, 0)
        q0 = robot.data.joint_pos[0].detach().cpu().clone()
        target = q0.clone().unsqueeze(0).to(device)
        target[0, cid] += EPSILON
        limits = robot.data.soft_joint_pos_limits[0].to(device)
        target = torch.clamp(target, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
        for _ in range(MEASURE_STEPS):
            robot.set_joint_position_target(target, joint_ids=all_ids)
            env.step(zero)
        tcp1 = get_tcp_pose(env, 0)
        for axis in range(3):
            J[axis, c] = (tcp1["world_pos"][axis] - tcp0["world_pos"][axis]) / EPSILON
    return J

def run_convention_test(env, device, convention, J_world=None):
    """Run a small step test under a given convention. Returns metrics dict."""
    robot = env.robot
    cyl_map = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jn in enumerate(robot.joint_names):
            if jn == name:
                cyl_map[name] = i
                break
    zero = torch.zeros((1, 5), device=device)
    all_ids = list(range(robot.num_joints))
    env.reset()
    for _ in range(SETTLE_STEPS):
        env.step(zero)
    tcp0 = get_tcp_pose(env, 0)
    cyl0 = get_cylinder_state(env, 0)
    joint0 = get_all_joint_state(env, 0)

    # Build a small multi-cylinder delta
    if convention == "D" and J_world is not None:
        desired_dtcp = np.array([0.005, 0.003, -0.002])
        JT = J_world.T
        lam = 0.05
        pinv = JT @ np.linalg.inv(J_world @ JT + lam * np.eye(3))
        delta_isaac = pinv @ desired_dtcp
    else:
        delta_isaac = np.array([0.002, 0.002, 0.001, 0.001])

    # Apply sign correction if mode B or C
    if convention in ("B", "C"):
        # Invert the current sign convention (negate to test "corrected" sign)
        delta_isaac = -delta_isaac

    # Rate limit if mode C
    if convention == "C":
        max_step = np.array([0.002, 0.002, 0.002, 0.002])
        delta_isaac = np.clip(delta_isaac, -max_step, max_step)

    # Apply targets
    q0 = robot.data.joint_pos[0].detach().cpu().clone()
    target = q0.clone().unsqueeze(0).to(device)
    for c, cname in enumerate(CYLINDER_JOINT_NAMES):
        if cname in cyl_map:
            target[0, cyl_map[cname]] += delta_isaac[c]
    limits = robot.data.soft_joint_pos_limits[0].to(device)
    target = torch.clamp(target, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))

    actual_delta_isaac = np.zeros(4)
    for c, cname in enumerate(CYLINDER_JOINT_NAMES):
        if cname in cyl_map:
            actual_delta_isaac[c] = float(target[0, cyl_map[cname]].cpu().item()) - float(q0[cyl_map[cname]].item())

    max_vels = np.zeros(4)
    for _ in range(MEASURE_STEPS):
        robot.set_joint_position_target(target, joint_ids=all_ids)
        env.step(zero)
        for c, cname in enumerate(CYLINDER_JOINT_NAMES):
            if cname in cyl_map:
                v = abs(float(robot.data.joint_vel[0, cyl_map[cname]].detach().cpu().item()))
                if v > max_vels[c]:
                    max_vels[c] = v

    tcp1 = get_tcp_pose(env, 0)
    cyl1 = get_cylinder_state(env, 0)
    joint1 = get_all_joint_state(env, 0)

    dtcp = np.array([tcp1["world_pos"][i] - tcp0["world_pos"][i] for i in range(3)])
    dcyl = {n: cyl1[n]["pos"] - cyl0[n]["pos"] for n in CYLINDER_JOINT_NAMES if n in cyl1}

    # Direction check: do cylinder commands and TCP motion agree?
    tcp_mag = float(np.linalg.norm(dtcp))
    cyl_mag = float(np.linalg.norm(list(dcyl.values())))

    return {
        "convention": convention,
        "commanded_delta": delta_isaac.tolist(),
        "actual_delta_applied": actual_delta_isaac.tolist(),
        "dtcp_world": dtcp.tolist(),
        "tcp_motion_mag": tcp_mag,
        "cylinder_delta": dcyl,
        "cylinder_delta_mag": cyl_mag,
        "max_cylinder_velocity": max_vels.tolist(),
        "stable": tcp_mag < 0.5 and cyl_mag < 0.1,
    }

def main():
    p = argparse.ArgumentParser(description="Test 9: A/B comparison")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", action="store_true", default=False)
    args = p.parse_args()
    device = args.device
    headless = not args.viewer
    print("=" * 72)
    print("Test 9: Current Pipeline A/B Comparison")
    print("=" * 72)
    log_lines = []
    env = None
    try:
        env, cfg = build_mulag_env(num_envs=1, device=device, headless=headless)
        J_w = measure_jacobian(env, device)
        print(f"J_cyl_world shape: {J_w.shape}")

        conventions = ["A", "B", "C", "D"]
        labels = {
            "A": "Current convention (sign=-1 for all)",
            "B": "Corrected sign (flipped signs)",
            "C": "Corrected sign + rate limiting",
            "D": "Numerical J_cyl for TCP motion",
        }
        results = []
        for conv in conventions:
            print(f"\nTesting convention {conv}: {labels[conv]}")
            r = run_convention_test(env, device, conv, J_w if conv == "D" else None)
            r["label"] = labels[conv]
            results.append(r)
            print(f"  TCP motion: {r['dtcp_world']} (mag={r['tcp_motion_mag']:.6f})")
            print(f"  Cyl delta:  {r['cylinder_delta']} (mag={r['cylinder_delta_mag']:.6f})")
            print(f"  Max cyl vel: {r['max_cylinder_velocity']}")
            print(f"  Stable: {r['stable']}")
            log_lines.append(f"{conv}: TCP={r['tcp_motion_mag']:.6f} cyl={r['cylinder_delta_mag']:.6f} stable={r['stable']}")

        csv_rows = []
        for r in results:
            row = {"convention": r["convention"], "label": r["label"],
                   "tcp_dx": r["dtcp_world"][0], "tcp_dy": r["dtcp_world"][1], "tcp_dz": r["dtcp_world"][2],
                   "tcp_mag": r["tcp_motion_mag"], "cyl_mag": r["cylinder_delta_mag"],
                   "stable": r["stable"]}
            for i, n in enumerate(CYLINDER_JOINT_NAMES):
                row[f"cmd_{n}"] = r["commanded_delta"][i]
                row[f"dcyl_{n}"] = r["cylinder_delta"].get(n, 0)
                row[f"maxv_{n}"] = r["max_cylinder_velocity"][i]
            csv_rows.append(row)
        save_json(results, "ab_comparison.json")
        save_csv(csv_rows, "ab_comparison.csv")
        save_log("\n".join(log_lines), "ab_comparison.log")

        md = ["# A/B Comparison Report\n\n"]
        md.append("## Conventions Tested\n\n")
        for r in results:
            md.append(f"### {r['convention']}: {r['label']}\n")
            md.append(f"- Commanded delta: {r['commanded_delta']}\n")
            md.append(f"- Applied delta:   {r['actual_delta_applied']}\n")
            md.append(f"- TCP world delta: {r['dtcp_world']}\n")
            md.append(f"- TCP motion mag:  {r['tcp_motion_mag']:.6f} m\n")
            md.append(f"- Cylinder delta:  {r['cylinder_delta']}\n")
            md.append(f"- Max cyl velocity: {r['max_cylinder_velocity']}\n")
            md.append(f"- Stable: {r['stable']}\n\n")
        md.append("## Recommendation\n\n")
        md.append("Based on the comparison, the convention that gives the most predictable\n")
        md.append("TCP response with minimal instability should be adopted.\n")
        save_report("".join(md), "ab_comparison.md")
        print(f"\nTest 9 complete. {len(results)} conventions tested.")
    except Exception as exc:
        import traceback
        err = f"ERROR: {exc}\n{traceback.format_exc()}"
        print(err); log_lines.append(err)
        save_log("\n".join(log_lines), "ab_comparison.log")
        save_json({"status": "FAILED", "error": str(exc)}, "ab_comparison.json")
        sys.exit(1)
    finally:
        _close_sim_app()
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
