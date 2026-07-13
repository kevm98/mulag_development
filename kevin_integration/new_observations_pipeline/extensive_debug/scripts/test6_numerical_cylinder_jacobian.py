#!/usr/bin/env python3
"""Test 6: Numerical cylinder Jacobian.

Builds J_cyl = d(TCP) / d(s_cylinder) by finite differences.
Saves J_cyl_world and J_cyl_base, validates with damped pseudoinverse.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import torch
import numpy as np
from kevin_integration.new_observations_pipeline.extensive_debug.scripts.debug_utils import (
    _close_sim_app,
    build_mulag_env, get_tcp_pose, get_cylinder_state,
    save_json, save_csv, save_report, save_log, CYLINDER_JOINT_NAMES,
)

EPSILON = 0.002  # m perturbation
SETTLE_STEPS = 30
MEASURE_STEPS = 8
DAMPING = 0.05

def measure_jacobian(env, device):
    """Measure J_cyl via central differences: J[:, c] = (TCP(+eps) - TCP(-eps)) / (2*eps)."""
    robot = env.robot
    cyl_map = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jn in enumerate(robot.joint_names):
            if jn == name:
                cyl_map[name] = i
                break
    zero = torch.zeros((1, 5), device=device)
    all_ids = list(range(robot.num_joints))
    J_world = np.zeros((3, 4))
    J_base = np.zeros((3, 4))
    for c, cname in enumerate(CYLINDER_JOINT_NAMES):
        if cname not in cyl_map:
            continue
        cid = cyl_map[cname]
        # +eps
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.step(zero)
        q0 = robot.data.joint_pos[0].detach().cpu().clone()
        target_p = q0.clone().unsqueeze(0).to(device)
        target_p[0, cid] += EPSILON
        limits = robot.data.soft_joint_pos_limits[0].to(device)
        target_p = torch.clamp(target_p, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
        for _ in range(MEASURE_STEPS):
            robot.set_joint_position_target(target_p, joint_ids=all_ids)
            env.step(zero)
        tcp_p = get_tcp_pose(env, 0)
        # -eps
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.step(zero)
        q0n = robot.data.joint_pos[0].detach().cpu().clone()
        target_n = q0n.clone().unsqueeze(0).to(device)
        target_n[0, cid] -= EPSILON
        target_n = torch.clamp(target_n, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
        for _ in range(MEASURE_STEPS):
            robot.set_joint_position_target(target_n, joint_ids=all_ids)
            env.step(zero)
        tcp_n = get_tcp_pose(env, 0)
        for axis in range(3):
            J_world[axis, c] = (tcp_p["world_pos"][axis] - tcp_n["world_pos"][axis]) / (2 * EPSILON)
            J_base[axis, c] = (tcp_p["base_pos"][axis] - tcp_n["base_pos"][axis]) / (2 * EPSILON)
    return J_world, J_base

def validate_jacobian(env, J_world, device):
    """Test Jacobian prediction: command delta_s = damped_pinv(J) * desired_dtcp."""
    robot = env.robot
    cyl_map = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jn in enumerate(robot.joint_names):
            if jn == name:
                cyl_map[name] = i
                break
    zero = torch.zeros((1, 5), device=device)
    all_ids = list(range(robot.num_joints))
    JT = J_world.T
    JJT = J_world @ JT
    lam_I = DAMPING * np.eye(3)
    pinv = JT @ np.linalg.inv(JJT + lam_I)
    desired_dtcp = np.array([0.01, 0.0, 0.0])  # 1cm in X
    delta_s = pinv @ desired_dtcp
    env.reset()
    for _ in range(SETTLE_STEPS):
        env.step(zero)
    tcp0 = get_tcp_pose(env, 0)
    q0 = robot.data.joint_pos[0].detach().cpu().clone()
    target = q0.clone().unsqueeze(0).to(device)
    for c, cname in enumerate(CYLINDER_JOINT_NAMES):
        if cname in cyl_map:
            target[0, cyl_map[cname]] += delta_s[c]
    limits = robot.data.soft_joint_pos_limits[0].to(device)
    target = torch.clamp(target, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
    for _ in range(MEASURE_STEPS):
        robot.set_joint_position_target(target, joint_ids=all_ids)
        env.step(zero)
    tcp1 = get_tcp_pose(env, 0)
    actual_dtcp = np.array([tcp1["world_pos"][i] - tcp0["world_pos"][i] for i in range(3)])
    predicted_dtcp = J_world @ delta_s
    return {
        "desired_dtcp": desired_dtcp.tolist(),
        "predicted_dtcp": predicted_dtcp.tolist(),
        "actual_dtcp": actual_dtcp.tolist(),
        "delta_s_command": delta_s.tolist(),
        "prediction_error": (actual_dtcp - desired_dtcp).tolist(),
        "prediction_error_norm": float(np.linalg.norm(actual_dtcp - desired_dtcp)),
    }

def main():
    p = argparse.ArgumentParser(description="Test 6: Numerical cylinder Jacobian")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", action="store_true", default=False)
    p.add_argument("--eps", type=float, default=EPSILON)
    p.add_argument("--damping", type=float, default=DAMPING)
    p.add_argument("--skip_validation", action="store_true")
    args = p.parse_args()
    device = args.device
    headless = not args.viewer
    print("=" * 72)
    print(f"Test 6: Numerical Cylinder Jacobian (eps={args.eps} m)")
    print("=" * 72)
    log_lines = []
    env = None
    try:
        env, cfg = build_mulag_env(num_envs=1, device=device, headless=headless)
        print("Measuring J_cyl via central differences...")
        J_w, J_b = measure_jacobian(env, device)
        print(f"\nJ_cyl_world (3x4):\n{J_w}")
        print(f"\nJ_cyl_base (3x4):\n{J_b}")
        S_w = np.linalg.svd(J_w, compute_uv=False)
        S_b = np.linalg.svd(J_b, compute_uv=False)
        print(f"Singular values (world): {S_w}")
        print(f"Singular values (base):  {S_b}")
        cond_w = S_w[0] / S_w[-1] if S_w[-1] > 1e-9 else float("inf")
        cond_b = S_b[0] / S_b[-1] if S_b[-1] > 1e-9 else float("inf")
        print(f"Condition number (world): {cond_w:.2f}")
        print(f"Condition number (base):  {cond_b:.2f}")
        log_lines.append(f"J_world:\n{J_w}")
        log_lines.append(f"J_base:\n{J_b}")
        log_lines.append(f"SVs world: {S_w}, cond={cond_w:.2f}")
        log_lines.append(f"SVs base: {S_b}, cond={cond_b:.2f}")

        jac_data = {"J_world": J_w.tolist(), "J_base": J_b.tolist(),
                    "singular_values_world": S_w.tolist(), "singular_values_base": S_b.tolist(),
                    "condition_number_world": float(cond_w), "condition_number_base": float(cond_b),
                    "epsilon_m": args.eps}

        validation = None
        if not args.skip_validation:
            print(f"\nValidating Jacobian via damped pseudoinverse (lambda={args.damping})...")
            validation = validate_jacobian(env, J_w, device)
            jac_data["validation"] = validation
            print(f"  Desired  dTCP: {validation['desired_dtcp']}")
            print(f"  Predicted dTCP: {validation['predicted_dtcp']}")
            print(f"  Actual   dTCP: {validation['actual_dtcp']}")
            print(f"  Error:         {validation['prediction_error']} (norm={validation['prediction_error_norm']:.6f})")
            log_lines.append(f"Validation: desired={validation['desired_dtcp']} actual={validation['actual_dtcp']} err={validation['prediction_error_norm']:.6f}")

        save_json(jac_data, "numerical_cylinder_jacobian.json")
        csv_rows = []
        for c in range(4):
            csv_rows.append({"cylinder": CYLINDER_JOINT_NAMES[c],
                            "Jw_x": J_w[0,c], "Jw_y": J_w[1,c], "Jw_z": J_w[2,c],
                            "Jb_x": J_b[0,c], "Jb_y": J_b[1,c], "Jb_z": J_b[2,c]})
        save_csv(csv_rows, "numerical_cylinder_jacobian.csv")
        if validation:
            save_csv([validation], "numerical_jacobian_validation.csv")
            md = []
            md.append("# Numerical Jacobian Validation\n\n")
            md.append(f"## Jacobian\n\nJ_world:\n```\n{J_w}\n```\n\nJ_base:\n```\n{J_b}\n```\n\n")
            md.append(f"Singular values (world): {S_w}\n")
            md.append(f"Condition number: {cond_w:.2f}\n\n")
            md.append(f"## Damped Pseudoinverse Validation (lambda={args.damping})\n\n")
            md.append(f"- Desired dTCP:  {validation['desired_dtcp']}\n")
            md.append(f"- Predicted dTCP: {validation['predicted_dtcp']}\n")
            md.append(f"- Actual dTCP:    {validation['actual_dtcp']}\n")
            md.append(f"- δs command:     {validation['delta_s_command']}\n")
            md.append(f"- Error norm:     {validation['prediction_error_norm']:.6f} m\n")
            save_report("".join(md), "numerical_jacobian_validation.md")
        save_log("\n".join(log_lines), "numerical_cylinder_jacobian.log")
        print(f"\nTest 6 complete.")
    except Exception as exc:
        import traceback
        err = f"ERROR: {exc}\n{traceback.format_exc()}"
        print(err); log_lines.append(err)
        save_log("\n".join(log_lines), "numerical_cylinder_jacobian.log")
        save_json({"status": "FAILED", "error": str(exc)}, "numerical_cylinder_jacobian.json")
        sys.exit(1)
    finally:
        _close_sim_app()
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
