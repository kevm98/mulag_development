#!/usr/bin/env python3
"""Test 7: q4/q5 passive coupling check.

Investigates whether Messerkopf_joint (joint 5) is passive/coupled with
Messerkopf_Schwenk_joint (joint 4). Measures whether joint 5 is controllable.
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
    save_json, save_csv, save_report, save_log, ARM_JOINT_NAMES,
)

EPSILON = 0.005  # rad
SETTLE_STEPS = 30
MEASURE_STEPS = 10

def main():
    p = argparse.ArgumentParser(description="Test 7: q4/q5 coupling check")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", action="store_true", default=False)
    args = p.parse_args()
    device = args.device
    headless = not args.viewer
    print("=" * 72)
    print("Test 7: q4/q5 Passive Coupling Check")
    print("=" * 72)
    log_lines = []
    env = None
    try:
        env, cfg = build_mulag_env(num_envs=1, device=device, headless=headless)
        robot = env.robot
        # Find joint indices
        jmap = {}
        for name in ARM_JOINT_NAMES:
            for i, jn in enumerate(robot.joint_names):
                if jn == name:
                    jmap[name] = i
                    break
        q4_name = "Messerkopf_Schwenk_joint"
        q5_name = "Messerkopf_joint"
        if q4_name not in jmap or q5_name not in jmap:
            print(f"ERROR: Could not find {q4_name} or {q5_name}")
            sys.exit(1)
        j4 = jmap[q4_name]
        j5 = jmap[q5_name]
        zero = torch.zeros((1, 5), device=device)
        all_ids = list(range(robot.num_joints))

        # Test A: Command joint 4 positive, observe joint 5
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.step(zero)
        initial_q = robot.data.joint_pos[0].detach().cpu().clone()
        target_a = initial_q.clone().unsqueeze(0).to(device)
        target_a[0, j4] += EPSILON
        limits = robot.data.soft_joint_pos_limits[0].to(device)
        target_a = torch.clamp(target_a, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
        for _ in range(MEASURE_STEPS):
            robot.set_joint_position_target(target_a, joint_ids=all_ids)
            env.step(zero)
        final_a = robot.data.joint_pos[0].detach().cpu().clone()
        dq4_a = float(final_a[j4] - initial_q[j4])
        dq5_a = float(final_a[j5] - initial_q[j5])

        # Test B: Command joint 5 positive, observe joint 4
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.step(zero)
        initial_qb = robot.data.joint_pos[0].detach().cpu().clone()
        target_b = initial_qb.clone().unsqueeze(0).to(device)
        target_b[0, j5] += EPSILON
        target_b = torch.clamp(target_b, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
        for _ in range(MEASURE_STEPS):
            robot.set_joint_position_target(target_b, joint_ids=all_ids)
            env.step(zero)
        final_b = robot.data.joint_pos[0].detach().cpu().clone()
        dq4_b = float(final_b[j4] - initial_qb[j4])
        dq5_b = float(final_b[j5] - initial_qb[j5])

        # Test C: Estimate coupling ratio
        env.reset()
        for _ in range(SETTLE_STEPS):
            env.step(zero)
        q0 = robot.data.joint_pos[0].detach().cpu().clone()
        # Larger j4 perturbation for coupling estimate
        target_c = q0.clone().unsqueeze(0).to(device)
        target_c[0, j4] += EPSILON * 3
        target_c = torch.clamp(target_c, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
        for _ in range(MEASURE_STEPS * 2):
            robot.set_joint_position_target(target_c, joint_ids=all_ids)
            env.step(zero)
        final_c = robot.data.joint_pos[0].detach().cpu().clone()
        dq4_c = float(final_c[j4] - q0[j4])
        dq5_c = float(final_c[j5] - q0[j5])
        ratio = dq5_c / dq4_c if abs(dq4_c) > 1e-9 else float("inf")

        results = {
            "test_a_command_q4": {"dq4": dq4_a, "dq5": dq5_a, "interpretation": "Command q4 → observe q5"},
            "test_b_command_q5": {"dq4": dq4_b, "dq5": dq5_b, "interpretation": "Command q5 → observe q4"},
            "test_c_coupling_estimate": {"dq4": dq4_c, "dq5": dq5_c, "ratio_q5_per_q4": ratio},
            "conclusion": "q5 IS coupled to q4" if abs(ratio) > 0.01 else "q5 is independent",
        }

        print(f"\nTest A (command q4=+{EPSILON}): dq4={dq4_a:+.6f}, dq5={dq5_a:+.6f}")
        print(f"Test B (command q5=+{EPSILON}): dq4={dq4_b:+.6f}, dq5={dq5_b:+.6f}")
        print(f"Test C (large q4): dq4={dq4_c:+.6f}, dq5={dq5_c:+.6f}, ratio={ratio:+.4f}")
        log_lines.append(f"A: dq4={dq4_a:+.6f} dq5={dq5_a:+.6f}")
        log_lines.append(f"B: dq4={dq4_b:+.6f} dq5={dq5_b:+.6f}")
        log_lines.append(f"C: dq4={dq4_c:+.6f} dq5={dq5_c:+.6f} ratio={ratio:+.4f}")

        save_json(results, "q4_q5_coupling.json")
        save_csv([{"test": "A_cmd_q4", "dq4": dq4_a, "dq5": dq5_a},
                  {"test": "B_cmd_q5", "dq4": dq4_b, "dq5": dq5_b},
                  {"test": "C_coupling", "dq4": dq4_c, "dq5": dq5_c, "ratio": ratio}],
                 "q4_q5_coupling.csv")
        save_log("\n".join(log_lines), "q4_q5_coupling.log")

        md = []
        md.append("# q4/q5 Coupling Report\n\n")
        md.append(f"## Test A: Command q4 +{EPSILON} rad\n")
        md.append(f"- dq4 = {dq4_a:+.6f} rad\n")
        md.append(f"- dq5 = {dq5_a:+.6f} rad\n\n")
        md.append(f"## Test B: Command q5 +{EPSILON} rad\n")
        md.append(f"- dq4 = {dq4_b:+.6f} rad\n")
        md.append(f"- dq5 = {dq5_b:+.6f} rad\n\n")
        md.append(f"## Test C: Coupling Ratio Estimate\n")
        md.append(f"- dq4 = {dq4_c:+.6f} rad\n")
        md.append(f"- dq5 = {dq5_c:+.6f} rad\n")
        md.append(f"- **q5 ≈ {ratio:+.4f} × q4**\n\n")
        if abs(ratio) > 0.01:
            md.append("## Conclusion\n")
            md.append(f"Joint 5 (Messerkopf_joint) IS mechanically coupled to joint 4 with ratio ~{ratio:+.4f}.\n")
            md.append("A 5-DOF Jacobian that treats them independently would be rank-deficient.\n")
            md.append("An effective 4-input Jacobian (4 hydraulic cylinders → TCP) is the correct approach.\n")
        else:
            md.append("## Conclusion\n")
            md.append("Joint 5 appears independent — no significant coupling detected.\n")
        save_report("".join(md), "q4_q5_coupling.md")
        print(f"\nTest 7 complete.")
    except Exception as exc:
        import traceback
        err = f"ERROR: {exc}\n{traceback.format_exc()}"
        print(err); log_lines.append(err)
        save_log("\n".join(log_lines), "q4_q5_coupling.log")
        save_json({"status": "FAILED", "error": str(exc)}, "q4_q5_coupling.json")
        sys.exit(1)
    finally:
        _close_sim_app()
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
