#!/usr/bin/env python3
"""Test 8: Safe velocity/cylinder tracking debug controller.

Implements a conservative tracking controller:
    s_ddot_des = s_ddot_ref + 2*zeta*omega_n*(s_dot_ref - s_dot) + omega_n^2*(s_ref - s)

Runs hold, step, sine, and TCP references with conservative limits.
"""
from __future__ import annotations
import sys, json, argparse, time, math
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import torch
import numpy as np
from kevin_integration.new_observations_pipeline.extensive_debug.scripts.debug_utils import (
    _close_sim_app,
    build_mulag_env, get_tcp_pose, get_cylinder_state,
    save_json, save_csv, save_log, CYLINDER_JOINT_NAMES,
)

ZETA = 1.2
OMEGA_N = 2.0
S_DOT_MAX = [0.02, 0.025, 0.025, 0.015]
S_DDOT_MAX = [0.15, 0.18, 0.18, 0.10]
DT = 1.0 / 120.0
STEPS = 120  # ~1 second
SETTLE = 30

def run_tracking_test(env, device, ref_type, ref_func, label):
    """Run tracking with a reference generator. ref_func(t) -> s_ref[4], s_dot_ref[4]."""
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
    for _ in range(SETTLE):
        env.step(zero)
    s0 = torch.tensor([get_cylinder_state(env, 0)[n]["pos"] for n in CYLINDER_JOINT_NAMES], dtype=torch.float64)
    s_dot = torch.zeros(4, dtype=torch.float64)
    s = s0.clone()
    log_data = []
    for step in range(STEPS):
        t = step * DT
        s_ref, s_dot_ref, s_ddot_ref = ref_func(t, s0)
        s_ref_t = torch.as_tensor(s_ref, dtype=torch.float64)
        s_dot_ref_t = torch.as_tensor(s_dot_ref, dtype=torch.float64)
        s_ddot_ref_t = torch.as_tensor(s_ddot_ref, dtype=torch.float64)
        # PD + feedforward
        s_ddot_des = (s_ddot_ref_t +
                      2 * ZETA * OMEGA_N * (s_dot_ref_t - s_dot) +
                      OMEGA_N**2 * (s_ref_t - s))
        s_ddot_des = torch.clamp(s_ddot_des,
                                 torch.tensor([-m for m in S_DDOT_MAX], dtype=torch.float64),
                                 torch.tensor(S_DDOT_MAX, dtype=torch.float64))
        s_dot = s_dot + s_ddot_des * DT
        s_dot = torch.clamp(s_dot,
                            torch.tensor([-m for m in S_DOT_MAX], dtype=torch.float64),
                            torch.tensor(S_DOT_MAX, dtype=torch.float64))
        s = s + s_dot * DT
        # Apply to simulation
        q0 = robot.data.joint_pos[0].detach().cpu().to(torch.float64)
        target = q0.clone().unsqueeze(0).to(device)
        for c, cname in enumerate(CYLINDER_JOINT_NAMES):
            if cname in cyl_map:
                target[0, cyl_map[cname]] = s[c].item()
        limits = robot.data.soft_joint_pos_limits[0].to(device)
        target = torch.clamp(target, limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0))
        robot.set_joint_position_target(target, joint_ids=all_ids)
        env.step(zero)
        # Measure actual
        cyl_state = get_cylinder_state(env, 0)
        s_actual = np.array([cyl_state[n]["pos"] for n in CYLINDER_JOINT_NAMES])
        s_dot_actual = np.array([cyl_state[n]["vel"] for n in CYLINDER_JOINT_NAMES])
        tcp = get_tcp_pose(env, 0)
        log_data.append({
            "step": step, "t": t,
            "s_ref_0": s_ref[0], "s_ref_1": s_ref[1], "s_ref_2": s_ref[2], "s_ref_3": s_ref[3],
            "s_actual_0": s_actual[0], "s_actual_1": s_actual[1], "s_actual_2": s_actual[2], "s_actual_3": s_actual[3],
            "s_dot_actual_0": s_dot_actual[0], "s_dot_actual_1": s_dot_actual[1],
            "s_dot_actual_2": s_dot_actual[2], "s_dot_actual_3": s_dot_actual[3],
            "tcp_wx": tcp["world_pos"][0], "tcp_wy": tcp["world_pos"][1], "tcp_wz": tcp["world_pos"][2],
        })
        # Update measurement
        s = torch.tensor(s_actual, dtype=torch.float64)
        s_dot = torch.tensor(s_dot_actual, dtype=torch.float64)
    return log_data

def make_hold_ref(s0):
    def ref(t, s0_local):
        return (s0_local.numpy(), np.zeros(4), np.zeros(4))
    return ref

def make_step_ref(s0, cyl_idx, delta=0.003):
    def ref(t, s0_local):
        sr = s0_local.clone().numpy()
        if t > 0.2:
            sr[cyl_idx] += delta
        return (sr, np.zeros(4), np.zeros(4))
    return ref

def make_sine_ref(s0, cyl_idx, amp=0.003, freq=0.5):
    def ref(t, s0_local):
        sr = s0_local.clone().numpy()
        phase = 2 * math.pi * freq * t
        sr[cyl_idx] += amp * math.sin(phase)
        sdr = np.zeros(4)
        sdr[cyl_idx] = amp * 2 * math.pi * freq * math.cos(phase)
        sddr = np.zeros(4)
        sddr[cyl_idx] = -amp * (2 * math.pi * freq)**2 * math.sin(phase)
        return (sr, sdr, sddr)
    return ref

def make_tcp_ref(s0):
    """Slow diagonal XY displacement using approximate cylinder mapping. Simplified."""
    def ref(t, s0_local):
        sr = s0_local.clone().numpy()
        if t > 0.3:
            sr[0] += 0.001  # small base rotation
            sr[1] += 0.001  # small boom 1
        return (sr, np.zeros(4), np.zeros(4))
    return ref

def main():
    p = argparse.ArgumentParser(description="Test 8: Safe velocity tracking")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", action="store_true", default=False)
    p.add_argument("--steps", type=int, default=STEPS)
    args = p.parse_args()
    device = args.device
    headless = not args.viewer
    print("=" * 72)
    print("Test 8: Safe Velocity/Cylinder Tracking")
    print("=" * 72)
    log_lines = []
    env = None
    try:
        env, cfg = build_mulag_env(num_envs=1, device=device, headless=headless)
        tests = [
            ("hold", make_hold_ref, "safe_velocity_tracking_hold.csv"),
            ("step_cyl0", lambda s0: make_step_ref(s0, 0), "safe_velocity_tracking_step_cyl0.csv"),
            ("step_cyl1", lambda s0: make_step_ref(s0, 1), "safe_velocity_tracking_step_cyl1.csv"),
            ("step_cyl2", lambda s0: make_step_ref(s0, 2), "safe_velocity_tracking_step_cyl2.csv"),
            ("step_cyl3", lambda s0: make_step_ref(s0, 3), "safe_velocity_tracking_step_cyl3.csv"),
            ("sine_cyl1", lambda s0: make_sine_ref(s0, 1), "safe_velocity_tracking_sine.csv"),
            ("tcp_small", make_tcp_ref, "safe_velocity_tracking_tcp.csv"),
        ]
        all_results = {}
        for label, ref_gen_fn, csv_name in tests:
            print(f"\nRunning {label}...")
            data = run_tracking_test(env, device, label, ref_gen_fn, label)
            save_csv(data, csv_name)
            all_results[label] = {"steps": len(data), "final_s_actual": [data[-1][f"s_actual_{i}"] for i in range(4)] if data else None}
            log_lines.append(f"{label}: {len(data)} steps")
        save_json(all_results, "safe_velocity_tracking_results.json")
        save_log("\n".join(log_lines), "safe_velocity_tracking.log")
        print(f"\nTest 8 complete. {len(tests)} tracking tests run.")
    except Exception as exc:
        import traceback
        err = f"ERROR: {exc}\n{traceback.format_exc()}"
        print(err); log_lines.append(err)
        save_log("\n".join(log_lines), "safe_velocity_tracking.log")
        save_json({"status": "FAILED", "error": str(exc)}, "safe_velocity_tracking_results.json")
        sys.exit(1)
    finally:
        _close_sim_app()
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
