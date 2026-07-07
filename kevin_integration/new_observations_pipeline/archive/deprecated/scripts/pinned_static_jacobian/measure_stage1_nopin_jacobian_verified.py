#!/usr/bin/env python3
"""Measure no-pin effective J_yz using actual cylinder displacement, with drift correction and validation."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch, math

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--force_rate_limit_fraction", type=float, default=0.01)
parser.add_argument("--velocity_filter_beta", type=float, default=0.8)
parser.add_argument("--dx_values", type=str, default="0.0025,0.005,0.010")
parser.add_argument("--n_steps_values", type=str, default="1,5,10,30")
parser.add_argument("--n_repeats", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage1_yz_tracking_env_wrapper import Stage1YZTrackingEnvWrapper

device = args_cli.device
dx_list = [float(x) for x in args_cli.dx_values.split(",")]
n_steps_list = [int(x) for x in args_cli.n_steps_values.split(",")]
n_repeats = int(args_cli.n_repeats)

print(f"MEASURING NO-PIN J_yz", flush=True)
print(f"kp={args_cli.kp_scale} kd={args_cli.kd_scale} dx={dx_list} N={n_steps_list} repeats={n_repeats}", flush=True)

# Build single env
cfg = MulagUpperLevelEnvCfg(); cfg.scene.num_envs = 1
env = Stage1YZTrackingEnvWrapper(cfg, horizon_steps=10, action_scale_m=0.05, device=device,
                                  kp_scale=args_cli.kp_scale, kd_scale=args_cli.kd_scale,
                                  force_rate_limit=args_cli.force_rate_limit_fraction,
                                  velocity_filter_beta=args_cli.velocity_filter_beta,
                                  trajectory_mode="sequence", target_hold_steps=999,
                                  post_reset_settle_steps=5)
env._target_y_range = 0.03; env._target_z_range = 0.01
B = 1; za = torch.zeros((B, 2), device=device)
cyl_ids = env._cyl_ids
H = env._H

# Store all measurements
all_results = {}  # (dx, n_steps) -> list of per-cylinder measurements

def reset_and_settle():
    env._env.reset()
    env.reset()
    for _ in range(10):
        env.step(za)

def run_baseline(n_steps):
    """Run N policy steps with zero action, measure EE and cylinder drift."""
    ee_start = env._tcp_yz()[0].cpu().numpy().copy()
    q_start = env._uw.robot.data.joint_pos[0, cyl_ids].cpu().numpy().copy()
    for _ in range(n_steps):
        env.step(za)
    ee_end = env._tcp_yz()[0].cpu().numpy().copy()
    q_end = env._uw.robot.data.joint_pos[0, cyl_ids].cpu().numpy().copy()
    return ee_start, ee_end, q_start, q_end

def apply_perturbation(cyl_idx, dx, n_steps):
    """Apply +dx or -dx to cylinder cyl_idx, run N steps, measure response."""
    q_before = env._uw.robot.data.joint_pos[0, cyl_ids].cpu().numpy().copy()
    ee_before = env._tcp_yz()[0].cpu().numpy().copy()

    cqh = env._uw.robot.data.joint_pos[:, cyl_ids].clone()
    cqh[0, cyl_idx] += dx
    for _ in range(n_steps * H):
        env._uw.set_debug_manual_cylinder_targets(cqh, tool_position_target=env._tool_target)
        env._env.step(torch.zeros((B, 5), device=device))

    q_after = env._uw.robot.data.joint_pos[0, cyl_ids].cpu().numpy().copy()
    ee_after = env._tcp_yz()[0].cpu().numpy().copy()
    return q_before, q_after, ee_before, ee_after

cyl_names = ["Drehzapfen", "Ausleger_I", "Ausleger_II", "Messerkopf_Schwenk"]

for dx in dx_list:
    for n_steps in n_steps_list:
        print(f"\n--- dx={dx:.4f} N={n_steps} ---", flush=True)
        config_key = (dx, n_steps)
        all_results[config_key] = []

        for cyl_idx in range(4):
            cyl_data = {"cyl": cyl_idx, "name": cyl_names[cyl_idx], "dx": dx, "n_steps": n_steps,
                        "valid": [], "J_cmd": [], "J_actual": [], "actual_dq_ratio": []}

            for rep in range(n_repeats):
                # Baseline drift
                reset_and_settle()
                _, ee_bl_end, _, q_bl_end = run_baseline(n_steps)

                # +dx
                reset_and_settle()
                qp_b, qp_a, eep_b, eep_a = apply_perturbation(cyl_idx, +dx, n_steps)

                # -dx
                reset_and_settle()
                qm_b, qm_a, eem_b, eem_a = apply_perturbation(cyl_idx, -dx, n_steps)

                # Drift-corrected deltas
                ee_delta = (eep_a - eem_a) / 2  # symmetric difference
                ee_bl_drift = ee_bl_end - eep_b  # baseline drift during measurement
                ee_corrected = ee_delta - ee_bl_drift

                # Actual cylinder displacement
                actual_dq_plus = qp_a[cyl_idx] - qp_b[cyl_idx]
                actual_dq_minus = qm_a[cyl_idx] - qm_b[cyl_idx]
                actual_sym = (actual_dq_plus - actual_dq_minus) / 2

                # Jacobian columns
                J_cmd = ee_corrected / (2 * dx)  # using commanded dx

                valid = True
                J_actual = np.zeros(2)
                ratio = 0.0

                if abs(actual_sym) > 0.2 * dx:
                    J_actual = ee_corrected / actual_sym
                    ratio = actual_sym / dx
                else:
                    valid = False

                # Antisymmetry check
                antisym = abs(actual_dq_plus + actual_dq_minus) < 0.5 * abs(actual_sym)
                if not antisym:
                    valid = False

                cyl_data["valid"].append(valid)
                cyl_data["J_cmd"].append(J_cmd)
                if valid:
                    cyl_data["J_actual"].append(J_actual)
                cyl_data["actual_dq_ratio"].append(ratio)

                status = "VALID" if valid else "INVALID"
                print(f"  cyl {cyl_idx} [{cyl_names[cyl_idx]}] rep {rep}: "
                      f"actual_dq={actual_sym:.6f} ratio={ratio:.2f} "
                      f"J_cmd=({J_cmd[0]:.2f},{J_cmd[1]:.2f}) "
                      f"J_actual=({J_actual[0]:.2f},{J_actual[1]:.2f}) "
                      f"{status}", flush=True)

            all_results[config_key].append(cyl_data)

# ====================================================================
# Summary: find best config
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY: best config per cylinder", flush=True)
print(f"{'='*60}", flush=True)

best_J = np.zeros((2, 4))
best_configs = {}

for cyl_idx in range(4):
    best_score = -1
    for (dx, ns), cyl_list in all_results.items():
        cd = cyl_list[cyl_idx]
        n_valid = sum(cd["valid"])
        if n_valid == 0:
            continue
        # Score: prefer high validity, stable signs, actual_dq close to dx
        ratios = [r for r, v in zip(cd["actual_dq_ratio"], cd["valid"]) if v]
        if not ratios:
            continue
        mean_ratio = float(np.mean(ratios))
        # Penalize ratio far from 1.0
        ratio_score = 1.0 - abs(mean_ratio - 1.0)
        score = n_valid * 10 + ratio_score * 5

        if score > best_score:
            best_score = score
            valid_js = [j for j, v in zip(cd["J_cmd"], cd["valid"]) if v]
            if valid_js:
                best_J[:, cyl_idx] = np.median(valid_js, axis=0)
                best_configs[cyl_idx] = (dx, ns, n_valid, mean_ratio)

for cyl_idx in range(4):
    if cyl_idx in best_configs:
        dx, ns, nv, ratio = best_configs[cyl_idx]
        print(f"  cyl {cyl_idx} [{cyl_names[cyl_idx]}]: dx={dx:.4f} N={ns} valid={nv}/{n_repeats} "
              f"ratio={ratio:.2f} J=({best_J[0,cyl_idx]:.2f},{best_J[1,cyl_idx]:.2f})", flush=True)
    else:
        print(f"  cyl {cyl_idx} [{cyl_names[cyl_idx]}]: NO VALID MEASUREMENT", flush=True)

print(f"\nBest J_yz (no-pin):", flush=True)
print(f"  {best_J.tolist()}", flush=True)

# Compare with old
J_old = np.array([[-1.74, 29.81, 20.45, 20.29], [-1.51, -24.73, -30.88, -31.66]])
print(f"Old J_yz (pinned):", flush=True)
print(f"  {J_old.tolist()}", flush=True)

# Rank and conditioning
U, S, Vt = np.linalg.svd(best_J)
print(f"\nSVD: singular values={S} rank={np.sum(S > 1e-6)} cond={S[0]/S[-1] if S[-1] > 1e-10 else 'inf'}", flush=True)

# ====================================================================
# Validation: random cylinder perturbations
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"VALIDATION: 10 random cylinder perturbations", flush=True)
print(f"{'='*60}", flush=True)

cosines = []; sign_ok_y = 0; sign_ok_z = 0

for v in range(10):
    reset_and_settle()
    dq = np.random.uniform(-0.0025, 0.0025, 4)
    ee_before = env._tcp_yz()[0].cpu().numpy().copy()
    q_before = env._uw.robot.data.joint_pos[0, cyl_ids].cpu().numpy().copy()

    cqh = env._uw.robot.data.joint_pos[:, cyl_ids].clone()
    for c in range(4):
        cqh[0, c] += dq[c]
    for _ in range(10 * H):
        env._uw.set_debug_manual_cylinder_targets(cqh, tool_position_target=env._tool_target)
        env._env.step(torch.zeros((B, 5), device=device))

    q_after = env._uw.robot.data.joint_pos[0, cyl_ids].cpu().numpy().copy()
    ee_after = env._tcp_yz()[0].cpu().numpy().copy()
    actual_dq = q_after - q_before
    ee_measured = ee_after - ee_before
    ee_pred = best_J @ actual_dq

    cos = np.dot(ee_measured, ee_pred) / (np.linalg.norm(ee_measured) * np.linalg.norm(ee_pred) + 1e-10)
    cosines.append(cos)
    if np.sign(ee_measured[0]) == np.sign(ee_pred[0]): sign_ok_y += 1
    if np.sign(ee_measured[1]) == np.sign(ee_pred[1]): sign_ok_z += 1

    print(f"  {v}: measured=({ee_measured[0]:.4f},{ee_measured[1]:.4f}) "
          f"pred=({ee_pred[0]:.4f},{ee_pred[1]:.4f}) cos={cos:.3f}", flush=True)

print(f"\n  median cosine: {float(np.median(cosines)):.3f}", flush=True)
print(f"  sign_ok_Y: {sign_ok_y}/10  sign_ok_Z: {sign_ok_z}/10", flush=True)

env.close()
simulation_app.close()
