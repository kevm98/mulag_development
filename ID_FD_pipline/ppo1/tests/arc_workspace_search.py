#!/usr/bin/env python3
"""Deterministic maximum-safe-arc sweep. Tests arc lengths 0.88-1.50 m."""

import argparse, json, math, sys, numpy as np
from pathlib import Path
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--speed", type=float, default=0.05)
p.add_argument("--cycles", type=int, default=3)

AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch
import ID_FD_pipline.tasks
from isaaclab_tasks.utils import parse_env_cfg

# Fixed geometry from circle fit
CX, CY = 4.327, -3.224
R = 1.286
THETA_CENTER_DEG = -8.7  # center of the verified ground-band arc

CANDIDATES = [0.88, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]

results = []

for target_L in CANDIDATES:
    delta_theta_rad = target_L / R
    delta_deg = math.degrees(delta_theta_rad)
    ta = THETA_CENTER_DEG - 0.5 * delta_deg
    tb = THETA_CENTER_DEG + 0.5 * delta_deg
    
    # Compute chord and spans
    chord = 2 * R * math.sin(0.5 * delta_theta_rad)
    xA = CX + R * math.cos(math.radians(ta))
    yA = CY + R * math.sin(math.radians(ta))
    xB = CX + R * math.cos(math.radians(tb))
    yB = CY + R * math.sin(math.radians(tb))
    xs = abs(xB - xA); ys = abs(yB - yA)

    steps = max(int(3 * 2 * target_L / float(a.speed) / (1/60)), 1200)
    
    print(f"\n=== ARC CANDIDATE L={target_L:.2f}m theta=[{ta:.1f},{tb:.1f}]° "
          f"span={delta_deg:.1f}° chord={chord:.3f}m Xspan={xs:.3f}m Yspan={ys:.3f}m steps={steps} ===", flush=True)

    cfg = parse_env_cfg("Kevin-Mulag-IDFD-EEYZResidual-PLAY-v0", device=a.device, num_envs=a.num_envs)
    cfg.seed = int(a.seed)
    cfg.residual_mow_shape = "arc"
    cfg.residual_min_accumulated_path_m = float(a.cycles) * 2 * target_L
    cfg.residual_path_speed_mps = float(a.speed)
    cfg.residual_arc_center_x = CX; cfg.residual_arc_center_y = CY
    cfg.residual_arc_radius_m = R
    cfg.residual_arc_theta_a_deg = ta; cfg.residual_arc_theta_b_deg = tb
    cfg.episode_length_s = steps / 60.0 + 10
    env = gym.make("Kevin-Mulag-IDFD-EEYZResidual-PLAY-v0", cfg=cfg).unwrapped
    
    try:
        zero = torch.zeros((a.num_envs, 3), device=env.device)
        env.reset(seed=int(a.seed))
        
        errors = []; heights = []; phases = []
        goals_np = []; ees_np = []
        
        for step in range(steps):
            obs, reward, term, trunc, _ = env.step(zero)
            ee = env._compute_ee_pos_local()
            nominal = env._get_nominal_goal()
            err = float(torch.linalg.norm(nominal - ee, dim=-1)[0].item())
            z = float(env._compute_cutter_world_z()[0].item())
            phase = int(env._state_machine.state.phase[0].item())
            errors.append(err); heights.append(z); phases.append(phase)
            goals_np.append(nominal[0].detach().cpu().numpy().copy())
            ees_np.append(ee[0].detach().cpu().numpy().copy())
            if term[0] or trunc[0]:
                break
        
        errs = np.array(errors)
        # Trim first 1000 steps (settle + deploy ~16.7s), only score mowing
        mow_mask = np.isin(phases, [2, 3])
        settle_trim = 3400
        if len(errs) > settle_trim:
            errs = errs[settle_trim:]
            mow_mask = mow_mask[settle_trim:]
            heights = heights[settle_trim:]
        mow_errs = errs[mow_mask] if mow_mask.any() else errs
        h = np.array(heights)
        ground_frac = np.mean(np.abs(h) <= 0.10) if len(h) > 0 else 0
        
        p95 = np.percentile(mow_errs, 95) if len(mow_errs) > 0 else 999
        mean_e = float(mow_errs.mean()) if len(mow_errs) > 0 else 999
        
        # Endpoint check: first and last mowing positions
        goals_a = np.array(goals_np)
        ees_a = np.array(ees_np)
        A_err = np.linalg.norm(goals_a[0] - ees_a[0]) if len(goals_a) > 0 else 999
        B_err = np.linalg.norm(goals_a[300] - ees_a[300]) if len(goals_a) > 300 else 999
        
        passed = bool(p95 < 0.05 and ground_frac >= 0.95 and A_err < 0.05 and B_err < 0.05)
        
        status = "PASS" if passed else "FAIL"
        print(f"[ARC-SEARCH] L={target_L:.2f}m {status}: mean={mean_e:.4f} p95={p95:.4f} "
              f"A_err={A_err:.4f} B_err={B_err:.4f} ground={ground_frac:.2f} "
              f"mow_steps={mow_mask.sum()}", flush=True)
        
        results.append({
            "L": target_L, "passed": passed, "mean": mean_e, "p95": p95,
            "A_err": A_err, "B_err": B_err, "ground_frac": ground_frac,
            "chord": chord, "span_x": xs, "span_y": ys,
        })
        
        if not passed:
            print(f"[ARC-SEARCH] Stopping: first failure at L={target_L:.2f}m", flush=True)
            break
            
    finally:
        env.close()

print("\n=== ARC SEARCH RESULTS ===")
print(f"{'L [m]':>8} {'Pass':>6} {'Mean':>8} {'p95':>8} {'A_err':>8} {'B_err':>8} {'Grnd':>6} {'Chord':>8} {'Xspan':>8} {'Yspan':>8}")
for r in results:
    print(f"{r['L']:8.2f} {'YES' if r['passed'] else 'NO':>6} {r['mean']:8.4f} {r['p95']:8.4f} "
          f"{r['A_err']:8.4f} {r['B_err']:8.4f} {r['ground_frac']:6.2f} "
          f"{r['chord']:8.3f} {r['span_x']:8.3f} {r['span_y']:8.3f}")

passed = [r for r in results if r['passed']]
if passed:
    best = max(passed, key=lambda r: r['L'])
    print(f"\nMAX SAFE ARC: L={best['L']:.2f}m one-way, chord={best['chord']:.3f}m, "
          f"cycle={2*best['L']:.2f}m, p95={best['p95']:.4f}m")
else:
    print("\nNO CANDIDATE PASSED")

sim_app.close()
