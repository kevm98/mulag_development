#!/usr/bin/env python3
"""Characterize the REAL plant with ID/FD bypassed (direct PhysX cylinder drive).

Answers, per cylinder and for the EE:
  1. STEP RESPONSE to a 0.10 m cylinder setpoint  -> can the plant even get there?
  2. MAX cylinder speed (both directions)          -> the true rate limit
  3. Per-step delta actually realised              -> THE SPEC THE FD MUST DELIVER
  4. EE trajectory tracking with exact IK          -> the achievable ceiling

Run:
  ./rrlab.sh -p plant_study/characterize_plant.py --headless
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--setpoint_m", type=float, default=0.10)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch  # noqa: E402
torch.manual_seed(a.seed)
import ID_FD_pipline.tasks  # noqa: E402,F401
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 1e6
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset(seed=int(a.seed))

drv = DirectCylinderDriver(env)
drv.settle(80)
step_dt = float(env.step_dt)
x0 = drv.measured_x().clone()
print("\n================ PLANT CHARACTERIZATION (ID/FD BYPASSED) ================", flush=True)
print(f"[P] settled cylinder x0 = {np.round(x0[0].cpu().numpy(), 4).tolist()}", flush=True)
print(f"[P] step_dt = {step_dt:.5f} s\n", flush=True)

# ---------- 1+2+3: per-cylinder step response, both directions ----------
print("--- STEP RESPONSE: cylinder target = x0 +/- setpoint, ID/FD bypassed ---", flush=True)
print(f"{'cyl':>3} {'dir':>4} {'reached_m':>10} {'of_goal':>8} {'t63_s':>7} "
      f"{'v_max_m/s':>10} {'mean_step_delta_mm':>19}", flush=True)
spec = []
for c in range(4):
    for sgn, tag in ((+1.0, "+"), (-1.0, "-")):
        env.reset(seed=int(a.seed)); drv.settle(80)
        base = drv.measured_x().clone()
        goal = base.clone(); goal[:, c] += sgn * a.setpoint_m
        hist = []
        for _ in range(int(a.steps)):
            drv.step(goal)
            hist.append(float(drv.measured_x()[0, c].item()))
        h = np.asarray(hist) - float(base[0, c].item())
        target = sgn * a.setpoint_m
        reached = h[-1]
        frac = reached / target
        v = np.abs(np.diff(h)) / step_dt
        vmax = float(np.percentile(v, 95))
        # time to 63% of the achieved value (first-order time constant)
        t63 = None
        thr = 0.63 * reached
        idx = np.where((h >= thr) if target > 0 else (h <= thr))[0]
        if len(idx):
            t63 = float(idx[0] * step_dt)
        mean_step = float(np.mean(np.abs(np.diff(h)))) * 1000.0
        spec.append((abs(reached), vmax, mean_step))
        print(f"{c:>3} {tag:>4} {reached:>+10.4f} {frac:>7.1%} "
              f"{(t63 if t63 is not None else float('nan')):>7.2f} {vmax:>10.4f} {mean_step:>19.3f}", flush=True)

reach = [s[0] for s in spec]; vmaxs = [s[1] for s in spec]; deltas = [s[2] for s in spec]
print(f"\n[P] reach: min={min(reach):.4f} max={max(reach):.4f} m  (goal {a.setpoint_m:.3f} m)", flush=True)
print(f"[P] cylinder v_max: min={min(vmaxs):.4f} max={max(vmaxs):.4f} m/s", flush=True)

# ---------- 4: what the FD must deliver ----------
print("\n================ SPEC FOR THE ID/FD MODELS ================", flush=True)
print("The FD's job is to produce, per control step, the cylinder delta that the", flush=True)
print("PhysX plant demonstrably realises above. Required per-step delta magnitude:", flush=True)
print(f"  median realised |delta| per step = {np.median(deltas):.3f} mm", flush=True)
print(f"  peak    realised |delta| per step = {max(deltas):.3f} mm", flush=True)
print(f"  => FD delta_extension must reach ~{max(deltas):.1f} mm/step "
      f"({max(vmaxs):.3f} m/s) at full valve, in BOTH directions, for all 4 cylinders.", flush=True)
print("Compare this against the FD's actual residual valve effect to see the gap.", flush=True)
print("===========================================================\n", flush=True)
env.close(); sim_app.close()
