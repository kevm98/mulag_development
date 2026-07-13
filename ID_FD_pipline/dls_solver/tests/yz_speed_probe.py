#!/usr/bin/env python3
"""Measure the arm's MAX achievable EE base-Y/Z speed through the full
ID/FD + cylinder-DLS pipeline.

Trajectory speed must be set below this, or the EE can never follow the path
(symptom: follow_ratio < 1, i.e. the EE travels less path than the goal).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch  # noqa: E402
torch.manual_seed(a.seed)
import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSController  # noqa: E402
from ID_FD_pipline.dls_solver.ee_cascade import EECascadeController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 300.0
cfg.low_level_servo_mode = "direct_position_delta"
cfg.direct_position_delta_clip_m = 0.0015
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset(seed=int(a.seed))
zero = torch.zeros((1, 4), device=env.device)
for _ in range(30):
    env.step(zero)

step_dt = float(env.step_dt)
controller = CylinderDLSController(env)
cascade = EECascadeController(env, cylinder_controller=controller)
cascade.reset()
anchor = env._compute_ee_pos_local().detach().clone()

print("\n========== MAX EE Y/Z SPEED (through full pipeline) ==========", flush=True)
results = {}
for name, direction in [("+Y", (1.0, 0.0)), ("-Y", (-1.0, 0.0)), ("+Z", (0.0, 1.0)), ("-Z", (0.0, -1.0))]:
    env.reset(seed=int(a.seed))
    for _ in range(30):
        env.step(zero)
    cascade.reset()
    anchor = env._compute_ee_pos_local().detach().clone()
    # far goal in the commanded direction (beyond reach) -> saturates the arm
    goal = anchor.clone()
    goal[:, 1] += 0.60 * direction[0]
    goal[:, 2] += 0.60 * direction[1]
    ee_hist = []
    for _ in range(int(a.steps)):
        cg = cascade.cylinder_goal(goal)
        controller.step(cg)
        ee_hist.append(env._compute_ee_pos_local()[0, 1:3].detach().cpu().numpy().copy())
    E = np.asarray(ee_hist)
    v = np.linalg.norm(np.diff(E, axis=0), axis=1) / step_dt
    # steady max speed = 90th pct of the transient (ignore the very first ramp)
    vmax = float(np.percentile(v[5:], 90))
    travel = float(np.linalg.norm(E[-1] - E[0]))
    results[name] = vmax
    print(f"[SPD] {name}: peak90 EE speed = {vmax:.4f} m/s   travel over "
          f"{a.steps} steps = {travel:.3f} m", flush=True)

vmin = min(results.values())
print(f"\n[SPD] SLOWEST direction max speed = {vmin:.4f} m/s", flush=True)
print(f"[SPD] RECOMMENDED path speed (60% of slowest) = {0.6 * vmin:.4f} m/s", flush=True)
print("==============================================================\n", flush=True)
env.close(); sim_app.close()
