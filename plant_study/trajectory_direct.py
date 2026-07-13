#!/usr/bin/env python3
"""EE line/curve/sweep tracking with ID/FD BYPASSED (direct PhysX cylinder drive)
+ non-invasive pose-adaptive DLS using the native PhysX Jacobian.

This is the ACHIEVABLE CEILING. Whatever error we get here is the best the arm
can do; anything worse in the real pipeline is the ID/FD's fault, and the gap is
the spec those models must close.
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
p.add_argument("--steps", type=int, default=600)
p.add_argument("--warmup", type=int, default=60)
p.add_argument("--speed", type=float, default=0.05)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch  # noqa: E402
torch.manual_seed(a.seed)
import ID_FD_pipline.tasks  # noqa: E402,F401
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402
from ID_FD_pipline.dls_solver import ee_yz_jacobian  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 1e6
cfg.ee_trajectory_speed_mps = a.speed
cfg.ee_trajectory_length_m = 0.24
cfg.ee_trajectory_radius_m = 0.10
cfg.ee_trajectory_sweep_amp_m = 0.15
cfg.ee_trajectory_sweep_wavelength_m = 0.30
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset(seed=int(a.seed))
drv = DirectCylinderDriver(env)
step_dt = float(env.step_dt)


@torch.no_grad()
def dls_pinv(J, ratio=0.03):
    u, s, vh = torch.linalg.svd(J, full_matrices=False)
    d = ratio * s[0]
    return vh.T @ torch.diag(s / (s.square() + d.square())) @ u.T


print("\n=========== EE TRAJECTORY — ID/FD BYPASSED (ACHIEVABLE CEILING) ===========", flush=True)
results = {}
for mode in ("line", "curve", "sweep"):
    env.reset(seed=int(a.seed))
    drv.settle(80)
    anchor = drv.ee_local().detach().clone()
    env._traj_mode = mode
    env._traj_anchor[:] = anchor
    env._traj_u[:] = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    env._traj_v[:] = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    env.episode_length_buf.zero_()

    J = ee_yz_jacobian(env)[0]
    pinv = dls_pinv(J)
    errs, G, E = [], [], []
    for step in range(int(a.steps)):
        env.episode_length_buf += 1
        env._update_ee_trajectory_goal()
        goal = env._ee_goal_local[:, 1:3].detach().clone()
        ee_before = drv.ee_local()
        if step % 5 == 0:
            J = ee_yz_jacobian(env)[0]
            pinv = dls_pinv(J)
        error = goal - ee_before[:, 1:3]
        dx = torch.clamp(0.6 * (error @ pinv.T), -0.01, 0.01)
        x_target = torch.clamp(drv.measured_x() + dx, drv.lo, drv.hi)
        drv.step(x_target)
        ee = drv.ee_local()
        errs.append(float(torch.linalg.norm(env._ee_goal_local[0, 1:3] - ee[0, 1:3]).item()))
        G.append(env._ee_goal_local[0, 1:3].cpu().numpy().copy())
        E.append(ee[0, 1:3].cpu().numpy().copy())

    s = np.asarray(errs[a.warmup:]); G = np.asarray(G); E = np.asarray(E)
    gp = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    gs = float(np.linalg.norm(G.max(0) - G.min(0)))
    es = float(np.linalg.norm(E.max(0) - E.min(0)))
    mean, p95 = float(s.mean()), float(np.percentile(s, 95))
    ok = p95 < 0.05 and gp > 0.25 and mean < 0.25 * gs
    results[mode] = mean
    print(f"[CEIL] {mode:>5}: mean={mean:.4f} p95={p95:.4f} m | goal_path={gp:.3f} "
          f"goal_span={gs:.3f} ee_span={es:.3f} err/span={mean/max(gs,1e-9):.2f} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)

print(f"\n[CEIL] This is the best the arm can do with a PERFECT actuator.", flush=True)
print(f"[CEIL] The real ID/FD pipeline must match these numbers.", flush=True)
print("===========================================================================\n", flush=True)
env.close(); sim_app.close()
